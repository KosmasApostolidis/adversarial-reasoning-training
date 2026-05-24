"""Outer adversarial training loop.

Per outer step:
  1. Inner PGD crafts `x_adv` that maximises teacher-forced CE on the
     gold trajectory (attacks-repo `PGDAttack`, used as-is).
  2. Clean forward + adv forward over the full teacher-forced sequence.
  3. Loss = selector(TRADES | PGD-AT | OAAT) over (logits_clean,
     logits_adv, input_ids, task_mask, traj_mask).
  4. Backprop, gradient-accumulate, step optimizer + scheduler.

Periodic checkpoint + optional dev evaluation between epochs.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ..attacks.inner_pgd import (
    InnerPgdConfig,
    epsilon_for_epoch,
    run_inner_pgd,
    validate_eps_schedule,
)
from ..data.collator import TFCollator
from ..losses.selector import LossCallResult
from ..trajectory.teacher_force import TeacherForcedBatch
from ..utils.constants import DEFAULT_PGD_ALPHA_RATIO, EPS_4_255
from ..utils.mem import current_memory_stats, reset_peak_memory
from .ckpt import CheckpointRegistry


@dataclass
class TrainerConfig:
    epochs: int = 5
    grad_accum: int = 8
    log_every: int = 20
    eval_every: int = 200
    save_every: int = 0  # 0 disables; >0 saves ckpt every N global_steps
    grad_clip_norm: float = 1.0
    amp_dtype: str = "bf16"  # bf16 | fp16 | fp32
    eps_schedule: list[dict[str, Any]] | None = None
    default_epsilon: float = EPS_4_255
    alpha_ratio: float = DEFAULT_PGD_ALPHA_RATIO
    pgd_steps: int = 20  # match canonical configs/defenses.yaml
    pgd_random_restarts: int = 1
    pgd_attack_mode: str = "pgd"  # pgd | apgd
    pgd_momentum: float = 0.75
    pgd_rho: float = 0.75
    # Seed for the train DataLoader RNG (controls shuffle order). Threaded
    # from ``--seed`` so a re-run with the same seed observes identical
    # mini-batch ordering. Without this, even a fully seeded torch / numpy
    # / random still ships a non-deterministic DataLoader because torch
    # falls back to a fresh entropy source.
    loader_seed: int = 0
    run_dir: Path = Path("runs/default")
    final_save_include_optimizer: bool = False  # match canonical training.yaml: weights-only save


class AdvTrainer:
    """Adversarial reasoning trainer. Single-GPU, micro-batch 1 + accum.

    Parameters
    ----------
    vlm : attacks-repo VLMBase-compatible object with `forward_with_logits`.
    model : the underlying `torch.nn.Module` whose params we optimise.
    collator : TFCollator producing TeacherForcedBatch.
    loss_fn : closure from `losses.selector.build_loss`.
    optimizer, scheduler : torch optim + LR scheduler.
    config : TrainerConfig.
    device : torch device.
    evaluator : optional callable (global_step, epoch) -> metrics dict.
        Called on `eval_every` cadence + at training end.
    metric_for_best : key in evaluator output used to pick best ckpt.
    """

    def __init__(
        self,
        *,
        vlm: Any,
        model: torch.nn.Module,
        collator: TFCollator,
        loss_fn: Callable[..., LossCallResult],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        config: TrainerConfig,
        device: torch.device | str = "cuda",
        evaluator: Callable[[int, int], dict[str, float]] | None = None,
        metric_for_best: str = "tool_name_acc",
    ) -> None:
        self.vlm = vlm
        self.model = model
        self.collator = collator
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = torch.device(device)
        self.evaluator = evaluator
        self.metric_for_best = metric_for_best

        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt = CheckpointRegistry(self.config.run_dir / "ckpt")
        self.log_path = self.config.run_dir / "train_log.jsonl"
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=(config.amp_dtype == "fp16")
        )
        # Surface malformed eps_schedule entries before any training starts —
        # a typo would otherwise crash mid-epoch and lose progress.
        validate_eps_schedule(self.config.eps_schedule, n_epochs=self.config.epochs)

        # Per-fit mutable state. ``accum_count`` tracks valid (non-NaN)
        # micro-batches in the current grad-accum window; partial tails
        # are flushed by an epoch-end drain. NaN-skip resets to zero so
        # the bad window restarts cleanly rather than stepping on a
        # sub-window of valid grads (which would silently scale the
        # gradient by post_skip / grad_accum and mis-report ``avg_loss``).
        self._global_step = 0
        self._accum_loss_acc = 0.0
        self._accum_count = 0

    # --- utilities --------------------------------------------------------

    def _amp_dtype(self) -> torch.dtype:
        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.config.amp_dtype]

    def _append_log_record(self, record: dict[str, Any]) -> None:
        """Append one JSON-line record to ``self.log_path``."""
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- forward helpers --------------------------------------------------

    def _forward_logits(
        self,
        batch: TeacherForcedBatch,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Call vlm.forward_with_logits with full teacher-forced sequence.

        `pixel_values` is passed positionally as the image tensor; all
        other forward_kwargs (image_grid_thw, attention_mask) are spread
        via **kwargs with the original `pixel_values` stripped to avoid
        double-passing.
        """
        fwd_kwargs = {
            k: v
            for k, v in batch.forward_kwargs.items()
            if k != "pixel_values" and v is not None
        }
        return self.vlm.forward_with_logits(pixel_values, batch.input_ids, **fwd_kwargs)

    # --- main step --------------------------------------------------------

    def _outer_step(
        self,
        batch: TeacherForcedBatch,
        epsilon: float,
    ) -> tuple[LossCallResult, dict[str, float]]:
        inner_cfg = InnerPgdConfig(
            epsilon=epsilon,
            alpha_ratio=self.config.alpha_ratio,
            steps=self.config.pgd_steps,
            random_restarts=self.config.pgd_random_restarts,
            attack_mode=self.config.pgd_attack_mode,
            momentum=self.config.pgd_momentum,
            rho=self.config.pgd_rho,
        )
        pixel_values = batch.forward_kwargs["pixel_values"].to(
            self.device, dtype=next(self.model.parameters()).dtype
        )

        # Switch to eval mode for the PGD craft so dropout / BN do not perturb
        # the deterministic forward the attacker needs, then restore the prior
        # mode before the OUTER forward+backward. Pre-fix this never restored,
        # so the entire adv-FT run executed in eval mode (dropout disabled).
        was_training = self.model.training
        self.model.train(False)
        try:
            attack_result = run_inner_pgd(self.vlm, pixel_values, batch, inner_cfg)
        finally:
            self.model.train(was_training)
        x_adv = attack_result.perturbed_image.detach().to(self.device)
        if x_adv.ndim == 3:
            x_adv = x_adv.unsqueeze(0)

        # PGD divergence (non-finite loss) means run_inner_pgd substituted the
        # clean image — silently making this micro-batch a clean step. Surface
        # it as a distinct log event so post-hoc analysis can count how many
        # adversarial signals were lost during the run, and propagate the
        # flag into the train_step log via diagnostics.
        attack_loss_value = float(attack_result.loss_final)
        attack_diverged = math.isnan(attack_loss_value) or math.isinf(attack_loss_value)
        if attack_diverged:
            self._append_log_record({
                "event": "attack_diverged",
                "epsilon": epsilon,
                "attack_iterations": int(attack_result.iterations),
            })

        with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype()):
            logits_clean = self._forward_logits(batch, pixel_values)
            logits_adv = self._forward_logits(batch, x_adv)
            loss_out = self.loss_fn(
                logits_clean, logits_adv,
                batch.input_ids, batch.task_mask, batch.traj_mask,
            )

        diagnostics = {
            "attack_loss_final": attack_loss_value,
            "attack_iterations": int(attack_result.iterations),
            "attack_diverged": int(attack_diverged),
            "epsilon": epsilon,
        }
        return loss_out, diagnostics

    # --- optimizer step --------------------------------------------------

    def _clip_grads(self) -> torch.Tensor:
        """Clip trainable params per ``config.grad_clip_norm``.

        Returns the pre-clip total norm (or 0 when clipping is disabled).
        Caller must have unscaled the optimizer's grads first under fp16
        AMP — otherwise the clip would operate on scaled magnitudes.
        """
        if self.config.grad_clip_norm and self.config.grad_clip_norm > 0:
            return torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=self.config.grad_clip_norm,
            )
        return torch.tensor(0.0)

    def _handle_inf_grad(self, epoch: int, reason: str) -> None:
        """Log the inf-grad skip and reset accumulators.

        ``scaler.update`` must still run on this branch so the AMP
        loss-scale halves on the inf event; otherwise the scaler stays
        at the pre-skip scale and every subsequent micro-batch
        overflows the same way.
        """
        self._append_log_record({
            "event": "skipped_nan_grad",
            "global_step": self._global_step,
            "epoch": epoch,
            "grad_norm": float("nan"),
            "accum_count": self._accum_count,
            "reason": reason,
        })
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.update()
        self._accum_loss_acc = 0.0
        self._accum_count = 0

    def _log_train_step(
        self,
        *,
        epoch: int,
        avg_loss: float,
        steps_in_window: int,
        total_norm: torch.Tensor,
        reason: str,
        start_time: float,
        loss_out: LossCallResult,
        diag: dict[str, float],
    ) -> None:
        if self._global_step % self.config.log_every == 0:
            mem = current_memory_stats()
            self._append_log_record({
                "event": "train_step",
                "global_step": self._global_step,
                "epoch": epoch,
                "avg_loss": avg_loss,
                "accum_count": steps_in_window,
                "wall_s": time.time() - start_time,
                "grad_norm": float(total_norm),
                "step_reason": reason,
                **loss_out.components,
                **diag,
                **mem.as_dict(),
            })

    def _maybe_save_periodic(self, epoch: int) -> None:
        should_save = (
            self.config.save_every > 0
            and self._global_step % self.config.save_every == 0
        )
        if should_save:
            self.ckpt.save_weights_only(
                model=self.model,
                step=self._global_step,
                epoch=epoch,
                metric_value=None,
                extra={"reason": "save_every"},
            )

    def _maybe_eval_periodic(self, epoch: int) -> None:
        should_eval = (
            self.config.eval_every > 0
            and self._global_step % self.config.eval_every == 0
        )
        if should_eval:
            self._run_eval_and_save(epoch)

    def _run_eval_and_save(self, epoch: int) -> None:
        metrics = self.evaluator(self._global_step, epoch) if self.evaluator else {}
        metric_value = metrics.get(self.metric_for_best) if metrics else None
        self._append_log_record({
            "event": "eval",
            "global_step": self._global_step,
            "epoch": epoch,
            "metrics": metrics,
        })
        self.ckpt.save(
            model=self.model,
            optimizer=self.optimizer,
            step=self._global_step,
            epoch=epoch,
            metric_value=metric_value,
            extra={"metrics": metrics},
        )

    def _apply_optimizer_step(
        self,
        *,
        epoch: int,
        loss_out: LossCallResult,
        diag: dict[str, float],
        reason: str,
        start_time: float,
    ) -> None:
        """Clip → finite-check → step → log → ckpt. Shared between
        in-window trigger and end-of-epoch drain.
        """
        if self._accum_count == 0:
            return
        # Under fp16 AMP, gradients carry the loss-scale factor. PyTorch
        # contract: scale → backward → unscale_ → clip → step → update.
        # If scaler is disabled (bf16/fp32) unscale_ is a no-op so this
        # is safe to always call. Skipping unscale_ here would clip the
        # scaled magnitudes, silently miscalibrating ``grad_clip_norm``
        # by orders of magnitude under fp16.
        self.scaler.unscale_(self.optimizer)
        total_norm = self._clip_grads()
        if not torch.isfinite(total_norm):
            self._handle_inf_grad(epoch, reason)
            return

        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._global_step += 1
        avg_loss = self._accum_loss_acc / self._accum_count
        steps_in_window = self._accum_count
        self._accum_loss_acc = 0.0
        self._accum_count = 0

        self._log_train_step(
            epoch=epoch,
            avg_loss=avg_loss,
            steps_in_window=steps_in_window,
            total_norm=total_norm,
            reason=reason,
            start_time=start_time,
            loss_out=loss_out,
            diag=diag,
        )
        self._maybe_save_periodic(epoch)
        self._maybe_eval_periodic(epoch)

    # --- main loop --------------------------------------------------------

    def _init_beta_schedule(self) -> tuple[Any, float, float]:
        """Read TRADES beta annealing endpoints off ``loss_fn.config``.

        Returns ``(loss_cfg, beta_start, beta_end)``. ``loss_cfg`` is
        ``None`` when the loss closure exposes no config (gold/teacher-
        forced gates), in which case both endpoints fall back to 6.0
        — the historical default that matches the pre-refactor literal.
        """
        loss_cfg = getattr(self.loss_fn, "config", None)
        if loss_cfg is None:
            return None, 6.0, 6.0
        return loss_cfg, loss_cfg.beta, getattr(loss_cfg, "beta_end", loss_cfg.beta)

    def _anneal_beta(
        self,
        loss_cfg: Any,
        beta_start: float,
        beta_end: float,
        epoch: int,
    ) -> None:
        """Linear-interpolate ``loss_cfg.beta`` across the run.

        No-op when there is no config or only a single epoch (the
        ``epochs - 1`` denominator would be zero).
        """
        if loss_cfg is None or self.config.epochs <= 1:
            return
        frac = (epoch - 1) / (self.config.epochs - 1)
        loss_cfg.beta = beta_start + frac * (beta_end - beta_start)

    def _handle_nan_loss(
        self,
        *,
        epoch: int,
        micro_idx: int,
        diag: dict[str, float],
    ) -> None:
        """Drop grads + reset window after a NaN/Inf micro-batch.

        A single bad batch must not poison every subsequent step. Resetting
        the window counter forces the next valid micro-batches to form a
        full grad_accum window before stepping; otherwise the step would
        apply on a sub-window with mis-scaled gradient.
        """
        self._append_log_record({
            "event": "skipped_nan_loss",
            "global_step": self._global_step,
            "epoch": epoch,
            "micro_idx": micro_idx,
            "accum_count_before_skip": self._accum_count,
            **diag,
        })
        self.optimizer.zero_grad(set_to_none=True)
        self._accum_loss_acc = 0.0
        self._accum_count = 0
        self.scaler.update()

    def _drain_partial_window(
        self,
        *,
        epoch: int,
        last_loss_out: LossCallResult | None,
        last_diag: dict[str, float],
        start_time: float,
    ) -> None:
        """Apply trailing micro-batches before crossing the epoch boundary.

        If we skip the drain, their grads leak into the next epoch's first
        window and the final epoch's tail never updates the model at all.
        """
        if self._accum_count > 0 and last_loss_out is not None:
            self._apply_optimizer_step(
                epoch=epoch,
                loss_out=last_loss_out,
                diag=last_diag,
                reason="epoch_end_drain",
                start_time=start_time,
            )

    def _run_epoch(
        self,
        *,
        epoch: int,
        loader: DataLoader,
        epsilon: float,
        last_loss_out: LossCallResult | None,
        last_diag: dict[str, float],
        start_time: float,
    ) -> tuple[LossCallResult | None, dict[str, float]]:
        """Run one epoch: micro-batch loop + end-of-epoch drain.

        Returns the last (loss_out, diag) seen so the drain can fire on
        a partial trailing window without re-running ``_outer_step``.
        """
        for micro_idx, batch in enumerate(loader):
            batch_t: TeacherForcedBatch = batch.to(self.device)
            loss_out, diag = self._outer_step(batch_t, epsilon)

            if not torch.isfinite(loss_out.total):
                self._handle_nan_loss(epoch=epoch, micro_idx=micro_idx, diag=diag)
                continue

            self.scaler.scale(loss_out.total / self.config.grad_accum).backward()
            self._accum_loss_acc += float(loss_out.total.detach())
            self._accum_count += 1
            last_loss_out, last_diag = loss_out, diag

            if self._accum_count >= self.config.grad_accum:
                self._apply_optimizer_step(
                    epoch=epoch,
                    loss_out=loss_out,
                    diag=diag,
                    reason="window_full",
                    start_time=start_time,
                )

        self._drain_partial_window(
            epoch=epoch, last_loss_out=last_loss_out,
            last_diag=last_diag, start_time=start_time,
        )
        return last_loss_out, last_diag

    def _finalize_training(self, *, total_outer: int, start_time: float) -> None:
        """Final ckpt save + fit_done log + train_meta.json dump.

        Compute-transparency metadata: ``scripts/figures/compute_summary.py``
        walks each run dir for ``train_meta.json`` to render the H200-
        hours/peak-GiB LaTeX table. Keys mirror the gate JSON schema so
        the same reader works across T0/T1/T2/T3 + trainer outputs.
        """
        global_step = self._global_step
        try:
            metrics = self.evaluator(global_step, self.config.epochs) if self.evaluator else {}
        except Exception:
            logger.exception(
                "evaluator crashed at step=%d — proceeding with empty metrics "
                "to save final checkpoint", global_step,
            )
            metrics = {}
        metric_value = metrics.get(self.metric_for_best) if metrics else None
        if self.config.final_save_include_optimizer:
            self.ckpt.save(
                model=self.model,
                optimizer=self.optimizer,
                step=max(1, global_step),
                epoch=self.config.epochs,
                metric_value=metric_value,
                extra={"metrics": metrics, "final": True},
            )
        else:
            self.ckpt.save_weights_only(
                model=self.model,
                step=max(1, global_step),
                epoch=self.config.epochs,
                metric_value=metric_value,
                extra={"metrics": metrics, "final": True},
            )
        duration_s = time.time() - start_time
        final_mem = current_memory_stats()
        self._append_log_record({
            "event": "fit_done",
            "global_step": global_step,
            "wall_s": duration_s,
            "metrics": metrics,
            "total_outer": total_outer,
            **final_mem.as_dict(),
        })
        meta = {
            "duration_s": duration_s,
            "peak_memory_gb": final_mem.peak_allocated_gb,
            "peak_reserved_gb": final_mem.peak_reserved_gb,
            "global_step": global_step,
            "total_outer": total_outer,
            "epochs": self.config.epochs,
            "device": final_mem.device,
        }
        meta_path = self.config.run_dir / "train_meta.json"
        try:
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        except OSError:
            logger.warning(
                "Failed to write train_meta.json to %s — disk full?",
                meta_path,
            )

    def fit(self, dataset: Dataset) -> None:
        loader_gen = torch.Generator()
        loader_gen.manual_seed(int(self.config.loader_seed))
        loader = DataLoader(
            dataset, batch_size=1, shuffle=True,
            collate_fn=self.collator, generator=loader_gen,
        )
        total_outer = len(loader) * self.config.epochs
        self._global_step = 0
        self._accum_loss_acc = 0.0
        self._accum_count = 0
        start_time = time.time()
        reset_peak_memory()

        last_loss_out: LossCallResult | None = None
        last_diag: dict[str, float] = {}
        loss_cfg, beta_start, beta_end = self._init_beta_schedule()

        for epoch in range(1, self.config.epochs + 1):
            self._anneal_beta(loss_cfg, beta_start, beta_end, epoch)
            epsilon = epsilon_for_epoch(
                epoch, self.config.eps_schedule, default_eps=self.config.default_epsilon,
            )
            last_loss_out, last_diag = self._run_epoch(
                epoch=epoch,
                loader=loader,
                epsilon=epsilon,
                last_loss_out=last_loss_out,
                last_diag=last_diag,
                start_time=start_time,
            )

        self._finalize_training(total_outer=total_outer, start_time=start_time)
