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
    default_epsilon: float = 4.0 / 255.0
    alpha_ratio: float = 0.25
    pgd_steps: int = 7
    run_dir: Path = Path("runs/default")
    final_save_include_optimizer: bool = True


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
        validate_eps_schedule(self.config.eps_schedule)

    # --- utilities --------------------------------------------------------

    def _amp_dtype(self) -> torch.dtype:
        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.config.amp_dtype]

    def _log(self, record: dict[str, Any]) -> None:
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
            random_restarts=1,
        )
        pixel_values = batch.forward_kwargs["pixel_values"].to(self.device)

        self.model.train(False)
        attack_result = run_inner_pgd(self.vlm, pixel_values, batch, inner_cfg)
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
            self._log({
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

    # --- main loop --------------------------------------------------------

    def fit(self, dataset: Dataset) -> None:
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=self.collator)
        total_outer = len(loader) * self.config.epochs
        # Mutable trainer-step state — wrapped in a dict so the
        # ``_apply_optimizer_step`` helper below can mutate it cleanly.
        state: dict[str, Any] = {
            "global_step": 0,
            "accum_loss_acc": 0.0,
            # accum_count tracks how many *valid* (non-NaN) micro-batches
            # have been backpropped into the current window. ``do_step``
            # waits for ``accum_count >= grad_accum``; partial tails at
            # epoch end are flushed by an explicit drain call. NaN-skip
            # zeros the counter so the bad window restarts cleanly
            # rather than stepping on a sub-window of valid grads
            # (which would silently scale the gradient by
            # post_skip / grad_accum and mis-report ``avg_loss``).
            "accum_count": 0,
        }
        start_time = time.time()
        reset_peak_memory()

        def _apply_optimizer_step(
            *,
            epoch: int,
            loss_out: LossCallResult,
            diag: dict[str, float],
            reason: str,
        ) -> None:
            """Clip → finite-check → step → log → ckpt. Shared between
            in-window trigger and end-of-epoch drain.
            """
            if state["accum_count"] == 0:
                return
            # Under fp16 AMP, gradients carry the loss-scale factor. PyTorch
            # contract: scale → backward → unscale_ → clip → step → update.
            # If scaler is disabled (bf16/fp32) unscale_ is a no-op so this
            # is safe to always call. Skipping unscale_ here would clip the
            # scaled magnitudes, silently miscalibrating ``grad_clip_norm``
            # by orders of magnitude under fp16.
            self.scaler.unscale_(self.optimizer)
            if self.config.grad_clip_norm and self.config.grad_clip_norm > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=self.config.grad_clip_norm,
                )
            else:
                total_norm = torch.tensor(0.0)

            if not torch.isfinite(total_norm):
                self._log({
                    "event": "skipped_nan_grad",
                    "global_step": state["global_step"],
                    "epoch": epoch,
                    "grad_norm": float("nan"),
                    "accum_count": state["accum_count"],
                    "reason": reason,
                })
                self.optimizer.zero_grad(set_to_none=True)
                # Even though we skip ``scaler.step``, ``update`` must run so
                # the loss-scale halves on this inf-grad event; otherwise the
                # scaler stays at the pre-skip scale and every subsequent
                # micro-batch overflows the same way.
                self.scaler.update()
                state["accum_loss_acc"] = 0.0
                state["accum_count"] = 0
                return

            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            state["global_step"] += 1
            avg_loss = state["accum_loss_acc"] / state["accum_count"]
            steps_in_window = state["accum_count"]
            state["accum_loss_acc"] = 0.0
            state["accum_count"] = 0

            if state["global_step"] % self.config.log_every == 0:
                mem = current_memory_stats()
                self._log({
                    "event": "train_step",
                    "global_step": state["global_step"],
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

            if self.config.save_every > 0 and state["global_step"] % self.config.save_every == 0:
                self.ckpt.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    step=state["global_step"],
                    epoch=epoch,
                    metric_value=None,
                    extra={"reason": "save_every"},
                    include_optimizer=False,
                )

            if self.config.eval_every > 0 and state["global_step"] % self.config.eval_every == 0:
                metrics = (
                    self.evaluator(state["global_step"], epoch) if self.evaluator else {}
                )
                metric_value = metrics.get(self.metric_for_best) if metrics else None
                self._log({
                    "event": "eval",
                    "global_step": state["global_step"],
                    "epoch": epoch,
                    "metrics": metrics,
                })
                self.ckpt.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    step=state["global_step"],
                    epoch=epoch,
                    metric_value=metric_value,
                    extra={"metrics": metrics},
                )

        last_loss_out: LossCallResult | None = None
        last_diag: dict[str, float] = {}

        loss_cfg = getattr(self.loss_fn, "config", None)
        beta_start: float = 6.0
        beta_end: float = 6.0
        if loss_cfg is not None:
            beta_start = loss_cfg.beta
            beta_end = getattr(loss_cfg, "beta_end", loss_cfg.beta)

        for epoch in range(1, self.config.epochs + 1):
            if loss_cfg is not None and self.config.epochs > 1:
                frac = (epoch - 1) / (self.config.epochs - 1)
                loss_cfg.beta = beta_start + frac * (beta_end - beta_start)

            epsilon = epsilon_for_epoch(
                epoch, self.config.eps_schedule, default_eps=self.config.default_epsilon,
            )
            for micro_idx, batch in enumerate(loader):
                batch_t: TeacherForcedBatch = batch.to(self.device)
                loss_out, diag = self._outer_step(batch_t, epsilon)

                # Finite-loss guard: a single bad batch must not poison every
                # subsequent step. Drop grads for this micro-batch AND reset
                # the window counter so the next valid micro-batches form a
                # full grad_accum window before stepping (otherwise the step
                # would apply on a sub-window with mis-scaled gradient).
                if not torch.isfinite(loss_out.total):
                    self._log({
                        "event": "skipped_nan_loss",
                        "global_step": state["global_step"],
                        "epoch": epoch,
                        "micro_idx": micro_idx,
                        "accum_count_before_skip": state["accum_count"],
                        **diag,
                    })
                    self.optimizer.zero_grad(set_to_none=True)
                    state["accum_loss_acc"] = 0.0
                    state["accum_count"] = 0
                    continue

                self.scaler.scale(loss_out.total / self.config.grad_accum).backward()
                state["accum_loss_acc"] += float(loss_out.total.detach())
                state["accum_count"] += 1
                last_loss_out, last_diag = loss_out, diag

                if state["accum_count"] >= self.config.grad_accum:
                    _apply_optimizer_step(
                        epoch=epoch,
                        loss_out=loss_out,
                        diag=diag,
                        reason="window_full",
                    )

            # End-of-epoch drain: trailing micro-batches in a partial
            # window must be applied before crossing the epoch boundary,
            # else their grads leak into the next epoch's first window
            # and the final epoch's tail never updates the model at all.
            if state["accum_count"] > 0 and last_loss_out is not None:
                _apply_optimizer_step(
                    epoch=epoch,
                    loss_out=last_loss_out,
                    diag=last_diag,
                    reason="epoch_end_drain",
                )

        global_step = state["global_step"]

        # --- final checkpoint + end-of-training evaluation
        metrics = self.evaluator(global_step, self.config.epochs) if self.evaluator else {}
        metric_value = metrics.get(self.metric_for_best) if metrics else None
        self.ckpt.save(
            model=self.model,
            optimizer=self.optimizer,
            step=max(1, global_step),
            epoch=self.config.epochs,
            metric_value=metric_value,
            extra={"metrics": metrics, "final": True},
            include_optimizer=self.config.final_save_include_optimizer,
        )
        duration_s = time.time() - start_time
        final_mem = current_memory_stats()
        self._log({
            "event": "fit_done",
            "global_step": global_step,
            "wall_s": duration_s,
            "metrics": metrics,
            "total_outer": total_outer,
            **final_mem.as_dict(),
        })
        # Compute-transparency metadata: scripts/figures/compute_summary.py
        # walks each run dir for this file to render the H200-hours/peak-GiB
        # LaTeX table. Keys mirror the gate JSON schema so the same reader
        # works across T0/T1/T2/T3 + trainer outputs.
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
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
