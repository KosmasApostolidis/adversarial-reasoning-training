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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset

from ..attacks.inner_pgd import InnerPgdConfig, epsilon_for_epoch, run_inner_pgd
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
    grad_clip_norm: float = 1.0
    amp_dtype: str = "bf16"  # bf16 | fp16 | fp32
    eps_schedule: list[dict[str, Any]] | None = None
    default_epsilon: float = 4.0 / 255.0
    alpha_ratio: float = 0.25
    pgd_steps: int = 7
    run_dir: Path = Path("runs/default")


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

        with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype()):
            logits_clean = self._forward_logits(batch, pixel_values)
            logits_adv = self._forward_logits(batch, x_adv)
            loss_out = self.loss_fn(
                logits_clean, logits_adv,
                batch.input_ids, batch.task_mask, batch.traj_mask,
            )

        diagnostics = {
            "attack_loss_final": float(attack_result.loss_final),
            "attack_iterations": int(attack_result.iterations),
            "epsilon": epsilon,
        }
        return loss_out, diagnostics

    # --- main loop --------------------------------------------------------

    def fit(self, dataset: Dataset) -> None:
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=self.collator)
        total_outer = len(loader) * self.config.epochs
        global_step = 0
        accum_loss_acc = 0.0
        start_time = time.time()
        reset_peak_memory()

        for epoch in range(1, self.config.epochs + 1):
            epsilon = epsilon_for_epoch(
                epoch, self.config.eps_schedule, default_eps=self.config.default_epsilon,
            )
            for micro_idx, batch in enumerate(loader):
                batch_t: TeacherForcedBatch = batch.to(self.device)
                loss_out, diag = self._outer_step(batch_t, epsilon)
                (loss_out.total / self.config.grad_accum).backward()
                accum_loss_acc += float(loss_out.total.detach())

                do_step = ((micro_idx + 1) % self.config.grad_accum) == 0
                if do_step:
                    if self.config.grad_clip_norm and self.config.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            (p for p in self.model.parameters() if p.requires_grad),
                            max_norm=self.config.grad_clip_norm,
                        )
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    avg_loss = accum_loss_acc / self.config.grad_accum
                    accum_loss_acc = 0.0

                    if global_step % self.config.log_every == 0:
                        mem = current_memory_stats()
                        self._log({
                            "event": "train_step",
                            "global_step": global_step,
                            "epoch": epoch,
                            "avg_loss": avg_loss,
                            "wall_s": time.time() - start_time,
                            **loss_out.components,
                            **diag,
                            **mem.as_dict(),
                        })

                    if self.config.eval_every > 0 and global_step % self.config.eval_every == 0:
                        metrics = (
                            self.evaluator(global_step, epoch) if self.evaluator else {}
                        )
                        metric_value = metrics.get(self.metric_for_best) if metrics else None
                        self._log({
                            "event": "eval",
                            "global_step": global_step,
                            "epoch": epoch,
                            "metrics": metrics,
                        })
                        self.ckpt.save(
                            model=self.model,
                            optimizer=self.optimizer,
                            step=global_step,
                            epoch=epoch,
                            metric_value=metric_value,
                            extra={"metrics": metrics},
                        )

        # --- final checkpoint + end-of-training evaluation
        metrics = self.evaluator(global_step, self.config.epochs) if self.evaluator else {}
        metric_value = metrics.get(self.metric_for_best) if metrics else None
        self.ckpt.save(
            model=self.model,
            optimizer=self.optimizer,
            step=global_step,
            epoch=self.config.epochs,
            metric_value=metric_value,
            extra={"metrics": metrics, "final": True},
        )
        self._log({
            "event": "fit_done",
            "global_step": global_step,
            "wall_s": time.time() - start_time,
            "metrics": metrics,
            "total_outer": total_outer,
        })
