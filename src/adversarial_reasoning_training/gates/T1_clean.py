"""T1 — clean teacher-forced fine-tune convergence gate.

Runs a short clean-only FT loop (PGD disabled) on the gold oracle
trajectories and checks that tool-name accuracy + answer-EM reach the
configured thresholds on a dev split. This proves:

  * the data loader produces gradient-carrying batches.
  * the segment masks are correctly weighted (learning tools > thoughts).
  * the optimizer schedule isn't broken.
  * the oracle's trajectory templates are learnable at all.

Writes ``runs/<id>/gates/T1.json``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset

from ..data.collator import TFCollator
from ..losses.task_ce import task_ce


@dataclass
class T1Thresholds:
    tool_name_acc_min: float = 0.85
    answer_em_min: float = 0.70
    max_steps: int = 200


@dataclass
class T1Result:
    passed: bool
    tool_name_acc: float
    answer_em: float
    train_loss_final: float
    steps: int
    duration_s: float
    thresholds: dict[str, float]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_forward(vlm: Any, batch: Any, device: torch.device) -> torch.Tensor:
    fwd_kwargs = {
        k: v for k, v in batch.forward_kwargs.items()
        if k != "pixel_values" and v is not None
    }
    pixel_values = batch.forward_kwargs["pixel_values"].to(device)
    return vlm.forward_with_logits(pixel_values, batch.input_ids, **fwd_kwargs)


def run_t1(
    *,
    vlm: Any,
    model: torch.nn.Module,
    train_ds: Dataset,
    collator: TFCollator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    evaluator: Callable[[int, int], dict[str, float]],
    thresholds: T1Thresholds,
    out_path: Path,
    device: str = "cuda",
    amp_dtype: torch.dtype = torch.bfloat16,
    grad_accum: int = 8,
    tool_name_metric: str = "tool_name_acc",
    answer_em_metric: str = "answer_em",
) -> T1Result:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device)
    start = time.time()

    model.train(True)
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collator)
    step = 0
    micro = 0
    loss_final = float("nan")

    for batch in loader:
        if step >= thresholds.max_steps:
            break
        batch = batch.to(device_t)
        with torch.autocast(device_type=device_t.type, dtype=amp_dtype):
            logits = _clean_forward(vlm, batch, device_t)
            loss = task_ce(logits, batch.input_ids, batch.task_mask)

        (loss / grad_accum).backward()
        micro += 1
        if micro % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), max_norm=1.0,
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            loss_final = float(loss.detach())

    metrics = evaluator(step, 1) or {}
    tool_acc = float(metrics.get(tool_name_metric, 0.0))
    answer_em = float(metrics.get(answer_em_metric, 0.0))
    passed = (
        tool_acc >= thresholds.tool_name_acc_min
        and answer_em >= thresholds.answer_em_min
    )
    notes: list[str] = []
    if tool_acc < thresholds.tool_name_acc_min:
        notes.append(f"tool_name_acc {tool_acc:.3f} < {thresholds.tool_name_acc_min}")
    if answer_em < thresholds.answer_em_min:
        notes.append(f"answer_em {answer_em:.3f} < {thresholds.answer_em_min}")

    result = T1Result(
        passed=passed,
        tool_name_acc=tool_acc,
        answer_em=answer_em,
        train_loss_final=loss_final,
        steps=step,
        duration_s=time.time() - start,
        thresholds={
            "tool_name_acc_min": thresholds.tool_name_acc_min,
            "answer_em_min": thresholds.answer_em_min,
            "max_steps": thresholds.max_steps,
        },
        notes=notes,
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    return result
