"""PGD-AT (Madry 2018): supervised CE computed on adversarial inputs only."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .task_ce import task_ce


@dataclass(frozen=True)
class PgdAtOutput:
    total: torch.Tensor
    task_adv: torch.Tensor


def pgd_at_loss(
    logits_adv: torch.Tensor,
    input_ids: torch.Tensor,
    task_mask: torch.Tensor,
) -> PgdAtOutput:
    task = task_ce(logits_adv, input_ids, task_mask)
    return PgdAtOutput(total=task, task_adv=task)
