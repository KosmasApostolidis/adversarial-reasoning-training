"""OAAT: one-shot adversarial augmentation with α-mixing against clean CE."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .task_ce import task_ce


@dataclass(frozen=True)
class OaatOutput:
    total: torch.Tensor
    task_clean: torch.Tensor
    task_adv: torch.Tensor
    alpha: float


def oaat_loss(
    logits_clean: torch.Tensor,
    logits_adv: torch.Tensor,
    input_ids: torch.Tensor,
    task_mask: torch.Tensor,
    *,
    alpha: float = 0.5,
) -> OaatOutput:
    """L = α · task_ce(clean) + (1-α) · task_ce(adv)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"OAAT alpha must lie in [0, 1]; got {alpha}")
    task_clean = task_ce(logits_clean, input_ids, task_mask)
    task_adv = task_ce(logits_adv, input_ids, task_mask)
    total = alpha * task_clean + (1.0 - alpha) * task_adv
    return OaatOutput(total=total, task_clean=task_clean, task_adv=task_adv, alpha=alpha)
