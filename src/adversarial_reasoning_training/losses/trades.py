"""TRADES loss: supervised CE on clean + β·KL(clean ‖ adv)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .task_ce import task_ce
from .traj_kl import traj_kl


@dataclass(frozen=True)
class TradesOutput:
    total: torch.Tensor
    task: torch.Tensor
    kl: torch.Tensor
    beta: float


def trades_loss(
    logits_clean: torch.Tensor,
    logits_adv: torch.Tensor,
    input_ids: torch.Tensor,
    task_mask: torch.Tensor,
    traj_mask: torch.Tensor,
    *,
    beta: float = 6.0,
    temperature: float = 2.0,
) -> TradesOutput:
    """L = task_ce(clean) + β · traj_kl(clean ‖ adv)."""
    task = task_ce(logits_clean, input_ids, task_mask)
    kl = traj_kl(logits_clean, logits_adv, traj_mask, temperature=temperature)
    total = task + beta * kl
    return TradesOutput(total=total, task=task, kl=kl, beta=beta)
