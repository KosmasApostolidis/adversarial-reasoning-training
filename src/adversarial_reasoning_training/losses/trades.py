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
    task_weight: float = 1.0,
) -> TradesOutput:
    """L = task_weight · task_ce(clean) + β · traj_kl(stop_grad(clean) ‖ adv).

    The natural-image logits are detached inside the KL term so the
    robustness regulariser only updates the adversarial branch. This
    matches the reference TRADES (Zhang et al., ICML 2019) — without
    the detach, KL puts a spurious "match adv" gradient on the clean
    logits that competes with task_ce's "match labels" pull.

    `task_weight` defaults to 1.0 (canonical TRADES). Setting it to 0.0
    isolates the trajectory-KL signal — the "pure traj-KL" novelty
    ablation — by removing the clean-CE term entirely while keeping
    the inner-PGD adversary intact.
    """
    task = task_ce(logits_clean, input_ids, task_mask)
    kl = traj_kl(
        logits_clean.detach(), logits_adv, traj_mask, temperature=temperature,
    )
    total = task_weight * task + beta * kl
    return TradesOutput(total=total, task=task, kl=kl, beta=beta)
