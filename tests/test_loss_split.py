"""TRADES / PGD-AT / OAAT outputs match expected formulas on toy tensors."""

from __future__ import annotations

import torch

from adversarial_reasoning_training.losses.oaat import oaat_loss
from adversarial_reasoning_training.losses.pgd_at import pgd_at_loss
from adversarial_reasoning_training.losses.task_ce import task_ce
from adversarial_reasoning_training.losses.traj_kl import traj_kl
from adversarial_reasoning_training.losses.trades import trades_loss


def _toy() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    B, T, V = 1, 8, 32
    logits_clean = torch.randn(B, T, V, requires_grad=True)
    logits_adv = torch.randn(B, T, V, requires_grad=True)
    input_ids = torch.randint(0, V, (B, T))
    task_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0]], dtype=torch.float32)
    traj_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0]], dtype=torch.float32)
    return logits_clean, logits_adv, input_ids, task_mask, traj_mask


def test_pgd_at_equals_task_ce_on_adv() -> None:
    lc, la, ids, tm, _ = _toy()
    direct = task_ce(la, ids, tm)
    via_loss = pgd_at_loss(la, ids, tm)
    assert torch.allclose(direct, via_loss.total)


def test_oaat_is_convex_combo() -> None:
    lc, la, ids, tm, _ = _toy()
    alpha = 0.3
    expected = alpha * task_ce(lc, ids, tm) + (1 - alpha) * task_ce(la, ids, tm)
    out = oaat_loss(lc, la, ids, tm, alpha=alpha)
    assert torch.allclose(expected, out.total)


def test_trades_decomposes() -> None:
    lc, la, ids, tm, trm = _toy()
    beta = 4.0
    out = trades_loss(lc, la, ids, tm, trm, beta=beta, temperature=2.0)
    expected_total = task_ce(lc, ids, tm) + beta * traj_kl(lc, la, trm, temperature=2.0)
    assert torch.allclose(expected_total, out.total)
    assert out.beta == beta


def test_traj_kl_zero_when_logits_equal() -> None:
    lc, _, _, _, trm = _toy()
    kl = traj_kl(lc, lc.detach().clone(), trm, temperature=2.0)
    assert kl.item() == 0.0 or kl.item() < 1e-6
