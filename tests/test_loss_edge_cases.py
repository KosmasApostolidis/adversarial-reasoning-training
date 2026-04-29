"""Regression tests for loss-math edge cases.

* **H2** — ``task_ce`` / ``traj_kl`` used to clamp the divisor to
  ``min=1.0`` so a fully-masked batch silently returned ``0.0``,
  making bug-shaped data look like trivially-easy data. The trainer's
  NaN-skip never fired because the result was finite. The fix lets
  the divisor reach zero so 0/0 → NaN and the trainer's finite-loss
  guard at ``adv_trainer.fit`` catches it.
* **H3** — TRADES used to call ``traj_kl(logits_clean, logits_adv)``
  with no ``detach`` on the natural side. Reference TRADES (Zhang et
  al., ICML 2019) detaches the natural logits inside the
  robustness-regularisation term. Without detach the gradient path
  through ``logits_clean`` adds a spurious "match adv" pull that
  competes with ``task_ce``'s "match labels" pull on the same
  parameters.
"""

from __future__ import annotations

import pytest
import torch

from adversarial_reasoning_training.losses.task_ce import task_ce
from adversarial_reasoning_training.losses.traj_kl import traj_kl
from adversarial_reasoning_training.losses.trades import trades_loss


@pytest.mark.unit
def test_task_ce_returns_nan_when_mask_is_all_zero() -> None:
    """H2: empty mask must surface as NaN (so trainer NaN-skip fires),
    not silently as 0.0 from a clamped denominator.
    """
    torch.manual_seed(0)
    B, T, V = 1, 6, 8
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    task_mask = torch.zeros(B, T)  # nothing to score → empty mask
    out = task_ce(logits, input_ids, task_mask)
    assert torch.isnan(out), (
        "task_ce on an all-zero mask must produce NaN so the trainer "
        "treats it as a degenerate batch and skips with a logged "
        "event, not silently report 0.0 loss"
    )


@pytest.mark.unit
def test_traj_kl_returns_nan_when_mask_is_all_zero() -> None:
    """H2 sibling for traj_kl."""
    torch.manual_seed(0)
    B, T, V = 1, 6, 8
    a = torch.randn(B, T, V)
    b = torch.randn(B, T, V)
    mask = torch.zeros(B, T)
    out = traj_kl(a, b, mask)
    assert torch.isnan(out)


@pytest.mark.unit
def test_task_ce_finite_with_partial_mask() -> None:
    """Sanity: any non-zero mask still produces finite loss."""
    torch.manual_seed(0)
    B, T, V = 1, 6, 8
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    mask = torch.tensor([[0, 1, 0, 1, 0, 0]], dtype=torch.float32)
    out = task_ce(logits, input_ids, mask)
    assert torch.isfinite(out)
    assert out > 0.0


@pytest.mark.unit
def test_trades_kl_does_not_backprop_into_clean_logits() -> None:
    """H3: TRADES' KL term must NOT update the clean logits via
    backprop. Reference implementation detaches the natural side so
    only the adv branch contributes gradients to the robustness
    regulariser.
    """
    torch.manual_seed(0)
    B, T, V = 1, 6, 8
    logits_clean = torch.randn(B, T, V, requires_grad=True)
    logits_adv = torch.randn(B, T, V, requires_grad=True)
    input_ids = torch.randint(0, V, (B, T))
    task_mask = torch.tensor([[0, 1, 1, 1, 0, 0]], dtype=torch.float32)
    traj_mask = torch.tensor([[0, 1, 1, 1, 1, 0]], dtype=torch.float32)

    out = trades_loss(
        logits_clean,
        logits_adv,
        input_ids,
        task_mask,
        traj_mask,
        beta=4.0,
        temperature=2.0,
    )
    # Backprop ONLY the KL term so we can isolate its gradient flow.
    out.kl.backward()

    # Clean logits MUST have zero gradient from the KL term — only the
    # task_ce term in TRADES (which we did not call backward on) can
    # legitimately push gradients into clean logits.
    assert logits_clean.grad is None or torch.allclose(
        logits_clean.grad, torch.zeros_like(logits_clean.grad)
    ), (
        "TRADES KL must detach the clean logits; otherwise the "
        "robustness regulariser puts a spurious gradient on the "
        "natural-image branch and competes with the task-CE pull."
    )
    # Adv logits MUST receive non-zero gradient (KL is the whole point).
    assert logits_adv.grad is not None
    assert logits_adv.grad.abs().sum().item() > 0.0
