"""Unit tests for trainer/optim — factory + scheduler shape."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from adversarial_reasoning_training.trainer.optim import (
    OptimConfig,
    ScheduleConfig,
    build_optimizer,
    build_scheduler,
)


class _RoleStubModel(nn.Module):
    """Tiny stand-in whose param names span all three role buckets:
    vit / projector / lm. Names match `_VIT_PATTERNS`, `_PROJECTOR_PATTERNS`,
    `_LM_PATTERNS` substrings used by `param_groups_by_role`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = nn.Linear(4, 4)
        self.mm_projector = nn.Linear(4, 4)
        self.language_model = nn.Linear(4, 4)


def test_build_optimizer_adamw_returns_per_role_groups() -> None:
    model = _RoleStubModel()
    cfg = OptimConfig(kind="adamw", lr_lm=1.0e-5, lr_projector=2.0e-5, lr_vit=3.0e-6)
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    lrs = sorted(group["lr"] for group in opt.param_groups)
    assert lrs == sorted([1.0e-5, 2.0e-5, 3.0e-6])


def test_build_optimizer_unknown_kind_raises() -> None:
    model = _RoleStubModel()
    with pytest.raises(ValueError, match="Unknown optimizer kind"):
        build_optimizer(model, OptimConfig(kind="lion"))


def test_build_optimizer_empty_param_groups_raises() -> None:
    model = _RoleStubModel()
    for p in model.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="No trainable parameter groups"):
        build_optimizer(model, OptimConfig(kind="adamw"))


def test_build_optimizer_adamw8bit_branch() -> None:
    bnb = pytest.importorskip("bitsandbytes")
    model = _RoleStubModel()
    opt = build_optimizer(model, OptimConfig(kind="adamw8bit"))
    assert isinstance(opt, bnb.optim.AdamW8bit)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused AdamW needs CUDA")
def test_build_optimizer_adamw_fused_branch() -> None:
    model = _RoleStubModel().cuda()
    opt = build_optimizer(model, OptimConfig(kind="adamw_fused"))
    assert isinstance(opt, torch.optim.AdamW)
    # Confirm fused flag was wired through (param_groups carry it on PT 2.x).
    assert any(group.get("fused") for group in opt.param_groups)


def _step_lr(scheduler) -> float:
    return scheduler.get_last_lr()[0]


def test_build_scheduler_warmup_then_decay_cosine() -> None:
    model = _RoleStubModel()
    opt = build_optimizer(model, OptimConfig(kind="adamw"))
    base_lr = opt.param_groups[0]["lr"]
    sched = build_scheduler(opt, ScheduleConfig(total_steps=100, warmup_pct=0.1, kind="cosine"))
    # Step 0 → ~0 (just before first warmup step, scaled by lambda)
    assert _step_lr(sched) == pytest.approx(0.0, abs=1e-9)
    # Walk through warmup — at warmup_steps the lambda hits 1.0.
    for _ in range(10):
        opt.step()
        sched.step()
    assert _step_lr(sched) == pytest.approx(base_lr, rel=1e-6)
    # Walk to the very end and assert near-zero (cosine decays to 0 at progress=1).
    for _ in range(90):
        opt.step()
        sched.step()
    assert _step_lr(sched) == pytest.approx(0.0, abs=1e-9)


def test_build_scheduler_linear_decay_to_zero() -> None:
    model = _RoleStubModel()
    opt = build_optimizer(model, OptimConfig(kind="adamw"))
    sched = build_scheduler(opt, ScheduleConfig(total_steps=20, warmup_pct=0.0, kind="linear"))
    for _ in range(20):
        opt.step()
        sched.step()
    assert _step_lr(sched) == pytest.approx(0.0, abs=1e-9)


def test_build_scheduler_constant_holds_base_lr() -> None:
    model = _RoleStubModel()
    opt = build_optimizer(model, OptimConfig(kind="adamw"))
    base_lr = opt.param_groups[0]["lr"]
    sched = build_scheduler(opt, ScheduleConfig(total_steps=50, warmup_pct=0.0, kind="constant"))
    # After warmup (single step, since warmup_steps clamps to >=1), constant
    # path returns 1.0 → base_lr.
    for _ in range(10):
        opt.step()
        sched.step()
    assert _step_lr(sched) == pytest.approx(base_lr, rel=1e-6)


def test_build_scheduler_cosine_midpoint_is_half_base() -> None:
    """Cosine schedule with no warmup → at progress=0.5 the LR is base/2.

    Validates the trig wiring rather than just the endpoints.
    """
    model = _RoleStubModel()
    opt = build_optimizer(model, OptimConfig(kind="adamw"))
    base_lr = opt.param_groups[0]["lr"]
    sched = build_scheduler(opt, ScheduleConfig(total_steps=100, warmup_pct=0.01, kind="cosine"))
    for _ in range(50):
        opt.step()
        sched.step()
    assert _step_lr(sched) == pytest.approx(base_lr * 0.5, rel=0.05)
    _ = math  # silence unused warning if test trimmed in future


def test_build_scheduler_warmup_pct_zero_starts_at_base_lr() -> None:
    """B11: ``warmup_pct=0.0`` must actually disable warmup. Pre-fix
    ``warmup_steps = max(1, ...)`` forced a single zero-LR warmup step so
    the operator could never opt out of warmup. With the clamp removed,
    step 0 lambda returns the decay function at progress=0 → 1.0.
    """
    model = _RoleStubModel()
    opt = build_optimizer(model, OptimConfig(kind="adamw"))
    base_lr = opt.param_groups[0]["lr"]
    sched = build_scheduler(
        opt, ScheduleConfig(total_steps=100, warmup_pct=0.0, kind="constant")
    )
    # Step 0 effective lr must equal base_lr, not 0.
    assert _step_lr(sched) == pytest.approx(base_lr, rel=1e-6)


def test_build_scheduler_warmup_pct_nonzero_still_starts_at_zero() -> None:
    """Behaviour preservation: when warmup IS requested, step 0 lambda
    is ``step/warmup_steps == 0`` exactly as before — the B11 fix only
    affects the ``warmup_pct=0`` opt-out path."""
    model = _RoleStubModel()
    opt = build_optimizer(model, OptimConfig(kind="adamw"))
    sched = build_scheduler(
        opt, ScheduleConfig(total_steps=100, warmup_pct=0.1, kind="cosine")
    )
    assert _step_lr(sched) == pytest.approx(0.0, abs=1e-9)
