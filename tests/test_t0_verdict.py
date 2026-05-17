"""T0 verdict — freeze-aware per-role grad check (B06).

The T0 environment gate exists to catch "the freeze strategy did not
accidentally disconnect a subgraph". Pre-fix the verdict only failed
when ALL three role buckets reported a zero grad norm — but a real
freeze regression typically severs ONE subgraph (e.g., projector grads
vanish under a fragile patcher) while the other two roles still
receive grads, so the gate silently passed.

These tests pin down the post-fix contract: every role that has any
trainable parameter must receive a non-zero gradient. Roles whose
parameters are intentionally frozen are exempt.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adversarial_reasoning_training.gates.T0_env import (
    _evaluate_t0_verdict,
    _RoleGradNorms,
    _trainable_roles,
)


class _ThreeRoleStub(nn.Module):
    """Tiny model whose param names match vit / projector / lm patterns."""

    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = nn.Linear(4, 4)
        self.mm_projector = nn.Linear(4, 4)
        self.language_model = nn.Linear(4, 4)


def test_evaluate_verdict_fails_when_trainable_role_has_zero_grad() -> None:
    """All three roles trainable, projector grad accidentally zero →
    pre-fix passed (only `all_zero` triggers fail). Post-fix must fail
    with a note naming the disconnected role."""
    grads = _RoleGradNorms(vit=1.0, projector=0.0, lm=1.0)
    loss = torch.tensor(0.5)
    passed, notes = _evaluate_t0_verdict(
        loss_total=loss,
        grad_norms=grads,
        peak_gb=1.0,
        peak_memory_limit_gb=120.0,
        trainable_roles=frozenset({"vit", "projector", "lm"}),
    )
    assert passed is False
    assert any("projector" in n for n in notes), notes


def test_evaluate_verdict_passes_when_only_frozen_role_has_zero_grad() -> None:
    """Strategy=vit_only freezes the vision backbone. ViT grad=0 is the
    correct outcome; verdict must still pass when projector+lm have grads."""
    grads = _RoleGradNorms(vit=0.0, projector=1.0, lm=1.0)
    loss = torch.tensor(0.5)
    passed, notes = _evaluate_t0_verdict(
        loss_total=loss,
        grad_norms=grads,
        peak_gb=1.0,
        peak_memory_limit_gb=120.0,
        trainable_roles=frozenset({"projector", "lm"}),
    )
    assert passed is True, notes
    assert notes == []


def test_evaluate_verdict_fails_when_all_zero_legacy_path() -> None:
    """Behaviour preservation: the pre-fix `all_zero` failure mode is
    a strict subset of the new per-role check — when no role has grads
    AND every role was supposed to be trainable, the verdict must fail."""
    grads = _RoleGradNorms(vit=0.0, projector=0.0, lm=0.0)
    loss = torch.tensor(0.5)
    passed, notes = _evaluate_t0_verdict(
        loss_total=loss,
        grad_norms=grads,
        peak_gb=1.0,
        peak_memory_limit_gb=120.0,
        trainable_roles=frozenset({"vit", "projector", "lm"}),
    )
    assert passed is False
    assert notes  # at least one note explaining the failure


def test_trainable_roles_picks_up_unfrozen_buckets() -> None:
    """Helper used by `_finalize_t0` — must return exactly the role
    names whose pattern matches at least one trainable param."""
    model = _ThreeRoleStub()
    for n, p in model.named_parameters():
        if "vision_tower" in n:
            p.requires_grad_(False)
    roles = _trainable_roles(model)
    assert roles == frozenset({"projector", "lm"})


def test_trainable_roles_empty_when_fully_frozen() -> None:
    model = _ThreeRoleStub()
    for p in model.parameters():
        p.requires_grad_(False)
    assert _trainable_roles(model) == frozenset()
