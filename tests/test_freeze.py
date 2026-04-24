"""Freeze strategies must toggle requires_grad correctly per role."""

from __future__ import annotations

import torch

from adversarial_reasoning_training.trainer.freeze import (
    FreezeConfig,
    apply_freeze,
    param_groups_by_role,
)


class _DummyVLM(torch.nn.Module):
    """Module with named parameters that match the role patterns."""

    def __init__(self) -> None:
        super().__init__()
        self.visual = torch.nn.Linear(8, 8)            # vit
        self.mm_projector = torch.nn.Linear(8, 8)      # projector
        self.language_model = torch.nn.Linear(8, 8)    # lm


def test_freeze_none_keeps_everything_trainable() -> None:
    m = _DummyVLM()
    counts = apply_freeze(m, FreezeConfig(strategy="none"))
    assert counts["frozen"] == 0
    assert all(p.requires_grad for p in m.parameters())


def test_freeze_vit_only_freezes_only_vit() -> None:
    m = _DummyVLM()
    apply_freeze(m, FreezeConfig(strategy="vit_only"))
    assert not any(p.requires_grad for p in m.visual.parameters())
    assert all(p.requires_grad for p in m.mm_projector.parameters())
    assert all(p.requires_grad for p in m.language_model.parameters())


def test_param_groups_split_by_role() -> None:
    m = _DummyVLM()
    apply_freeze(m, FreezeConfig(strategy="none"))
    pg = param_groups_by_role(m, lr_lm=1.0, lr_projector=2.0, lr_vit=3.0)
    by_lr = {g["lr"]: g for g in pg}
    assert 1.0 in by_lr and 2.0 in by_lr and 3.0 in by_lr
    # Each group must be non-empty
    for g in pg:
        assert len(g["params"]) > 0
