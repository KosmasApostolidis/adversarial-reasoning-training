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


class _QwenLikeVLM(torch.nn.Module):
    """Qwen2.5/3-VL-style naming: the multimodal projector lives at
    ``visual.merger.*``, whose name matches BOTH the vit pattern
    (``visual``) and the projector pattern (``merger``)."""

    def __init__(self) -> None:
        super().__init__()
        self.visual = torch.nn.Module()
        self.visual.patch_embed = torch.nn.Linear(8, 8)  # pure vit
        self.visual.merger = torch.nn.Linear(8, 8)       # projector (under visual.*)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.Linear(8, 8)        # lm


def test_qwen_merger_is_projector_not_vit() -> None:
    """Regression: ``visual.merger.*`` must be bucketed to the projector
    group (lr_projector), not the vit group. The vit-first dispatch order
    previously captured it as vit, training the projector at lr_vit (10x
    off) and leaving the projector group empty."""
    m = _QwenLikeVLM()
    apply_freeze(m, FreezeConfig(strategy="none"))
    pg = param_groups_by_role(m, lr_lm=1.0, lr_projector=2.0, lr_vit=3.0)
    by_lr = {g["lr"]: list(g["params"]) for g in pg}
    merger_params = set(m.visual.merger.parameters())
    projector_group = by_lr.get(2.0, [])
    vit_group = by_lr.get(3.0, [])
    assert merger_params.issubset(set(projector_group)), (
        "visual.merger.* must land in the projector group at lr_projector"
    )
    assert not (merger_params & set(vit_group)), (
        "visual.merger.* must NOT be in the vit group"
    )
