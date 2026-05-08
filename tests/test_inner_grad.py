"""Verify that gradients flow through the inner-PGD wrapper.

The PGD attack is supposed to maximise teacher-forced CE wrt the
input image tensor, so its `_loss(x)` must produce a non-None gradient
for `x` inside its own optimization loop. We test the wrapper layer
that builds prompt_tokens / target_tokens from a TeacherForcedBatch.
"""

from __future__ import annotations

import torch

from adversarial_reasoning_training.attacks.inner_pgd import _split_prompt_target
from adversarial_reasoning_training.trajectory.segments import SegmentKind


def _toy_batch():
    """Build a minimal `TeacherForcedBatch`-shaped object."""
    from dataclasses import dataclass

    @dataclass
    class _B:
        input_ids: torch.Tensor
        task_mask: torch.Tensor
        traj_mask: torch.Tensor
        attention_mask: torch.Tensor
        labels: torch.Tensor
        segment_ids: torch.Tensor
        forward_kwargs: dict
        segments: list

    T = 12
    seg = torch.full((1, T), SegmentKind.USER.value, dtype=torch.int32)
    seg[0, 6:] = SegmentKind.TOOL_NAME.value
    return _B(
        input_ids=torch.arange(T, dtype=torch.long).unsqueeze(0),
        task_mask=torch.tensor([[0] * 6 + [1] * 6], dtype=torch.float32),
        traj_mask=torch.tensor([[0] * 6 + [1] * 6], dtype=torch.float32),
        attention_mask=torch.ones((1, T), dtype=torch.long),
        labels=torch.arange(T, dtype=torch.long).unsqueeze(0),
        segment_ids=seg,
        forward_kwargs={"pixel_values": torch.zeros((1, 3, 8, 8))},
        segments=[],
    )


def test_split_prompt_target_shapes() -> None:
    batch = _toy_batch()
    prompt, target, mask = _split_prompt_target(batch)
    assert prompt.shape == (1, 6)
    assert target.shape == (1, 6)
    assert mask.shape == (1, 6)
    assert (mask > 0).all()


def test_split_prompt_target_raises_when_no_task() -> None:
    """If task_mask is all zero, the splitter must raise rather than silently returning everything."""
    import pytest

    batch = _toy_batch()
    batch.task_mask = torch.zeros_like(batch.task_mask)
    with pytest.raises(ValueError):
        _split_prompt_target(batch)


# --- run_inner_pgd ---------------------------------------------------------


class _StubAttack:
    """Stand-in for ``PGDAttack`` whose ``run`` returns a configurable result.

    The test patches the attacks-repo PGDAttack symbol that ``run_inner_pgd``
    imports, so the wrapper exercises only its own glue logic (split → invoke
    → finite-image guard) without spinning up the real PGD optimization.
    """

    def __init__(self, *, perturbed: torch.Tensor, loss_final: float) -> None:
        self._perturbed = perturbed
        self._loss_final = loss_final

    @classmethod
    def make_factory(cls, *, perturbed: torch.Tensor, loss_final: float):
        def _ctor(**_ignored):
            return cls(perturbed=perturbed, loss_final=loss_final)
        return _ctor

    def run(self, **_ignored) -> object:
        from adversarial_reasoning.attacks.base import AttackResult
        return AttackResult(
            perturbed_image=self._perturbed,
            delta=torch.zeros_like(self._perturbed),
            loss_final=self._loss_final,
            loss_trajectory=[self._loss_final],
            iterations=1,
            success=True,
            metadata={},
        )


def test_run_inner_pgd_falls_back_to_clean_when_perturbed_is_nan(
    monkeypatch,
) -> None:
    """If PGD diverges to non-finite pixels, wrapper must substitute the
    clean image and emit NaN loss so downstream code can flag the event.
    """
    from adversarial_reasoning_training.attacks import inner_pgd as ip

    clean = torch.full((1, 3, 8, 8), 0.5)
    nan_pixels = torch.full_like(clean, float("nan"))
    monkeypatch.setattr(
        ip,
        "PGDAttack",
        _StubAttack.make_factory(perturbed=nan_pixels, loss_final=0.7),
    )

    result = ip.run_inner_pgd(
        vlm=object(),
        image_tensor=clean,
        batch=_toy_batch(),
        config=ip.InnerPgdConfig(epsilon=0.01, steps=1, random_restarts=1),
    )

    assert torch.equal(result.perturbed_image, clean)
    import math
    assert math.isnan(result.loss_final)


def test_run_inner_pgd_passes_finite_perturbation_through(monkeypatch) -> None:
    """Finite perturbed pixels are returned untouched; only NaN/inf trigger
    the clean-image fallback.
    """
    from adversarial_reasoning_training.attacks import inner_pgd as ip

    clean = torch.full((1, 3, 8, 8), 0.5)
    finite = clean + 0.01
    monkeypatch.setattr(
        ip,
        "PGDAttack",
        _StubAttack.make_factory(perturbed=finite, loss_final=1.5),
    )

    result = ip.run_inner_pgd(
        vlm=object(),
        image_tensor=clean,
        batch=_toy_batch(),
        config=ip.InnerPgdConfig(epsilon=0.01, steps=1, random_restarts=1),
    )

    assert torch.equal(result.perturbed_image, finite)
    assert result.loss_final == 1.5
