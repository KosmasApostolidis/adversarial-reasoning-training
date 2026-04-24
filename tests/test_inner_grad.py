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
