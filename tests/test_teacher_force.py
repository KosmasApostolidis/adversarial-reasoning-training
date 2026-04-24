"""Round-trip + segment integrity tests for the teacher-force assembler."""

from __future__ import annotations

import pytest

from adversarial_reasoning_training.trajectory.mask import build_masks
from adversarial_reasoning_training.trajectory.segments import (
    DEFAULT_MASK_WEIGHTS,
    SegmentKind,
)


def test_segment_kinds_unique() -> None:
    values = [k.value for k in SegmentKind]
    assert len(values) == len(set(values)), "SegmentKind values must be unique"


def test_default_mask_weights_cover_all_kinds() -> None:
    weights = DEFAULT_MASK_WEIGHTS.task
    for kind in (SegmentKind.TOOL_NAME, SegmentKind.TOOL_ARGS, SegmentKind.ANSWER):
        assert kind in weights, f"task weights missing {kind}"


def test_build_masks_observation_zeroed() -> None:
    """Observations must be masked to 0 in both task and traj masks."""
    import torch

    seg = torch.tensor(
        [[
            SegmentKind.SYSTEM.value,
            SegmentKind.USER.value,
            SegmentKind.THOUGHT.value,
            SegmentKind.TOOL_NAME.value,
            SegmentKind.OBSERVATION.value,
            SegmentKind.ANSWER.value,
        ]],
        dtype=torch.int32,
    )
    task_mask, traj_mask = build_masks(seg, DEFAULT_MASK_WEIGHTS)
    obs_idx = 4
    assert task_mask[0, obs_idx].item() == 0.0
    assert traj_mask[0, obs_idx].item() == 0.0


def test_build_masks_weights_match_config() -> None:
    import torch

    seg = torch.tensor(
        [[SegmentKind.TOOL_NAME.value, SegmentKind.TOOL_ARGS.value, SegmentKind.ANSWER.value]],
        dtype=torch.int32,
    )
    task_mask, _ = build_masks(seg, DEFAULT_MASK_WEIGHTS)
    assert task_mask[0, 0].item() == pytest.approx(
        DEFAULT_MASK_WEIGHTS.task[SegmentKind.TOOL_NAME]
    )
    assert task_mask[0, 1].item() == pytest.approx(
        DEFAULT_MASK_WEIGHTS.task[SegmentKind.TOOL_ARGS]
    )
    assert task_mask[0, 2].item() == pytest.approx(
        DEFAULT_MASK_WEIGHTS.task[SegmentKind.ANSWER]
    )
