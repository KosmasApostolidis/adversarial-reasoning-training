"""Unit tests for ``adversarial_reasoning_training.data.collator.TFCollator``.

The collator is a thin wrapper over ``trajectory.teacher_force.assemble``,
so we test its single-sample contract and argument forwarding via a
mocked ``assemble`` to avoid pulling in a real HF processor.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from adversarial_reasoning_training.data.collator import TFCollator
from adversarial_reasoning_training.trajectory.segments import DEFAULT_MASK_WEIGHTS, MaskWeights


@dataclass
class _FakeSample:
    prompt: str
    trajectory: object
    image: object


def _fake_batch_sentinel() -> object:
    return SimpleNamespace(sentinel="teacher-forced-batch")


def test_collator_rejects_multi_sample_batch() -> None:
    collator = TFCollator(family="qwen2_5_vl", processor=object())
    with pytest.raises(NotImplementedError, match="micro_batch=1"):
        collator([_FakeSample("p1", object(), object()), _FakeSample("p2", object(), object())])


def test_collator_forwards_to_assemble_with_default_weights() -> None:
    processor = object()
    collator = TFCollator(family="qwen2_5_vl", processor=processor)
    sample = _FakeSample(prompt="hello", trajectory="traj-stub", image="image-stub")
    fake_batch = _fake_batch_sentinel()

    with patch(
        "adversarial_reasoning_training.data.collator.assemble",
        return_value=fake_batch,
    ) as mock_assemble:
        out = collator([sample])

    assert out is fake_batch
    mock_assemble.assert_called_once_with(
        "qwen2_5_vl",
        "hello",
        "traj-stub",
        "image-stub",
        processor,
        system_prompt=None,
        weights=DEFAULT_MASK_WEIGHTS,
    )


def test_collator_passes_custom_weights_and_system_prompt() -> None:
    processor = object()
    custom_weights = MaskWeights(
        task=DEFAULT_MASK_WEIGHTS.task,
        traj=DEFAULT_MASK_WEIGHTS.traj,
    )
    collator = TFCollator(
        family="qwen2_5_vl",
        processor=processor,
        system_prompt="be helpful",
        weights=custom_weights,
    )
    sample = _FakeSample("p", object(), object())

    with patch(
        "adversarial_reasoning_training.data.collator.assemble",
        return_value=_fake_batch_sentinel(),
    ) as mock_assemble:
        collator([sample])

    _, kwargs = mock_assemble.call_args
    assert kwargs["system_prompt"] == "be helpful"
    assert kwargs["weights"] is custom_weights


def test_collator_rejects_empty_batch() -> None:
    collator = TFCollator(family="qwen2_5_vl", processor=object())
    with pytest.raises(NotImplementedError):
        collator([])
