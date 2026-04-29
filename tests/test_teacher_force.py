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


def test_assemble_llava_next_smoke() -> None:
    """Smoke-test LLaVA-NeXT assembler with a stub processor + tokenizer.

    Verifies the assembler dispatches, returns a TeacherForcedBatch with
    matching shapes, threads pixel_values + image_sizes via forward_kwargs,
    and emits the expected image-token expansion (1 image_token_id per
    pre-expanded slot, BOS prepended, EOS at every turn boundary).
    """
    import torch
    from PIL import Image

    from adversarial_reasoning.agents.base import ToolCall, Trajectory
    from adversarial_reasoning_training.trajectory.teacher_force import assemble

    class _StubTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        unk_token_id = 0
        _IMAGE_TOKEN = "<image>"
        _IMAGE_TOKEN_ID = 32000

        def convert_tokens_to_ids(self, token: str) -> int:
            return self._IMAGE_TOKEN_ID if token == self._IMAGE_TOKEN else 99

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # Simple deterministic per-character mapping for the smoke test.
            return [ord(c) % 1000 + 100 for c in text]

    class _StubImageProcessor:
        def __call__(self, images, return_tensors: str = "pt"):
            return {
                "pixel_values": torch.zeros(1, 5, 3, 336, 336),
                "image_sizes": torch.tensor([[336, 336]]),
            }

    class _StubProcessor:
        image_token = "<image>"

        def __init__(self) -> None:
            self.tokenizer = _StubTokenizer()
            self.image_processor = _StubImageProcessor()

        def __call__(self, *, text: str, images, return_tensors: str = "pt"):
            # Pretend the processor expanded one <image> placeholder into 5.
            ids = [self.tokenizer._IMAGE_TOKEN_ID] * 5
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    image = Image.new("RGB", (336, 336), color="white")
    traj = Trajectory(
        task_id="t0",
        model_id="llava7b",
        seed=0,
        tool_calls=[
            ToolCall(step=0, name="zoom", args={"x": 1}, result="ok"),
            ToolCall(step=1, name="ocr", args={"y": 2}, result="hello"),
        ],
        final_answer="answer-42",
        reasoning_trace="step1\n---\nstep2",
    )

    batch = assemble(
        family="llava_next",
        user_prompt="describe the image",
        trajectory=traj,
        image=image,
        processor=_StubProcessor(),
    )

    T = batch.input_ids.shape[1]
    assert batch.input_ids.shape == (1, T)
    assert batch.segment_ids.shape == (1, T)
    assert batch.task_mask.shape == (1, T)
    assert batch.traj_mask.shape == (1, T)
    assert batch.labels.shape == (1, T)
    assert int(batch.input_ids[0, 0].item()) == _StubTokenizer.bos_token_id
    image_token_count = (batch.input_ids == _StubTokenizer._IMAGE_TOKEN_ID).sum().item()
    assert image_token_count == 5
    eos_count = (batch.input_ids == _StubTokenizer.eos_token_id).sum().item()
    # 2 tool turns + final answer turn → 3 EOS tokens.
    assert eos_count == 3
    assert "pixel_values" in batch.forward_kwargs
    assert "image_sizes" in batch.forward_kwargs
    assert "attention_mask" in batch.forward_kwargs


def test_assemble_internvl2_smoke() -> None:
    """Smoke-test InternVL2 assembler with a stub VLM wrapper.

    Verifies the dispatcher routes to ``assemble_internvl``, the IMG_CONTEXT
    sentinel expands to ``num_patches * num_image_token`` image-token ids,
    pixel_values shape is preserved, and one EOS-equivalent ``<|im_end|>``
    token closes every assistant/tool/final turn.
    """
    import torch
    from PIL import Image

    from adversarial_reasoning.agents.base import ToolCall, Trajectory
    from adversarial_reasoning_training.trajectory.teacher_force import assemble

    class _StubTokenizer:
        unk_token_id = 0
        _IMG_CONTEXT_ID = 92546
        _IM_END_ID = 92547

        def convert_tokens_to_ids(self, token: str) -> int:
            if token == "<IMG_CONTEXT>":
                return self._IMG_CONTEXT_ID
            return 99

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # Map "<|im_end|>" deterministically to EOS-like id; everything else
            # to per-character ids so segments stay distinguishable.
            if text == "<|im_end|>":
                return [self._IM_END_ID]
            if text.startswith("<|im_end|>"):
                return [self._IM_END_ID] + [
                    ord(c) % 1000 + 100 for c in text[len("<|im_end|>"):]
                ]
            return [ord(c) % 1000 + 100 for c in text]

    class _StubVLM:
        family = "internvl2"

        def __init__(self) -> None:
            self.tokenizer = _StubTokenizer()
            # 4 patches with 8 image tokens per patch → 32 IMG_CONTEXT tokens.
            self._num_image_token = 8

        def preprocess_image(self, image):
            return torch.zeros(4, 3, 448, 448)

    image = Image.new("RGB", (448, 448), color="white")
    traj = Trajectory(
        task_id="t0",
        model_id="internvl2_8b",
        seed=0,
        tool_calls=[
            ToolCall(step=0, name="zoom", args={"x": 1}, result="ok"),
            ToolCall(step=1, name="ocr", args={"y": 2}, result="hello"),
        ],
        final_answer="answer-internvl",
        reasoning_trace="step1\n---\nstep2",
    )

    batch = assemble(
        family="internvl2",
        user_prompt="describe the image",
        trajectory=traj,
        image=image,
        processor=_StubVLM(),
    )

    T = batch.input_ids.shape[1]
    assert batch.input_ids.shape == (1, T)
    assert batch.segment_ids.shape == (1, T)
    img_ctx_count = (batch.input_ids == _StubTokenizer._IMG_CONTEXT_ID).sum().item()
    # 4 patches * 8 num_image_token = 32 IMG_CONTEXT slots.
    assert img_ctx_count == 32
    im_end_count = (batch.input_ids == _StubTokenizer._IM_END_ID).sum().item()
    # system close + user close + 2x assistant turn close + 2x tool obs close
    # + final assistant close = 7 <|im_end|> tokens.
    assert im_end_count == 7
    assert "pixel_values" in batch.forward_kwargs
    assert tuple(batch.forward_kwargs["pixel_values"].shape) == (4, 3, 448, 448)
    assert "attention_mask" in batch.forward_kwargs


def test_assemble_unknown_family_raises() -> None:
    from PIL import Image

    from adversarial_reasoning.agents.base import Trajectory
    from adversarial_reasoning_training.trajectory.teacher_force import assemble

    image = Image.new("RGB", (32, 32))
    traj = Trajectory(
        task_id="t0", model_id="x", seed=0,
        tool_calls=[], final_answer="x", reasoning_trace="",
    )
    with pytest.raises(ValueError, match="bogus_family"):
        assemble(
            family="bogus_family",
            user_prompt="p",
            trajectory=traj,
            image=image,
            processor=object(),
        )
