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


def _make_smoke_trajectory(*, model_id: str, final_answer: str):
    """Build a two-step Trajectory shared across teacher-force smoke tests."""
    from adversarial_reasoning.agents.base import ToolCall, Trajectory

    return Trajectory(
        task_id="t0",
        model_id=model_id,
        seed=0,
        tool_calls=[
            ToolCall(step=0, name="zoom", args={"x": 1}, result="ok"),
            ToolCall(step=1, name="ocr", args={"y": 2}, result="hello"),
        ],
        final_answer=final_answer,
        reasoning_trace="step1\n---\nstep2",
    )


def _make_llava_next_stub_processor():
    """Construct a LLaVA-NeXT-shaped stub processor for the smoke test.

    Returns the processor and the tokenizer-id constants the test asserts on.
    """
    import torch

    class _StubTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        unk_token_id = 0
        _IMAGE_TOKEN = "<image>"
        _IMAGE_TOKEN_ID = 32000

        def convert_tokens_to_ids(self, token: str) -> int:
            return self._IMAGE_TOKEN_ID if token == self._IMAGE_TOKEN else 99

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
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
            ids = [self.tokenizer._IMAGE_TOKEN_ID] * 5
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    return _StubProcessor(), _StubTokenizer


def _make_internvl2_stub_vlm():
    """Construct an InternVL2-shaped stub VLM wrapper for the smoke test."""
    import torch

    class _StubTokenizer:
        unk_token_id = 0
        _IMG_CONTEXT_ID = 92546
        _IM_END_ID = 92547

        def convert_tokens_to_ids(self, token: str) -> int:
            if token == "<IMG_CONTEXT>":
                return self._IMG_CONTEXT_ID
            return 99

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
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
            self._num_image_token = 8  # 4 patches * 8 → 32 IMG_CONTEXT tokens

        def preprocess_image(self, image):
            return torch.zeros(4, 3, 448, 448)

    return _StubVLM(), _StubTokenizer


def test_assemble_llava_next_smoke() -> None:
    """Smoke-test LLaVA-NeXT assembler with a stub processor + tokenizer.

    Verifies the assembler dispatches, returns a TeacherForcedBatch with
    matching shapes, threads pixel_values + image_sizes via forward_kwargs,
    and emits the expected image-token expansion (1 image_token_id per
    pre-expanded slot, BOS prepended, EOS at every turn boundary).
    """
    from PIL import Image

    from adversarial_reasoning_training.trajectory.teacher_force import assemble

    processor, stub_tok = _make_llava_next_stub_processor()
    image = Image.new("RGB", (336, 336), color="white")
    traj = _make_smoke_trajectory(model_id="llava7b", final_answer="answer-42")

    batch = assemble(
        family="llava_next", user_prompt="describe the image",
        trajectory=traj, image=image, processor=processor,
    )

    T = batch.input_ids.shape[1]
    assert batch.input_ids.shape == (1, T)
    assert batch.segment_ids.shape == (1, T)
    assert batch.task_mask.shape == (1, T)
    assert batch.traj_mask.shape == (1, T)
    assert batch.labels.shape == (1, T)
    assert int(batch.input_ids[0, 0].item()) == stub_tok.bos_token_id
    image_token_count = (batch.input_ids == stub_tok._IMAGE_TOKEN_ID).sum().item()
    assert image_token_count == 5
    eos_count = (batch.input_ids == stub_tok.eos_token_id).sum().item()
    assert eos_count == 3  # 2 tool turns + final answer turn
    assert {"pixel_values", "image_sizes", "attention_mask"} <= set(batch.forward_kwargs)


def test_assemble_internvl2_smoke() -> None:
    """Smoke-test InternVL2 assembler with a stub VLM wrapper.

    Verifies the dispatcher routes to ``assemble_internvl``, the IMG_CONTEXT
    sentinel expands to ``num_patches * num_image_token`` image-token ids,
    pixel_values shape is preserved, and one EOS-equivalent ``<|im_end|>``
    token closes every assistant/tool/final turn.
    """
    from PIL import Image

    from adversarial_reasoning_training.trajectory.teacher_force import assemble

    vlm, stub_tok = _make_internvl2_stub_vlm()
    image = Image.new("RGB", (448, 448), color="white")
    traj = _make_smoke_trajectory(model_id="internvl3_8b", final_answer="answer-internvl")

    batch = assemble(
        family="internvl2", user_prompt="describe the image",
        trajectory=traj, image=image, processor=vlm,
    )

    T = batch.input_ids.shape[1]
    assert batch.input_ids.shape == (1, T)
    assert batch.segment_ids.shape == (1, T)
    img_ctx_count = (batch.input_ids == stub_tok._IMG_CONTEXT_ID).sum().item()
    assert img_ctx_count == 32  # 4 patches * 8 num_image_token
    im_end_count = (batch.input_ids == stub_tok._IM_END_ID).sum().item()
    # system + user + 2x assistant + 2x tool + final assistant = 7 <|im_end|>
    assert im_end_count == 7
    assert "pixel_values" in batch.forward_kwargs
    assert tuple(batch.forward_kwargs["pixel_values"].shape) == (4, 3, 448, 448)
    assert "attention_mask" in batch.forward_kwargs


def test_assemble_unknown_family_raises() -> None:
    from adversarial_reasoning.agents.base import Trajectory
    from PIL import Image

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
