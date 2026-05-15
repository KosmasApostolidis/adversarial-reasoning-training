"""Qwen2.5-VL teacher-forced template assembler."""

from __future__ import annotations

import json
from typing import Any

import torch
from adversarial_reasoning.agents.base import Trajectory
from PIL import Image

from ..segments import DEFAULT_MASK_WEIGHTS, MaskWeights, Segment, SegmentKind
from ..teacher_force import TeacherForcedBatch, _format_observation, _split_thoughts
from ._common import IM_END, IM_START, pack_teacher_forced_batch

QWEN_IM_START = IM_START
QWEN_IM_END = IM_END
QWEN_VIS_START = "<|vision_start|>"
QWEN_VIS_END = "<|vision_end|>"
QWEN_IMAGE_PAD = "<|image_pad|>"
QWEN_TOOL_OPEN = "<tool_call>"
QWEN_TOOL_CLOSE = "</tool_call>"


def _append_qwen_system(segments: list[Segment], sys_text: str) -> None:
    """Append the system turn that prefaces every Qwen conversation."""
    segments.append(Segment(f"{QWEN_IM_START}system\n", SegmentKind.SEPARATOR))
    segments.append(Segment(sys_text, SegmentKind.SYSTEM))
    segments.append(Segment(f"\n{QWEN_IM_END}\n", SegmentKind.SEPARATOR))


def _append_qwen_user_with_image(
    segments: list[Segment], user_prompt: str, num_image_pad: int,
) -> None:
    """Append the user turn carrying ``<|vision_*|>``-wrapped image pads + prompt."""
    segments.append(Segment(f"{QWEN_IM_START}user\n", SegmentKind.SEPARATOR))
    image_pad_run = QWEN_IMAGE_PAD * max(1, num_image_pad)
    segments.append(
        Segment(f"{QWEN_VIS_START}{image_pad_run}{QWEN_VIS_END}\n", SegmentKind.USER)
    )
    segments.append(Segment(user_prompt, SegmentKind.USER))
    segments.append(Segment(f"\n{QWEN_IM_END}\n", SegmentKind.SEPARATOR))


def _append_qwen_assistant_tool_call(
    segments: list[Segment], thought: str, call: Any,
) -> None:
    """Append one assistant turn that emits ``Thought:`` then a tool_call JSON."""
    segments.append(Segment(f"{QWEN_IM_START}assistant\n", SegmentKind.SEPARATOR))
    segments.append(Segment("Thought: ", SegmentKind.SEPARATOR))
    segments.append(Segment(thought, SegmentKind.THOUGHT))
    segments.append(Segment(f"\n{QWEN_TOOL_OPEN}\n", SegmentKind.SEPARATOR))
    segments.append(Segment('{"name": ', SegmentKind.SEPARATOR))
    segments.append(Segment(f'"{call.name}"', SegmentKind.TOOL_NAME))
    segments.append(Segment(', "arguments": ', SegmentKind.SEPARATOR))
    segments.append(
        Segment(json.dumps(call.args, ensure_ascii=False), SegmentKind.TOOL_ARGS)
    )
    segments.append(Segment("}", SegmentKind.SEPARATOR))
    segments.append(Segment(f"\n{QWEN_TOOL_CLOSE}\n", SegmentKind.SEPARATOR))
    segments.append(Segment(f"{QWEN_IM_END}\n", SegmentKind.SEPARATOR))


def _append_qwen_tool_observation(segments: list[Segment], call: Any) -> None:
    """Append the tool-observation turn that follows an assistant tool_call.

    Observation tokens get weight 0 via ``SegmentKind.OBSERVATION``.
    """
    segments.append(Segment(f"{QWEN_IM_START}tool\n", SegmentKind.SEPARATOR))
    segments.append(Segment(_format_observation(call), SegmentKind.OBSERVATION))
    segments.append(Segment(f"\n{QWEN_IM_END}\n", SegmentKind.SEPARATOR))


def _append_qwen_final_answer(segments: list[Segment], final_answer: str) -> None:
    """Append the closing assistant turn that emits the final answer."""
    segments.append(Segment(f"{QWEN_IM_START}assistant\n", SegmentKind.SEPARATOR))
    segments.append(Segment("Answer: ", SegmentKind.SEPARATOR))
    segments.append(Segment(final_answer, SegmentKind.ANSWER))
    segments.append(Segment(f"\n{QWEN_IM_END}", SegmentKind.SEPARATOR))


def _build_qwen_segments(
    user_prompt: str,
    trajectory: Trajectory,
    *,
    system_prompt: str | None = None,
    num_image_pad: int = 1,
) -> list[Segment]:
    """Linearize (prompt, trajectory) into ordered segments for Qwen2.5-VL.

    Layout (one assistant turn, interleaved with tool observations):

        <|im_start|>system\n{sys}\n<|im_end|>\n
        <|im_start|>user\n{VIS}{IMG}{/VIS}\n{prompt}\n<|im_end|>\n
        <|im_start|>assistant\n
            Thought: {thought_1}\n
            <tool_call>{...tool_1}</tool_call>\n
        <|im_end|>\n
        <|im_start|>tool\n{obs_1}\n<|im_end|>\n
        ...
        <|im_start|>assistant\n
            Answer: {final_answer}\n
        <|im_end|>

    Reasoning-trace text is split by ``\\n---\\n`` into per-step thoughts
    if present; otherwise every assistant turn gets an empty Thought.
    """
    sys_text = (
        system_prompt
        if system_prompt is not None
        else "You are a medical-imaging VLM agent. Reason step by step and call tools."
    )
    segments: list[Segment] = []

    _append_qwen_system(segments, sys_text)
    _append_qwen_user_with_image(segments, user_prompt, num_image_pad)

    thoughts = _split_thoughts(
        trajectory.reasoning_trace, n_steps=len(trajectory.tool_calls),
    )
    for i, call in enumerate(trajectory.tool_calls):
        _append_qwen_assistant_tool_call(segments, thoughts[i], call)
        _append_qwen_tool_observation(segments, call)

    _append_qwen_final_answer(segments, trajectory.final_answer)
    return segments


def _tokenize_segments(
    segments: list[Segment],
    tokenizer: Any,
) -> tuple[list[int], list[int]]:
    """Tokenize each segment separately and concatenate; return (ids, seg_ids)."""
    all_ids: list[int] = []
    all_seg: list[int] = []
    for seg in segments:
        if seg.text == "":
            continue
        ids = tokenizer.encode(seg.text, add_special_tokens=False)
        if not ids:
            continue
        all_ids.extend(ids)
        all_seg.extend([int(seg.kind)] * len(ids))
    return all_ids, all_seg


def _process_image_qwen(
    image: Image.Image,
    processor: Any,
) -> tuple[dict[str, torch.Tensor], int]:
    """Run image side of Qwen's processor. Returns (kwargs, num_image_pad).

    Qwen2.5-VL expects ``<|image_pad|>`` repeated ``image_grid_thw.prod() //
    merge_size**2`` times between ``<|vision_start|>`` and ``<|vision_end|>``,
    matching the post-merge patch count emitted by the vision encoder.
    """
    img_proc = getattr(processor, "image_processor", None)
    if img_proc is None:
        raise RuntimeError("Qwen processor missing .image_processor; cannot prepare image.")
    out = img_proc(images=[image], return_tensors="pt")
    kwargs: dict[str, torch.Tensor] = {"pixel_values": out["pixel_values"]}
    num_image_pad = 1
    if "image_grid_thw" in out:
        kwargs["image_grid_thw"] = out["image_grid_thw"]
        merge_size = int(getattr(img_proc, "merge_size", 2))
        grid = out["image_grid_thw"][0]
        num_image_pad = int(grid.prod().item() // (merge_size * merge_size))
    return kwargs, num_image_pad


def assemble_qwen(
    user_prompt: str,
    trajectory: Trajectory,
    image: Image.Image,
    processor: Any,
    *,
    system_prompt: str | None = None,
    weights: MaskWeights = DEFAULT_MASK_WEIGHTS,
) -> TeacherForcedBatch:
    """Family-specific assembler for Qwen2.5-VL.

    Parameters
    ----------
    user_prompt : The task prompt presented to the agent.
    trajectory : Gold trajectory whose tokens become the supervised target.
    image : PIL image passed as the visual context.
    processor : HF `AutoProcessor` compatible with Qwen2.5-VL.
    system_prompt : Optional override for the system message.
    weights : Per-segment task and trajectory mask weights.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    forward_kwargs, num_image_pad = _process_image_qwen(image, processor)
    segments = _build_qwen_segments(
        user_prompt, trajectory,
        system_prompt=system_prompt, num_image_pad=num_image_pad,
    )
    ids, seg = _tokenize_segments(segments, tokenizer)
    if not ids:
        raise ValueError("assemble_qwen produced zero tokens; empty trajectory?")

    return pack_teacher_forced_batch(
        ids=ids, seg=seg, segments=segments,
        forward_kwargs=forward_kwargs, weights=weights,
    )
