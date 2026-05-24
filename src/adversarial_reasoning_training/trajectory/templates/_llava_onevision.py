"""LLaVA-OneVision (Qwen2-7B backbone) teacher-forced template assembler.

Qwen2 chat format with LLaVA ``<image>`` placeholder. Multi-turn ReAct
tool-calling via ``<|im_start|>`` / ``<|im_end|>`` role delimiters.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import torch
from adversarial_reasoning.agents.base import ToolCall, Trajectory
from PIL import Image

from ..segments import DEFAULT_MASK_WEIGHTS, MaskWeights, Segment, SegmentKind
from ..teacher_force import TeacherForcedBatch, _format_observation, _split_thoughts
from ._common import pack_teacher_forced_batch

logger = logging.getLogger(__name__)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
LLAVA_OV_DEFAULT_SYSTEM = (
    "You are a medical-imaging VLM agent. Reason step by step and call tools."
)

_IMG_SENTINEL = "<__LLAVA_OV_IMG__>"


def _process_image_llava_ov(
    image: Image.Image,
    processor: Any,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Run LLaVA-OneVision image processor; return (forward_kwargs, image_token_id, num_image_tokens)."""
    img_proc = getattr(processor, "image_processor", None)
    if img_proc is None:
        raise RuntimeError(
            "LLaVA-OneVision processor missing .image_processor; cannot prepare image."
        )
    out = img_proc(images=[image], return_tensors="pt")
    forward_kwargs: dict[str, torch.Tensor] = {"pixel_values": out["pixel_values"]}
    if "image_sizes" in out:
        forward_kwargs["image_sizes"] = out["image_sizes"]

    tokenizer = getattr(processor, "tokenizer", processor)
    image_token_text = getattr(processor, "image_token", "<image>")
    image_token_id = tokenizer.convert_tokens_to_ids(image_token_text)
    if image_token_id is None or image_token_id == tokenizer.unk_token_id:
        raise RuntimeError(
            f"LLaVA-OneVision processor missing image_token={image_token_text!r}."
        )

    num_image_tokens = 1
    try:
        proc_out = processor(text=image_token_text, images=image, return_tensors="pt")
        ids = proc_out["input_ids"][0].tolist()
        count = sum(1 for t in ids if t == image_token_id)
        if count >= 1:
            num_image_tokens = count
    except (KeyError, IndexError) as exc:
        logger.warning(
            "LLaVA-OneVision processor did not pre-expand image placeholder "
            "(%s: %s); falling back to num_image_tokens=1 — training "
            "data may be misaligned.",
            type(exc).__name__, exc,
        )
        num_image_tokens = 1

    return forward_kwargs, int(image_token_id), num_image_tokens


def _emit_llava_ov_assistant_step(
    segments: list[Segment],
    thought: str,
    call: ToolCall,
) -> None:
    """Append one assistant turn body: Thought + JSON tool-call (no role separators)."""
    segments.append(Segment("Thought: ", SegmentKind.SEPARATOR))
    segments.append(Segment(thought, SegmentKind.THOUGHT))
    segments.append(Segment('\n{"tool": ', SegmentKind.SEPARATOR))
    segments.append(Segment(f'"{call.name}"', SegmentKind.TOOL_NAME))
    segments.append(Segment(', "args": ', SegmentKind.SEPARATOR))
    segments.append(
        Segment(json.dumps(call.args, ensure_ascii=False), SegmentKind.TOOL_ARGS)
    )
    segments.append(Segment("}", SegmentKind.SEPARATOR))


def _append_llava_ov_system(segments: list[Segment], sys_text: str) -> None:
    """Append the system turn prefacing every LLaVA-OneVision conversation."""
    segments.append(Segment(f"{IM_START}system\n", SegmentKind.SEPARATOR))
    segments.append(Segment(sys_text, SegmentKind.SYSTEM))
    segments.append(Segment(f"\n{IM_END}\n", SegmentKind.SEPARATOR))


def _append_llava_ov_user(segments: list[Segment], user_prompt: str) -> None:
    """Append the user turn carrying the image sentinel + prompt."""
    segments.append(Segment(f"{IM_START}user\n", SegmentKind.SEPARATOR))
    segments.append(Segment(_IMG_SENTINEL, SegmentKind.USER))
    segments.append(Segment(f"\n{user_prompt}", SegmentKind.USER))
    segments.append(Segment(f"\n{IM_END}\n", SegmentKind.SEPARATOR))


def _append_llava_ov_assistant_turn(
    segments: list[Segment], thought: str, call: ToolCall,
) -> None:
    """Append a complete assistant tool-call turn (role separators + body)."""
    segments.append(Segment(f"{IM_START}assistant\n", SegmentKind.SEPARATOR))
    _emit_llava_ov_assistant_step(segments, thought, call)
    segments.append(Segment(f"\n{IM_END}\n", SegmentKind.SEPARATOR))


def _append_llava_ov_tool_observation(segments: list[Segment], call: ToolCall) -> None:
    """Append the tool-observation turn that follows an assistant tool_call."""
    segments.append(Segment(f"{IM_START}tool\n", SegmentKind.SEPARATOR))
    segments.append(Segment(_format_observation(call), SegmentKind.OBSERVATION))
    segments.append(Segment(f"\n{IM_END}\n", SegmentKind.SEPARATOR))


def _append_llava_ov_final_answer(segments: list[Segment], final_answer: str) -> None:
    """Append the closing assistant turn that emits the final answer."""
    segments.append(Segment(f"{IM_START}assistant\n", SegmentKind.SEPARATOR))
    segments.append(Segment("Answer: ", SegmentKind.SEPARATOR))
    segments.append(Segment(final_answer, SegmentKind.ANSWER))
    segments.append(Segment(f"\n{IM_END}", SegmentKind.SEPARATOR))


def _append_llava_ov_empty_trajectory_answer(
    segments: list[Segment], final_answer: str,
) -> None:
    """Append a bare assistant answer turn when the trajectory has no tool calls."""
    segments.append(Segment(f"{IM_START}assistant\n", SegmentKind.SEPARATOR))
    segments.append(Segment(final_answer, SegmentKind.ANSWER))
    segments.append(Segment(f"\n{IM_END}", SegmentKind.SEPARATOR))


def _build_llava_ov_segments(
    user_prompt: str,
    trajectory: Trajectory,
    *,
    system_prompt: str | None = None,
) -> list[Segment]:
    """Linearize (prompt, trajectory) into Qwen2-style chat segments.

    Layout::

        <|im_start|>system\n{sys}\n<|im_end|>\n
        <|im_start|>user\n<image>\n{prompt}\n<|im_end|>\n
        <|im_start|>assistant\nThought: {t1}\n{"tool": ...}\n<|im_end|>\n
        <|im_start|>tool\n{obs1}\n<|im_end|>\n
        <|im_start|>assistant\nThought: {t2}\n{"tool": ...}\n<|im_end|>\n
        ...
        <|im_start|>assistant\nAnswer: {final}\n<|im_end|>
    """
    sys_text = system_prompt if system_prompt is not None else LLAVA_OV_DEFAULT_SYSTEM
    segments: list[Segment] = []

    _append_llava_ov_system(segments, sys_text)
    _append_llava_ov_user(segments, user_prompt)

    n_calls = len(trajectory.tool_calls)
    if n_calls == 0:
        _append_llava_ov_empty_trajectory_answer(segments, trajectory.final_answer)
        return segments

    thoughts = _split_thoughts(trajectory.reasoning_trace, n_steps=n_calls)
    _append_llava_ov_assistant_turn(segments, thoughts[0], trajectory.tool_calls[0])
    for i in range(1, n_calls):
        _append_llava_ov_tool_observation(segments, trajectory.tool_calls[i - 1])
        _append_llava_ov_assistant_turn(segments, thoughts[i], trajectory.tool_calls[i])

    _append_llava_ov_tool_observation(segments, trajectory.tool_calls[-1])
    _append_llava_ov_final_answer(segments, trajectory.final_answer)
    return segments


def _tokenize_llava_ov_segments(
    segments: list[Segment],
    tokenizer: Any,
    image_token_id: int,
    num_image_tokens: int,
) -> tuple[list[int], list[int]]:
    """Tokenize segments; substitute image sentinel with literal token ids."""
    all_ids: list[int] = []
    all_seg: list[int] = []
    for seg in segments:
        if seg.text == _IMG_SENTINEL:
            all_ids.extend([image_token_id] * num_image_tokens)
            all_seg.extend([int(seg.kind)] * num_image_tokens)
            continue
        if seg.text == "":
            continue
        ids = tokenizer.encode(seg.text, add_special_tokens=False)
        if not ids:
            continue
        all_ids.extend(ids)
        all_seg.extend([int(seg.kind)] * len(ids))
    return all_ids, all_seg


def assemble_llava_onevision(
    user_prompt: str,
    trajectory: Trajectory,
    image: Image.Image,
    processor: Any,
    *,
    system_prompt: str | None = None,
    weights: MaskWeights = DEFAULT_MASK_WEIGHTS,
) -> TeacherForcedBatch:
    """Family-specific assembler for LLaVA-OneVision (Qwen2-7B backbone).

    ``processor`` is a ``LlavaOnevisionProcessor`` carrying ``.tokenizer``
    and ``.image_processor``. The returned ``forward_kwargs`` carry
    ``pixel_values``, ``image_sizes``, and ``attention_mask``.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    forward_kwargs, image_token_id, num_image_tokens = _process_image_llava_ov(
        image, processor
    )
    segments = _build_llava_ov_segments(
        user_prompt, trajectory, system_prompt=system_prompt,
    )
    ids, seg = _tokenize_llava_ov_segments(
        segments, tokenizer, image_token_id, num_image_tokens,
    )
    if not ids:
        raise ValueError(
            "assemble_llava_onevision produced zero tokens; empty trajectory?"
        )

    return pack_teacher_forced_batch(
        ids=ids, seg=seg, segments=segments,
        forward_kwargs=forward_kwargs, weights=weights,
    )
