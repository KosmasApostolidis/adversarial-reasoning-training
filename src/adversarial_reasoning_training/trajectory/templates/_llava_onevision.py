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

from ..mask import build_masks, labels_from_input_ids
from ..segments import DEFAULT_MASK_WEIGHTS, MaskWeights, Segment, SegmentKind
from ..teacher_force import TeacherForcedBatch, _format_observation, _split_thoughts

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
    except (KeyError, IndexError, RuntimeError, ValueError, AttributeError, TypeError):
        logger.debug(
            "LLaVA-OneVision processor did not pre-expand image placeholder; "
            "falling back to num_image_tokens=1.",
            exc_info=True,
        )
        num_image_tokens = 1

    return forward_kwargs, int(image_token_id), num_image_tokens


def _emit_llava_ov_assistant_step(
    segments: list[Segment],
    thought: str,
    call: ToolCall,
) -> None:
    """Append one assistant turn: Thought + JSON tool-call."""
    segments.append(Segment("Thought: ", SegmentKind.SEPARATOR))
    segments.append(Segment(thought, SegmentKind.THOUGHT))
    segments.append(Segment('\n{"tool": ', SegmentKind.SEPARATOR))
    segments.append(Segment(f'"{call.name}"', SegmentKind.TOOL_NAME))
    segments.append(Segment(', "args": ', SegmentKind.SEPARATOR))
    segments.append(
        Segment(json.dumps(call.args, ensure_ascii=False), SegmentKind.TOOL_ARGS)
    )
    segments.append(Segment("}", SegmentKind.SEPARATOR))


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

    # System
    segments.append(Segment(f"{IM_START}system\n", SegmentKind.SEPARATOR))
    segments.append(Segment(sys_text, SegmentKind.SYSTEM))
    segments.append(Segment(f"\n{IM_END}\n", SegmentKind.SEPARATOR))

    # User (image + prompt)
    segments.append(Segment(f"{IM_START}user\n", SegmentKind.SEPARATOR))
    segments.append(Segment(_IMG_SENTINEL, SegmentKind.USER))
    segments.append(Segment(f"\n{user_prompt}", SegmentKind.USER))
    segments.append(Segment(f"\n{IM_END}\n", SegmentKind.SEPARATOR))

    n_calls = len(trajectory.tool_calls)
    if n_calls == 0:
        segments.append(Segment(f"{IM_START}assistant\n", SegmentKind.SEPARATOR))
        segments.append(Segment(trajectory.final_answer, SegmentKind.ANSWER))
        segments.append(Segment(f"\n{IM_END}", SegmentKind.SEPARATOR))
        return segments

    thoughts = _split_thoughts(trajectory.reasoning_trace, n_steps=n_calls)

    # First assistant turn
    segments.append(Segment(f"{IM_START}assistant\n", SegmentKind.SEPARATOR))
    _emit_llava_ov_assistant_step(segments, thoughts[0], trajectory.tool_calls[0])
    segments.append(Segment(f"\n{IM_END}\n", SegmentKind.SEPARATOR))

    # Middle assistant turns, each preceded by a tool observation
    for i in range(1, n_calls):
        prev_obs = _format_observation(trajectory.tool_calls[i - 1])
        segments.append(Segment(f"{IM_START}tool\n", SegmentKind.SEPARATOR))
        segments.append(Segment(prev_obs, SegmentKind.OBSERVATION))
        segments.append(Segment(f"\n{IM_END}\n", SegmentKind.SEPARATOR))
        segments.append(Segment(f"{IM_START}assistant\n", SegmentKind.SEPARATOR))
        _emit_llava_ov_assistant_step(segments, thoughts[i], trajectory.tool_calls[i])
        segments.append(Segment(f"\n{IM_END}\n", SegmentKind.SEPARATOR))

    # Final observation + answer
    last_obs = _format_observation(trajectory.tool_calls[-1])
    segments.append(Segment(f"{IM_START}tool\n", SegmentKind.SEPARATOR))
    segments.append(Segment(last_obs, SegmentKind.OBSERVATION))
    segments.append(Segment(f"\n{IM_END}\n", SegmentKind.SEPARATOR))
    segments.append(Segment(f"{IM_START}assistant\n", SegmentKind.SEPARATOR))
    segments.append(Segment("Answer: ", SegmentKind.SEPARATOR))
    segments.append(Segment(trajectory.final_answer, SegmentKind.ANSWER))
    segments.append(Segment(f"\n{IM_END}", SegmentKind.SEPARATOR))

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

    input_ids = torch.tensor([ids], dtype=torch.long)
    segment_ids = torch.tensor([seg], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    task_mask, traj_mask = build_masks(segment_ids, weights)
    labels = labels_from_input_ids(input_ids, task_mask)

    forward_kwargs["attention_mask"] = attention_mask

    return TeacherForcedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        task_mask=task_mask,
        traj_mask=traj_mask,
        labels=labels,
        forward_kwargs=forward_kwargs,
        segments=segments,
    )
