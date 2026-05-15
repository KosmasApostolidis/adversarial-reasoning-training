"""LLaVA-NeXT (Mistral-Instruct backbone) teacher-forced template assembler."""

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

LLAVA_INST_OPEN = "[INST] "
LLAVA_INST_CLOSE = " [/INST] "
LLAVA_DEFAULT_SYSTEM = (
    "You are a medical-imaging VLM agent. Reason step by step and call tools."
)

# Sentinel strings injected at segment build time so the LLaVA tokenizer pass
# can substitute them with literal token ids (image-token expansion, BOS, EOS)
# without relying on tokenizer behavior over reserved-glyph strings.
_LLAVA_BOS_SENTINEL = "<__LLAVA_BOS__>"
_LLAVA_EOS_SENTINEL = "<__LLAVA_EOS__>"
_LLAVA_IMG_SENTINEL = "<__LLAVA_IMG__>"


def _process_image_llava(
    image: Image.Image,
    processor: Any,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Run image side of LlavaNext's processor and compute image-token expansion.

    LLaVA-NeXT uses anyres tiling: ``pixel_values`` has shape
    ``[1, num_patches, 3, H, W]`` and ``image_sizes`` carries the original
    H/W needed for the merge step. The number of image tokens that should
    appear in ``input_ids`` after the processor's pre-expansion depends on
    ``image_sizes`` and the vision tower config.

    Rather than reimplementing that math, we ask the processor itself by
    running it on a single-token placeholder text and counting how many
    ``image_token_id`` rows it produced.
    """
    img_proc = getattr(processor, "image_processor", None)
    if img_proc is None:
        raise RuntimeError(
            "LLaVA-NeXT processor missing .image_processor; cannot prepare image."
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
            f"LLaVA-NeXT processor does not expose image_token={image_token_text!r}."
        )

    num_image_tokens = _count_llava_image_tokens(
        processor, image, image_token_text, image_token_id,
    )
    return forward_kwargs, int(image_token_id), num_image_tokens


def _count_llava_image_tokens(
    processor: Any,
    image: Image.Image,
    image_token_text: str,
    image_token_id: int,
) -> int:
    """Ask the processor how many image tokens it would emit; fall back to 1.

    Older transformers versions don't pre-expand the image placeholder; the
    model handles expansion at forward time, so a single image token is
    sufficient and keeps input_ids length aligned with output logits. The
    fallback is logged at DEBUG so it stays auditable without spamming
    production runs.
    """
    try:
        proc_out = processor(text=image_token_text, images=image, return_tensors="pt")
        ids = proc_out["input_ids"][0].tolist()
        count = sum(1 for t in ids if t == image_token_id)
        if count >= 1:
            return count
    except (KeyError, IndexError, RuntimeError, ValueError, AttributeError, TypeError):
        logger.debug(
            "LLaVA-NeXT processor did not pre-expand image placeholder; "
            "falling back to num_image_tokens=1.",
            exc_info=True,
        )
    return 1


def _emit_llava_assistant_step(
    segments: list[Segment],
    thought: str,
    call: ToolCall,
) -> None:
    """Append one assistant-turn body (Thought + tool-call JSON + EOS)."""
    segments.append(Segment("Thought: ", SegmentKind.SEPARATOR))
    segments.append(Segment(thought, SegmentKind.THOUGHT))
    segments.append(Segment('\n{"tool": ', SegmentKind.SEPARATOR))
    segments.append(Segment(f'"{call.name}"', SegmentKind.TOOL_NAME))
    segments.append(Segment(', "args": ', SegmentKind.SEPARATOR))
    segments.append(
        Segment(json.dumps(call.args, ensure_ascii=False), SegmentKind.TOOL_ARGS)
    )
    segments.append(Segment("}", SegmentKind.SEPARATOR))
    segments.append(Segment(_LLAVA_EOS_SENTINEL, SegmentKind.SEPARATOR))


def _append_llava_user_prelude(
    segments: list[Segment], sys_text: str, user_prompt: str,
) -> None:
    """Append BOS + [INST] + image sentinel + system/prompt body + [/INST]."""
    body = f"{sys_text}\n\n{user_prompt}" if sys_text else user_prompt
    segments.append(Segment(_LLAVA_BOS_SENTINEL, SegmentKind.SEPARATOR))
    segments.append(Segment(LLAVA_INST_OPEN, SegmentKind.SEPARATOR))
    segments.append(Segment(_LLAVA_IMG_SENTINEL, SegmentKind.USER))
    segments.append(Segment(f"\n{body}", SegmentKind.USER))
    segments.append(Segment(LLAVA_INST_CLOSE, SegmentKind.SEPARATOR))


def _append_llava_observation_block(
    segments: list[Segment], call: ToolCall,
) -> None:
    """Append the [INST] Observation: {obs} [/INST] block following a tool_call."""
    segments.append(Segment("[INST] Observation: ", SegmentKind.SEPARATOR))
    segments.append(Segment(_format_observation(call), SegmentKind.OBSERVATION))
    segments.append(Segment(LLAVA_INST_CLOSE, SegmentKind.SEPARATOR))


def _append_llava_final_answer(
    segments: list[Segment], final_answer: str,
) -> None:
    """Append the final assistant answer followed by EOS sentinel."""
    segments.append(Segment(final_answer, SegmentKind.ANSWER))
    segments.append(Segment(_LLAVA_EOS_SENTINEL, SegmentKind.SEPARATOR))


def _build_llava_segments(
    user_prompt: str,
    trajectory: Trajectory,
    *,
    system_prompt: str | None = None,
) -> list[Segment]:
    """Linearize (prompt, trajectory) into ordered segments for LLaVA-NeXT.

    Mistral-Instruct is single-role; we emit a multi-turn ReAct conversation
    matching the inference template baked into ``LlavaNext._format_prompt``:

        <BOS>[INST] <IMG>\\n{system}\\n\\n{prompt} [/INST] Thought: {t1}
        {"tool": "<name>", "args": {...}}<EOS>
        [INST] Observation: {obs1} [/INST] Thought: {t2}
        {"tool": "<name>", "args": {...}}<EOS>
        ...
        [INST] Observation: {obsN} [/INST] {final_answer}<EOS>

    BOS / EOS / image-token positions are emitted as sentinel segments so
    ``_tokenize_llava_segments`` can substitute them with literal token ids.
    """
    sys_text = system_prompt if system_prompt is not None else LLAVA_DEFAULT_SYSTEM
    segments: list[Segment] = []

    _append_llava_user_prelude(segments, sys_text, user_prompt)

    n_calls = len(trajectory.tool_calls)
    if n_calls == 0:
        _append_llava_final_answer(segments, trajectory.final_answer)
        return segments

    thoughts = _split_thoughts(trajectory.reasoning_trace, n_steps=n_calls)
    _emit_llava_assistant_step(segments, thoughts[0], trajectory.tool_calls[0])
    for i in range(1, n_calls):
        _append_llava_observation_block(segments, trajectory.tool_calls[i - 1])
        _emit_llava_assistant_step(segments, thoughts[i], trajectory.tool_calls[i])

    _append_llava_observation_block(segments, trajectory.tool_calls[-1])
    _append_llava_final_answer(segments, trajectory.final_answer)
    return segments


def _tokenize_llava_segments(
    segments: list[Segment],
    tokenizer: Any,
    image_token_id: int,
    num_image_tokens: int,
) -> tuple[list[int], list[int]]:
    """Tokenize segments; substitute BOS / EOS / image sentinels with token ids."""
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    all_ids: list[int] = []
    all_seg: list[int] = []
    for seg in segments:
        if seg.text == _LLAVA_IMG_SENTINEL:
            all_ids.extend([image_token_id] * num_image_tokens)
            all_seg.extend([int(seg.kind)] * num_image_tokens)
            continue
        if seg.text == _LLAVA_BOS_SENTINEL:
            if bos_id is not None:
                all_ids.append(int(bos_id))
                all_seg.append(int(seg.kind))
            continue
        if seg.text == _LLAVA_EOS_SENTINEL:
            if eos_id is not None:
                all_ids.append(int(eos_id))
                all_seg.append(int(seg.kind))
            continue
        if seg.text == "":
            continue
        ids = tokenizer.encode(seg.text, add_special_tokens=False)
        if not ids:
            continue
        all_ids.extend(ids)
        all_seg.extend([int(seg.kind)] * len(ids))
    return all_ids, all_seg


def _require_llava_special_tokens(tokenizer: Any) -> None:
    """Verify the LLaVA-NeXT tokenizer exposes both BOS and EOS token ids.

    Mistral-Instruct's chat template needs both to bracket turns: dropping BOS
    mis-routes the first user prompt; dropping EOS collapses assistant-turn
    boundaries so the loss mask leaks across turns. The tokenize step is
    permissive (sentinels skipped silently when ids are None), so we fail loud
    here instead of producing quietly corrupt teacher-forced sequences.
    """
    if getattr(tokenizer, "bos_token_id", None) is None:
        raise RuntimeError(
            "LLaVA-NeXT tokenizer is missing bos_token_id; cannot build "
            "teacher-forced sequence. Check that the processor was loaded "
            "from a Mistral-Instruct-compatible checkpoint."
        )
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise RuntimeError(
            "LLaVA-NeXT tokenizer is missing eos_token_id; cannot delimit "
            "assistant turns. Check that the processor was loaded from a "
            "Mistral-Instruct-compatible checkpoint."
        )


def assemble_llava_next(
    user_prompt: str,
    trajectory: Trajectory,
    image: Image.Image,
    processor: Any,
    *,
    system_prompt: str | None = None,
    weights: MaskWeights = DEFAULT_MASK_WEIGHTS,
) -> TeacherForcedBatch:
    """Family-specific assembler for LLaVA-NeXT (Mistral-Instruct backbone).

    Parameters mirror :func:`assemble_qwen`. ``processor`` must be a
    ``LlavaNextProcessor``. The returned ``forward_kwargs`` carry
    ``pixel_values``, ``image_sizes``, and ``attention_mask`` — the trio
    accepted by ``LlavaNext.forward_with_logits``.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    _require_llava_special_tokens(tokenizer)
    forward_kwargs, image_token_id, num_image_tokens = _process_image_llava(
        image, processor
    )
    segments = _build_llava_segments(
        user_prompt, trajectory, system_prompt=system_prompt
    )
    ids, seg = _tokenize_llava_segments(
        segments, tokenizer, image_token_id, num_image_tokens
    )
    if not ids:
        raise ValueError("assemble_llava_next produced zero tokens; empty trajectory?")

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
