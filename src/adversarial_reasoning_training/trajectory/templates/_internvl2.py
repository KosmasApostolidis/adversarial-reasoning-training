"""InternVL2 (InternLM2-Chat backbone) teacher-forced template assembler.

InternVL2-8B reuses Qwen-style ``<|im_start|>`` / ``<|im_end|>`` role markers
but expands one ``<image>`` placeholder into
``<img>`` + N copies of ``<IMG_CONTEXT>`` + ``</img>`` where
``N == num_patches * num_image_token``. Both are runtime-determined per
image (anyres tiling + vision-tower output count), so a sentinel-driven
tokenization pass is reused.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from adversarial_reasoning.agents.base import ToolCall, Trajectory
from PIL import Image

from ..segments import DEFAULT_MASK_WEIGHTS, MaskWeights, Segment, SegmentKind
from ..teacher_force import TeacherForcedBatch, _format_observation, _split_thoughts
from ._common import IM_END, IM_START, pack_teacher_forced_batch

INTERNVL_IMG_START = "<img>"
INTERNVL_IMG_END = "</img>"
INTERNVL_IMG_CONTEXT = "<IMG_CONTEXT>"
INTERNVL_DEFAULT_SYSTEM = (
    "You are a medical-imaging VLM agent. Reason step by step and call tools."
)
_INTERNVL_IMG_SENTINEL = "<__INTERNVL_IMG__>"


def _process_image_internvl(
    image: Image.Image,
    vlm: Any,
) -> tuple[dict[str, torch.Tensor], int, int, int]:
    """Run InternVL2's tile preprocessor and read off image-token sizing.

    InternVL2 has no formal HF processor — the wrapper class exposes
    :meth:`preprocess_image` (tile + normalise → ``[num_patches, 3, H, W]``)
    and a ``_num_image_token`` attribute (typ. 256, the per-patch
    post-merge image-token count). Both values plus the
    ``<IMG_CONTEXT>`` token id are needed to expand the image placeholder
    in ``input_ids`` so the model's IMG_CONTEXT-replacement step in
    forward sees the right number of slots.
    """
    if not hasattr(vlm, "preprocess_image"):
        raise RuntimeError(
            "InternVL2 assembler expected vlm with .preprocess_image(); got "
            f"{type(vlm).__name__}."
        )
    pixel_values = vlm.preprocess_image(image)
    if pixel_values.dim() != 4:
        raise RuntimeError(
            "InternVL2 preprocess_image must return [num_patches, 3, H, W]; "
            f"got shape {tuple(pixel_values.shape)}."
        )
    num_patches = int(pixel_values.shape[0])

    tokenizer = getattr(vlm, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("InternVL2 assembler requires vlm.tokenizer.")

    num_image_token = int(getattr(vlm, "_num_image_token", 0))
    if num_image_token <= 0:
        # Fall back to model attribute (set by the canonical OpenGVLab code).
        model = getattr(vlm, "model", None)
        num_image_token = int(getattr(model, "num_image_token", 0)) if model else 0
    if num_image_token <= 0:
        raise RuntimeError(
            "InternVL2 assembler could not resolve num_image_token from vlm "
            "or vlm.model; refusing to emit zero IMG_CONTEXT tokens."
        )

    img_context_id = tokenizer.convert_tokens_to_ids(INTERNVL_IMG_CONTEXT)
    if img_context_id is None or img_context_id == getattr(tokenizer, "unk_token_id", -1):
        raise RuntimeError(
            f"InternVL2 tokenizer does not expose {INTERNVL_IMG_CONTEXT!r} as a token."
        )

    forward_kwargs: dict[str, torch.Tensor] = {"pixel_values": pixel_values}
    return forward_kwargs, int(img_context_id), num_image_token, num_patches


def _emit_internvl_assistant_step(
    segments: list[Segment],
    thought: str,
    call: ToolCall,
) -> None:
    """Append one assistant-turn body (Thought + tool-call JSON), no closing
    role marker — caller decides whether to close with <|im_end|> or fold
    into the next observation block.
    """
    segments.append(Segment("Thought: ", SegmentKind.SEPARATOR))
    segments.append(Segment(thought, SegmentKind.THOUGHT))
    segments.append(Segment('\n{"tool": ', SegmentKind.SEPARATOR))
    segments.append(Segment(f'"{call.name}"', SegmentKind.TOOL_NAME))
    segments.append(Segment(', "args": ', SegmentKind.SEPARATOR))
    segments.append(
        Segment(json.dumps(call.args, ensure_ascii=False), SegmentKind.TOOL_ARGS)
    )
    segments.append(Segment("}", SegmentKind.SEPARATOR))


def _append_internvl_system(segments: list[Segment], sys_text: str) -> None:
    """Append the system turn prefacing every InternVL2 conversation."""
    segments.append(Segment(f"{IM_START}system\n", SegmentKind.SEPARATOR))
    segments.append(Segment(sys_text, SegmentKind.SYSTEM))
    segments.append(Segment(f"{IM_END}\n", SegmentKind.SEPARATOR))


def _append_internvl_user(segments: list[Segment], user_prompt: str) -> None:
    """Append the user turn carrying the <img><CTX>...<CTX></img> + prompt."""
    segments.append(Segment(f"{IM_START}user\n", SegmentKind.SEPARATOR))
    segments.append(Segment(INTERNVL_IMG_START, SegmentKind.SEPARATOR))
    segments.append(Segment(_INTERNVL_IMG_SENTINEL, SegmentKind.USER))
    segments.append(Segment(INTERNVL_IMG_END, SegmentKind.SEPARATOR))
    segments.append(Segment(f"\n{user_prompt}", SegmentKind.USER))
    segments.append(Segment(f"{IM_END}\n", SegmentKind.SEPARATOR))


def _append_internvl_assistant_turn(
    segments: list[Segment], thought: str, call: ToolCall,
) -> None:
    """Append a complete assistant tool-call turn (role separators + body)."""
    segments.append(Segment(f"{IM_START}assistant\n", SegmentKind.SEPARATOR))
    _emit_internvl_assistant_step(segments, thought, call)
    segments.append(Segment(f"{IM_END}\n", SegmentKind.SEPARATOR))


def _append_internvl_tool_observation(segments: list[Segment], call: ToolCall) -> None:
    """Append the tool-observation turn following an assistant tool_call."""
    segments.append(Segment(f"{IM_START}tool\n", SegmentKind.SEPARATOR))
    segments.append(Segment(_format_observation(call), SegmentKind.OBSERVATION))
    segments.append(Segment(f"{IM_END}\n", SegmentKind.SEPARATOR))


def _append_internvl_final_answer(segments: list[Segment], final_answer: str) -> None:
    """Append the closing assistant turn emitting the final answer."""
    segments.append(Segment(f"{IM_START}assistant\n", SegmentKind.SEPARATOR))
    segments.append(Segment(final_answer, SegmentKind.ANSWER))
    segments.append(Segment(IM_END, SegmentKind.SEPARATOR))


def _build_internvl_segments(
    user_prompt: str,
    trajectory: Trajectory,
    *,
    system_prompt: str | None = None,
) -> list[Segment]:
    """Linearize (prompt, trajectory) into ordered segments for InternVL2.

    InternLM2-Chat layout (multi-turn ReAct, image embedded in user turn):

        <|im_start|>system\\n{sys}<|im_end|>\\n
        <|im_start|>user\\n<img><CTX>...<CTX></img>\\n{prompt}<|im_end|>\\n
        <|im_start|>assistant\\n
            Thought: {t1}\\n{"tool":...,"args":...}<|im_end|>\\n
        <|im_start|>tool\\n{obs1}<|im_end|>\\n
        <|im_start|>assistant\\n
            Thought: {t2}\\n{"tool":...,"args":...}<|im_end|>\\n
        ...
        <|im_start|>assistant\\n{final_answer}<|im_end|>

    The ``<CTX>...<CTX>`` run is emitted as a single sentinel segment;
    ``_tokenize_internvl_segments`` substitutes it with N copies of the
    ``<IMG_CONTEXT>`` token id where N = num_patches * num_image_token.
    """
    sys_text = system_prompt if system_prompt is not None else INTERNVL_DEFAULT_SYSTEM
    segments: list[Segment] = []

    _append_internvl_system(segments, sys_text)
    _append_internvl_user(segments, user_prompt)

    n_calls = len(trajectory.tool_calls)
    if n_calls == 0:
        _append_internvl_final_answer(segments, trajectory.final_answer)
        return segments

    thoughts = _split_thoughts(trajectory.reasoning_trace, n_steps=n_calls)
    for i, call in enumerate(trajectory.tool_calls):
        _append_internvl_assistant_turn(segments, thoughts[i], call)
        _append_internvl_tool_observation(segments, call)

    _append_internvl_final_answer(segments, trajectory.final_answer)
    return segments


def _tokenize_internvl_segments(
    segments: list[Segment],
    tokenizer: Any,
    img_context_id: int,
    num_image_tokens: int,
) -> tuple[list[int], list[int]]:
    """Tokenize segments; substitute the image sentinel with N IMG_CONTEXT ids."""
    # NOTE: add_special_tokens=False means BOS is NOT prepended.  InternLM2
    # chat models use ``<|im_start|>`` as the sequence delimiter, which
    # subsumes the BOS role.  If the tokenizer's chat template changes,
    # verify that BOS handling here matches what ``generate()`` produces.
    all_ids: list[int] = []
    all_seg: list[int] = []
    for seg in segments:
        if seg.text == _INTERNVL_IMG_SENTINEL:
            all_ids.extend([img_context_id] * num_image_tokens)
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


def assemble_internvl(
    user_prompt: str,
    trajectory: Trajectory,
    image: Image.Image,
    vlm: Any,
    *,
    system_prompt: str | None = None,
    weights: MaskWeights = DEFAULT_MASK_WEIGHTS,
) -> TeacherForcedBatch:
    """Family-specific assembler for InternVL2 (InternLM2-Chat backbone).

    Unlike :func:`assemble_qwen` / :func:`assemble_llava_next`, this takes
    the full attacks-repo VLM wrapper (``vlm``) instead of a HF processor.
    InternVL2 ships custom modeling code without a first-class
    ``AutoProcessor``; the wrapper's :meth:`preprocess_image` does the
    OpenGVLab dynamic-tile recipe and ``_num_image_token`` carries the
    per-patch image-token count needed for the IMG_CONTEXT expansion. The
    returned ``forward_kwargs`` thread ``pixel_values`` and
    ``attention_mask`` — the pair accepted by
    ``InternVL2.forward_with_logits``.
    """
    forward_kwargs, img_context_id, num_image_token, num_patches = (
        _process_image_internvl(image, vlm)
    )
    tokenizer = vlm.tokenizer
    segments = _build_internvl_segments(
        user_prompt, trajectory, system_prompt=system_prompt
    )
    total_image_tokens = num_image_token * num_patches
    ids, seg = _tokenize_internvl_segments(
        segments, tokenizer, img_context_id, total_image_tokens
    )
    if not ids:
        raise ValueError("assemble_internvl produced zero tokens; empty trajectory?")

    return pack_teacher_forced_batch(
        ids=ids, seg=seg, segments=segments,
        forward_kwargs=forward_kwargs, weights=weights,
    )
