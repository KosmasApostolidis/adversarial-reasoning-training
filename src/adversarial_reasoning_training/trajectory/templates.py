"""Per-family teacher-forced template assemblers.

Each VLM family (Qwen2.5-VL, LLaVA-NeXT, InternVL2) has its own chat
formatting and image-token expansion rules. The public ``assemble_*``
functions here turn a (prompt, trajectory, image, processor) tuple into
a :class:`TeacherForcedBatch` that the trainer can feed back into
``forward_with_logits``.

The orchestrator + shared helpers (``TeacherForcedBatch``,
``_split_thoughts``, ``_format_observation``, ``assemble``) live in
``teacher_force.py``; this module is import-side dependent on those.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from adversarial_reasoning.agents.base import ToolCall, Trajectory
from PIL import Image

from .mask import build_masks, labels_from_input_ids
from .segments import DEFAULT_MASK_WEIGHTS, MaskWeights, Segment, SegmentKind
from .teacher_force import TeacherForcedBatch, _format_observation, _split_thoughts

# --- Qwen2.5-VL chat template literals --------------------------------------

QWEN_IM_START = "<|im_start|>"
QWEN_IM_END = "<|im_end|>"
QWEN_VIS_START = "<|vision_start|>"
QWEN_VIS_END = "<|vision_end|>"
QWEN_IMAGE_PAD = "<|image_pad|>"
QWEN_TOOL_OPEN = "<tool_call>"
QWEN_TOOL_CLOSE = "</tool_call>"


# --- LLaVA-NeXT (Mistral-Instruct) chat template literals -------------------

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


# --- InternVL2 (InternLM2-Chat backbone) chat template literals -------------
# InternVL2-8B reuses Qwen-style ``<|im_start|>``/``<|im_end|>`` role markers
# but expands one ``<image>`` placeholder into
# ``<img>`` + N copies of ``<IMG_CONTEXT>`` + ``</img>`` where
# ``N == num_patches * num_image_token``. Both are runtime-determined per
# image (anyres tiling + vision-tower output count), so a sentinel-driven
# tokenization pass is reused.

INTERNVL_IMG_START = "<img>"
INTERNVL_IMG_END = "</img>"
INTERNVL_IMG_CONTEXT = "<IMG_CONTEXT>"
INTERNVL_DEFAULT_SYSTEM = (
    "You are a medical-imaging VLM agent. Reason step by step and call tools."
)
_INTERNVL_IMG_SENTINEL = "<__INTERNVL_IMG__>"


# --- Qwen2.5-VL -------------------------------------------------------------


def _format_tool_call_json(call: ToolCall) -> str:
    """Serialize a ToolCall into Qwen's `<tool_call>` JSON body."""
    payload = {"name": call.name, "arguments": call.args}
    return json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))


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
        <|im_start|>assistant\n
            Thought: {thought_2}\n
            <tool_call>{...tool_2}</tool_call>\n
        <|im_end|>\n
        ...
        <|im_start|>assistant\n
            Answer: {final_answer}\n
        <|im_end|>

    Reasoning-trace text is split by `\\n---\\n` into per-step thoughts
    if present; otherwise every assistant turn gets an empty Thought.
    """
    sys_text = (
        system_prompt
        if system_prompt is not None
        else "You are a medical-imaging VLM agent. Reason step by step and call tools."
    )
    segments: list[Segment] = []

    # --- System
    segments.append(Segment(f"{QWEN_IM_START}system\n", SegmentKind.SEPARATOR))
    segments.append(Segment(sys_text, SegmentKind.SYSTEM))
    segments.append(Segment(f"\n{QWEN_IM_END}\n", SegmentKind.SEPARATOR))

    # --- User (image + prompt)
    segments.append(Segment(f"{QWEN_IM_START}user\n", SegmentKind.SEPARATOR))
    image_pad_run = QWEN_IMAGE_PAD * max(1, num_image_pad)
    segments.append(
        Segment(f"{QWEN_VIS_START}{image_pad_run}{QWEN_VIS_END}\n", SegmentKind.USER)
    )
    segments.append(Segment(user_prompt, SegmentKind.USER))
    segments.append(Segment(f"\n{QWEN_IM_END}\n", SegmentKind.SEPARATOR))

    # --- Assistant turns, one per tool_call + a final answer turn
    thoughts = _split_thoughts(trajectory.reasoning_trace, n_steps=len(trajectory.tool_calls))
    for i, call in enumerate(trajectory.tool_calls):
        segments.append(Segment(f"{QWEN_IM_START}assistant\n", SegmentKind.SEPARATOR))
        segments.append(Segment("Thought: ", SegmentKind.SEPARATOR))
        segments.append(Segment(thoughts[i], SegmentKind.THOUGHT))
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

        # Tool observation — weight 0 via OBSERVATION kind.
        obs_text = _format_observation(call)
        segments.append(Segment(f"{QWEN_IM_START}tool\n", SegmentKind.SEPARATOR))
        segments.append(Segment(obs_text, SegmentKind.OBSERVATION))
        segments.append(Segment(f"\n{QWEN_IM_END}\n", SegmentKind.SEPARATOR))

    # --- Final answer turn
    segments.append(Segment(f"{QWEN_IM_START}assistant\n", SegmentKind.SEPARATOR))
    segments.append(Segment("Answer: ", SegmentKind.SEPARATOR))
    segments.append(Segment(trajectory.final_answer, SegmentKind.ANSWER))
    segments.append(Segment(f"\n{QWEN_IM_END}", SegmentKind.SEPARATOR))

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


# --- LLaVA-NeXT -------------------------------------------------------------


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

    num_image_tokens = 1
    try:
        proc_out = processor(text=image_token_text, images=image, return_tensors="pt")
        ids = proc_out["input_ids"][0].tolist()
        count = sum(1 for t in ids if t == image_token_id)
        if count >= 1:
            num_image_tokens = count
    except Exception:
        # Older transformers versions don't pre-expand the image placeholder; the
        # model handles expansion at forward time, so a single image token is
        # sufficient and keeps input_ids length aligned with output logits.
        num_image_tokens = 1

    return forward_kwargs, int(image_token_id), num_image_tokens


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
    body = f"{sys_text}\n\n{user_prompt}" if sys_text else user_prompt

    segments: list[Segment] = []
    segments.append(Segment(_LLAVA_BOS_SENTINEL, SegmentKind.SEPARATOR))
    segments.append(Segment(LLAVA_INST_OPEN, SegmentKind.SEPARATOR))
    segments.append(Segment(_LLAVA_IMG_SENTINEL, SegmentKind.USER))
    segments.append(Segment(f"\n{body}", SegmentKind.USER))
    segments.append(Segment(LLAVA_INST_CLOSE, SegmentKind.SEPARATOR))

    n_calls = len(trajectory.tool_calls)
    if n_calls == 0:
        segments.append(Segment(trajectory.final_answer, SegmentKind.ANSWER))
        segments.append(Segment(_LLAVA_EOS_SENTINEL, SegmentKind.SEPARATOR))
        return segments

    thoughts = _split_thoughts(trajectory.reasoning_trace, n_steps=n_calls)
    _emit_llava_assistant_step(segments, thoughts[0], trajectory.tool_calls[0])

    for i in range(1, n_calls):
        prev_obs = _format_observation(trajectory.tool_calls[i - 1])
        segments.append(Segment("[INST] Observation: ", SegmentKind.SEPARATOR))
        segments.append(Segment(prev_obs, SegmentKind.OBSERVATION))
        segments.append(Segment(LLAVA_INST_CLOSE, SegmentKind.SEPARATOR))
        _emit_llava_assistant_step(segments, thoughts[i], trajectory.tool_calls[i])

    last_obs = _format_observation(trajectory.tool_calls[-1])
    segments.append(Segment("[INST] Observation: ", SegmentKind.SEPARATOR))
    segments.append(Segment(last_obs, SegmentKind.OBSERVATION))
    segments.append(Segment(LLAVA_INST_CLOSE, SegmentKind.SEPARATOR))
    segments.append(Segment(trajectory.final_answer, SegmentKind.ANSWER))
    segments.append(Segment(_LLAVA_EOS_SENTINEL, SegmentKind.SEPARATOR))

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
    # Mistral-Instruct's chat template needs both BOS and EOS to bracket
    # turns: dropping BOS mis-routes the first user prompt; dropping EOS
    # collapses assistant-turn boundaries and the loss mask leaks across
    # turns. The tokenize step is permissive (sentinels skipped silently
    # when ids are None), so we fail loud here instead of producing
    # quietly corrupt teacher-forced sequences.
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


# --- InternVL2 --------------------------------------------------------------


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
    segments.append(Segment(f"{QWEN_IM_START}system\n", SegmentKind.SEPARATOR))
    segments.append(Segment(sys_text, SegmentKind.SYSTEM))
    segments.append(Segment(f"{QWEN_IM_END}\n", SegmentKind.SEPARATOR))

    segments.append(Segment(f"{QWEN_IM_START}user\n", SegmentKind.SEPARATOR))
    segments.append(Segment(INTERNVL_IMG_START, SegmentKind.SEPARATOR))
    segments.append(Segment(_INTERNVL_IMG_SENTINEL, SegmentKind.USER))
    segments.append(Segment(INTERNVL_IMG_END, SegmentKind.SEPARATOR))
    segments.append(Segment(f"\n{user_prompt}", SegmentKind.USER))
    segments.append(Segment(f"{QWEN_IM_END}\n", SegmentKind.SEPARATOR))

    n_calls = len(trajectory.tool_calls)
    if n_calls == 0:
        segments.append(Segment(f"{QWEN_IM_START}assistant\n", SegmentKind.SEPARATOR))
        segments.append(Segment(trajectory.final_answer, SegmentKind.ANSWER))
        segments.append(Segment(QWEN_IM_END, SegmentKind.SEPARATOR))
        return segments

    thoughts = _split_thoughts(trajectory.reasoning_trace, n_steps=n_calls)
    for i, call in enumerate(trajectory.tool_calls):
        segments.append(Segment(f"{QWEN_IM_START}assistant\n", SegmentKind.SEPARATOR))
        _emit_internvl_assistant_step(segments, thoughts[i], call)
        segments.append(Segment(f"{QWEN_IM_END}\n", SegmentKind.SEPARATOR))

        obs_text = _format_observation(call)
        segments.append(Segment(f"{QWEN_IM_START}tool\n", SegmentKind.SEPARATOR))
        segments.append(Segment(obs_text, SegmentKind.OBSERVATION))
        segments.append(Segment(f"{QWEN_IM_END}\n", SegmentKind.SEPARATOR))

    segments.append(Segment(f"{QWEN_IM_START}assistant\n", SegmentKind.SEPARATOR))
    segments.append(Segment(trajectory.final_answer, SegmentKind.ANSWER))
    segments.append(Segment(QWEN_IM_END, SegmentKind.SEPARATOR))

    return segments


def _tokenize_internvl_segments(
    segments: list[Segment],
    tokenizer: Any,
    img_context_id: int,
    num_image_tokens: int,
) -> tuple[list[int], list[int]]:
    """Tokenize segments; substitute the image sentinel with N IMG_CONTEXT ids."""
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


__all__ = [
    "assemble_internvl",
    "assemble_llava_next",
    "assemble_qwen",
]
