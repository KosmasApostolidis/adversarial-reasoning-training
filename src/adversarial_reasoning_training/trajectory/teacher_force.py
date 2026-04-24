"""Teacher-forced trajectory linearization.

Turns a `Trajectory` (tool calls + thoughts + final answer) and a user
prompt into:

    (input_ids, attention_mask, segment_ids, task_mask, traj_mask, labels)

plus the model-specific `forward_kwargs` (pixel_values, image_grid_thw)
needed to replay the forward pass through a VLM. The result is fed into
a single `forward_with_logits` call — no autoregressive sampling during
training.

The assembler is family-dispatched: each VLM family (Qwen2.5-VL today,
LLaVA-v1.6 in Phase 2) needs its own per-segment chat formatting, but
the mask bookkeeping is shared.

This module is the hinge of the whole training loop: if segment ids
or token offsets drift by even one position, both the task loss and
the PGD inner objective become silently wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

from adversarial_reasoning.agents.base import ToolCall, Trajectory

from .mask import build_masks, labels_from_input_ids
from .segments import DEFAULT_MASK_WEIGHTS, MaskWeights, Segment, SegmentKind


# --- Qwen2.5-VL chat template literals --------------------------------------

QWEN_IM_START = "<|im_start|>"
QWEN_IM_END = "<|im_end|>"
QWEN_VIS_START = "<|vision_start|>"
QWEN_VIS_END = "<|vision_end|>"
QWEN_IMAGE_PAD = "<|image_pad|>"
QWEN_TOOL_OPEN = "<tool_call>"
QWEN_TOOL_CLOSE = "</tool_call>"


@dataclass(frozen=True)
class TeacherForcedBatch:
    """Payload of `assemble()`.

    Fields
    ------
    input_ids : LongTensor [1, T]. Full teacher-forced token sequence.
    attention_mask : LongTensor [1, T].
    segment_ids : LongTensor [1, T]. Values from `SegmentKind`.
    task_mask : FloatTensor [1, T]. Weights for the supervised CE term.
    traj_mask : FloatTensor [1, T]. Weights for the clean-vs-adv KL term.
    labels : LongTensor [1, T]. `input_ids` with task_mask==0 set to -100.
    forward_kwargs : dict. Extra kwargs required by the VLM's forward pass
        (e.g., pixel_values, image_grid_thw, cross-attention masks).
    segments : list[Segment]. Source-of-truth ordered segment list for
        debugging / tests.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    segment_ids: torch.Tensor
    task_mask: torch.Tensor
    traj_mask: torch.Tensor
    labels: torch.Tensor
    forward_kwargs: dict[str, Any]
    segments: list[Segment] = field(default_factory=list)

    def to(self, device: torch.device | str) -> TeacherForcedBatch:
        return TeacherForcedBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            segment_ids=self.segment_ids.to(device),
            task_mask=self.task_mask.to(device),
            traj_mask=self.traj_mask.to(device),
            labels=self.labels.to(device),
            forward_kwargs={
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in self.forward_kwargs.items()
            },
            segments=self.segments,
        )


# --- Segment builders -------------------------------------------------------


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


def _split_thoughts(reasoning_trace: str, n_steps: int) -> list[str]:
    """Split reasoning_trace into per-step thoughts.

    We accept any of: an empty string, a single block, or a block with
    `\\n---\\n` delimiters. Shorter-than-expected splits are right-padded
    with the empty string; longer splits are right-truncated.
    """
    if n_steps <= 0:
        return []
    if not reasoning_trace:
        return [""] * n_steps
    parts = [p.strip() for p in reasoning_trace.split("\n---\n")]
    if len(parts) == 1 and n_steps > 1:
        parts = parts + [""] * (n_steps - 1)
    if len(parts) < n_steps:
        parts = parts + [""] * (n_steps - len(parts))
    return parts[:n_steps]


def _format_observation(call: ToolCall) -> str:
    """Stringify a ToolCall's observation (`result` or `error`)."""
    if call.error:
        return f"ERROR: {call.error}"
    if call.result is None:
        return "(no result)"
    if isinstance(call.result, (dict, list)):
        return json.dumps(call.result, ensure_ascii=False)
    return str(call.result)


# --- Core assembler ---------------------------------------------------------


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


def assemble(
    family: str,
    user_prompt: str,
    trajectory: Trajectory,
    image: Image.Image,
    processor: Any,
    *,
    system_prompt: str | None = None,
    weights: MaskWeights = DEFAULT_MASK_WEIGHTS,
) -> TeacherForcedBatch:
    """Family-dispatched assembler.

    Today: `family=="qwen_vl"` → `assemble_qwen`. Other families raise;
    LLaVA-v1.6 (family="llava_next") will be added in Phase 2 once Qwen
    training is green.
    """
    if family == "qwen_vl":
        return assemble_qwen(
            user_prompt, trajectory, image, processor,
            system_prompt=system_prompt, weights=weights,
        )
    raise NotImplementedError(
        f"Teacher-forced assembler for family={family!r} not implemented yet."
    )
