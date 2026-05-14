"""Teacher-forced trajectory linearization — orchestrator.

Turns a `Trajectory` (tool calls + thoughts + final answer) and a user
prompt into:

    (input_ids, attention_mask, segment_ids, task_mask, traj_mask, labels)

plus the model-specific `forward_kwargs` (pixel_values, image_grid_thw)
needed to replay the forward pass through a VLM. The result is fed into
a single `forward_with_logits` call — no autoregressive sampling during
training.

The assembler is family-dispatched: each VLM family (Qwen2.5-VL,
LLaVA-NeXT, InternVL2) needs its own per-segment chat formatting, but
the mask bookkeeping is shared. Family-specific logic lives in
``trajectory.templates``; this module owns the public ``TeacherForcedBatch``
dataclass and the ``assemble()`` dispatch entrypoint.

This module is the hinge of the whole training loop: if segment ids
or token offsets drift by even one position, both the task loss and
the PGD inner objective become silently wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch
from adversarial_reasoning.agents.base import ToolCall, Trajectory
from PIL import Image

from ..utils.constants import VLMFamily
from .segments import DEFAULT_MASK_WEIGHTS, MaskWeights


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
    segments: list[Any] = field(default_factory=list)

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

    Supported families:

    * ``qwen_vl`` → :func:`templates.assemble_qwen` — ``processor`` is a
      Qwen2.5-VL ``AutoProcessor`` (carries ``.tokenizer`` + ``.image_processor``).
    * ``llava_next`` → :func:`templates.assemble_llava_next` — ``processor``
      is a ``LlavaNextProcessor``.
    * ``internvl2`` → :func:`templates.assemble_internvl` — ``processor`` is
      the attacks-repo ``InternVL2`` wrapper itself (carries ``.tokenizer``,
      ``.preprocess_image``, ``._num_image_token``); InternVL2 has no
      first-class HF processor.
    """
    # Deferred import: templates imports TeacherForcedBatch from this module,
    # so a top-level import would create a cycle. Late binding keeps the
    # module-load graph acyclic.
    from .templates import (
        assemble_internvl,
        assemble_llava_next,
        assemble_llava_onevision,
        assemble_qwen,
    )

    # Centralised dispatch — keeps the canonical family-name set in one
    # place so callers get a clear ``ValueError`` listing valid options on
    # a typo (e.g. "qwen2_5_vl") instead of the older
    # ``NotImplementedError`` which was indistinguishable from "support
    # not yet wired up". When adding a new family, register it here.
    dispatch = {
        VLMFamily.QWEN_VL.value: assemble_qwen,
        VLMFamily.LLAVA_NEXT.value: assemble_llava_next,
        VLMFamily.LLAVA_ONEVISION.value: assemble_llava_onevision,
        VLMFamily.INTERNVL2.value: assemble_internvl,
    }
    fn = dispatch.get(family)
    if fn is None:
        raise ValueError(
            f"Unknown family={family!r}; expected one of "
            f"{sorted(dispatch)}."
        )
    return fn(
        user_prompt, trajectory, image, processor,
        system_prompt=system_prompt, weights=weights,
    )
