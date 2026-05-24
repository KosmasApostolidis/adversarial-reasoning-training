"""Chat template markers shared across the Qwen and InternVL2 assemblers.

Qwen2.5-VL and InternVL2 both ride InternLM-style ``<|im_start|>`` /
``<|im_end|>`` role brackets — the latter inherits the convention from
InternLM2-Chat. Keeping the literals here avoids the cross-module
duplication the pre-split ``templates.py`` carried (it defined
``QWEN_IM_START`` once and InternVL2 re-used the symbol by name).

LLaVA-NeXT uses Mistral-Instruct ``[INST] ... [/INST]`` and has its own
markers in :mod:`._llava_next`.
"""

from __future__ import annotations

import torch

from ..mask import build_masks, labels_from_input_ids
from ..segments import MaskWeights, Segment
from ..teacher_force import TeacherForcedBatch

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def pack_teacher_forced_batch(
    *,
    ids: list[int],
    seg: list[int],
    segments: list[Segment],
    forward_kwargs: dict[str, torch.Tensor],
    weights: MaskWeights,
) -> TeacherForcedBatch:
    """Wrap tokenized ids/segments into a TeacherForcedBatch.

    Builds the three masks (task / traj / attention), packs everything into
    1xT tensors, and threads attention_mask through ``forward_kwargs`` so the
    HF wrapper sees it. Shared across all family assemblers because the tail
    of every ``assemble_*`` is mechanically identical.
    """
    input_ids = torch.tensor([ids], dtype=torch.long)
    segment_ids = torch.tensor([seg], dtype=torch.long)
    # B=1 invariant: no padding exists, so attention_mask is always all-ones.
    # If B>1 batching is ever supported, this must be derived from actual
    # padding positions (TFCollator already raises NotImplementedError for B>1).
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
