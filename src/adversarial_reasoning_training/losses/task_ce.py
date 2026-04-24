"""Token-level cross-entropy masked by a per-token weight tensor.

Causal-LM invariant: `logits[:, i, :]` predicts `input_ids[:, i+1]`.
We shift accordingly and weight each per-position CE by `task_mask[:, i+1]`
so that only tool_name / tool_args / answer / thought tokens contribute.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def task_ce(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    task_mask: torch.Tensor,
) -> torch.Tensor:
    """Weighted masked token CE.

    Parameters
    ----------
    logits : FloatTensor [B, T, V]
    input_ids : LongTensor [B, T]
    task_mask : FloatTensor [B, T]

    Returns
    -------
    scalar tensor. Sum of (per-token CE × shifted mask) divided by
    mask-sum, so the scale is invariant to sequence length.
    """
    if logits.dim() != 3 or input_ids.dim() != 2 or task_mask.dim() != 2:
        raise ValueError("Expected logits [B,T,V], input_ids [B,T], task_mask [B,T]")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = task_mask[:, 1:].contiguous()
    vocab = shift_logits.size(-1)
    ce = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    weighted = ce * shift_mask
    denom = shift_mask.sum().clamp_min(1.0)
    return weighted.sum() / denom
