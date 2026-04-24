"""Mask-tensor construction from a segment-id stream."""

from __future__ import annotations

import torch

from .segments import MaskWeights, SegmentKind


def build_masks(
    segment_ids: torch.Tensor,
    weights: MaskWeights,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a per-token segment-id tensor into (task_mask, traj_mask).

    Parameters
    ----------
    segment_ids : LongTensor of shape [B, T] with values from `SegmentKind`.
    weights : per-kind scalar weights.

    Returns
    -------
    task_mask, traj_mask : FloatTensors of shape [B, T].
    """
    task_mask = torch.zeros_like(segment_ids, dtype=torch.float32)
    traj_mask = torch.zeros_like(segment_ids, dtype=torch.float32)
    for kind in SegmentKind:
        if kind == SegmentKind.PAD:
            continue
        idx = segment_ids == int(kind)
        if idx.any():
            task_mask[idx] = weights.for_task(kind)
            traj_mask[idx] = weights.for_traj(kind)
    return task_mask, traj_mask


def labels_from_input_ids(
    input_ids: torch.Tensor,
    task_mask: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Build teacher-forced labels: positions with mask==0 become `ignore_index`.

    Intended for use with `F.cross_entropy(..., ignore_index=-100)` if you
    prefer the HF convention. Our own `task_ce` loss uses the mask directly
    and does not rely on ignore_index, but returning labels is convenient
    for compatibility with HF `Trainer`-style callers.
    """
    labels = input_ids.clone()
    labels[task_mask == 0.0] = ignore_index
    return labels
