"""Teacher-forced trajectory linearization — the load-bearing module.

Turns a `Trajectory` (tool calls + thoughts + answer) into one token
sequence with per-position segment IDs so the whole ReAct chain can be
scored by a single forward pass.
"""

from .mask import build_masks, labels_from_input_ids
from .segments import (
    DEFAULT_MASK_WEIGHTS,
    MaskWeights,
    Segment,
    SegmentKind,
)
from .teacher_force import TeacherForcedBatch, assemble
from .templates import assemble_qwen

__all__ = [
    "DEFAULT_MASK_WEIGHTS",
    "MaskWeights",
    "Segment",
    "SegmentKind",
    "TeacherForcedBatch",
    "assemble",
    "assemble_qwen",
    "build_masks",
    "labels_from_input_ids",
]
