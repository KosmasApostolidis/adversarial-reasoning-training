"""Segment kinds + per-segment mask-weight configuration.

A `Segment` is a contiguous stretch of the teacher-forced sequence whose
tokens share the same role (system prompt, user prompt, CoT thought,
tool name, tool args, tool observation, final answer, or special
separator). Per-role loss weights let us score tool calls and the final
answer strongly while de-weighting reasoning prose and masking tool
observations entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class SegmentKind(IntEnum):
    PAD = 0
    SYSTEM = 1
    USER = 2
    THOUGHT = 3
    TOOL_NAME = 4
    TOOL_ARGS = 5
    OBSERVATION = 6
    ANSWER = 7
    SEPARATOR = 8


@dataclass(frozen=True)
class Segment:
    """A (text, kind) pair. Tokenized separately then concatenated."""

    text: str
    kind: SegmentKind


@dataclass(frozen=True)
class MaskWeights:
    """Per-segment weights for the task-CE mask and the trajectory-KL mask.

    task_mask: drives the supervised CE term (loss_task).
    traj_mask: drives the clean-vs-adv KL consistency term (loss_traj).

    Both masks default to 0 for segments that the model does not *produce*
    (SYSTEM, USER, OBSERVATION). Observations are deterministic functions
    of tool args; penalising the model for their tokens would teach it to
    memorise tool outputs instead of choosing tools well.
    """

    task: dict[SegmentKind, float] = field(
        default_factory=lambda: {
            SegmentKind.TOOL_NAME: 1.0,
            SegmentKind.TOOL_ARGS: 0.5,
            SegmentKind.ANSWER: 1.0,
            SegmentKind.THOUGHT: 0.25,
            SegmentKind.SEPARATOR: 0.0,
        }
    )
    traj: dict[SegmentKind, float] = field(
        default_factory=lambda: {
            SegmentKind.TOOL_NAME: 1.0,
            SegmentKind.TOOL_ARGS: 0.5,
            SegmentKind.ANSWER: 1.0,
            SegmentKind.THOUGHT: 0.25,
            SegmentKind.SEPARATOR: 0.0,
        }
    )

    def for_task(self, kind: SegmentKind) -> float:
        return float(self.task.get(kind, 0.0))

    def for_traj(self, kind: SegmentKind) -> float:
        return float(self.traj.get(kind, 0.0))


DEFAULT_MASK_WEIGHTS = MaskWeights()
