"""Collator that turns TrainSample(s) into TeacherForcedBatch.

For the full-FT adversarial loop we run micro_batch=1 + grad_accum=N,
so this collator assumes B=1 per call. A proper B>1 batched collator
would need length padding on input_ids / segment_ids / masks AND
image-grid batching — punt to Phase 2.
"""

from __future__ import annotations

from typing import Any

from ..trajectory.segments import DEFAULT_MASK_WEIGHTS, MaskWeights
from ..trajectory.teacher_force import TeacherForcedBatch, assemble
from .dataset import TrainSample


class TFCollator:
    """Assemble teacher-forced training batches from TrainSample objects."""

    def __init__(
        self,
        family: str,
        processor: Any,
        *,
        system_prompt: str | None = None,
        weights: MaskWeights = DEFAULT_MASK_WEIGHTS,
    ) -> None:
        self.family = family
        self.processor = processor
        self.system_prompt = system_prompt
        self.weights = weights

    def __call__(self, batch: list[TrainSample]) -> TeacherForcedBatch:
        if len(batch) != 1:
            raise NotImplementedError(
                "TFCollator currently supports micro_batch=1 only. "
                "Use gradient accumulation (configs/training.yaml:grad_accum) for effective batching."
            )
        s = batch[0]
        return assemble(
            self.family,
            s.prompt,
            s.trajectory,
            s.image,
            self.processor,
            system_prompt=self.system_prompt,
            weights=self.weights,
        )
