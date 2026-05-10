"""Per-family teacher-forced template assemblers.

Each VLM family (Qwen2.5-VL, LLaVA-NeXT, InternVL2) has its own chat
formatting and image-token expansion rules. The public ``assemble_*``
functions here turn a (prompt, trajectory, image, processor) tuple into
a :class:`TeacherForcedBatch` that the trainer can feed back into
``forward_with_logits``.

Layout: this used to be a single 717-LOC ``templates.py``; the
Clean-Code sweep split it into one private module per family plus a
shared :mod:`._common` for chat markers reused across families.
External callers import the three public ``assemble_*`` symbols
unchanged (``from .templates import assemble_qwen, …``).

The orchestrator + shared helpers (``TeacherForcedBatch``,
``_split_thoughts``, ``_format_observation``, ``assemble``) live in
``teacher_force.py``; this package is import-side dependent on those.
"""

from __future__ import annotations

from ._internvl2 import assemble_internvl
from ._llava_next import assemble_llava_next
from ._qwen import assemble_qwen

__all__ = [
    "assemble_internvl",
    "assemble_llava_next",
    "assemble_qwen",
]
