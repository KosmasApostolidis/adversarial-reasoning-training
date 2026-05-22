"""Shared numerical constants used across attacks, trainer, gates, and CLI.

Defining these in one place avoids drift between hardcoded literals (e.g.
multiple call sites carrying their own ``4.0 / 255.0``) and matches the
configuration schema used in ``configs/defenses.yaml``.
"""

from __future__ import annotations

from enum import Enum

# Pixel byte scale: ``1 / 255`` maps an 8-bit byte delta to a [0,1] image-space
# delta. Used to convert ε expressed in byte units (2, 4, 8) into image-space ε.
BYTE_SCALE: float = 1.0 / 255.0

# Standard ε-ball radii for PGD adversarial training/eval.
EPS_2_255: float = 2.0 * BYTE_SCALE
EPS_4_255: float = 4.0 * BYTE_SCALE

# PGD inner-step ratio: α = ratio · ε. 0.25 is the canonical value used by
# both inner_pgd.InnerPGDConfig and the AdvTrainer / CLI defaults.
DEFAULT_PGD_ALPHA_RATIO: float = 0.25

# Default global gradient-norm clip used by gates/T1 fine-tune and the
# adversarial trainer outer step. Anchoring it here prevents the value
# drifting between the two call sites.
GRAD_CLIP_NORM: float = 1.0

# Default freeze strategy when a gate config does not name one explicitly.
DEFAULT_FREEZE_STRATEGY: str = "none"

# Canonical config paths for the gate runners. Gates accept overrides on
# the command line; this dict is the source of truth for the defaults.
DEFAULT_GATE_CONFIG_PATHS: dict[str, str] = {
    "defenses": "configs/defenses.yaml",
    "data": "configs/data.yaml",
    "gold": "configs/gold.yaml",
}

# Default gradient accumulation steps used by gate CLI defaults.
GRAD_ACCUM_DEFAULT: int = 8

# Logging and evaluation cadence defaults.
LOG_EVERY_DEFAULT: int = 20
EVAL_EVERY_DEFAULT: int = 200


class VLMFamily(str, Enum):
    """Canonical VLM family identifiers used by the dispatch logic.

    Inheriting from ``str`` keeps backwards compatibility with the existing
    ``vlm.family == "internvl2"`` style equality checks and YAML round-trips.
    """

    QWEN_VL = "qwen_vl"
    LLAVA_NEXT = "llava_next"
    LLAVA_ONEVISION = "llava_onevision"
    INTERNVL2 = "internvl2"


__all__ = [
    "BYTE_SCALE",
    "DEFAULT_FREEZE_STRATEGY",
    "DEFAULT_GATE_CONFIG_PATHS",
    "DEFAULT_PGD_ALPHA_RATIO",
    "EPS_2_255",
    "EPS_4_255",
    "GRAD_ACCUM_DEFAULT",
    "GRAD_CLIP_NORM",
    "VLMFamily",
]
