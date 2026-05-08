"""Shared numerical constants used across attacks, trainer, gates, and CLI.

Defining these in one place avoids drift between hardcoded literals (e.g.
multiple call sites carrying their own ``4.0 / 255.0``) and matches the
configuration schema used in ``configs/defenses.yaml``.
"""

from __future__ import annotations

# Pixel byte scale: ``1 / 255`` maps an 8-bit byte delta to a [0,1] image-space
# delta. Used to convert ε expressed in byte units (2, 4, 8) into image-space ε.
BYTE_SCALE: float = 1.0 / 255.0

# Standard ε-ball radii for PGD adversarial training/eval.
EPS_2_255: float = 2.0 * BYTE_SCALE
EPS_4_255: float = 4.0 * BYTE_SCALE
EPS_8_255: float = 8.0 * BYTE_SCALE

# PGD inner-step ratio: α = ratio · ε. 0.25 is the canonical value used by
# both inner_pgd.InnerPGDConfig and the AdvTrainer / CLI defaults.
DEFAULT_PGD_ALPHA_RATIO: float = 0.25

__all__ = [
    "BYTE_SCALE",
    "DEFAULT_PGD_ALPHA_RATIO",
    "EPS_2_255",
    "EPS_4_255",
    "EPS_8_255",
]
