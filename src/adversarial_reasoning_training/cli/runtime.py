"""Process-level setup helpers used by scripts: seed, device, run dir."""

from __future__ import annotations

from pathlib import Path

import torch

from ..utils.paths import normalize_run_dir
from ..utils.seed import seed_everything


def setup_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python/NumPy/PyTorch via ``utils.seed.seed_everything``."""
    seed_everything(seed, deterministic=deterministic)


def setup_device(name: str = "cuda") -> torch.device:
    """Return a ``torch.device``, falling back to CPU if CUDA is requested but unavailable.

    Scripts that want strict CUDA-required behavior should validate
    ``torch.cuda.is_available()`` themselves; this helper never raises.
    """
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def setup_run_dir(path: str | Path) -> Path:
    """Create the run directory (and parents) and return its resolved ``Path``."""
    return normalize_run_dir(path)
