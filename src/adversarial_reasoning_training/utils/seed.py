"""Deterministic seeding across Python, NumPy, PyTorch, and CUDA."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs relevant to this pipeline.

    Parameters
    ----------
    seed : int
        RNG seed applied to Python, NumPy, and PyTorch (CPU + CUDA).
    deterministic : bool
        If True, flip CuDNN / PyTorch flags to favor determinism over speed.
        Some ops still have non-deterministic CUDA kernels; this only
        reduces, not eliminates, nondeterminism.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):
            pass
