"""Deterministic seeding across Python, NumPy, PyTorch, and CUDA."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    warn_only: bool = False,
) -> None:
    """Seed all RNGs relevant to this pipeline.

    Parameters
    ----------
    seed : int
        RNG seed applied to Python, NumPy, and PyTorch (CPU + CUDA).
    deterministic : bool
        If True, flip CuDNN / PyTorch flags to favor determinism over speed.
    warn_only : bool
        Passed through to ``torch.use_deterministic_algorithms``. Default
        ``False`` is publication-grade — any op without a deterministic
        kernel under the seed raises instead of silently falling back to
        a non-deterministic one (which would invalidate seed-replication
        claims in T1/T3 reports). Set ``True`` for smoke runs that
        accept best-effort determinism.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS needs an explicit workspace-config hint to keep matmul
    # outputs deterministic across kernel selection. Without it,
    # ``torch.use_deterministic_algorithms(True)`` raises on the first
    # backward pass on H200. ``setdefault`` so an operator override wins.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # transformers maintains its own RNG layer used by .generate() and
    # some tokenizer fast paths; missing this leaks non-determinism
    # into T1/T3 evaluations even when torch+numpy+random are seeded.
    try:
        from transformers import set_seed as hf_set_seed  # type: ignore
    except ImportError:
        pass
    else:
        hf_set_seed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except (AttributeError, RuntimeError):
            pass
