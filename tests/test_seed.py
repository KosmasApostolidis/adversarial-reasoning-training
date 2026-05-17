"""Unit tests for utils/seed — cross-RNG determinism."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from adversarial_reasoning_training.utils.seed import seed_everything


def _draw_python_numpy_torch() -> tuple[float, float, float]:
    return (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1).item()),
    )


def test_seed_everything_makes_python_numpy_torch_deterministic() -> None:
    seed_everything(42, deterministic=False)
    a = _draw_python_numpy_torch()
    seed_everything(42, deterministic=False)
    b = _draw_python_numpy_torch()
    assert a == b


def test_seed_everything_different_seeds_diverge() -> None:
    seed_everything(0, deterministic=False)
    a = _draw_python_numpy_torch()
    seed_everything(1, deterministic=False)
    b = _draw_python_numpy_torch()
    assert a != b


def test_seed_everything_does_not_touch_pythonhashseed(
    monkeypatch,
) -> None:
    """B07: pre-fix ``seed_everything`` assigned ``PYTHONHASHSEED`` in-process,
    but Python's hash randomization is locked in at interpreter startup —
    the assignment was a no-op that misled readers into thinking hash-stable
    dict iteration was guaranteed. Removed; this regression test asserts
    the env var is left alone so the false-confidence line cannot return.
    """
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    seed_everything(123, deterministic=False)
    assert "PYTHONHASHSEED" not in os.environ, (
        "seed_everything must not set PYTHONHASHSEED — it's a no-op "
        "after interpreter startup and gives false determinism guarantees."
    )


def test_seed_everything_deterministic_flag_flips_cudnn() -> None:
    seed_everything(7, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_seed_everything_non_deterministic_path_runs() -> None:
    # Smoke: deterministic=False should leave cudnn flags untouched-by-this-call
    # (we don't assert on prior global state — just that the path executes).
    seed_everything(99, deterministic=False)
    # Subsequent draws must still be reproducible despite the relaxed flag.
    a = _draw_python_numpy_torch()
    seed_everything(99, deterministic=False)
    b = _draw_python_numpy_torch()
    assert a == b
