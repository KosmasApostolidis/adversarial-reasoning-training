"""Unit tests for the robust-comparison figure renderer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_PATH = REPO_ROOT / "scripts" / "figures" / "make_figures.py"


def _load_make_figures():
    spec = importlib.util.spec_from_file_location("_make_figures_under_test", FIG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def make_figures():
    return _load_make_figures()


def test_render_robust_comparison_writes_non_empty_png(tmp_path: Path, make_figures) -> None:
    baseline = {
        "tool_name_acc": [0.0, 0.0, 1.0, 0.0, 0.0],
        "answer_em": [0.0, 0.0, 0.0, 1.0, 0.0],
        "traj_edit_distance": [0.10, 0.20, 0.05, 0.15, 0.25],
    }
    defended = {
        "tool_name_acc": [1.0, 1.0, 1.0, 1.0, 1.0],
        "answer_em": [1.0, 1.0, 1.0, 1.0, 1.0],
        "traj_edit_distance": [0.92, 0.88, 0.95, 0.85, 0.90],
    }
    t3_payload = {
        "passed": True,
        "significant_metrics": ["tool_name_acc", "traj_edit_distance"],
    }
    out = tmp_path / "robust.png"
    make_figures.render_robust_comparison(baseline, defended, t3_payload, out)

    assert out.exists()
    assert out.stat().st_size > 4096


def test_bootstrap_ci_deterministic_under_same_seed(make_figures) -> None:
    samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    lo_a = make_figures._bootstrap_ci(samples, lo=True, seed=42, n_resamples=500)
    lo_b = make_figures._bootstrap_ci(samples, lo=True, seed=42, n_resamples=500)
    hi_a = make_figures._bootstrap_ci(samples, lo=False, seed=42, n_resamples=500)
    hi_b = make_figures._bootstrap_ci(samples, lo=False, seed=42, n_resamples=500)

    assert lo_a == lo_b
    assert hi_a == hi_b
    assert lo_a < hi_a


def test_bootstrap_ci_handles_empty(make_figures) -> None:
    import math

    result = make_figures._bootstrap_ci([], lo=True, seed=0, n_resamples=10)
    assert math.isnan(result)


def test_render_skips_metrics_missing_from_either_side(tmp_path: Path, make_figures) -> None:
    baseline = {
        "tool_name_acc": [0.0, 0.0, 0.0],
        "traj_edit_distance": [0.1, 0.2, 0.3],
    }
    defended = {
        "tool_name_acc": [1.0, 1.0, 1.0],
        "answer_em": [1.0, 1.0, 1.0],
    }
    t3_payload = {"passed": True, "significant_metrics": ["tool_name_acc"]}
    out = tmp_path / "robust.png"
    make_figures.render_robust_comparison(baseline, defended, t3_payload, out)

    assert out.exists()
    assert out.stat().st_size > 4096
