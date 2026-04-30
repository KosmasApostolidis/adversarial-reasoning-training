"""Unit coverage for the figure / aggregation helpers under scripts/figures/.

These scripts are not packaged so we load them by file path. The contract
tested here is the one ``run_pipeline.sh`` Phase 4-5 depends on:

* ``aggregate_seeds.aggregate`` accepts a ``shared_t1`` path that overrides
  per-seed ``T1.json`` lookup.
* ``aggregate_seeds`` CLI honours ``--min-seeds`` (warn) and ``--strict``
  (non-zero exit).
* ``compute_summary.summarise_run`` accepts both ``peak_memory_gb`` (gate
  schema) and ``peak_allocated_gb`` (trainer ``train_meta.json``).
* ``make_figures.render_headline_3model`` writes a non-empty PNG when
  given one or more ``aggregate.json`` payloads.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "scripts" / "figures"


def _load_script(name: str) -> Any:
    """Load scripts/figures/<name>.py as an importable module."""
    spec = importlib.util.spec_from_file_location(name, FIGURES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def aggregate_seeds_mod() -> Any:
    return _load_script("aggregate_seeds")


@pytest.fixture()
def compute_summary_mod() -> Any:
    return _load_script("compute_summary")


@pytest.fixture()
def make_figures_mod() -> Any:
    return _load_script("make_figures")


def _write_t1(path: Path, *, tool_acc: float = 0.9, em: float = 0.8, passed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "passed": passed,
        "tool_name_acc": tool_acc,
        "answer_em": em,
    }))


def _write_t2(path: Path, *, tool_acc: float = 0.88, passed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "passed": passed,
        "per_metric": {
            "tool_name_acc": {"current": tool_acc, "drop": 0.02},
            "answer_em":     {"current": 0.78,     "drop": 0.03},
            "args_iou":      {"current": 0.85,     "drop": 0.01},
        },
    }))


def _write_t3(path: Path, *, delta: float = 0.30, passed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "passed": passed,
        "per_metric": {
            "tool_name_acc":      {"baseline_mean": 0.40, "defended_mean": 0.40 + delta,
                                   "delta": delta, "p_value": 0.001, "p_adj": 0.001},
            "args_iou":           {"baseline_mean": 0.50, "defended_mean": 0.50 + delta,
                                   "delta": delta, "p_value": 0.01,  "p_adj": 0.02},
            "answer_em":          {"baseline_mean": 0.30, "defended_mean": 0.30 + delta,
                                   "delta": delta, "p_value": 0.001, "p_adj": 0.002},
            "traj_edit_distance": {"baseline_mean": 0.20, "defended_mean": 0.20 + delta,
                                   "delta": delta, "p_value": 0.005, "p_adj": 0.01},
        },
    }))


@pytest.fixture()
def three_seed_layout(tmp_path: Path) -> tuple[Path, list[Path]]:
    """Three seed dirs, one shared T1.json above them."""
    t1 = tmp_path / "t1_qwen" / "gates" / "T1.json"
    _write_t1(t1)
    seeds: list[Path] = []
    for s in range(3):
        sd = tmp_path / f"seed{s}"
        _write_t2(sd / "gates" / "T2.json", tool_acc=0.86 + 0.01 * s)
        _write_t3(sd / "gates" / "T3.json", delta=0.30 + 0.02 * s)
        seeds.append(sd)
    return t1, seeds


# -----------------------------------------------------------------------------
# aggregate_seeds
# -----------------------------------------------------------------------------


def test_shared_t1_is_used_when_per_seed_t1_missing(
    aggregate_seeds_mod: Any, three_seed_layout: tuple[Path, list[Path]],
) -> None:
    t1, seeds = three_seed_layout
    # No per-seed T1.json on any seed directory.
    payload = aggregate_seeds_mod.aggregate(seeds, shared_t1=t1)
    summary = payload["summary"]
    # T1.tool_name_acc is populated from the shared file across all 3 seeds.
    assert "T1.tool_name_acc" in summary
    assert summary["T1.tool_name_acc"]["n"] == 3
    assert summary["T1.tool_name_acc"]["mean"] == pytest.approx(0.9)


def test_shared_t1_falls_back_to_per_seed_when_none(
    aggregate_seeds_mod: Any, three_seed_layout: tuple[Path, list[Path]],
) -> None:
    _, seeds = three_seed_layout
    # Drop a per-seed T1 into seed0 only.
    _write_t1(seeds[0] / "gates" / "T1.json", tool_acc=0.77)
    payload = aggregate_seeds_mod.aggregate(seeds, shared_t1=None)
    # Only 1 seed contributed T1; n==1 in that case.
    assert payload["summary"]["T1.tool_name_acc"]["n"] == 1
    assert payload["summary"]["T1.tool_name_acc"]["mean"] == pytest.approx(0.77)


def test_gate_pass_rate_uses_shared_t1(
    aggregate_seeds_mod: Any, three_seed_layout: tuple[Path, list[Path]],
) -> None:
    t1, seeds = three_seed_layout
    payload = aggregate_seeds_mod.aggregate(seeds, shared_t1=t1)
    assert payload["gate_pass_rate"]["T1"]["rate"] == pytest.approx(1.0)
    assert payload["gate_pass_rate"]["T1"]["n_seeds"] == 3


def test_aggregate_cli_min_seeds_warns_only(
    three_seed_layout: tuple[Path, list[Path]], tmp_path: Path,
) -> None:
    t1, seeds = three_seed_layout
    out = tmp_path / "agg.json"
    cmd = [
        sys.executable, str(FIGURES_DIR / "aggregate_seeds.py"),
        "--seeds", *(str(s) for s in seeds[:1]),  # 1 seed, below threshold
        "--shared-t1", str(t1),
        "--min-seeds", "3",
        "--out", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert "WARN" in proc.stderr
    assert "min-seeds=3" in proc.stderr


def test_aggregate_cli_strict_fails_below_threshold(
    three_seed_layout: tuple[Path, list[Path]], tmp_path: Path,
) -> None:
    t1, seeds = three_seed_layout
    out = tmp_path / "agg.json"
    cmd = [
        sys.executable, str(FIGURES_DIR / "aggregate_seeds.py"),
        "--seeds", *(str(s) for s in seeds[:2]),
        "--shared-t1", str(t1),
        "--min-seeds", "3",
        "--strict",
        "--out", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 2
    assert not out.exists()


def test_aggregate_cli_at_min_seeds_succeeds(
    three_seed_layout: tuple[Path, list[Path]], tmp_path: Path,
) -> None:
    t1, seeds = three_seed_layout
    out = tmp_path / "agg.json"
    cmd = [
        sys.executable, str(FIGURES_DIR / "aggregate_seeds.py"),
        "--seeds", *(str(s) for s in seeds),
        "--shared-t1", str(t1),
        "--min-seeds", "3",
        "--strict",
        "--out", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text())
    assert payload["n_seeds"] == 3


# -----------------------------------------------------------------------------
# compute_summary
# -----------------------------------------------------------------------------


def test_compute_summary_accepts_peak_allocated_gb(
    compute_summary_mod: Any, tmp_path: Path,
) -> None:
    """train_meta.json uses MemoryStats schema (peak_allocated_gb).
    compute_summary must read it as the same field as the gate-side
    peak_memory_gb so the LaTeX table is correct."""
    run = tmp_path / "qwen_main_seed0"
    run.mkdir()
    (run / "train_meta.json").write_text(json.dumps({
        "duration_s": 7200.0,            # 2 h
        "peak_allocated_gb": 87.5,
        "global_step": 1234,
    }))
    row = compute_summary_mod.summarise_run(run, "train_meta.json")
    assert row["has_data"] is True
    assert row["duration_h"] == pytest.approx(2.0)
    assert row["peak_gb"] == pytest.approx(87.5)


def test_compute_summary_prefers_gate_schema_when_both_present(
    compute_summary_mod: Any, tmp_path: Path,
) -> None:
    run = tmp_path / "t0"
    gates = run / "gates"
    gates.mkdir(parents=True)
    (gates / "T0.json").write_text(json.dumps({
        "passed": True,
        "duration_s": 60.0,
        "peak_memory_gb": 12.0,           # gate schema
        "peak_allocated_gb": 999.0,       # adversarial sentinel — should be ignored
    }))
    row = compute_summary_mod.summarise_run(run, "train_meta.json")
    assert row["peak_gb"] == pytest.approx(12.0)
    assert "T0✓" in row["status"]


# -----------------------------------------------------------------------------
# make_figures (headline mode)
# -----------------------------------------------------------------------------


def _seed_aggregate_payload(delta: float) -> dict[str, Any]:
    return {
        "summary": {
            f"T3.{m}_delta": {
                "n": 3, "mean": delta, "std": 0.05,
                "ci_lo": delta - 0.06, "ci_hi": delta + 0.06,
            }
            for m in ("tool_name_acc", "args_iou", "answer_em", "traj_edit_distance")
        },
    }


def test_render_headline_3model_writes_png(
    make_figures_mod: Any, tmp_path: Path,
) -> None:
    paths: list[Path] = []
    for label, delta in [("qwen_main", 0.30), ("llava_main", 0.20), ("llama_main", 0.10)]:
        p = tmp_path / label / "aggregate.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(_seed_aggregate_payload(delta)))
        paths.append(p)
    out = tmp_path / "headline.png"
    rc = make_figures_mod.render_headline_3model(paths, out)
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_headline_3model_handles_single_model(
    make_figures_mod: Any, tmp_path: Path,
) -> None:
    p = tmp_path / "qwen_main" / "aggregate.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(_seed_aggregate_payload(0.25)))
    out = tmp_path / "headline.png"
    assert make_figures_mod.render_headline_3model([p], out) == 0
    assert out.exists()


def test_render_headline_3model_errors_on_missing_input(
    make_figures_mod: Any, tmp_path: Path,
) -> None:
    out = tmp_path / "headline.png"
    rc = make_figures_mod.render_headline_3model([tmp_path / "nope.json"], out)
    assert rc == 1
    assert not out.exists()


def test_make_figures_cli_aggregate_requires_out(tmp_path: Path) -> None:
    p = tmp_path / "qwen_main" / "aggregate.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(_seed_aggregate_payload(0.20)))
    cmd = [
        sys.executable, str(FIGURES_DIR / "make_figures.py"),
        "--aggregate", str(p),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 2
    assert "--out" in proc.stderr
