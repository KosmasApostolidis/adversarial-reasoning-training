"""Regression tests for the T3 NaN-metric drop logic.

Two bugs the current implementation would silently exhibit:

* **C5** — operator has no signal that a metric was dropped. The
  ``_drop_nan_metrics`` helper empties a metric list when any value is
  NaN (e.g. legacy records lacking the ``tool_calls`` field) but does
  not record which metric(s) were dropped, so a passing T3 with 3/4
  metrics quietly missing looks identical to a passing T3 with all 4.
* **H1** — paired-alignment poisoning. ``align_per_sample`` calls
  ``_drop_nan_metrics`` on baseline and defended **independently**.
  If a NaN exists only on one side, the other side's array stays
  populated → downstream consumers compare empty vs populated lists
  with mismatched lengths and undefined Wilcoxon behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adversarial_reasoning_training.eval import robust_eval as re_mod


def _write_records(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _record(
    sample_id: str,
    *,
    epsilon: float = 4 / 255,
    attack_mode: str = "pgd",
    seed: int = 0,
    benign_seq: list[str] | None = None,
    attacked_seq: list[str] | None = None,
    benign_calls: list[dict] | None = None,
    attacked_calls: list[dict] | None = None,
    benign_ans: str = "yes",
    attacked_ans: str = "yes",
    edit_distance_norm: float = 0.0,
) -> dict:
    benign = {
        "tool_sequence": benign_seq or ["t_a", "t_b"],
        "final_answer": benign_ans,
    }
    attacked = {
        "tool_sequence": attacked_seq or ["t_a", "t_b"],
        "final_answer": attacked_ans,
    }
    if benign_calls is not None:
        benign["tool_calls"] = benign_calls
    if attacked_calls is not None:
        attacked["tool_calls"] = attacked_calls
    return {
        "sample_id": sample_id,
        "epsilon": epsilon,
        "attack_mode": attack_mode,
        "seed": seed,
        "benign": benign,
        "attacked": attacked,
        "edit_distance_norm": edit_distance_norm,
    }


@pytest.mark.unit
def test_drop_nan_metrics_returns_dropped_names() -> None:
    """C5: ``_drop_nan_metrics`` must report which metrics it emptied."""
    metrics: dict[str, list[float]] = {
        "tool_name_acc": [1.0, 0.0, 1.0],
        "args_iou": [0.5, float("nan"), 0.7],  # poisoned by NaN
        "answer_em": [1.0, 1.0, 0.0],
        "traj_edit_distance": [0.2, 0.3, float("nan")],  # also poisoned
    }
    cleaned, dropped = re_mod._drop_nan_metrics(metrics)
    assert sorted(dropped) == ["args_iou", "traj_edit_distance"], (
        "must surface every metric whose list was emptied so an "
        "operator can tell at a glance which evidence was lost"
    )
    assert cleaned["args_iou"] == []
    assert cleaned["traj_edit_distance"] == []
    assert cleaned["tool_name_acc"] == [1.0, 0.0, 1.0]
    assert cleaned["answer_em"] == [1.0, 1.0, 0.0]


@pytest.mark.unit
def test_align_per_sample_drops_metric_from_both_sides_when_either_has_nan(
    tmp_path: Path,
) -> None:
    """H1: a NaN on the baseline side must cause that metric to be
    emptied on the defended side too — keeping paired arrays
    length-matched. The current code drops sides independently and
    leaves a populated list paired with an empty one.
    """
    baseline_path = tmp_path / "baseline.jsonl"
    defended_path = tmp_path / "defended.jsonl"

    # sample_1 baseline lacks tool_calls (legacy schema) → args_iou=NaN.
    # Both sides have valid tool_calls for sample_2 → args_iou is finite.
    baseline = [
        _record("s1", benign_calls=None, attacked_calls=None),
        _record(
            "s2",
            benign_calls=[{"name": "t_a", "args": {"k": 1}}],
            attacked_calls=[{"name": "t_a", "args": {"k": 1}}],
        ),
    ]
    defended = [
        _record(
            "s1",
            benign_calls=[{"name": "t_a", "args": {"k": 1}}],
            attacked_calls=[{"name": "t_a", "args": {"k": 1}}],
        ),
        _record(
            "s2",
            benign_calls=[{"name": "t_a", "args": {"k": 1}}],
            attacked_calls=[{"name": "t_a", "args": {"k": 1}}],
        ),
    ]
    _write_records(baseline_path, baseline)
    _write_records(defended_path, defended)

    b, d, shared = re_mod.align_per_sample(baseline_path, defended_path)
    assert len(shared) == 2

    # Even though defended-side args_iou values are all finite, the
    # presence of a NaN on baseline must drop args_iou from BOTH sides.
    assert b["args_iou"] == [], "baseline args_iou must be empty (had NaN)"
    assert d["args_iou"] == [], (
        "defended args_iou must ALSO be empty for paired alignment, "
        "even though its own values were finite"
    )
    # Other metrics remain finite + length-matched on both sides.
    assert len(b["tool_name_acc"]) == len(d["tool_name_acc"]) == 2
    assert len(b["answer_em"]) == len(d["answer_em"]) == 2
    assert len(b["traj_edit_distance"]) == len(d["traj_edit_distance"]) == 2


@pytest.mark.unit
def test_align_per_sample_with_drops_exposes_dropped_set(
    tmp_path: Path,
) -> None:
    """C5: exposes the dropped-metric set so the T3 layer can record it."""
    baseline_path = tmp_path / "baseline.jsonl"
    defended_path = tmp_path / "defended.jsonl"
    _write_records(baseline_path, [_record("s1", benign_calls=None)])
    _write_records(
        defended_path,
        [
            _record(
                "s1",
                benign_calls=[{"name": "t_a", "args": {"k": 1}}],
                attacked_calls=[{"name": "t_a", "args": {"k": 1}}],
            )
        ],
    )

    assert hasattr(re_mod, "align_per_sample_with_drops"), (
        "robust_eval must expose align_per_sample_with_drops so callers "
        "(cli/eval_robust.py, T3 gate) can record dropped metric names"
    )
    b, d, _shared, dropped = re_mod.align_per_sample_with_drops(
        baseline_path, defended_path
    )
    assert "args_iou" in dropped
    assert b["args_iou"] == [] and d["args_iou"] == []
