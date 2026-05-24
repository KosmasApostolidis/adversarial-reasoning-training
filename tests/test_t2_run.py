"""Unit tests for gates/T2_no_collapse.run_t2 + _read_metrics — branch coverage.

The heavy `_main()` body (HF model load + checkpoint + dev-set eval) is out
of scope for cheap unit tests; covered by the integration smoke runs
documented in docs/EXPERIMENT_RUNS.md instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from adversarial_reasoning_training.gates.T2_no_collapse import (
    T2Thresholds,
    _read_metrics,
    run_t2,
)


def _write_t1(path: Path, metrics: dict[str, float], *, nested: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics} if nested else metrics
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_read_metrics_flat_layout(tmp_path: Path) -> None:
    p = tmp_path / "T1.json"
    _write_t1(p, {"tool_name_acc": 0.85, "answer_em": 0.7})
    out = _read_metrics(p, ("tool_name_acc", "answer_em"))
    assert out == {"tool_name_acc": 0.85, "answer_em": 0.7}


def test_read_metrics_nested_metrics_layout(tmp_path: Path) -> None:
    p = tmp_path / "T1.json"
    _write_t1(p, {"tool_name_acc": 0.9, "answer_em": 0.8}, nested=True)
    out = _read_metrics(p, ("tool_name_acc", "answer_em"))
    assert out == {"tool_name_acc": 0.9, "answer_em": 0.8}


def test_read_metrics_skips_missing_keys(tmp_path: Path) -> None:
    p = tmp_path / "T1.json"
    _write_t1(p, {"tool_name_acc": 0.5})
    out = _read_metrics(p, ("tool_name_acc", "answer_em"))
    assert out == {"tool_name_acc": 0.5}


def test_run_t2_passes_within_tolerance(tmp_path: Path) -> None:
    t1 = tmp_path / "T1.json"
    _write_t1(t1, {"tool_name_acc": 0.90, "answer_em": 0.80})
    out = tmp_path / "gates" / "T2.json"

    def evaluator() -> dict[str, float]:
        return {"tool_name_acc": 0.88, "answer_em": 0.78}  # 2pp drop on each

    result = run_t2(
        adv_clean_evaluator=evaluator,
        t1_result_path=t1,
        out_path=out,
        thresholds=T2Thresholds(tolerance_pp=3.0, metrics=("tool_name_acc", "answer_em")),
    )
    assert result.passed is True
    for m in ("tool_name_acc", "answer_em"):
        assert result.per_metric[m]["ok"] is True
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk["passed"] is True


def test_run_t2_fails_when_drop_exceeds_tolerance(tmp_path: Path) -> None:
    t1 = tmp_path / "T1.json"
    _write_t1(t1, {"tool_name_acc": 0.90, "answer_em": 0.80})

    def evaluator() -> dict[str, float]:
        return {"tool_name_acc": 0.50, "answer_em": 0.40}  # 40pp drop — way over

    result = run_t2(
        adv_clean_evaluator=evaluator,
        t1_result_path=t1,
        out_path=tmp_path / "T2.json",
        thresholds=T2Thresholds(tolerance_pp=3.0),
    )
    assert result.passed is False
    assert any("drop" in n for n in result.notes)


def test_run_t2_notes_metric_missing_from_t1(tmp_path: Path) -> None:
    t1 = tmp_path / "T1.json"
    _write_t1(t1, {"tool_name_acc": 0.9})  # answer_em missing

    def evaluator() -> dict[str, float]:
        return {"tool_name_acc": 0.88, "answer_em": 0.7}

    result = run_t2(
        adv_clean_evaluator=evaluator,
        t1_result_path=t1,
        out_path=tmp_path / "T2.json",
        thresholds=T2Thresholds(metrics=("tool_name_acc", "answer_em")),
    )
    # answer_em was absent from T1 → noted but does not flip passed.
    assert any("answer_em" in n and "missing" in n for n in result.notes)
    assert result.passed is True


def test_run_t2_evaluator_returning_none_treated_as_zero(tmp_path: Path) -> None:
    t1 = tmp_path / "T1.json"
    _write_t1(t1, {"tool_name_acc": 0.90})

    def evaluator() -> dict[str, float]:
        # type: ignore[return-value]
        return None  # type: ignore[return-value]

    result = run_t2(
        adv_clean_evaluator=evaluator,  # type: ignore[arg-type]
        t1_result_path=t1,
        out_path=tmp_path / "T2.json",
        thresholds=T2Thresholds(metrics=("tool_name_acc",)),
    )
    # Null evaluator → NaN current — guaranteed to fail regardless of ceiling.
    assert result.passed is False
