"""Unit tests for the robust-eval bridge (records.jsonl -> per_sample dict)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adversarial_reasoning_training.eval.robust_eval import (
    T3_METRICS,
    align_per_sample,
    load_records,
    records_to_per_sample,
    save_per_sample,
)
from adversarial_reasoning_training.gates.T3_robust import T3Thresholds, run_t3


def _record(
    sample_id: str,
    epsilon: float,
    edit_distance_norm: float,
    benign_seq: list[str],
    attacked_seq: list[str],
    benign_answer: str,
    attacked_answer: str,
    *,
    attack_mode: str = "pgd",
    seed: int = 0,
) -> dict:
    return {
        "model_key": "m",
        "task_id": "t",
        "sample_id": sample_id,
        "attack_name": "pgd",
        "attack_mode": attack_mode,
        "epsilon": epsilon,
        "seed": seed,
        "benign": {
            "task_id": "t",
            "model_id": "m",
            "seed": seed,
            "tool_sequence": benign_seq,
            "final_answer": benign_answer,
            "metadata": {},
        },
        "attacked": {
            "task_id": "t",
            "model_id": "m",
            "seed": seed,
            "tool_sequence": attacked_seq,
            "final_answer": attacked_answer,
            "metadata": {},
        },
        "edit_distance_norm": edit_distance_norm,
        "elapsed_s": 1.0,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_records_to_per_sample_preserves_three_metrics(tmp_path: Path) -> None:
    records = [
        _record("s1", 0.0078, 0.5, ["a", "b"], ["a", "b"], "yes", "yes"),
        _record("s2", 0.0078, 0.0, ["x"], ["x"], "ok", "ok"),
    ]
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, records)

    per_sample = records_to_per_sample(path)

    assert set(per_sample.keys()) == set(T3_METRICS)
    assert per_sample["tool_name_acc"] == [1.0, 1.0]
    assert per_sample["answer_em"] == [1.0, 1.0]
    assert per_sample["traj_edit_distance"] == pytest.approx([0.5, 1.0])


def test_traj_edit_distance_is_similarity_higher_is_better(tmp_path: Path) -> None:
    records = [
        _record("s1", 0.0078, 0.0, ["a"], ["a"], "x", "x"),
        _record("s2", 0.0078, 1.0, ["a"], ["b"], "x", "y"),
    ]
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, records)

    per_sample = records_to_per_sample(path)

    assert per_sample["traj_edit_distance"] == pytest.approx([1.0, 0.0])


def test_align_per_sample_intersects_on_pair_key(tmp_path: Path) -> None:
    base = [
        _record("s1", 0.0078, 0.8, ["a"], ["b"], "foo", "bar"),
        _record("s2", 0.0078, 0.7, ["a"], ["c"], "foo", "baz"),
        _record("s3", 0.0078, 0.6, ["a"], ["d"], "foo", "qux"),
    ]
    defended = [
        _record("s1", 0.0078, 0.1, ["a"], ["a"], "foo", "foo"),
        _record("s2", 0.0078, 0.0, ["a"], ["a"], "foo", "foo"),
    ]
    base_path = tmp_path / "baseline.jsonl"
    def_path = tmp_path / "defended.jsonl"
    _write_jsonl(base_path, base)
    _write_jsonl(def_path, defended)

    baseline_ps, defended_ps, shared = align_per_sample(base_path, def_path)

    assert len(shared) == 2
    assert [k[0] for k in shared] == ["s1", "s2"]
    assert baseline_ps["tool_name_acc"] == [0.0, 0.0]
    assert defended_ps["tool_name_acc"] == [1.0, 1.0]


def test_clamps_out_of_range_edit_distance(tmp_path: Path) -> None:
    rec = _record("s1", 0.0078, 1.5, ["a"], ["b"], "x", "y")
    rec_neg = _record("s2", 0.0078, -0.2, ["a"], ["a"], "x", "x")
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, [rec, rec_neg])

    per_sample = records_to_per_sample(path)

    assert per_sample["traj_edit_distance"] == pytest.approx([0.0, 1.0])


def test_handles_empty_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")

    assert load_records(path) == []
    per_sample = records_to_per_sample(path)
    for key in T3_METRICS:
        assert per_sample[key] == []


def test_save_per_sample_round_trips(tmp_path: Path) -> None:
    per_sample = {
        "tool_name_acc": [1.0, 0.0, 1.0],
        "answer_em": [0.0, 0.0, 1.0],
        "traj_edit_distance": [0.5, 0.25, 0.9],
    }
    out = tmp_path / "nested" / "per_sample.json"
    save_per_sample(out, per_sample)

    loaded = json.loads(out.read_text())
    assert loaded == per_sample


def test_t3_pass_end_to_end_under_clean_robustness(tmp_path: Path) -> None:
    base_recs = [
        _record(f"s{i}", 0.0078, 0.85, ["a", "b"], ["c", "d"], "foo", "bar")
        for i in range(8)
    ]
    def_recs = [
        _record(f"s{i}", 0.0078, 0.10, ["a", "b"], ["a", "b"], "foo", "foo")
        for i in range(8)
    ]
    base_path = tmp_path / "baseline.jsonl"
    def_path = tmp_path / "defended.jsonl"
    _write_jsonl(base_path, base_recs)
    _write_jsonl(def_path, def_recs)

    baseline_ps, defended_ps, shared = align_per_sample(base_path, def_path)
    assert len(shared) == 8

    out_path = tmp_path / "T3.json"
    result = run_t3(
        baseline_per_sample=baseline_ps,
        defended_per_sample=defended_ps,
        out_path=out_path,
        thresholds=T3Thresholds(
            min_traj_edit_delta=0.10,
            alpha=0.05,
            min_significant_metrics=3,
        ),
    )

    assert result.passed is True
    assert set(result.significant_metrics) == {
        "tool_name_acc",
        "answer_em",
        "traj_edit_distance",
    }
    assert any("args_iou" in note for note in result.notes)
