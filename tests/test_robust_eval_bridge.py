"""Unit tests for the robust-eval bridge (records.jsonl -> per_sample dict)."""

from __future__ import annotations

import json
from dataclasses import dataclass
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


@dataclass
class EvalRecord:
    """Test helper: sample-level eval record.

    Bundles the 11 params that ``_record()`` required into a single
    dataclass so call sites read ``EvalRecord(sample_id=..., ...)``
    instead of ``_record("s1", 0.0078, 0.5, [...], [...], "yes", "yes")``.
    """

    sample_id: str
    epsilon: float
    edit_distance_norm: float
    benign_seq: list[str]
    attacked_seq: list[str]
    benign_answer: str
    attacked_answer: str
    attack_mode: str = "pgd"
    seed: int = 0
    benign_calls: list[dict] | None = None
    attacked_calls: list[dict] | None = None


def _toolcall(step: int, name: str, args: dict) -> dict:
    return {"step": step, "name": name, "args": args, "result": None, "error": None}


def _record_from(rec: EvalRecord) -> dict:
    benign_node: dict = {
        "task_id": "t",
        "model_id": "m",
        "seed": rec.seed,
        "tool_sequence": rec.benign_seq,
        "final_answer": rec.benign_answer,
        "metadata": {},
    }
    attacked_node: dict = {
        "task_id": "t",
        "model_id": "m",
        "seed": rec.seed,
        "tool_sequence": rec.attacked_seq,
        "final_answer": rec.attacked_answer,
        "metadata": {},
    }
    if rec.benign_calls is not None:
        benign_node["tool_calls"] = rec.benign_calls
    if rec.attacked_calls is not None:
        attacked_node["tool_calls"] = rec.attacked_calls
    return {
        "model_key": "m",
        "task_id": "t",
        "sample_id": rec.sample_id,
        "attack_name": "pgd",
        "attack_mode": rec.attack_mode,
        "epsilon": rec.epsilon,
        "seed": rec.seed,
        "benign": benign_node,
        "attacked": attacked_node,
        "edit_distance_norm": rec.edit_distance_norm,
        "elapsed_s": 1.0,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_records_to_per_sample_preserves_all_metrics(tmp_path: Path) -> None:
    records = [
        _record_from(EvalRecord("s1", 0.0078, 0.5, ["a", "b"], ["a", "b"], "yes", "yes")),
        _record_from(EvalRecord("s2", 0.0078, 0.0, ["x"], ["x"], "ok", "ok")),
    ]
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, records)

    per_sample = records_to_per_sample(path)

    assert set(per_sample.keys()) == set(T3_METRICS)
    assert per_sample["tool_name_acc"] == [1.0, 1.0]
    assert per_sample["answer_em"] == [1.0, 1.0]
    assert per_sample["traj_edit_distance"] == pytest.approx([0.5, 1.0])
    assert per_sample["args_iou"] == []


def test_traj_edit_distance_is_similarity_higher_is_better(tmp_path: Path) -> None:
    records = [
        _record_from(EvalRecord("s1", 0.0078, 0.0, ["a"], ["a"], "x", "x")),
        _record_from(EvalRecord("s2", 0.0078, 1.0, ["a"], ["b"], "x", "y")),
    ]
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, records)

    per_sample = records_to_per_sample(path)

    assert per_sample["traj_edit_distance"] == pytest.approx([1.0, 0.0])


def test_align_per_sample_intersects_on_pair_key(tmp_path: Path) -> None:
    base = [
        _record_from(EvalRecord("s1", 0.0078, 0.8, ["a"], ["b"], "foo", "bar")),
        _record_from(EvalRecord("s2", 0.0078, 0.7, ["a"], ["c"], "foo", "baz")),
        _record_from(EvalRecord("s3", 0.0078, 0.6, ["a"], ["d"], "foo", "qux")),
    ]
    defended = [
        _record_from(EvalRecord("s1", 0.0078, 0.1, ["a"], ["a"], "foo", "foo")),
        _record_from(EvalRecord("s2", 0.0078, 0.0, ["a"], ["a"], "foo", "foo")),
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
    rec = _record_from(EvalRecord("s1", 0.0078, 1.5, ["a"], ["b"], "x", "y"))
    rec_neg = _record_from(EvalRecord("s2", 0.0078, -0.2, ["a"], ["a"], "x", "x"))
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
        "args_iou": [1.0, 0.5, 0.0],
        "answer_em": [0.0, 0.0, 1.0],
        "traj_edit_distance": [0.5, 0.25, 0.9],
    }
    out = tmp_path / "nested" / "per_sample.json"
    save_per_sample(out, per_sample)

    loaded = json.loads(out.read_text())
    assert loaded == per_sample


def test_args_iou_perfect_match_with_full_calls(tmp_path: Path) -> None:
    benign_calls = [
        _toolcall(0, "search", {"q": "x"}),
        _toolcall(1, "lookup", {"id": 7}),
    ]
    attacked_calls = [
        _toolcall(0, "search", {"q": "x"}),
        _toolcall(1, "lookup", {"id": 7}),
    ]
    rec = _record_from(EvalRecord(
        "s1", 0.0078, 0.0, ["search", "lookup"], ["search", "lookup"], "ok", "ok",
        benign_calls=benign_calls, attacked_calls=attacked_calls,
    ))
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, [rec])

    per_sample = records_to_per_sample(path)

    assert per_sample["args_iou"] == [1.0]


def test_args_iou_jaccard_math(tmp_path: Path) -> None:
    benign_calls = [_toolcall(0, "f", {"a": 1, "b": 2})]
    attacked_calls = [_toolcall(0, "f", {"a": 1, "c": 3})]
    rec = _record_from(EvalRecord(
        "s1", 0.0078, 0.0, ["f"], ["f"], "ok", "ok",
        benign_calls=benign_calls, attacked_calls=attacked_calls,
    ))
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, [rec])

    per_sample = records_to_per_sample(path)

    # {(a,1),(b,2)} ∩ {(a,1),(c,3)} = {(a,1)}; ∪ size 3 → 1/3
    assert per_sample["args_iou"] == pytest.approx([1.0 / 3.0])


def test_args_iou_dropped_when_old_schema(tmp_path: Path) -> None:
    """Records lacking tool_calls produce NaN args_iou, which _drop_nan_metrics
    empties so T3 trips its missing-metric branch."""
    rec = _record_from(EvalRecord("s1", 0.0078, 0.0, ["a"], ["a"], "x", "x"))
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, [rec])

    per_sample = records_to_per_sample(path)

    assert per_sample["args_iou"] == []
    assert per_sample["tool_name_acc"] == [1.0]


def test_args_iou_paired_steps_average(tmp_path: Path) -> None:
    benign_calls = [
        _toolcall(0, "f", {"a": 1}),
        _toolcall(1, "g", {"b": 2}),
    ]
    attacked_calls = [
        _toolcall(0, "f", {"a": 1}),       # IoU 1.0
        _toolcall(1, "g", {"b": 99}),       # IoU 0.0 (different repr)
    ]
    rec = _record_from(EvalRecord(
        "s1", 0.0078, 0.0, ["f", "g"], ["f", "g"], "ok", "ok",
        benign_calls=benign_calls, attacked_calls=attacked_calls,
    ))
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, [rec])

    per_sample = records_to_per_sample(path)

    assert per_sample["args_iou"] == pytest.approx([0.5])


def test_t3_pass_end_to_end_under_clean_robustness(tmp_path: Path) -> None:
    base_recs = [
        _record_from(EvalRecord(
            f"s{i}", 0.0078, 0.85, ["a", "b"], ["c", "d"], "foo", "bar",
        ))
        for i in range(8)
    ]
    def_recs = [
        _record_from(EvalRecord(
            f"s{i}", 0.0078, 0.10, ["a", "b"], ["a", "b"], "foo", "foo",
        ))
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
        undefended_per_sample=baseline_ps,
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
