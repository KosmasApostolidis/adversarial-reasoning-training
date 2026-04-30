"""Behavior contract for ``align_per_sample``.

The legacy public-API test only checks that the symbol exists on
``robust_eval``. This test pins the documented runtime contract that
``cli/eval_robust.py`` and downstream T3 consumers depend on:

* Returns a 3-tuple ``(baseline_per_sample, defended_per_sample, shared_keys)``.
* ``baseline_per_sample`` and ``defended_per_sample`` are dicts keyed by the
  T3 metric family with parallel-indexed value lists.
* ``shared_keys`` is the sorted intersection of (sample_id, epsilon,
  attack_mode, seed) on the two record files.
* Records present on only one side are silently dropped — the alignment
  is what makes Wilcoxon a valid pairwise test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from adversarial_reasoning_training.eval.robust_eval import (
    T3_METRICS,
    align_per_sample,
)


def _record(
    *,
    sample_id: str,
    epsilon: float,
    attack_mode: str = "pgd_linf",
    seed: int = 0,
    benign_seq: list[str],
    attacked_seq: list[str],
    benign_ans: str = "",
    attacked_ans: str = "",
    edit_distance_norm: float = 0.0,
) -> dict:
    return {
        "sample_id": sample_id,
        "epsilon": epsilon,
        "attack_mode": attack_mode,
        "seed": seed,
        "edit_distance_norm": edit_distance_norm,
        "benign": {
            "tool_sequence": benign_seq,
            "tool_calls": [{"args": {}} for _ in benign_seq],
            "final_answer": benign_ans,
        },
        "attacked": {
            "tool_sequence": attacked_seq,
            "tool_calls": [{"args": {}} for _ in attacked_seq],
            "final_answer": attacked_ans,
        },
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.fixture()
def paired_records(tmp_path: Path) -> tuple[Path, Path]:
    base = [
        _record(sample_id="s1", epsilon=0.0157, benign_seq=["a"], attacked_seq=["a"]),
        _record(sample_id="s2", epsilon=0.0157, benign_seq=["a"], attacked_seq=["b"]),
        _record(sample_id="s3", epsilon=0.0314, benign_seq=["a"], attacked_seq=["a"]),
        # baseline-only — must be dropped from the alignment
        _record(sample_id="s4", epsilon=0.0157, benign_seq=["a"], attacked_seq=["a"]),
    ]
    defended = [
        _record(sample_id="s1", epsilon=0.0157, benign_seq=["a"], attacked_seq=["a"]),
        _record(sample_id="s2", epsilon=0.0157, benign_seq=["a"], attacked_seq=["a"]),
        _record(sample_id="s3", epsilon=0.0314, benign_seq=["a"], attacked_seq=["a"]),
        # defended-only — must be dropped from the alignment
        _record(sample_id="s5", epsilon=0.0157, benign_seq=["a"], attacked_seq=["a"]),
    ]
    base_path = tmp_path / "baseline.jsonl"
    def_path = tmp_path / "defended.jsonl"
    _write_jsonl(base_path, base)
    _write_jsonl(def_path, defended)
    return base_path, def_path


def test_returns_three_tuple(paired_records: tuple[Path, Path]) -> None:
    base_path, def_path = paired_records
    result = align_per_sample(base_path, def_path)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_per_sample_dicts_carry_t3_metrics(paired_records: tuple[Path, Path]) -> None:
    base_path, def_path = paired_records
    baseline, defended, _ = align_per_sample(base_path, def_path)
    for metric in T3_METRICS:
        assert metric in baseline, f"baseline missing T3 metric {metric}"
        assert metric in defended, f"defended missing T3 metric {metric}"
    # parallel indexing — each metric list has the same length on both sides
    for metric in T3_METRICS:
        assert len(baseline[metric]) == len(defended[metric])


def test_intersection_drops_orphans(paired_records: tuple[Path, Path]) -> None:
    base_path, def_path = paired_records
    _, _, shared = align_per_sample(base_path, def_path)
    # 3 paired samples (s1, s2, s3) survive; s4 baseline-only and s5 defended-only drop.
    assert len(shared) == 3
    sample_ids = {key[0] for key in shared}
    assert sample_ids == {"s1", "s2", "s3"}


def test_shared_keys_are_sorted(paired_records: tuple[Path, Path]) -> None:
    base_path, def_path = paired_records
    _, _, shared = align_per_sample(base_path, def_path)
    assert shared == sorted(shared), "shared_keys must be sorted for deterministic indexing"


def test_paired_indices_align(paired_records: tuple[Path, Path]) -> None:
    base_path, def_path = paired_records
    baseline, defended, shared = align_per_sample(base_path, def_path)
    # On (s2, 0.0157) baseline disagrees with itself (a vs b → tool_name_acc=0)
    # while defended agrees (a vs a → tool_name_acc=1). Index must be the same.
    s2_idx = next(i for i, k in enumerate(shared) if k[0] == "s2")
    assert baseline["tool_name_acc"][s2_idx] == 0.0
    assert defended["tool_name_acc"][s2_idx] == 1.0


def test_empty_intersection_returns_empty_lists(tmp_path: Path) -> None:
    base = tmp_path / "b.jsonl"
    defended = tmp_path / "d.jsonl"
    _write_jsonl(base, [_record(sample_id="x", epsilon=0.01, benign_seq=["a"], attacked_seq=["a"])])
    _write_jsonl(defended, [_record(sample_id="y", epsilon=0.01, benign_seq=["a"], attacked_seq=["a"])])
    baseline_ps, defended_ps, shared = align_per_sample(base, defended)
    assert shared == []
    for metric in T3_METRICS:
        assert baseline_ps[metric] == []
        assert defended_ps[metric] == []
