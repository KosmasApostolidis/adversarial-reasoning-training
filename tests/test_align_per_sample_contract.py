"""Behavior contract for ``align_per_sample``.

The legacy public-API test only checks that the symbol exists on
``robust_eval``. This test pins the documented runtime contract that
``cli/eval_robust.py`` and downstream T3 consumers depend on:

* Returns a 3-tuple ``(undefended_per_sample, defended_per_sample, shared_keys)``.
* ``undefended_per_sample`` and ``defended_per_sample`` are dicts keyed by the
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
        # undefended-only — must be dropped from the alignment
        _record(sample_id="s4", epsilon=0.0157, benign_seq=["a"], attacked_seq=["a"]),
    ]
    defended = [
        _record(sample_id="s1", epsilon=0.0157, benign_seq=["a"], attacked_seq=["a"]),
        _record(sample_id="s2", epsilon=0.0157, benign_seq=["a"], attacked_seq=["a"]),
        _record(sample_id="s3", epsilon=0.0314, benign_seq=["a"], attacked_seq=["a"]),
        # defended-only — must be dropped from the alignment
        _record(sample_id="s5", epsilon=0.0157, benign_seq=["a"], attacked_seq=["a"]),
    ]
    base_path = tmp_path / "undefended.jsonl"
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
    undefended, defended, _ = align_per_sample(base_path, def_path)
    for metric in T3_METRICS:
        assert metric in undefended, f"undefended missing T3 metric {metric}"
        assert metric in defended, f"defended missing T3 metric {metric}"
    # parallel indexing — each metric list has the same length on both sides
    for metric in T3_METRICS:
        assert len(undefended[metric]) == len(defended[metric])


def test_intersection_drops_orphans(paired_records: tuple[Path, Path]) -> None:
    base_path, def_path = paired_records
    _, _, shared = align_per_sample(base_path, def_path)
    # 3 paired samples (s1, s2, s3) survive; s4 undefended-only and s5 defended-only drop.
    assert len(shared) == 3
    sample_ids = {key[0] for key in shared}
    assert sample_ids == {"s1", "s2", "s3"}


def test_shared_keys_are_sorted(paired_records: tuple[Path, Path]) -> None:
    base_path, def_path = paired_records
    _, _, shared = align_per_sample(base_path, def_path)
    assert shared == sorted(shared), "shared_keys must be sorted for deterministic indexing"


def test_paired_indices_align(paired_records: tuple[Path, Path]) -> None:
    base_path, def_path = paired_records
    undefended, defended, shared = align_per_sample(base_path, def_path)
    # On (s2, 0.0157) undefended disagrees with itself (a vs b → tool_name_acc=0)
    # while defended agrees (a vs a → tool_name_acc=1). Index must be the same.
    s2_idx = next(i for i, k in enumerate(shared) if k[0] == "s2")
    assert undefended["tool_name_acc"][s2_idx] == 0.0
    assert defended["tool_name_acc"][s2_idx] == 1.0


def test_empty_intersection_returns_empty_lists(tmp_path: Path) -> None:
    base = tmp_path / "b.jsonl"
    defended = tmp_path / "d.jsonl"
    _write_jsonl(base, [_record(sample_id="x", epsilon=0.01, benign_seq=["a"], attacked_seq=["a"])])
    _write_jsonl(defended, [_record(sample_id="y", epsilon=0.01, benign_seq=["a"], attacked_seq=["a"])])
    undefended_ps, defended_ps, shared = align_per_sample(base, defended)
    assert shared == []
    for metric in T3_METRICS:
        assert undefended_ps[metric] == []
        assert defended_ps[metric] == []


# ── InternVL3 fallback: tools embedded in final_answer / reasoning_trace ──


def _record_internvl3(
    *,
    sample_id: str,
    epsilon: float,
    attack_mode: str = "apgd_linf",
    seed: int = 0,
    benign_text: str = "",
    attacked_text: str = "",
    edit_distance_norm: float = 0.0,
) -> dict:
    """Simulate InternVL3-format records where tools live in text fields."""
    return {
        "sample_id": sample_id,
        "epsilon": epsilon,
        "attack_mode": attack_mode,
        "seed": seed,
        "edit_distance_norm": edit_distance_norm,
        "benign": {
            "tool_sequence": [],
            "tool_calls": [],
            "reasoning_trace": benign_text,
            "final_answer": benign_text,
        },
        "attacked": {
            "tool_sequence": [],
            "tool_calls": [],
            "reasoning_trace": attacked_text,
            "final_answer": attacked_text,
        },
    }


def test_internvl3_tool_sequence_fallback_match(tmp_path: Path) -> None:
    """Empty tool_sequence + identical InternVL3 text → acc = 1.0."""
    text = '{"!tool": "query!guidelines", "args!": {"x": 1}}'
    u_path = tmp_path / "u.jsonl"
    d_path = tmp_path / "d.jsonl"
    _write_jsonl(u_path, [_record_internvl3(sample_id="s1", epsilon=0.01, benign_text=text, attacked_text=text)])
    _write_jsonl(d_path, [_record_internvl3(sample_id="s1", epsilon=0.01, benign_text=text, attacked_text=text)])
    undefended, defended, _ = align_per_sample(u_path, d_path)
    assert undefended["tool_name_acc"] == [1.0]
    assert defended["tool_name_acc"] == [1.0]


def test_internvl3_fallback_real_args_keep_perfect_iou(tmp_path: Path) -> None:
    """Regression: when the InternVL3 text fallback recovers REAL args via
    raw_decode and both sides match, ``args_iou`` is a genuine 1.0 and must
    NOT be NaN'd by the fake-perfection guard. NaN-ing it empties the whole
    args_iou metric (via _drop_nan_metrics), silently wiping the evidence."""
    text = '{"!tool": "query!guidelines", "args!": {"x": 1}}'
    u_path = tmp_path / "u.jsonl"
    d_path = tmp_path / "d.jsonl"
    _write_jsonl(u_path, [_record_internvl3(sample_id="s1", epsilon=0.01, benign_text=text, attacked_text=text)])
    _write_jsonl(d_path, [_record_internvl3(sample_id="s1", epsilon=0.01, benign_text=text, attacked_text=text)])
    undefended, defended, _ = align_per_sample(u_path, d_path)
    assert undefended["args_iou"] == [1.0], (
        "real recovered args matching on both sides must yield args_iou=1.0, "
        f"not be NaN-dropped to {undefended['args_iou']!r}"
    )
    assert defended["args_iou"] == [1.0]


def test_internvl3_fallback_name_only_nans_iou(tmp_path: Path) -> None:
    """Counterpart: when the fallback recovers tool NAMES but no args (empty
    args on every recovered call), the 1.0 is fake and must be NaN'd —
    preserving the discriminating behavior the guard is meant to provide."""
    text = '{"!tool": "escalate_to_specialist", "args!": {}}'
    u_path = tmp_path / "u.jsonl"
    d_path = tmp_path / "d.jsonl"
    _write_jsonl(u_path, [_record_internvl3(sample_id="s1", epsilon=0.01, benign_text=text, attacked_text=text)])
    _write_jsonl(d_path, [_record_internvl3(sample_id="s1", epsilon=0.01, benign_text=text, attacked_text=text)])
    undefended, defended, _ = align_per_sample(u_path, d_path)
    # empty recovered args → fake-perfect args_iou → metric dropped to []
    assert undefended["args_iou"] == []
    assert defended["args_iou"] == []


def test_internvl3_tool_sequence_fallback_mismatch(tmp_path: Path) -> None:
    """Attack breaks output → benign has tools, attacked is empty → acc = 0.0."""
    benign_text = '{"!tool": "escalate_to_specialist", "args!": {}}'
    attacked_text = ""
    u_path = tmp_path / "u.jsonl"
    d_path = tmp_path / "d.jsonl"
    _write_jsonl(u_path, [_record_internvl3(sample_id="s1", epsilon=0.01, benign_text=benign_text, attacked_text=attacked_text)])
    _write_jsonl(d_path, [_record_internvl3(sample_id="s1", epsilon=0.01, benign_text=benign_text, attacked_text=attacked_text)])
    undefended, defended, _ = align_per_sample(u_path, d_path)
    assert undefended["tool_name_acc"] == [0.0]
    assert defended["tool_name_acc"] == [0.0]


def test_internvl3_fallback_never_triggers_when_structured_populated(tmp_path: Path) -> None:
    """Structured tool_sequence non-empty → InternVL3 parser skipped."""
    u_path = tmp_path / "u.jsonl"
    d_path = tmp_path / "d.jsonl"
    _write_jsonl(u_path, [_record(sample_id="s1", epsilon=0.01, benign_seq=["a", "b"], attacked_seq=["a"])])
    _write_jsonl(d_path, [_record(sample_id="s1", epsilon=0.01, benign_seq=["a", "b"], attacked_seq=["a"])])
    undefended, defended, _ = align_per_sample(u_path, d_path)
    assert undefended["tool_name_acc"] == [0.0]  # ["a","b"] != ["a"]
