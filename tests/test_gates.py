"""T2 / T3 unit tests on synthetic fixtures (no GPU, no model)."""

from __future__ import annotations

import json

from adversarial_reasoning_training.gates.T2_no_collapse import (
    T2Thresholds,
    run_t2,
)
from adversarial_reasoning_training.gates.T3_robust import (
    T3Thresholds,
    _bh_fdr,
    run_t3,
)


def test_t2_pass_when_within_tolerance(tmp_path) -> None:
    t1_path = tmp_path / "T1.json"
    t1_path.write_text(json.dumps({
        "tool_name_acc": 0.90,
        "answer_em": 0.80,
        "args_iou": 0.75,
    }))
    out = tmp_path / "T2.json"
    result = run_t2(
        adv_clean_evaluator=lambda: {
            "tool_name_acc": 0.88,
            "answer_em": 0.78,
            "args_iou": 0.74,
        },
        t1_result_path=t1_path,
        out_path=out,
        thresholds=T2Thresholds(tolerance_pp=3.0),
    )
    assert result.passed


def test_t2_fail_on_collapse(tmp_path) -> None:
    t1_path = tmp_path / "T1.json"
    t1_path.write_text(json.dumps({
        "tool_name_acc": 0.90,
        "answer_em": 0.80,
        "args_iou": 0.75,
    }))
    out = tmp_path / "T2.json"
    result = run_t2(
        adv_clean_evaluator=lambda: {
            "tool_name_acc": 0.50,  # huge drop
            "answer_em": 0.80,
            "args_iou": 0.75,
        },
        t1_result_path=t1_path,
        out_path=out,
        thresholds=T2Thresholds(tolerance_pp=3.0),
    )
    assert not result.passed
    assert any("tool_name_acc" in n for n in result.notes)


def test_bh_fdr_rejects_smallest_p() -> None:
    p = [0.001, 0.04, 0.20, 0.80]
    rejected = _bh_fdr(p, alpha=0.05)
    # 0.001 must be rejected; 0.80 must not be
    assert rejected[0] is True
    assert rejected[3] is False


def test_t3_passes_with_clear_robustness_gain(tmp_path) -> None:
    n = 30
    undefended = {
        "tool_name_acc": [0.30] * n,
        "args_iou": [0.20] * n,
        "answer_em": [0.25] * n,
        "traj_edit_distance": [0.10] * n,
    }
    defended = {
        "tool_name_acc": [0.70] * n,
        "args_iou": [0.55] * n,
        "answer_em": [0.65] * n,
        "traj_edit_distance": [0.45] * n,
    }
    out = tmp_path / "T3.json"
    result = run_t3(
        undefended_per_sample=undefended,
        defended_per_sample=defended,
        out_path=out,
        thresholds=T3Thresholds(min_traj_edit_delta=0.10, alpha=0.05),
    )
    assert result.passed, f"expected pass, notes={result.notes}"


def test_t3_fails_when_defense_significantly_degrades_metrics(tmp_path) -> None:
    """B05 regression: pre-fix the gate used a two-sided Wilcoxon and
    counted ANY significant change as "significant_bh", so a defense that
    significantly *worsened* tool_name_acc / args_iou / answer_em while
    keeping traj_edit_distance within the directional bound passed T3.
    The directional fix must flip this case to a failure: significant
    degradation does not count toward the min-significant-metrics threshold.
    """
    n = 30
    # Defense moves every per-metric value DOWN by 0.4 (clearly significant
    # under Wilcoxon at n=30) while traj_edit_distance stays inside the
    # -min_traj_edit_delta bound. This mirrors the BUG_HUNT reproduction.
    undefended = {
        "tool_name_acc": [0.70] * n,
        "args_iou": [0.55] * n,
        "answer_em": [0.65] * n,
        "traj_edit_distance": [0.45] * n,
    }
    defended = {
        "tool_name_acc": [0.30] * n,
        "args_iou": [0.15] * n,
        "answer_em": [0.25] * n,
        "traj_edit_distance": [0.40] * n,  # only 0.05 drop — within bound
    }
    out = tmp_path / "T3.json"
    result = run_t3(
        undefended_per_sample=undefended,
        defended_per_sample=defended,
        out_path=out,
        thresholds=T3Thresholds(min_traj_edit_delta=0.10, alpha=0.05),
    )
    assert not result.passed, (
        f"expected fail, notes={result.notes}, per_metric={result.per_metric}"
    )
    # Each degraded metric must be flagged as NOT a significant improvement,
    # even though its two-sided p-value is below alpha.
    for k in ("tool_name_acc", "args_iou", "answer_em"):
        assert not result.per_metric[k]["significant_bh"], (
            f"{k} should not count as significant improvement: "
            f"{result.per_metric[k]}"
        )


def test_t3_directional_check_counts_only_positive_deltas(tmp_path) -> None:
    """Three metrics with significant positive deltas + one with significant
    negative delta. Only the three positive-delta metrics count, so a 3-of-4
    pass criterion still holds. Verifies the count uses the directional flag."""
    n = 30
    undefended = {
        "tool_name_acc": [0.30] * n,
        "args_iou": [0.20] * n,
        "answer_em": [0.25] * n,
        "traj_edit_distance": [0.10] * n,
    }
    defended = {
        "tool_name_acc": [0.70] * n,    # improvement (+0.40)
        "args_iou": [0.55] * n,         # improvement (+0.35)
        "answer_em": [0.10] * n,        # degradation (-0.15) — must not count
        "traj_edit_distance": [0.45] * n,  # improvement (+0.35)
    }
    out = tmp_path / "T3.json"
    result = run_t3(
        undefended_per_sample=undefended,
        defended_per_sample=defended,
        out_path=out,
        thresholds=T3Thresholds(
            min_traj_edit_delta=0.10,
            alpha=0.05,
            min_significant_metrics=3,
        ),
    )
    # Three positive-direction metrics significant → meets the 3-of-4 floor.
    assert result.passed, f"notes={result.notes}, per_metric={result.per_metric}"
    assert not result.per_metric["answer_em"]["significant_bh"], (
        "negative-delta metric must not register as significant improvement"
    )
    assert len(result.significant_metrics) == 3
    assert "answer_em" not in result.significant_metrics
