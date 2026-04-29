"""Robust-eval bridge: attacks-repo records.jsonl -> T3-compatible per_sample.json.

Per-sample metrics use the *vs-benign-on-same-model* semantic:
each runner record (`runner.py:pair_record`) carries paired
`benign` / `attacked` trajectories from the same model. We measure
how much the attack disturbs the model's own behaviour, so the
metric is a *similarity* (higher = more robust).

Four T3 metrics are computed here:

  * ``tool_name_acc`` — 1.0 if benign and attacked tool sequences are
    identical, else 0.0. Exact-match on ordered tool-name list.
  * ``args_iou`` — mean Jaccard overlap of ``(key, repr(value))`` pairs
    between paired benign / attacked tool-call args, averaged over
    paired steps. Requires the records.jsonl to include ``tool_calls``
    (attacks-repo PR #1: trajectory_record extension). Records lacking
    that field are reported as ``args_iou: nan`` and dropped from
    the metric array, so T3 falls back to its missing-metric branch.
  * ``answer_em``    — 1.0 if benign and attacked final answers are
    string-equal after ``str.strip()``, else 0.0.
  * ``traj_edit_distance`` — ``1.0 - edit_distance_norm``. The runner
    already emits the normalised Levenshtein on tool sequences as
    ``edit_distance_norm`` in [0, 1] (lower = closer); we flip to
    similarity so T3's "higher = better" delta convention holds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

T3_METRICS: tuple[str, ...] = (
    "tool_name_acc",
    "args_iou",
    "answer_em",
    "traj_edit_distance",
)


def _args_iou_single(benign_args: dict, attacked_args: dict) -> float:
    """Jaccard over (key, repr(value)) pairs between two arg dicts.

    Both empty ⇒ 1.0 (perfect agreement on nothing). Either-but-not-both
    empty ⇒ 0.0. Otherwise ``|∩| / |∪|`` over the canonicalised set.
    """
    benign_set = {(k, repr(v)) for k, v in benign_args.items()}
    attacked_set = {(k, repr(v)) for k, v in attacked_args.items()}
    if not benign_set and not attacked_set:
        return 1.0
    union = benign_set | attacked_set
    if not union:
        return 1.0
    inter = benign_set & attacked_set
    return len(inter) / len(union)


def _args_iou_record(record: dict) -> float:
    """Mean Jaccard over paired tool-call args. NaN if either side
    lacks the ``tool_calls`` field, signalling old-schema records.
    """
    benign_calls = record.get("benign", {}).get("tool_calls")
    attacked_calls = record.get("attacked", {}).get("tool_calls")
    if benign_calls is None or attacked_calls is None:
        return float("nan")
    n_paired = min(len(benign_calls), len(attacked_calls))
    if n_paired == 0:
        if not benign_calls and not attacked_calls:
            return 1.0
        return 0.0
    total = 0.0
    for i in range(n_paired):
        b_args = benign_calls[i].get("args", {}) or {}
        a_args = attacked_calls[i].get("args", {}) or {}
        total += _args_iou_single(b_args, a_args)
    return total / n_paired


def _record_metrics(record: dict) -> dict[str, float]:
    benign = record["benign"]
    attacked = record["attacked"]

    benign_seq = list(benign.get("tool_sequence", []))
    attacked_seq = list(attacked.get("tool_sequence", []))
    tool_name_acc = 1.0 if benign_seq == attacked_seq else 0.0

    benign_ans = (benign.get("final_answer") or "").strip()
    attacked_ans = (attacked.get("final_answer") or "").strip()
    answer_em = 1.0 if benign_ans == attacked_ans else 0.0

    edit_distance_norm = float(record.get("edit_distance_norm", 1.0))
    edit_distance_norm = max(0.0, min(1.0, edit_distance_norm))
    traj_edit_distance = 1.0 - edit_distance_norm

    args_iou = _args_iou_record(record)

    return {
        "tool_name_acc": tool_name_acc,
        "args_iou": args_iou,
        "answer_em": answer_em,
        "traj_edit_distance": traj_edit_distance,
    }


def _pair_key(record: dict) -> tuple[str, float, str, int]:
    return (
        str(record.get("sample_id", "")),
        float(record.get("epsilon", 0.0)),
        str(record.get("attack_mode", "")),
        int(record.get("seed", 0)),
    )


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _drop_nan_metrics(
    metrics: dict[str, list[float]],
) -> tuple[dict[str, list[float]], list[str]]:
    """If any per-record value for a metric is NaN, empty its list so
    T3 trips its missing-metric branch instead of running Wilcoxon on
    NaN. This is how ``args_iou`` falls back gracefully when records
    pre-date the trajectory_record schema extension.

    Returns ``(cleaned, dropped_names)`` so callers can surface the
    dropped metric set to the operator (C5: silent drop made T3
    pass with quiet evidence loss).
    """
    cleaned: dict[str, list[float]] = {}
    dropped: list[str] = []
    for key, values in metrics.items():
        if any(math.isnan(v) for v in values):
            cleaned[key] = []
            dropped.append(key)
        else:
            cleaned[key] = values
    return cleaned, dropped


def _drop_nan_metrics_paired(
    *metric_dicts: dict[str, list[float]],
) -> tuple[tuple[dict[str, list[float]], ...], list[str]]:
    """Apply :func:`_drop_nan_metrics` symmetrically across N dicts:
    a metric is dropped from EVERY dict if ANY dict has a NaN for it.
    Required for paired comparisons (e.g. baseline vs defended) — the
    naive per-dict drop produces length-mismatched paired arrays
    (H1: empty list paired with populated list).
    """
    if not metric_dicts:
        return (), []
    drop_set: set[str] = set()
    for d in metric_dicts:
        for key, values in d.items():
            if any(math.isnan(v) for v in values):
                drop_set.add(key)
    cleaned_all: list[dict[str, list[float]]] = []
    for d in metric_dicts:
        cleaned_all.append({k: ([] if k in drop_set else v) for k, v in d.items()})
    return tuple(cleaned_all), sorted(drop_set)


def records_to_per_sample(path: Path) -> dict[str, list[float]]:
    """Convert one model's records.jsonl into a T3 per-sample dict.

    Records are sorted by (sample_id, epsilon, attack_mode, seed) so
    that two independent models' per-sample arrays line up index-for-
    index when both runs cover the same eval matrix. For paired
    baseline/defended evaluation, prefer :func:`align_per_sample`,
    which intersects on the pair-key and is robust to missing rows.
    """
    records = load_records(path)
    records.sort(key=_pair_key)
    metrics: dict[str, list[float]] = {key: [] for key in T3_METRICS}
    for record in records:
        per_record = _record_metrics(record)
        for key in T3_METRICS:
            metrics[key].append(per_record[key])
    cleaned, _ = _drop_nan_metrics(metrics)
    return cleaned


def align_per_sample(
    baseline_records: Path,
    defended_records: Path,
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[tuple[str, float, str, int]]]:
    """Load both records files and emit per-sample dicts on the
    intersection of their (sample_id, epsilon, attack_mode, seed)
    keys, sorted identically.

    Returns ``(baseline_per_sample, defended_per_sample, shared_keys)``.
    Records present on only one side are dropped silently; callers
    can compare ``len(shared_keys)`` against the input record counts
    to detect coverage holes.
    """
    base_recs = {_pair_key(r): r for r in load_records(baseline_records)}
    def_recs = {_pair_key(r): r for r in load_records(defended_records)}
    shared_keys = sorted(set(base_recs) & set(def_recs))

    baseline: dict[str, list[float]] = {key: [] for key in T3_METRICS}
    defended: dict[str, list[float]] = {key: [] for key in T3_METRICS}
    for key in shared_keys:
        b_metrics = _record_metrics(base_recs[key])
        d_metrics = _record_metrics(def_recs[key])
        for metric in T3_METRICS:
            baseline[metric].append(b_metrics[metric])
            defended[metric].append(d_metrics[metric])
    # Paired drop: a NaN on either side empties the metric on BOTH so
    # downstream consumers never see a populated list paired with []
    # (H1: independent drops produce length-mismatched paired arrays).
    (baseline, defended), _ = _drop_nan_metrics_paired(baseline, defended)
    return baseline, defended, shared_keys


def align_per_sample_with_drops(
    baseline_records: Path,
    defended_records: Path,
) -> tuple[
    dict[str, list[float]],
    dict[str, list[float]],
    list[tuple[str, float, str, int]],
    list[str],
]:
    """Like :func:`align_per_sample` but also returns the list of
    metric names that were dropped due to NaN on either side, so the
    T3 layer (or its caller) can surface them to the operator and
    refuse to silently pass with reduced evidence (C5).
    """
    base_recs = {_pair_key(r): r for r in load_records(baseline_records)}
    def_recs = {_pair_key(r): r for r in load_records(defended_records)}
    shared_keys = sorted(set(base_recs) & set(def_recs))

    baseline: dict[str, list[float]] = {key: [] for key in T3_METRICS}
    defended: dict[str, list[float]] = {key: [] for key in T3_METRICS}
    for key in shared_keys:
        b_metrics = _record_metrics(base_recs[key])
        d_metrics = _record_metrics(def_recs[key])
        for metric in T3_METRICS:
            baseline[metric].append(b_metrics[metric])
            defended[metric].append(d_metrics[metric])
    (baseline, defended), dropped = _drop_nan_metrics_paired(baseline, defended)
    return baseline, defended, shared_keys, dropped


def save_per_sample(path: Path, per_sample: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(per_sample, f, indent=2)
