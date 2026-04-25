"""Robust-eval bridge: attacks-repo records.jsonl -> T3-compatible per_sample.json.

Per-sample metrics use the *vs-benign-on-same-model* semantic:
each runner record (`runner.py:pair_record`) carries paired
`benign` / `attacked` trajectories from the same model. We measure
how much the attack disturbs the model's own behaviour, so the
metric is a *similarity* (higher = more robust).

Three of the four T3 metrics are computed here:

  * ``tool_name_acc`` — 1.0 if benign and attacked tool sequences are
    identical, else 0.0. Exact-match on ordered tool-name list.
  * ``answer_em``    — 1.0 if benign and attacked final answers are
    string-equal after ``str.strip()``, else 0.0.
  * ``traj_edit_distance`` — ``1.0 - edit_distance_norm``. The runner
    already emits the normalised Levenshtein on tool sequences as
    ``edit_distance_norm`` in [0, 1] (lower = closer); we flip to
    similarity so T3's "higher = better" delta convention holds.

``args_iou`` is **omitted** because the upstream
``trajectory_record`` (attacks-repo ``runner.py:286``) only persists
``tool_sequence`` (names), not full ``tool_calls`` with args dicts.
Computing args overlap requires a cross-repo schema extension; until
then T3 records ``args_iou`` as missing and runs BH-FDR on the
remaining three metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

T3_METRICS: tuple[str, ...] = ("tool_name_acc", "answer_em", "traj_edit_distance")


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

    return {
        "tool_name_acc": tool_name_acc,
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
    return metrics


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
    return baseline, defended, shared_keys


def save_per_sample(path: Path, per_sample: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(per_sample, f, indent=2)
