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
import logging
import math
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# InternVL3 stores tool calls as {"!tool": "name", "args!": {...}}
# inside final_answer / reasoning_trace text.
_INTERNVL_TOOL_RE = re.compile(r'"!tool"\s*:\s*"([^"]+)"')
# Locate the opening brace of each InternVL3 tool-call JSON blob.
_INTERNVL_BLOB_START_RE = re.compile(r'\{\s*"[!]tool"\s*:')

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

    All positions up to ``max(len(benign), len(attacked))`` contribute
    to the denominator. Extra tool calls on either side (beyond the
    common prefix) each contribute 0.0 — a mismatch, not a silent drop.
    """
    benign_calls = record.get("benign", {}).get("tool_calls")
    attacked_calls = record.get("attacked", {}).get("tool_calls")
    if benign_calls is None or attacked_calls is None:
        return float("nan")
    n_benign = len(benign_calls)
    n_attacked = len(attacked_calls)
    n_max = max(n_benign, n_attacked)
    if n_max == 0:
        return 1.0
    total = 0.0
    n_paired = min(n_benign, n_attacked)
    for i in range(n_paired):
        b_args = benign_calls[i].get("args", {}) or {}
        a_args = attacked_calls[i].get("args", {}) or {}
        total += _args_iou_single(b_args, a_args)
    # Unpaired positions (extra calls on either side) contribute 0.0.
    return total / n_max


def _levenshtein_norm(a: list[str], b: list[str]) -> float:
    """Normalised Levenshtein distance in [0, 1]. 0.0 = identical."""
    if not a and not b:
        return 0.0
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n] / max(m, n)


def _parse_tool_sequence_from_text(text: str) -> list[str]:
    """Extract ordered tool names from InternVL3-format text.

    InternVL3 embeds tool calls as ``{"!tool": "name", ...}`` JSON
    blobs inside ``final_answer`` and ``reasoning_trace``. Returns
    tool names in document order or an empty list when none found.
    """
    if not text:
        return []
    return _INTERNVL_TOOL_RE.findall(text)


def _parse_tool_calls_from_text(text: str) -> list[dict]:
    """Extract structured ``tool_calls`` from InternVL3-format text.

    Uses ``json.JSONDecoder.raw_decode`` to find JSON blob boundaries,
    which correctly handles braces inside string values (e.g.
    ``{"query": "use {search}"}``).  Falls back to brace counting when
    ``raw_decode`` raises, then to regex-only name extraction on failure.
    """
    if not text:
        return []
    tool_calls: list[dict] = []
    for m in _INTERNVL_BLOB_START_RE.finditer(text):
        start = m.start()
        try:
            obj, end = json.JSONDecoder().raw_decode(text, start)
            name = obj.get("!tool", "")
            args = obj.get("args!", {}) or {}
            tool_calls.append({"name": name, "args": args})
            continue
        except json.JSONDecodeError:
            pass

        # Fallback: brace-count to find the matching '}'
        depth = 0
        end = start
        for i in range(start, len(text)):
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if depth != 0:
            name_match = _INTERNVL_TOOL_RE.search(text[start:])
            if name_match:
                tool_calls.append({"name": name_match.group(1), "args": {}})
            continue
        blob = text[start:end]
        try:
            obj = json.loads(blob)
            name = obj.get("!tool", "")
            args = obj.get("args!", {}) or {}
            tool_calls.append({"name": name, "args": args})
        except json.JSONDecodeError:
            name_match = _INTERNVL_TOOL_RE.search(blob)
            if name_match:
                tool_calls.append({"name": name_match.group(1), "args": {}})
    return tool_calls


def _record_metrics(record: dict) -> dict[str, float]:
    # Take shallow copies so InternVL3 text-fallback mutations don't leak
    # into the caller's cached record.  Previously this wrote back into
    # the live dict, making re-processing silently different.
    benign = dict(record["benign"])
    attacked = dict(record["attacked"])

    benign_seq = list(benign.get("tool_sequence", []))
    attacked_seq = list(attacked.get("tool_sequence", []))

    # Fallback: InternVL3 stores tools in final_answer / reasoning_trace
    # text. Only trigger when the structured field is genuinely empty.
    # Also populate ``tool_calls`` from text so args_iou can be computed.
    fallback_triggered = False
    if not benign_seq:
        benign_text = "\n".join(
            [
                benign.get("reasoning_trace") or "",
                benign.get("final_answer") or "",
            ]
        )
        benign_seq = _parse_tool_sequence_from_text(benign_text)
        if not benign.get("tool_calls"):
            benign["tool_calls"] = _parse_tool_calls_from_text(benign_text)
        fallback_triggered = True
    if not attacked_seq:
        attacked_text = "\n".join(
            [
                attacked.get("reasoning_trace") or "",
                attacked.get("final_answer") or "",
            ]
        )
        attacked_seq = _parse_tool_sequence_from_text(attacked_text)
        if not attacked.get("tool_calls"):
            attacked["tool_calls"] = _parse_tool_calls_from_text(attacked_text)
        fallback_triggered = True

    tool_name_acc = 1.0 if benign_seq == attacked_seq else 0.0

    benign_ans = (benign.get("final_answer") or "").strip()
    attacked_ans = (attacked.get("final_answer") or "").strip()
    answer_em = 1.0 if benign_ans == attacked_ans else 0.0

    # When the text fallback was used the record-level edit_distance_norm
    # was computed from empty tool_sequences (0.0 → fake 1.0 similarity).
    # Compute from the parsed sequences instead so traj_edit_distance and
    # tool_name_acc agree on what the tool sequences actually were.
    if fallback_triggered:
        traj_edit_distance = 1.0 - _levenshtein_norm(benign_seq, attacked_seq)
    else:
        edit_distance_norm = float(record.get("edit_distance_norm", 1.0))
        edit_distance_norm = max(0.0, min(1.0, edit_distance_norm))
        traj_edit_distance = 1.0 - edit_distance_norm

    # Build a composite for _args_iou_record that merges text-parsed
    # tool_calls (from shallow copies) with the original record.  Without
    # this merge, _args_iou_record would only see the original record's
    # tool_calls (which may be missing for InternVL3 text-fallback data).
    record_for_iou = dict(record)
    record_for_iou["benign"] = dict(record["benign"])
    record_for_iou["attacked"] = dict(record["attacked"])
    if fallback_triggered:
        if benign.get("tool_calls"):
            record_for_iou["benign"]["tool_calls"] = benign["tool_calls"]
        if attacked.get("tool_calls"):
            record_for_iou["attacked"]["tool_calls"] = attacked["tool_calls"]

    args_iou = _args_iou_record(record_for_iou)
    # Detect fake-perfection: args_iou=1.0 is spurious ONLY when the text
    # parser (InternVL3 fallback) recovered tool names but no arguments —
    # i.e. every recovered call on both sides has empty args. ``raw_decode``
    # can recover real args from the blob (e.g. ``"args!": {"x": 1}``), in
    # which case a matching 1.0 is genuine and must be kept. Tools with
    # genuinely no required arguments (e.g. ``lookup_guideline``) also land
    # here with empty args, but for those the 1.0 carries no information
    # either way, so NaN-ing is the safe (evidence-absent) choice.
    text_parsed_tool_calls = (
        fallback_triggered
        and (bool(benign_seq) or bool(attacked_seq))
        and not record.get("benign", {}).get("tool_calls")
        and not record.get("attacked", {}).get("tool_calls")
    )
    if args_iou == 1.0 and text_parsed_tool_calls:
        # Inspect the MERGED calls _args_iou_record actually scored, not the
        # original record (whose tool_calls were empty). Only NaN when every
        # recovered call has empty args — otherwise the 1.0 reflects real
        # matched arguments and is preserved.
        merged_benign = record_for_iou["benign"].get("tool_calls") or []
        merged_attacked = record_for_iou["attacked"].get("tool_calls") or []
        all_empty_args = all(
            not (c.get("args") if isinstance(c, dict) else None)
            for c in (*merged_benign, *merged_attacked)
        )
        if all_empty_args:
            args_iou = float("nan")

    return {
        "tool_name_acc": tool_name_acc,
        "args_iou": args_iou,
        "answer_em": answer_em,
        "traj_edit_distance": traj_edit_distance,
    }


def _pair_key(record: dict) -> tuple[str, float, str, int]:
    eps_raw = record.get("epsilon", 0.0)
    try:
        eps = float(eps_raw)
    except (TypeError, ValueError):
        eps = 0.0
    if math.isnan(eps) or math.isinf(eps):
        # NaN ordering is undefined; substitute a sentinel so sort doesn't
        # crash.  The record is almost certainly malformed — warn loudly.
        logger.warning(
            "epsilon is %s for sample_id=%r; substituting 0.0 for sort key",
            eps_raw,
            record.get("sample_id", ""),
        )
        eps = 0.0
    return (
        str(record.get("sample_id", "")),
        eps,
        str(record.get("attack_mode", "")),
        int(record.get("seed", 0)),
    )


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
                logger.warning(
                    "skipping malformed JSON at %s:%d — %s",
                    path, lineno, line[:80],
                )
    if skipped:
        logger.warning(
            "load_records: %d / %d lines skipped from %s — "
            "downstream metrics computed on %d records",
            skipped, skipped + len(records), path.name, len(records),
        )
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
    Required for paired comparisons (e.g. undefended vs defended) — the
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
    undefended/defended evaluation, prefer :func:`align_per_sample`,
    which intersects on the pair-key and is robust to missing rows.
    """
    records = load_records(path)
    # Deduplicate by pair-key: duplicate (sample_id, ε, mode, seed) records
    # silently bias per-sample means if kept.  ``keep="last"`` matches the
    # semantics of ``align_per_sample_with_drops`` which naturally keeps the
    # last record for a given key via dict construction.
    seen: dict[tuple, int] = {}
    deduped: list[dict] = []
    for r in records:
        key = _pair_key(r)
        if key in seen:
            logger.warning(
                "records_to_per_sample: duplicate pair-key %s in %s "
                "(keeping last occurrence)",
                key, path.name,
            )
            deduped[seen[key]] = r
        else:
            seen[key] = len(deduped)
            deduped.append(r)
    deduped.sort(key=_pair_key)
    metrics: dict[str, list[float]] = {key: [] for key in T3_METRICS}
    for record in deduped:
        per_record = _record_metrics(record)
        for key in T3_METRICS:
            metrics[key].append(per_record[key])
    cleaned, dropped = _drop_nan_metrics(metrics)
    if dropped:
        logger.warning(
            "records_to_per_sample: dropped metrics %s from %s "
            "(all values were NaN — check record schema or parser)",
            sorted(dropped), path.name,
        )
    return cleaned


def align_per_sample(
    undefended_records: Path,
    defended_records: Path,
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[tuple[str, float, str, int]]]:
    """Like :func:`align_per_sample_with_drops` without dropped-metric tracking."""
    undefended, defended, shared_keys, _ = align_per_sample_with_drops(
        undefended_records, defended_records,
    )
    return undefended, defended, shared_keys


def align_per_sample_with_drops(
    undefended_records: Path,
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
    base_raw = load_records(undefended_records)
    def_raw = load_records(defended_records)
    base_recs: dict[tuple, dict] = {}
    def_recs: dict[tuple, dict] = {}
    for r in base_raw:
        key = _pair_key(r)
        if key in base_recs:
            logger.warning(
                "align_per_sample: duplicate pair-key %s in undefended "
                "records — keeping last occurrence", key,
            )
        base_recs[key] = r
    for r in def_raw:
        key = _pair_key(r)
        if key in def_recs:
            logger.warning(
                "align_per_sample: duplicate pair-key %s in defended "
                "records — keeping last occurrence", key,
            )
        def_recs[key] = r
    shared_keys = sorted(set(base_recs) & set(def_recs))

    undefended: dict[str, list[float]] = {key: [] for key in T3_METRICS}
    defended: dict[str, list[float]] = {key: [] for key in T3_METRICS}
    for key in shared_keys:
        b_metrics = _record_metrics(base_recs[key])
        d_metrics = _record_metrics(def_recs[key])
        for metric in T3_METRICS:
            undefended[metric].append(b_metrics[metric])
            defended[metric].append(d_metrics[metric])
    (undefended, defended), dropped = _drop_nan_metrics_paired(undefended, defended)
    return undefended, defended, shared_keys, dropped


def save_per_sample(path: Path, per_sample: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Replace NaN / Inf with null so json.dump doesn't crash.  Upstream
    # callers should have run _drop_nan_metrics first; this is a safety net.
    def _sanitize(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize(per_sample), f, indent=2)
