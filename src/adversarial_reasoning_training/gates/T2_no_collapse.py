"""T2 — no-collapse gate.

After adversarial fine-tune, clean-input metrics must remain within a
small tolerance of the T1 (clean-only FT) ceiling. This protects
against the classic adversarial-training failure mode where robustness
is bought by completely sacrificing clean accuracy.

Pass criterion per metric:
    adv_ft_clean >= clean_ft_ceiling - tolerance_pp / 100

Writes ``runs/<id>/gates/T2.json``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class T2Thresholds:
    tolerance_pp: float = 3.0  # allowed absolute drop in percentage points
    metrics: tuple[str, ...] = ("tool_name_acc", "answer_em", "args_iou")


@dataclass
class T2Result:
    passed: bool
    per_metric: dict[str, dict[str, float]]
    duration_s: float
    thresholds: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_metrics(path: Path, metric_keys: tuple[str, ...]) -> dict[str, float]:
    """Read metrics from a gate JSON payload.

    Accepts either the T1 result layout (flat keys) or an arbitrary
    evaluator payload stored under a ``metrics`` key.
    """
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    flat: dict[str, float] = {}
    source = payload.get("metrics", payload)
    for key in metric_keys:
        if key in payload:
            flat[key] = float(payload[key])
        elif key in source:
            flat[key] = float(source[key])
    return flat


def run_t2(
    *,
    adv_clean_evaluator: Callable[[], dict[str, float]],
    t1_result_path: Path,
    out_path: Path,
    thresholds: T2Thresholds = T2Thresholds(),
) -> T2Result:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    ceiling = _read_metrics(t1_result_path, thresholds.metrics)
    current = adv_clean_evaluator() or {}

    per_metric: dict[str, dict[str, float]] = {}
    notes: list[str] = []
    passed = True
    tol = thresholds.tolerance_pp / 100.0

    for key in thresholds.metrics:
        ceil = ceiling.get(key)
        cur = float(current.get(key, 0.0))
        if ceil is None:
            notes.append(f"metric {key} missing from T1 result")
            continue
        drop = ceil - cur
        per_metric[key] = {
            "ceiling": ceil,
            "current": cur,
            "drop": drop,
            "tolerance": tol,
            "ok": drop <= tol,
        }
        if drop > tol:
            passed = False
            notes.append(
                f"{key}: drop {drop * 100:.1f} pp exceeds tolerance {thresholds.tolerance_pp:.1f} pp"
            )

    result = T2Result(
        passed=passed,
        per_metric=per_metric,
        duration_s=time.time() - start,
        thresholds={
            "tolerance_pp": thresholds.tolerance_pp,
            "metrics": list(thresholds.metrics),
        },
        notes=notes,
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    return result
