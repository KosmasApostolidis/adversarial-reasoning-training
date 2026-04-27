"""Aggregate per-seed gate JSON artifacts into one ``aggregate.json``.

Each input directory is a single-seed run laid out as
``<seed_dir>/gates/{T1,T2,T3}.json``. We extract every numeric metric we
recognise across the three gate shapes, then emit per-metric mean / std /
95% percentile-bootstrap CI plus the underlying per-seed values.

Usage:
    python scripts/figures/aggregate_seeds.py \\
        --seeds runs/qwen_main_seed0 runs/qwen_main_seed1 ... \\
        --out   results/qwen_main/aggregate.json

The output schema is stable so ``make_ablation_tables.py`` and
``make_figures.py`` can consume it without re-deriving anything.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

T3_METRICS = ("tool_name_acc", "args_iou", "answer_em", "traj_edit_distance")
T1_METRICS = ("tool_name_acc", "answer_em")
T2_METRICS = ("tool_name_acc", "answer_em", "args_iou")


def _load(path: Path) -> dict[str, Any] | None:
    """Read a gate JSON. Returns None for missing OR corrupt files.

    Truncated / partially-written files (e.g. the trainer crashed
    mid-write) used to crash the entire seed-aggregation step; we now
    skip them with a stderr warning so a single bad file doesn't take
    out the whole aggregate.
    """
    if not path.exists():
        return None
    try:
        text = path.read_text().replace("NaN", "null")
        return json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARN: skipping corrupt JSON at {path}: {exc}", file=sys.stderr)
        return None


def _bootstrap_ci(
    samples: list[float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    if len(samples) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(samples)
    means = []
    for _ in range(n_resamples):
        means.append(sum(samples[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    # Symmetric percentile bootstrap: both bounds use ``int(p * N)``
    # without the asymmetric trailing ``-1`` that previously shifted
    # the upper bound one resample inward of the lower bound.
    lo_idx = max(0, min(n_resamples - 1, int(alpha * n_resamples)))
    hi_idx = max(0, min(n_resamples - 1, int((1.0 - alpha) * n_resamples)))
    return (means[lo_idx], means[hi_idx])


def _summarise(values: list[float]) -> dict[str, float | int]:
    finite = [v for v in values if v is not None and not math.isnan(v)]
    if not finite:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    mean = statistics.fmean(finite)
    std = statistics.stdev(finite) if len(finite) >= 2 else 0.0
    ci_lo, ci_hi = _bootstrap_ci(finite)
    return {"n": len(finite), "mean": mean, "std": std,
            "ci_lo": ci_lo, "ci_hi": ci_hi}


def _extract_seed_metrics(
    seed_dir: Path,
    shared_t1: Path | None = None,
) -> dict[str, dict[str, float | bool]]:
    """Pull every recognised metric + each gate's pass/fail from one seed.

    ``shared_t1`` lets the caller point at a single T1.json that is reused
    across every seed (T1 is per-model, not per-seed in this pipeline).
    Falls back to ``<seed_dir>/gates/T1.json`` when not provided.
    """
    out: dict[str, dict[str, float | bool]] = {"T1": {}, "T2": {}, "T3": {}}
    gates = seed_dir / "gates"

    t1_path = shared_t1 if shared_t1 is not None else (gates / "T1.json")
    t1 = _load(t1_path)
    if t1 is not None:
        out["T1"]["passed"] = bool(t1.get("passed", False))
        for k in T1_METRICS:
            if k in t1:
                out["T1"][k] = float(t1[k])

    t2 = _load(gates / "T2.json")
    if t2 is not None:
        out["T2"]["passed"] = bool(t2.get("passed", False))
        per = t2.get("per_metric", {})
        for k in T2_METRICS:
            if k in per and "current" in per[k]:
                out["T2"][k] = float(per[k]["current"])
                out["T2"][f"{k}_drop"] = float(per[k].get("drop", float("nan")))

    t3 = _load(gates / "T3.json")
    if t3 is not None:
        out["T3"]["passed"] = bool(t3.get("passed", False))
        per = t3.get("per_metric", {})
        for k in T3_METRICS:
            if k not in per:
                continue
            row = per[k]
            for field in ("baseline_mean", "defended_mean", "delta", "p_value", "p_adj"):
                if field in row:
                    out["T3"][f"{k}_{field}"] = float(row[field])

    return out


def aggregate(
    seed_dirs: list[Path],
    shared_t1: Path | None = None,
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for sd in seed_dirs:
        per_seed.append(
            {"seed_dir": str(sd), "metrics": _extract_seed_metrics(sd, shared_t1)}
        )

    by_metric: dict[str, list[float]] = {}
    pass_rates: dict[str, list[bool]] = {"T1": [], "T2": [], "T3": []}
    for entry in per_seed:
        for gate, vals in entry["metrics"].items():
            for k, v in vals.items():
                if k == "passed":
                    pass_rates[gate].append(bool(v))
                    continue
                by_metric.setdefault(f"{gate}.{k}", []).append(float(v))

    summary = {key: _summarise(vals) for key, vals in by_metric.items()}
    pass_summary = {
        gate: {
            "n_seeds": len(flags),
            "n_passed": sum(flags),
            "rate": (sum(flags) / len(flags)) if flags else float("nan"),
        }
        for gate, flags in pass_rates.items()
    }

    return {
        "n_seeds": len(seed_dirs),
        "seed_dirs": [str(p) for p in seed_dirs],
        "per_seed": per_seed,
        "summary": summary,
        "gate_pass_rate": pass_summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aggregate_seeds",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--seeds", type=Path, nargs="+", required=True,
                   help="One or more per-seed run directories (each holds gates/T*.json).")
    p.add_argument("--out", type=Path, required=True,
                   help="Path to aggregate.json output (created with parents).")
    p.add_argument("--shared-t1", type=Path, default=None,
                   help="Optional path to a single T1.json reused across every seed "
                        "(T1 gate is per-model, not per-seed).")
    p.add_argument("--min-seeds", type=int, default=3,
                   help="Warn (or fail with --strict) when fewer than this many seeds "
                        "are aggregated (default: 3).")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero when --min-seeds is not met.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    missing = [p for p in args.seeds if not p.exists()]
    if missing:
        print(f"ERROR: missing seed dirs: {[str(p) for p in missing]}", file=sys.stderr)
        return 1
    if args.shared_t1 is not None and not args.shared_t1.exists():
        print(f"ERROR: --shared-t1 not found: {args.shared_t1}", file=sys.stderr)
        return 1
    if len(args.seeds) < args.min_seeds:
        msg = (
            f"WARN: only {len(args.seeds)} seed(s) supplied; --min-seeds={args.min_seeds}. "
            "CIs will be wide / undefined and headline numbers should be treated as preview."
        )
        print(msg, file=sys.stderr)
        if args.strict:
            return 2
    payload = aggregate(args.seeds, shared_t1=args.shared_t1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}  (n_seeds={payload['n_seeds']}, "
          f"n_metrics={len(payload['summary'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
