"""Compute transparency report.

Walks ``--runs`` recursively for ``gates/T*.json`` artifacts and emits a
LaTeX ``tabular`` summarising wall-clock hours and peak GPU memory per
run, plus a totals row. Optional ``--meta-glob`` lets the caller include
trainer-side metadata files (e.g. ``train_meta.json``) that may carry
their own ``duration_s`` / ``peak_memory_gb`` fields.

Usage:
    python scripts/figures/compute_summary.py \\
        --runs runs/ \\
        --out  tables/compute.tex
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable


def _load(path: Path) -> dict | None:
    try:
        text = path.read_text().replace("NaN", "null")
        return json.loads(text)
    except (OSError, ValueError):
        return None


def _walk_jsons(run_dir: Path, meta_glob: str | None) -> Iterable[Path]:
    yield from sorted(run_dir.glob("gates/T*.json"))
    if meta_glob:
        yield from sorted(run_dir.glob(meta_glob))


def summarise_run(run_dir: Path, meta_glob: str | None) -> dict[str, float | bool | str]:
    duration_s = 0.0
    peak_gb = 0.0
    gate_status: dict[str, bool] = {}
    saw_any = False
    for path in _walk_jsons(run_dir, meta_glob):
        payload = _load(path)
        if not isinstance(payload, dict):
            continue
        saw_any = True
        d = payload.get("duration_s")
        if isinstance(d, (int, float)) and not math.isnan(float(d)):
            duration_s += float(d)
        # Accept either the gate-style ``peak_memory_gb`` (T0/T1 emit this)
        # or the trainer-side ``peak_allocated_gb`` from utils.mem.MemoryStats.
        p_val: float | None = None
        for key in ("peak_memory_gb", "peak_allocated_gb"):
            v = payload.get(key)
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                p_val = float(v)
                break
        if p_val is not None:
            peak_gb = max(peak_gb, p_val)
        if path.name.startswith("T") and path.name.endswith(".json"):
            tag = path.stem
            if "passed" in payload:
                gate_status[tag] = bool(payload["passed"])
    return {
        "name": run_dir.name,
        "duration_h": duration_s / 3600.0,
        "peak_gb": peak_gb,
        "status": ",".join(
            f"{tag}{'✓' if ok else '✗'}" for tag, ok in sorted(gate_status.items())
        ) or "--",
        "has_data": saw_any,
    }


def _fmt(v: float, places: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.{places}f}"


def render_latex(rows: list[dict[str, float | bool | str]]) -> str:
    out = [
        r"\begin{tabular}{lrrl}",
        r"\hline",
        r"Run & Duration (h) & Peak (GiB) & Gate status \\",
        r"\hline",
    ]
    total_h = 0.0
    max_gb = 0.0
    for row in rows:
        if not row["has_data"]:
            continue
        name_tex = str(row["name"]).replace("_", r"\_")
        out.append(
            f"{name_tex} & "
            f"{_fmt(float(row['duration_h']))} & "
            f"{_fmt(float(row['peak_gb']), 1)} & "
            f"{row['status']} \\\\"
        )
        total_h += float(row["duration_h"])
        max_gb = max(max_gb, float(row["peak_gb"]))
    out.append(r"\hline")
    out.append(
        rf"\textbf{{Total / max}} & {_fmt(total_h)} & {_fmt(max_gb, 1)} & --- \\"
    )
    out.append(r"\hline")
    out.append(r"\end{tabular}")
    return "\n".join(out) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compute_summary",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--runs", type=Path, required=True,
                   help="Root runs/ directory; each immediate subdir is one run.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output .tex path (created with parents).")
    p.add_argument("--meta-glob", type=str, default="train_meta.json",
                   help="Glob (relative to each run dir) for trainer metadata files.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.runs.exists():
        print(f"ERROR: --runs {args.runs} does not exist", file=sys.stderr)
        return 1
    run_dirs = sorted(p for p in args.runs.iterdir() if p.is_dir())
    rows = [summarise_run(d, args.meta_glob) for d in run_dirs]
    populated = [r for r in rows if r["has_data"]]
    if not populated:
        print(f"ERROR: no gate/metadata JSONs under {args.runs}", file=sys.stderr)
        return 1
    table = render_latex(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table)
    print(f"wrote {args.out}  ({len(populated)} runs with data of {len(rows)} dirs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
