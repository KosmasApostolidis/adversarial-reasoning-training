"""Render an ablation comparison LaTeX table from per-axis aggregates.

Each ``--inputs`` entry is an ``aggregate.json`` produced by
``aggregate_seeds.py`` for one ablation axis (e.g. loss, beta, freeze,
eps). We emit one LaTeX ``tabular`` showing the canonical T3 metric
family side-by-side with mean ± std and the bootstrap 95% CI.

Usage:
    python scripts/figures/make_ablation_tables.py \\
        --inputs results/qwen_abl_loss/aggregate.json \\
                 results/qwen_abl_beta/aggregate.json \\
        --out    tables/qwen_ablations.tex
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPORT_METRICS = (
    ("T2.tool_name_acc", "Tool acc (clean)"),
    ("T2.answer_em", "Answer EM (clean)"),
    ("T3.tool_name_acc_delta", "Δ Tool acc"),
    ("T3.args_iou_delta", "Δ Args IoU"),
    ("T3.answer_em_delta", "Δ Answer EM"),
    ("T3.traj_edit_distance_delta", "Δ Traj edit dist"),
)


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.3f}"


def _row(label: str, summary: dict[str, dict[str, float]]) -> str:
    cells = [label.replace("_", r"\_")]
    for key, _ in REPORT_METRICS:
        s = summary.get(key)
        if s is None:
            cells.append("--")
            continue
        mean = s.get("mean", float("nan"))
        std = s.get("std", float("nan"))
        cells.append(f"{_fmt(mean)} \\pm {_fmt(std)}")
    return " & ".join(f"${c}$" if "\\pm" in c else c for c in cells) + r" \\"


def _label_from_path(path: Path) -> str:
    parent = path.parent.name
    prefix = "qwen_abl_"
    return parent[len(prefix):] if parent.startswith(prefix) else parent


def _load_aggregate(path: Path) -> dict[str, Any] | None:
    """Read an aggregate.json. Returns None for missing OR corrupt files
    so the caller can skip that ablation cell with a warning rather
    than crashing the whole table render.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            text = f.read().replace("NaN", "null")
        return json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARN: skipping corrupt aggregate.json at {path}: {exc}",
              file=sys.stderr)
        return None


def render(inputs: list[Path]) -> str:
    header = ["Cell"] + [label for _, label in REPORT_METRICS]
    col_spec = "l" + "r" * len(REPORT_METRICS)
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\hline",
        " & ".join(h.replace("_", r"\_") for h in header) + r" \\",
        r"\hline",
    ]
    for path in inputs:
        agg = _load_aggregate(path)
        if agg is None:
            # Soft-skip: emit an em-dash row so the operator sees
            # which cell was lost without the table dropping silently.
            print(f"NOTE: rendering placeholder row for {path}",
                  file=sys.stderr)
            label = _label_from_path(path)
            cells = [label.replace("_", r"\_")] + ["--"] * len(REPORT_METRICS)
            lines.append(" & ".join(cells) + r" \\")
            continue
        summary = agg.get("summary", {})
        lines.append(_row(_label_from_path(path), summary))
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_ablation_tables",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--inputs", type=Path, nargs="+", required=True,
                   help="aggregate.json files (one per ablation cell).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output .tex path (created with parents).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        print(f"ERROR: missing aggregate files: {[str(p) for p in missing]}",
              file=sys.stderr)
        return 1
    table = render(args.inputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table)
    print(f"wrote {args.out}  ({len(args.inputs)} rows, "
          f"{len(REPORT_METRICS)} metric columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
