"""Render result figures from runs/ artifacts.

Two modes:

* ``legacy`` (default): walk ``--runs-dir`` for T0/T1 + per-run T3 inputs and
  emit ``fig04_gates_summary.png`` + per-run ``fig05_robust_comparison_*.png``.
* ``--aggregate``: render the headline 3-model bar chart from one or more
  ``aggregate.json`` files produced by ``aggregate_seeds.py`` and write it to
  ``--out``.

The two modes are mutually compatible — ``run_pipeline.sh`` calls the
aggregate mode in Phase 5; the legacy mode stays for ad-hoc inspection.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]

PLT_RC = {
    "figure.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

HEADLINE_METRICS = (
    ("T3.tool_name_acc_delta_mean", "Δ Tool acc"),
    ("T3.args_iou_delta_mean", "Δ Args IoU"),
    ("T3.answer_em_delta_mean", "Δ Answer EM"),
    ("T3.traj_edit_distance_delta_mean", "Δ Traj edit dist"),
)

COMPARISON_METRICS = (
    ("T3.tool_name_acc", "Tool acc"),
    ("T3.args_iou", "Args IoU"),
    ("T3.answer_em", "Answer EM"),
    ("T3.traj_edit_distance", "Traj edit dist"),
)


def load_json_lenient(path: Path) -> dict | None:
    """Read JSON tolerating bare ``NaN`` tokens. Returns None for
    missing OR corrupt files; callers must guard.
    """
    if not path.exists():
        return None
    try:
        text = path.read_text().replace("NaN", "null")
        return json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARN: skipping corrupt JSON at {path}: {exc}", file=sys.stderr)
        return None


def fmt_clean(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    return f"{value:.3f}"


def render_gates(t0: dict, t1: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    labels_t0 = ["vit", "projector", "lm"]
    vals_t0 = [t0["grad_norm_vit"], t0["grad_norm_projector"], t0["grad_norm_lm"]]
    bars = ax.bar(labels_t0, vals_t0, color=["#0b6efd", "#198754", "#dc3545"])
    for b, v in zip(bars, vals_t0):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("grad_norm")
    ax.set_yscale("log")
    status_t0 = "PASS" if t0["passed"] else "FAIL"
    ax.set_title(
        f"T0 env gate — {status_t0}  ({t0['duration_s']:.2f}s)\n"
        f"loss_total={t0['loss_total']:.3f}  loss_clean={fmt_clean(t0['loss_clean'])}  "
        f"peak={t0['peak_memory_gb']:.1f}/{t0['peak_memory_limit_gb']:.0f} GiB"
    )

    ax = axes[1]
    labels_t1 = ["tool_name_acc", "answer_em"]
    vals_t1 = [t1["tool_name_acc"], t1["answer_em"]]
    thrs = [t1["thresholds"]["tool_name_acc_min"], t1["thresholds"]["answer_em_min"]]
    x = list(range(len(labels_t1)))
    ax.bar(x, vals_t1, color=["#198754", "#0b6efd"], alpha=0.85, label="achieved")
    for xi, thr in zip(x, thrs):
        ax.hlines(thr, xi - 0.4, xi + 0.4, colors="#dc3545", lw=2, linestyles="--")
    for xi, v in zip(x, vals_t1):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_t1)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score")
    status_t1 = "PASS" if t1["passed"] else "FAIL"
    ax.set_title(
        f"T1 clean-FT gate — {status_t1}  ({t1['steps']} steps, {t1['duration_s']/60:.1f} min)\n"
        f"train_loss_final={t1['train_loss_final']:.2e}  (red dashed = threshold)"
    )

    fig.suptitle(
        "Gate summary — T0 (env sanity) + T1 (clean fine-tune)",
        y=1.02, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_robust_comparison(
    baseline_per_sample: dict[str, list[float]],
    defended_per_sample: dict[str, list[float]],
    t3_payload: dict,
    out_path: Path,
) -> None:
    """Render Δ = mean(defended) - mean(baseline) per T3 metric, with
    stars on BH-FDR-significant metrics. Bars include 95% bootstrap CIs
    on Δ to give a sense of effect size, not just p-value sign.
    """
    metrics = [
        m
        for m in ("tool_name_acc", "args_iou", "answer_em", "traj_edit_distance")
        if m in baseline_per_sample and m in defended_per_sample
    ]
    significant = set(t3_payload.get("significant_metrics", []))
    deltas: list[float] = []
    ci_lo: list[float] = []
    ci_hi: list[float] = []
    n_per_metric: list[int] = []
    for metric in metrics:
        baseline = baseline_per_sample[metric]
        defended = defended_per_sample[metric]
        n = min(len(baseline), len(defended))
        n_per_metric.append(n)
        if n == 0:
            deltas.append(float("nan"))
            ci_lo.append(float("nan"))
            ci_hi.append(float("nan"))
            continue
        per_sample_delta = [defended[i] - baseline[i] for i in range(n)]
        mean_delta = sum(per_sample_delta) / n
        deltas.append(mean_delta)
        ci_lo.append(_bootstrap_ci(per_sample_delta, lo=True))
        ci_hi.append(_bootstrap_ci(per_sample_delta, lo=False))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = list(range(len(metrics)))
    yerr = [
        [max(0.0, deltas[i] - ci_lo[i]) for i in range(len(metrics))],
        [max(0.0, ci_hi[i] - deltas[i]) for i in range(len(metrics))],
    ]
    ax.bar(x, deltas, yerr=yerr, capsize=4, color="#4c72b0", alpha=0.85)
    for i, metric in enumerate(metrics):
        if metric in significant:
            ax.text(
                x[i],
                deltas[i] + (yerr[1][i] if not math.isnan(yerr[1][i]) else 0.0) + 0.01,
                "*",
                ha="center",
                va="bottom",
                fontsize=14,
                color="#cc4c02",
            )
    ax.axhline(0.0, color="gray", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_ylabel("Δ = defended − baseline (similarity, higher is better)")
    n_total = max(n_per_metric) if n_per_metric else 0
    passed = bool(t3_payload.get("passed"))
    ax.set_title(
        f"Robust-eval Δ vs undefended baseline — n={n_total} paired "
        f"(samples × ε), T3 {'PASS' if passed else 'FAIL'}"
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_headline_3model(aggregates: list[Path], out_path: Path) -> int:
    """Headline figure: per-model Δ bars over T3 metrics with bootstrap CIs.

    Each ``aggregates`` entry is one ``aggregate.json`` (one model).
    Group label is the directory name minus the trailing ``_main`` suffix
    so ``results/qwen_main/aggregate.json`` → ``qwen``.
    """
    if not aggregates:
        print("ERROR: --aggregate requires at least one path", file=sys.stderr)
        return 1
    payloads: list[tuple[str, dict]] = []
    missing: list[Path] = []
    for path in aggregates:
        if not path.exists():
            print(f"WARN: missing aggregate (skipping): {path}", file=sys.stderr)
            missing.append(path)
            continue
        agg = load_json_lenient(path)
        if agg is None:
            # Corrupt JSON — load_json_lenient already logged the cause.
            missing.append(path)
            continue
        label = path.parent.name
        if label.endswith("_main"):
            label = label[:-5]
        payloads.append((label, agg))
    if not payloads:
        print(
            f"ERROR: no aggregates available — all {len(aggregates)} paths missing",
            file=sys.stderr,
        )
        return 1
    if missing:
        print(
            f"NOTE: rendering with {len(payloads)}/{len(aggregates)} models "
            f"(missing: {', '.join(str(p) for p in missing)})",
            file=sys.stderr,
        )

    metric_keys = [k for k, _ in HEADLINE_METRICS]
    metric_labels = [lab for _, lab in HEADLINE_METRICS]

    n_models = len(payloads)
    n_metrics = len(metric_keys)
    bar_w = 0.8 / max(1, n_models)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.8 * n_metrics), 4.0))

    palette = ["#0b6efd", "#198754", "#dc3545", "#cc4c02", "#6f42c1"]
    for mi, (label, agg) in enumerate(payloads):
        summary = agg.get("summary", {})
        means: list[float] = []
        lo_err: list[float] = []
        hi_err: list[float] = []
        for key in metric_keys:
            row = summary.get(key) or {}
            mean = float(row.get("mean", float("nan")))
            ci_lo = float(row.get("ci_lo", float("nan")))
            ci_hi = float(row.get("ci_hi", float("nan")))
            means.append(mean)
            lo_err.append(0.0 if math.isnan(ci_lo) or math.isnan(mean) else max(0.0, mean - ci_lo))
            hi_err.append(0.0 if math.isnan(ci_hi) or math.isnan(mean) else max(0.0, ci_hi - mean))
        x = [j + (mi - (n_models - 1) / 2.0) * bar_w for j in range(n_metrics)]
        ax.bar(
            x, means,
            width=bar_w,
            yerr=[lo_err, hi_err],
            capsize=3,
            label=label,
            color=palette[mi % len(palette)],
            alpha=0.85,
        )

    ax.axhline(0.0, color="gray", linewidth=0.7)
    ax.set_xticks(list(range(n_metrics)))
    ax.set_xticklabels(metric_labels, rotation=15, ha="right")
    ax.set_ylabel("Δ = defended − baseline (mean ± 95% CI)")
    ax.set_title(
        f"Headline robustness gains — {n_models} model(s), seed-aggregated T3"
    )
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({n_models} model(s), {n_metrics} metrics)")
    return 0


def render_baseline_vs_defended(aggregates: list[Path], out_path: Path) -> int:
    """Side-by-side baseline vs defended bars, one subplot per T3 metric."""
    if not aggregates:
        print("ERROR: --compare requires at least one path", file=sys.stderr)
        return 1
    payloads: list[tuple[str, dict]] = []
    for path in aggregates:
        if not path.exists():
            print(f"WARN: missing aggregate (skipping): {path}", file=sys.stderr)
            continue
        agg = load_json_lenient(path)
        if agg is None:
            continue
        label = path.parent.name
        if label.endswith("_main"):
            label = label[:-5]
        payloads.append((label, agg))
    if not payloads:
        print("ERROR: no aggregates available for comparison", file=sys.stderr)
        return 1

    metric_keys = [k for k, _ in COMPARISON_METRICS]
    metric_labels = [lab for _, lab in COMPARISON_METRICS]
    palette = {"baseline": "#6c757d", "defended": "#198754"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for mi, (ax, key, mlab) in enumerate(zip(axes.flat, metric_keys, metric_labels)):
        models_with_data: list[tuple[str, float, float, float, float]] = []
        for label, agg in payloads:
            summary = agg.get("summary", {})
            bl_key = f"{key}_baseline_mean"
            df_key = f"{key}_defended_mean"
            bl_row = summary.get(bl_key)
            df_row = summary.get(df_key)
            if not bl_row or not df_row:
                continue
            bl_mean = float(bl_row.get("mean", float("nan")))
            df_mean = float(df_row.get("mean", float("nan")))
            bl_ci_lo = float(bl_row.get("ci_lo", float("nan")))
            bl_ci_hi = float(bl_row.get("ci_hi", float("nan")))
            df_ci_lo = float(df_row.get("ci_lo", float("nan")))
            df_ci_hi = float(df_row.get("ci_hi", float("nan")))
            if math.isnan(bl_mean) or math.isnan(df_mean):
                continue
            models_with_data.append((
                label, bl_mean, df_mean,
                max(0.0, bl_mean - bl_ci_lo) if not math.isnan(bl_ci_lo) else 0.0,
                max(0.0, bl_ci_hi - bl_mean) if not math.isnan(bl_ci_hi) else 0.0,
                max(0.0, df_mean - df_ci_lo) if not math.isnan(df_ci_lo) else 0.0,
                max(0.0, df_ci_hi - df_mean) if not math.isnan(df_ci_hi) else 0.0,
            ))

        if not models_with_data:
            ax.text(0.5, 0.5, "No T3 data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(mlab)
            continue

        n = len(models_with_data)
        x = list(range(n))
        bar_w = 0.35
        for offset, cond, color in [(-bar_w / 2, 1, palette["baseline"]),
                                      (bar_w / 2, 2, palette["defended"])]:
            means = [m[cond] for m in models_with_data]
            lo_err = [m[3] if cond == 1 else m[5] for m in models_with_data]
            hi_err = [m[4] if cond == 1 else m[6] for m in models_with_data]
            ax.bar(
                [xi + offset for xi in x], means,
                width=bar_w,
                yerr=[lo_err, hi_err],
                capsize=3,
                color=color,
                alpha=0.85,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in models_with_data])
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.15)
        ax.set_title(mlab)

    # Shared legend
    handles = [
        Patch(facecolor=palette["baseline"], alpha=0.85, label="baseline"),
        Patch(facecolor=palette["defended"], alpha=0.85, label="defended"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)

    fig.suptitle(
        "Adversarially trained vs baseline — per-model per-metric comparison",
        y=1.01, fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({len(payloads)} model(s), {len(metric_keys)} metrics)")
    return 0


def _bootstrap_ci(
    samples: list[float],
    *,
    lo: bool,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> float:
    """Percentile bootstrap on the mean. Deterministic via fixed seed
    so figure rendering is reproducible. Returns the lo or hi
    boundary of the central ``confidence`` interval.
    """
    if not samples:
        return float("nan")
    import random

    rng = random.Random(seed)
    n = len(samples)
    means = []
    for _ in range(n_resamples):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    # Symmetric percentile bootstrap: both bounds use ``int(p * N)``
    # so the lo and hi sides are inset the same number of resamples
    # from the edges of the sorted distribution.
    idx = int(alpha * n_resamples) if lo else int((1.0 - alpha) * n_resamples)
    idx = max(0, min(n_resamples - 1, idx))
    return means[idx]


def render_legacy(runs: Path, out_dir: Path) -> int:
    t0_path = runs / "t0" / "gates" / "T0.json"
    t1_path = runs / "t1" / "gates" / "T1.json"
    for p in (t0_path, t1_path):
        if not p.exists():
            print(
                f"ERROR: missing artifact {p} — run the producing pipeline first",
                file=sys.stderr,
            )
            return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLT_RC)

    t0 = load_json_lenient(t0_path)
    t1 = load_json_lenient(t1_path)
    if t0 is None or t1 is None:
        print(
            f"ERROR: corrupt T0/T1 JSON; cannot render gates summary",
            file=sys.stderr,
        )
        return 1

    render_gates(t0, t1, out_dir / "fig04_gates_summary.png")

    n_figures = 1
    robust_t3_paths = sorted(runs.glob("*/gates/T3.json"))
    for t3_path in robust_t3_paths:
        gates_dir = t3_path.parent
        baseline_path = gates_dir / "baseline_per_sample.json"
        defended_path = gates_dir / "defended_per_sample.json"
        if not (baseline_path.exists() and defended_path.exists()):
            continue
        run_name = gates_dir.parent.name
        baseline_ps = load_json_lenient(baseline_path)
        defended_ps = load_json_lenient(defended_path)
        t3_payload = load_json_lenient(t3_path)
        if baseline_ps is None or defended_ps is None or t3_payload is None:
            print(
                f"WARN: skipping {run_name} (corrupt per-sample/T3 JSON)",
                file=sys.stderr,
            )
            continue
        out_path = out_dir / f"fig05_robust_comparison_{run_name}.png"
        render_robust_comparison(baseline_ps, defended_ps, t3_payload, out_path)
        n_figures += 1

    print(f"wrote {n_figures} figures to {out_dir}")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p}  ({p.stat().st_size // 1024} KiB)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_figures",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--aggregate", type=Path, nargs="+", default=None,
        help="aggregate.json file(s) → render headline 3-model figure to --out.",
    )
    p.add_argument(
        "--compare", type=Path, nargs="+", default=None,
        help="aggregate.json file(s) → render baseline-vs-defended comparison to --out.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Figure output PNG (required with --aggregate or --compare).",
    )
    p.add_argument(
        "--runs-dir", type=Path, default=ROOT / "runs",
        help="Legacy mode: walk this dir for T0/T1/T3 inputs (default: <repo>/runs).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=ROOT / "figures",
        help="Legacy mode: write per-run figures here (default: <repo>/figures).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    plt.rcParams.update(PLT_RC)
    if args.compare:
        if args.out is None:
            print("ERROR: --compare requires --out", file=sys.stderr)
            return 2
        return render_baseline_vs_defended(args.compare, args.out)
    if args.aggregate:
        if args.out is None:
            print("ERROR: --aggregate requires --out", file=sys.stderr)
            return 2
        return render_headline_3model(args.aggregate, args.out)
    return render_legacy(args.runs_dir, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
