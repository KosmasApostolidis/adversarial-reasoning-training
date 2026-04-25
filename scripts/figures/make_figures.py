"""Render result figures from runs/ artifacts.

Reads:
  runs/smoke/train_log.jsonl        (per-step metrics + fit_done)
  runs/t0/gates/T0.json             (env-gate verdict)
  runs/t1/gates/T1.json             (clean-FT-gate verdict)

Writes:
  figures/fig01_smoke_loss.png
  figures/fig02_smoke_grad_mem.png
  figures/fig03_smoke_attack.png
  figures/fig04_gates_summary.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = ROOT / "figures"

PLT_RC = {
    "figure.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_json_lenient(path: Path) -> dict:
    text = path.read_text().replace("NaN", "null")
    return json.loads(text)


def fmt_clean(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    return f"{value:.3f}"


def render_loss_curve(train_steps: list[dict], fit_done: dict, out_path: Path) -> None:
    steps = [r["global_step"] for r in train_steps]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(steps, [r["loss_total"] for r in train_steps], "o-", lw=2, label="loss_total", color="#0b6efd")
    ax.plot(steps, [r["loss_task"] for r in train_steps], "s--", lw=1.6, label="loss_task", color="#6f42c1")
    ax2 = ax.twinx()
    ax2.plot(steps, [r["loss_kl"] for r in train_steps], "^:", lw=1.6, label="loss_kl (right)", color="#198754")
    ax.set_xlabel("global_step")
    ax.set_ylabel("loss (task / total)")
    ax2.set_ylabel("loss_kl")
    ax.set_xticks(steps)
    ax.set_title(f"Smoke training — loss curves (β=6.0, {fit_done['wall_s']:.1f}s wall)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_grad_mem(train_steps: list[dict], out_path: Path) -> None:
    steps = [r["global_step"] for r in train_steps]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(steps, [r["grad_norm"] for r in train_steps], color="#fd7e14", alpha=0.75, label="grad_norm")
    ax.set_ylabel("grad_norm")
    ax.set_xlabel("global_step")
    ax.set_xticks(steps)
    ax2 = ax.twinx()
    ax2.plot(steps, [r["peak_allocated_gb"] for r in train_steps], "o-", color="#0b6efd", label="peak_allocated_gb")
    ax2.plot(steps, [r["peak_reserved_gb"] for r in train_steps], "s--", color="#dc3545", label="peak_reserved_gb")
    ax2.set_ylabel("GPU memory (GiB)")
    ax2.axhline(120.0, ls=":", color="#6c757d", lw=1, label="OOM limit (120 GiB)")
    ax.set_title("Smoke training — gradient norm & GPU memory")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_attack(train_steps: list[dict], out_path: Path) -> None:
    steps = [r["global_step"] for r in train_steps]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(steps, [r["attack_loss_final"] for r in train_steps], "o-", lw=2, color="#dc3545", label="attack_loss_final")
    ax.axhline(0, ls="--", color="#6c757d", lw=1)
    ax2 = ax.twinx()
    ax2.bar(steps, [r["attack_iterations"] for r in train_steps], alpha=0.3, color="#0b6efd", label="attack_iterations")
    ax.set_xlabel("global_step")
    ax.set_ylabel("attack_loss_final  (lower = stronger attack)")
    ax2.set_ylabel("attack_iterations")
    ax.set_xticks(steps)
    ax.set_title(f"Adversary — APGD inner loop (ε={train_steps[0]['epsilon']})")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


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
        "Gate summary — T0 (env smoke) + T1 (clean fine-tune)",
        y=1.02, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    smoke_path = RUNS / "smoke" / "train_log.jsonl"
    t0_path = RUNS / "t0" / "gates" / "T0.json"
    t1_path = RUNS / "t1" / "gates" / "T1.json"
    for p in (smoke_path, t0_path, t1_path):
        if not p.exists():
            print(
                f"ERROR: missing artifact {p.relative_to(ROOT)} — "
                "run the producing pipeline first",
                file=sys.stderr,
            )
            return 1

    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLT_RC)

    smoke = load_jsonl(smoke_path)
    train_steps = [r for r in smoke if r.get("event") == "train_step"]
    fit_done = next((r for r in smoke if r.get("event") == "fit_done"), None)
    if not train_steps or fit_done is None:
        print(
            f"ERROR: {smoke_path.relative_to(ROOT)} has no train_step / fit_done events",
            file=sys.stderr,
        )
        return 1

    t0 = load_json_lenient(t0_path)
    t1 = load_json_lenient(t1_path)

    render_loss_curve(train_steps, fit_done, OUT / "fig01_smoke_loss.png")
    render_grad_mem(train_steps, OUT / "fig02_smoke_grad_mem.png")
    render_attack(train_steps, OUT / "fig03_smoke_attack.png")
    render_gates(t0, t1, OUT / "fig04_gates_summary.png")

    print(f"wrote 4 figures to {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
