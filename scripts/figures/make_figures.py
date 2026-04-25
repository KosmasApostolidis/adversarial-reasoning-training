"""Render result figures from runs/ artifacts."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_jsonl(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_json_lenient(path: Path):
    text = path.read_text().replace("NaN", "null")
    return json.loads(text)


smoke = load_jsonl(RUNS / "smoke" / "train_log.jsonl")
train_steps = [r for r in smoke if r["event"] == "train_step"]
fit_done = next((r for r in smoke if r["event"] == "fit_done"), None)

t0 = load_json_lenient(RUNS / "t0" / "gates" / "T0.json")
t1 = load_json_lenient(RUNS / "t1" / "gates" / "T1.json")

steps = [r["global_step"] for r in train_steps]


# Figure 1: Loss curves
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
fig.savefig(OUT / "fig01_smoke_loss.png", bbox_inches="tight")
plt.close(fig)


# Figure 2: Grad norm + peak GPU memory
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
fig.savefig(OUT / "fig02_smoke_grad_mem.png", bbox_inches="tight")
plt.close(fig)


# Figure 3: Adversary metrics
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
fig.savefig(OUT / "fig03_smoke_attack.png", bbox_inches="tight")
plt.close(fig)


# Figure 4: T0 + T1 gate summary
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# T0 panel — grad-norm decomposition + memory headroom
ax = axes[0]
labels_t0 = ["vit", "projector", "lm"]
vals_t0 = [t0["grad_norm_vit"], t0["grad_norm_projector"], t0["grad_norm_lm"]]
bars = ax.bar(labels_t0, vals_t0, color=["#0b6efd", "#198754", "#dc3545"])
for b, v in zip(bars, vals_t0):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
clean = t0["loss_clean"]
clean_str = "NaN" if clean is None or (isinstance(clean, float) and math.isnan(clean)) else f"{clean:.3f}"
ax.set_ylabel("grad_norm")
ax.set_yscale("log")
status = "PASS" if t0["passed"] else "FAIL"
ax.set_title(
    f"T0 env gate — {status}  ({t0['duration_s']:.2f}s)\n"
    f"loss_total={t0['loss_total']:.3f}  loss_clean={clean_str}  peak={t0['peak_memory_gb']:.1f}/{t0['peak_memory_limit_gb']:.0f} GiB"
)

# T1 panel — eval metrics vs thresholds
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
status = "PASS" if t1["passed"] else "FAIL"
ax.set_title(
    f"T1 clean-FT gate — {status}  ({t1['steps']} steps, {t1['duration_s']/60:.1f} min)\n"
    f"train_loss_final={t1['train_loss_final']:.2e}  (red dashed = threshold)"
)

fig.suptitle("Gate summary — T0 (env smoke) + T1 (clean fine-tune)", y=1.02, fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig04_gates_summary.png", bbox_inches="tight")
plt.close(fig)


print(f"wrote 4 figures to {OUT}")
for p in sorted(OUT.glob("*.png")):
    print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size//1024} KiB)")
