# Experiment Runs — Publication Readiness

Runnable command sheet for the publication-readiness program described in
`.claude/plans/what-does-the-vectorized-codd.md`. **Do not auto-execute** —
each phase below is launched manually by the user.

Target venue: top-tier conference (NeurIPS / ICLR / CVPR-tier).
Model scope: `qwen2_5_vl_7b`, `llava_v1_6_mistral_7b`, `llama_3_2_vision_11b`.
Seeds per cell: 5 (`SEED ∈ {0,1,2,3,4}`).

---

## Project & Problem

Medical-imaging VLM agents (Qwen2.5-VL-7B, LLaVA-v1.6-Mistral-7B,
Llama-3.2-Vision-11B) reason on prostate-MRI tasks via a **ReAct loop** —
they pick a tool, choose tool arguments, read intermediate evidence, then
emit a final diagnosis. Defending only the final answer is insufficient:
white-box L∞ PGD pixel perturbations can flip an intermediate tool call
(e.g. mis-route a T2-weighted lookup) and propagate the error without
visibly altering final-answer logits. This repo trains the **whole
reasoning trajectory** to be robust.

**Threat model.** White-box, gradient-aware attacker. Pixel L∞ budget
`ε ∈ {2, 4, 8}/255`. Task scope = ProstateX prostate-MRI VLM agent.
Out of scope (cited as limitations): defense-aware adaptive attackers,
non-PGD attack families (AutoAttack / BPDA / etc.), tasks beyond ProstateX.

**Approach.** Trajectory-aware adversarial fine-tuning. Inner PGD wraps a
teacher-forced trajectory CE; the outer optimizer minimizes
task-loss + trajectory-consistency-loss using one of three loss families
(TRADES / PGD-AT / OAAT). Backprop through the full ReAct chain collapses
to one transformer forward with structured segment masks
(`src/adversarial_reasoning_training/trajectory/`). Full fine-tune (ViT +
projector + LM) with bnb 8-bit AdamW, gradient checkpointing, and an
ε curriculum 2/255 → 4/255 → 8/255.

**Validation = four gates** (T0 env → T1 clean-FT → T2 no-collapse →
T3 robustness). Downstream gates refuse stale upstream JSON; each gate
emits a pass/fail verdict + evidence under `runs/<id>/gates/`.

Source: `README.md`, `docs/PROJECT_REPORT.md`,
`src/adversarial_reasoning_training/{trajectory,losses,attacks,gates}/`.

---

## Defense & Training Defaults

Loss family is selected by `defense:` in `configs/training.yaml`.

| Family   | Formulation                                                            |
| -------- | ---------------------------------------------------------------------- |
| `trades` (default) | `L = task_ce(clean) + β · KL(p_clean ‖ p_adv)`               |
| `pgd_at` | `L = task_ce(adv)`                                                     |
| `oaat`   | `L = α · task_ce(clean) + (1 − α) · task_ce(adv)`                      |

Headline hyperparameters (`configs/training.yaml`):

| Knob                  | Value                                              |
| --------------------- | -------------------------------------------------- |
| epochs                | 50                                                 |
| micro-batch / accum   | 1 / 8 → effective batch 8                          |
| optimizer             | `adamw8bit`, β=(0.9, 0.999), wd=0.0                |
| LR (lm / proj / vit)  | 5e-6 / 1e-5 / 1e-6, cosine, warmup 3 %             |
| AMP                   | bf16, grad-clip 1.0, grad-checkpoint on            |
| ε curriculum          | 2/255 (e1–2) → 4/255 (e3) → 8/255 (e4–5+)          |
| TRADES β              | 6.0 (`configs/defenses.yaml`)                      |
| best-metric           | `tool_name_acc`                                    |
| checkpoint policy     | weights-only final save (avoids disk-full on 7B)   |

---

## Four-Gate Validation

| Gate | Module                                                  | Verifies                                                                                                                                                                            |
| ---- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| T0   | `adversarial_reasoning_training.gates.T0_env`           | Build + 1 fwd/bwd: total loss finite, every role-group (vit / projector / lm) gets non-zero grad, CUDA peak under ceiling.                                                          |
| T1   | `adversarial_reasoning_training.gates.T1_clean`         | Short clean-only FT (PGD off, ≤200 steps). Defaults: `tool_name_acc ≥ 0.85`, `answer_em ≥ 0.70`. Proves loaders, segment masks, optimizer, oracle templates are learnable.          |
| T2   | `adversarial_reasoning_training.gates.T2_no_collapse`   | Adv-FT clean metrics within `tolerance_pp = 3.0` of T1 ceiling on `{tool_name_acc, answer_em, args_iou}`. Catches the classic AT failure mode (robust but clean accuracy collapses). |
| T3   | `adversarial_reasoning_training.gates.T3_robust`        | Wilcoxon signed-rank per metric + BH-FDR over `{tool_name_acc, args_iou, answer_em, traj_edit_distance}`. Pass: `traj_edit_delta ≥ 0.10`, ≥ 3/4 metrics significant at α = 0.05.    |

Each gate writes `runs/<id>/gates/T*.json`. T2 + T3 currently land paired
records via the attacks-repo runner (`adversarial-reasoning-attacks`)
before evaluating.

---

## Prerequisites

Before any phase below:

```bash
# 1. Conda env
conda activate kosmasenv

# 2. Sibling deps first, then this repo, then optional flash-attn for Qwen
pip install -e ../adversarial-reasoning-attacks
pip install -e .[dev]
pip install -e .[flash]                       # optional, Qwen flash-attn

# 3. Console scripts (declared in pyproject.toml [project.scripts])
art-train       --help
art-eval-robust --help
art-make-gold   --help
```

Hardware: single H200 (≥ 120 GiB) per run; T0 enforces this ceiling.

Data: ProstateX preprocessed bundle at the path declared in
`configs/data.yaml` (preprocessing transfer reused from
`adversarial_reasoning.gates.preprocessing_transfer`).

Models: Hugging Face cache populated for `qwen2_5_vl_7b`,
`llava_v1_6_mistral_7b`, `llama_3_2_vision_11b`. Registry in
`../adversarial-reasoning-attacks/configs/models.yaml`.

---

## Step 0 — Gold Trajectories (one-time)

Materialize rule-based gold ReAct trajectories from ProstateX metadata +
the 50-case expert-reviewed probe. Re-run only when `configs/gold.yaml`
or the upstream metadata changes.

```bash
art-make-gold --config configs/gold.yaml --data configs/data.yaml
```

Underlying entry: `src/adversarial_reasoning_training/cli/make_gold.py`
(thin shim `scripts/make_gold_trajectories.py` is preserved for
backward compatibility).

---

## Run Layout

Every `art-train --run-dir runs/<id>` produces:

```
runs/<id>/
├── ckpt/             # CheckpointRegistry: best.pt + step-N.pt (ckpt_keep=2)
├── gates/            # T0.json … T3.json + figXX_*.png
├── records.jsonl     # per-sample defended records (from attacks-repo runner)
└── logs/             # train log + memory probe
```

`runs/baseline_<model>/records.jsonl` (undefended reference) is generated
**once per model** then reused by every defended seed/ablation cell —
do not regenerate per seed.

---

## CLI Reference

`art-train` (from `src/adversarial_reasoning_training/cli/train.py:45`):

```
art-train \
  --config <training.yaml> \
  --defenses <defenses.yaml> \
  --data <data.yaml> \
  --gold <gold.yaml> \
  --full-ft <full_ft.yaml> \
  --model <model_family_id> \
  --run-dir <runs/<id>> \
  [--device cuda] [--seed 0] \
  [--models-yaml ../adversarial-reasoning-attacks/configs/models.yaml]
```

`art-eval-robust` (from `src/adversarial_reasoning_training/cli/eval_robust.py:34`):

```
art-eval-robust \
  --baseline-records <records.jsonl> \
  --defended-records <records.jsonl> \
  --out-dir <dir> \
  [--alpha 0.05] [--min-traj-edit-delta 0.10] [--min-significant-metrics 3]
```

Per-sample `records.jsonl` is produced by the attacks-repo runner
(`adversarial-reasoning-attacks/src/adversarial_reasoning/runner.py`):

```
python -m adversarial_reasoning.runner \
  --config <experiments/<exp>.yaml> \
  --attacks-config configs/attacks.yaml \
  --split <dev|test> --max-steps 8 --pgd-steps 20 \
  --out <records.jsonl>
```

Model registry IDs (from `../adversarial-reasoning-attacks/configs/models.yaml`):
`qwen2_5_vl_7b`, `llava_v1_6_mistral_7b`, `llama_3_2_vision_11b`,
`defended_qwen2_5_vl_7b`.

Gates that run as standalone modules (per `gates/T*.py` docstrings):

```
python -m adversarial_reasoning_training.gates.T0_env --model <id> --out <T0.json>
python -m adversarial_reasoning_training.gates.T1_clean --model <id> [...] --out <T1.json>
python -m adversarial_reasoning_training.gates.T2_no_collapse --model <id> --ckpt <best.pt> --out <T2.json>
```

---

## Phase 0 — Pipeline unblock (single seed, single model)

Goal: one Qwen2.5-VL-7B run with all four gates green end-to-end.

```bash
cd /home/medadmin/kosmasapostolidis/adversarial-reasoning-training

# 0a. Re-run T0 to confirm cosmetic NaN fix from cb575f8.
python -m adversarial_reasoning_training.gates.T0_env \
  --model qwen2_5_vl_7b \
  --out runs/t0_recheck/T0.json

# 0b. T1 clean-FT gate (no adversary, ≤200 steps).
python -m adversarial_reasoning_training.gates.T1_clean \
  --model qwen2_5_vl_7b \
  --config configs/training.yaml \
  --data configs/data.yaml \
  --gold configs/gold.yaml \
  --full-ft configs/full_ft.yaml \
  --out runs/t1_qwen/T1.json

# 0c. First end-to-end Qwen adv-FT, seed=0, full ε curriculum 2/255 → 8/255.
art-train \
  --config configs/training.yaml \
  --defenses configs/defenses.yaml \
  --data configs/data.yaml \
  --gold configs/gold.yaml \
  --full-ft configs/full_ft.yaml \
  --model qwen2_5_vl_7b \
  --run-dir runs/qwen_full_seed0 \
  --seed 0

# 0d. T2 diversity gate against the resulting checkpoint.
python -m adversarial_reasoning_training.gates.T2_no_collapse \
  --model qwen2_5_vl_7b \
  --ckpt runs/qwen_full_seed0/ckpt/best.pt \
  --out runs/qwen_full_seed0/gates/T2.json

# 0e. Generate baseline (undefended) and defended records via attacks-repo runner.
# > ⚠️ Not yet in repo: `../adversarial-reasoning-attacks/configs/experiments/{baseline_qwen,defended_qwen}.yaml`.
# > Author both before this step. `defended_qwen.yaml` should point its
# > checkpoint at `runs/qwen_full_seed0/ckpt/best.pt`.
cd ../adversarial-reasoning-attacks
python -m adversarial_reasoning.runner \
  --config configs/experiments/baseline_qwen.yaml \
  --split dev \
  --out ../adversarial-reasoning-training/runs/baseline_qwen/records.jsonl

python -m adversarial_reasoning.runner \
  --config configs/experiments/defended_qwen.yaml \
  --split dev \
  --out ../adversarial-reasoning-training/runs/qwen_full_seed0/records.jsonl
cd -

# 0f. T3 robust-eval gate (Wilcoxon paired + BH-FDR).
art-eval-robust \
  --baseline-records runs/baseline_qwen/records.jsonl \
  --defended-records runs/qwen_full_seed0/records.jsonl \
  --out-dir runs/qwen_full_seed0/gates/
```

**Exit criterion:** `runs/qwen_full_seed0/gates/{T0,T1,T2,T3}.json` all PASS;
`fig05_robust_comparison_*.png` rendered.

---

## Phase 1 — Qwen 5-seed headline (Qwen only)

```bash
cd /home/medadmin/kosmasapostolidis/adversarial-reasoning-training

for SEED in 0 1 2 3 4; do
  art-train \
    --config configs/training.yaml \
    --defenses configs/defenses.yaml \
    --data configs/data.yaml \
    --gold configs/gold.yaml \
    --full-ft configs/full_ft.yaml \
    --model qwen2_5_vl_7b \
    --run-dir runs/qwen_main_seed${SEED} \
    --seed ${SEED}

  cd ../adversarial-reasoning-attacks
  python -m adversarial_reasoning.runner \
    --config configs/experiments/defended_qwen.yaml \
    --split test \
    --out ../adversarial-reasoning-training/runs/qwen_main_seed${SEED}/records.jsonl
  cd -

  art-eval-robust \
    --baseline-records runs/baseline_qwen/records.jsonl \
    --defended-records runs/qwen_main_seed${SEED}/records.jsonl \
    --out-dir runs/qwen_main_seed${SEED}/gates/
done

# Aggregate after all 5 seeds finish.
# > ⚠️ Not yet in repo: `scripts/figures/aggregate_seeds.py`. Author with
# > interface `--inputs <T3.json>... --out <aggregate.json>` collecting
# > per-metric mean ± stderr + Wilcoxon survivors across seeds.
python scripts/figures/aggregate_seeds.py \
  --inputs runs/qwen_main_seed{0,1,2,3,4}/gates/T3.json \
  --out results/qwen_main/aggregate.json

python scripts/figures/make_figures.py \
  --aggregate results/qwen_main/aggregate.json \
  --out figures/fig05_qwen_main_5seeds.png
```

**Exit criterion:** Wilcoxon q < 0.05 on ≥3 / 4 metrics (`tool_acc`,
`args_iou`, `answer_em`, `traj_edit`); `aggregate.json` reports
`n_seeds=5`.

---

## Phase 2 — Qwen ablation matrix

Each axis re-uses the Phase 1 seed loop. Configs live under `configs/ablations/`
(create per cell). Reference cell = Phase 1 (TRADES, β=6, full unfreeze, ε
curriculum 2→8) — do **not** re-run.

> ⚠️ Not yet in repo: every ablation YAML below must be authored before its
> axis can run:
> `configs/ablations/{loss_pgd_at,loss_oaat}.yaml`,
> `configs/ablations/beta_{0,1,12}.yaml`,
> `configs/ablations/full_ft_{lm_only,vit_proj_frozen}.yaml`,
> `configs/ablations/defenses_eps_fixed_8.yaml`.
> Each should diff minimally from `configs/{training,defenses,full_ft}.yaml`
> on the single knob being ablated.

### 2a. Loss family

```bash
for LOSS in pgd_at oaat; do
  for SEED in 0 1 2 3 4; do
    art-train \
      --config configs/ablations/loss_${LOSS}.yaml \
      --defenses configs/defenses.yaml \
      --data configs/data.yaml \
      --gold configs/gold.yaml \
      --full-ft configs/full_ft.yaml \
      --model qwen2_5_vl_7b \
      --run-dir runs/qwen_abl_loss_${LOSS}_seed${SEED} \
      --seed ${SEED}
  done
done
```

### 2b. Trajectory-KL weight β sweep

Apply same-step-count ceiling (cf. memory `feedback_t2_gate_budget_parity.md`).

```bash
for BETA in 0 1 12; do
  for SEED in 0 1 2 3 4; do
    art-train \
      --config configs/ablations/beta_${BETA}.yaml \
      --defenses configs/defenses.yaml \
      --data configs/data.yaml \
      --gold configs/gold.yaml \
      --full-ft configs/full_ft.yaml \
      --model qwen2_5_vl_7b \
      --run-dir runs/qwen_abl_beta_${BETA}_seed${SEED} \
      --seed ${SEED}
  done
done
```

### 2c. Freeze strategy

```bash
for FREEZE in lm_only vit_proj_frozen; do
  for SEED in 0 1 2 3 4; do
    art-train \
      --config configs/training.yaml \
      --defenses configs/defenses.yaml \
      --data configs/data.yaml \
      --gold configs/gold.yaml \
      --full-ft configs/ablations/full_ft_${FREEZE}.yaml \
      --model qwen2_5_vl_7b \
      --run-dir runs/qwen_abl_freeze_${FREEZE}_seed${SEED} \
      --seed ${SEED}
  done
done
```

### 2d. ε schedule

```bash
for SEED in 0 1 2 3 4; do
  art-train \
    --config configs/training.yaml \
    --defenses configs/ablations/defenses_eps_fixed_8.yaml \
    --data configs/data.yaml \
    --gold configs/gold.yaml \
    --full-ft configs/full_ft.yaml \
    --model qwen2_5_vl_7b \
    --run-dir runs/qwen_abl_eps_fixed_seed${SEED} \
    --seed ${SEED}
done
```

After every cell completes its 5 seeds, run the Phase 1 eval + aggregate
snippet, swapping `--inputs` and `--out`.

**Exit criterion per axis:** `results/qwen_abl_<axis>/aggregate.json` exists
with paired delta vs reference cell + significance markers.

---

## Phase 3 — LLaVA-v1.6-Mistral-7B replication

```bash
# Headline 5-seed sweep.
for SEED in 0 1 2 3 4; do
  art-train \
    --config configs/training.yaml \
    --defenses configs/defenses.yaml \
    --data configs/data.yaml \
    --gold configs/gold.yaml \
    --full-ft configs/full_ft.yaml \
    --model llava_v1_6_mistral_7b \
    --run-dir runs/llava_main_seed${SEED} \
    --seed ${SEED}
done

# Loss-family ablation only (3 cells × 5 seeds).
for LOSS in trades pgd_at oaat; do
  for SEED in 0 1 2 3 4; do
    art-train \
      --config configs/ablations/loss_${LOSS}.yaml \
      --defenses configs/defenses.yaml \
      --data configs/data.yaml \
      --gold configs/gold.yaml \
      --full-ft configs/full_ft.yaml \
      --model llava_v1_6_mistral_7b \
      --run-dir runs/llava_abl_loss_${LOSS}_seed${SEED} \
      --seed ${SEED}
  done
done

# Plus baseline_llava records and per-seed defended records (same shape as Phase 0e).
```

---

## Phase 4 — Llama-3.2-Vision-11B replication

Same shape as Phase 3, swap `--model llama_3_2_vision_11b`. Expect ~1.5×
wall-time per run; document any bf16-only / batch-size deviations in the
paper appendix.

```bash
for SEED in 0 1 2 3 4; do
  art-train \
    --config configs/training.yaml \
    --defenses configs/defenses.yaml \
    --data configs/data.yaml \
    --gold configs/gold.yaml \
    --full-ft configs/full_ft.yaml \
    --model llama_3_2_vision_11b \
    --run-dir runs/llama_main_seed${SEED} \
    --seed ${SEED}
done
# Loss-family ablation block identical to Phase 3, --model swapped.
```

---

## Phase 5 — Paper artifacts (no training)

```bash
# Headline 3-model figure.
python scripts/figures/make_figures.py \
  --aggregate results/qwen_main/aggregate.json \
              results/llava_main/aggregate.json \
              results/llama_main/aggregate.json \
  --out figures/fig_headline_3model.png

# Ablation tables (Qwen).
# > ⚠️ Not yet in repo: `scripts/figures/make_ablation_tables.py`.
# > Expected interface: `--inputs <aggregate.json>... --out <*.tex>`.
python scripts/figures/make_ablation_tables.py \
  --inputs results/qwen_abl_loss/aggregate.json \
           results/qwen_abl_beta/aggregate.json \
           results/qwen_abl_freeze/aggregate.json \
           results/qwen_abl_eps/aggregate.json \
  --out tables/qwen_ablations.tex

# Compute transparency report (H200 hours, peak GiB).
# > ⚠️ Not yet in repo: `scripts/figures/compute_summary.py`.
# > Expected interface: walk `runs/`, sum wall-time + peak memory per run,
# > emit a LaTeX table.
python scripts/figures/compute_summary.py \
  --runs runs/ \
  --out tables/compute.tex
```

---

## Compute budget

| Phase | Runs | Notes |
|---|---:|---|
| Phase 0 | 1 | single Qwen, seed 0 |
| Phase 1 | 5 | Qwen, headline |
| Phase 2 | 60 | Qwen ablations: 10 + 15 + 10 + 5 cells × 5 seeds, minus shared ref |
| Phase 3 | 20 | LLaVA: 5 headline + 15 loss-ablation |
| Phase 4 | 20 | Llama: 5 headline + 15 loss-ablation |
| **Total** | **~105** | plus ~210 `art-eval-robust` invocations |

At ~8 h / run on a single H200, sequential = ~840 h. **Decision required
before Phase 2:** multi-GPU, reduced epoch budget, or 3-seed ablation cells
(keep 5 only for headline).

---

## Out of scope (cite as limitations)

- Defense-aware adaptive attackers
- Non-PGD attack families (AutoAttack, BPDA, etc.)
- Tasks beyond ProstateX medical-imaging VLM agent setup
