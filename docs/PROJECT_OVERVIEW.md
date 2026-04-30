# Project Overview — Adversarial Reasoning Training for Medical-Imaging VLM Agents

> **Companion docs.** This file is the one-stop explainer. For internal
> architecture details see `[PROJECT_REPORT.md](./PROJECT_REPORT.md)`. For the
> canonical, copy-paste-ready experimental runbook see
> `[EXPERIMENT_RUNS.md](./EXPERIMENT_RUNS.md)`. The sibling repo
> `[adversarial-reasoning-attacks](../../adversarial-reasoning-attacks)`
> hosts the *measurement* code (VLM loaders, the ReAct agent, the PGD
> attacker, the trajectory metrics). This repo hosts the *training* code
> that hardens the same models.

---

## TL;DR

This project is the **defense-side** half of a two-repo research stack
on adversarial robustness of medical-imaging Vision-Language Model (VLM)
agents. It fine-tunes two open-weight medical VLM agents —
**Qwen2.5-VL-7B** and **LLaVA-v1.6-Mistral-7B** — so that their full
ReAct reasoning trajectory (tool selection → tool arguments →
intermediate evidence → final diagnosis) remains correct under
**white-box L∞ PGD pixel perturbations** on prostate-MRI tasks drawn
from the **BHI ProstateX** dataset. The contribution is a
**trajectory-aware adversarial fine-tuning loop** with a teacher-forced
trajectory linearization and a four-gate validation pipeline (T0 env,
T1 clean-FT, T2 no-collapse, T3 robust-eval with Wilcoxon + BH-FDR).

---

## 1. The problem — VLMs are vulnerable to adversarial attacks

Vision-Language Models inherit the classic adversarial fragility of
their CLIP-style vision encoders: a few pixels' worth of L∞-bounded
perturbation, invisible to humans, can flip captions, VQA answers, OCR
output, and — most relevant here — **tool selection in agentic
pipelines**. The same families of attacks that broke ImageNet
classifiers a decade ago (FGSM, PGD, C&W) still work against modern
multi-modal foundation models when the attacker has white-box access to
the image-conditioned forward pass.

The agentic-VLM setting makes the problem strictly harder. When a VLM
is wired into a **ReAct loop** (think → act → observe → repeat) it
emits a *trajectory* of intermediate decisions, not just a single
answer. **Defending only the final answer is not enough**: an attacker
can flip an *early* tool call — for example, mis-routing a T2-weighted
lookup to a different MRI sequence — and propagate the error all the
way to the final diagnosis without ever changing the final-answer
logits directly. Standard adversarial training, which optimises a loss
on the answer token only, fails to notice or defend against these
trajectory-level manipulations.

In medical imaging this becomes a **clinical-safety** problem rather
than a benchmark curiosity: a tool-call flip means the agent retrieves
the wrong sequence, queries the wrong segmentation model, or escalates
to the wrong specialist. The downstream diagnosis can change without
any visible perturbation in the image and without any anomaly in the
final-answer probability distribution. This is the gap this project
sets out to close.

**Threat model in concrete terms** (from
`[EXPERIMENT_RUNS.md](./EXPERIMENT_RUNS.md)`):

- **Attacker:** white-box, full gradient access through the VLM and
the ReAct loop.
- **Perturbation:** L∞ PGD on the input image, ε ∈ {2, 4, 8}/255.
- **Target:** flip an intermediate tool call (e.g.,
`escalate_to_specialist`) at a chosen step index `target-step-k`,
while the final-answer logits may or may not change.
- **Out of scope (acknowledged limitations):** defense-aware adaptive
attackers, non-PGD attack families (AutoAttack, BPDA, etc.), and
tasks beyond ProstateX.

---

## 2. Recent literature — what others do to make VLMs robust

The defense literature this work builds on falls into three layers.
The first two are widely-deployed; the third is essentially open.

### 2.1 Adversarial-training fundamentals (vision)

These are the classical defenses and the ones this repo implements and
ablates against:

- **PGD-AT** (Madry et al., *Towards Deep Learning Models Resistant to
Adversarial Attacks*, ICLR 2018) — minimise the worst-case loss
inside an ε-ball using projected gradient descent inner loop. The
baseline against which everything else is measured.
- **TRADES** (Zhang et al., *Theoretically Principled Trade-off between
Robustness and Accuracy*, ICML 2019) — decompose the loss into a
natural cross-entropy term and a robustness term that is the KL
divergence between clean and adversarial output distributions:
`L = CE(clean) + β · KL(p_clean ‖ p_adv)`. The β knob exposes the
robustness/accuracy frontier explicitly.
- **OAAT** (Sehwag et al., *Robust Learning Meets Generative Models*,
follow-up work on outer-adversarial training) — convex combination
of clean and adversarial cross-entropy:
`L = α · CE(clean) + (1 − α) · CE(adv)`. Simpler than TRADES,
often a strong baseline.
- **MART** (Wang et al., *Improving Adversarial Robustness Requires
Revisiting Misclassified Examples*, ICLR 2020) — re-weights the
adversarial term by misclassification.
- **AWP** (Wu et al., *Adversarial Weight Perturbation*, NeurIPS 2020)
— flatten the loss landscape in weight space, often stacked on top
of PGD-AT/TRADES.

This repo implements **PGD-AT, TRADES, and OAAT** as configurable
selectors (`configs/training.yaml: defense`). MART and AWP are not
implemented; they appear in the literature review as natural extensions.

### 2.2 VLM-specific robust fine-tuning (vision encoder)

A more recent line of work targets the vision encoder of a VLM
specifically:

- **TeCoA** (Mao et al., *Understanding Zero-Shot Adversarial
Robustness for Large-Scale Models*, ICLR 2023) — text-guided
contrastive adversarial fine-tuning for CLIP-style encoders.
- **FARE** (Schlarmann et al., *Robust CLIP: Unsupervised Adversarial
Fine-Tuning of Vision Embeddings for Robust Large Vision-Language
Models*, ICML 2024) — unsupervised AT on the vision encoder so the
embedding space stays robust without paired text labels.
- **AdvCLIP / RobustCLIP** family — variants that swap the CLIP
vision encoder for an adversarially-fine-tuned drop-in.

These methods harden the vision encoder *in isolation* and rely on the
downstream LM head to inherit the robustness. They do not optimise the
agentic chain end-to-end and they do not target multi-step reasoning.

### 2.3 Agent-level robustness (largely unexplored)

The third layer is the open ground. Existing work on attacking agents
focuses almost entirely on **text-side** vectors:

- **InjecAgent** (Zhan et al., *InjecAgent: Benchmarking Indirect
Prompt Injections in Tool-Integrated LLM Agents*, ACL 2024) —
benchmarks prompt injection delivered through tool outputs.
- **BadAgent** (Yang et al., 2024) — backdoors planted at fine-tuning
time that activate on tool-use prompts.

What is **missing**, and what this project addresses, is the
**visual** attack surface on agentic VLMs: white-box pixel
perturbations that target *intermediate* tool calls in a multi-step
medical-imaging workflow, plus a defense that is aware of the entire
trajectory rather than just the final-answer logits.

---

## 3. What this project does + the research gaps it fills

### 3.1 Defense in one sentence

**Trajectory-aware adversarial fine-tuning** of medical VLM agents:
joint optimisation of task cross-entropy and trajectory-consistency
loss under an ε curriculum of L∞ PGD perturbations, with the full
ReAct chain teacher-forced into a single backward pass.

### 3.2 Method components

1. **Teacher-forced trajectory linearization** — the entire ReAct
  chain (tool name → tool args → thought → final answer) is
   collapsed into one transformer forward pass with structured
   *segment masks* over each role span. This is load-bearing: it lets
   gradients reach every reasoning step instead of only the final
   answer token.
2. **Inner PGD** wrapped around the teacher-forced cross-entropy. The
  inner attacker maximises the gold-trajectory loss; the outer
   optimizer minimises a configurable defense loss on both the clean
   and adversarial branches.
3. **Three loss families**, selectable via `configs/training.yaml`:

  | Family             | Formulation                                       |
  | ------------------ | ------------------------------------------------- |
  | `trades` (default) | `L = task_ce(clean) + β · KL(p_clean ‖ p_adv)`    |
  | `pgd_at`           | `L = task_ce(adv)`                                |
  | `oaat`             | `L = α · task_ce(clean) + (1 − α) · task_ce(adv)` |

   plus an optional `traj_kl` term that aligns trajectory-segment
   distributions step by step.
4. **Full fine-tune** (ViT + projector + LM all unfrozen) with
  bitsandbytes 8-bit AdamW, gradient checkpointing, bf16 AMP,
   per-component learning rates (`lm = 5e-6`, `projector = 1e-5`,
   `vit = 1e-6`), cosine schedule with 3 % warmup, grad-clip 1.0,
   micro-batch 1 with grad-accum 8 (effective batch 8), 50 epochs.
5. **ε curriculum**: `2/255` for epochs 1–2, `4/255` for epoch 3,
  `8/255` for epochs 4+, declared in `configs/defenses.yaml`.
6. **Four-gate validation** before any result is reported:

  | Gate | Module                                       | Verifies                                                                                                                                                             |
  | ---- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | T0   | `gates.T0_env`                               | Env + 1 fwd/bwd: total loss finite, every role-group has non-zero grad, CUDA peak OK.                                                                                |
  | T1   | `gates.T1_clean`                             | Clean-only FT (PGD off, ≤200 steps): `tool_name_acc ≥ 0.85`, `answer_em ≥ 0.70`.                                                                                     |
  | T2   | `gates.T2_no_collapse`                       | Adv-FT clean metrics within `tolerance_pp = 3.0` of T1 ceiling — catches the classic AT failure mode (robust but clean accuracy collapses).                          |
  | T3   | `gates.T3_robust` (a.k.a. `art-eval-robust`) | Per-metric Wilcoxon signed-rank + BH-FDR over `{tool_name_acc, args_iou, answer_em, traj_edit_distance}`. Pass: `traj_edit_delta ≥ 0.10`, ≥ 3/4 metrics significant. |


### 3.3 Research gaps this project fills

1. **Agentic VLM robustness on medical-imaging tasks is essentially
  unstudied.** No prior benchmark or defense targets multi-step
   medical-imaging VLM agents under white-box L∞ PGD.
2. **Trajectory-consistency as a defense axis is novel.** Prior AT
  methods optimise final-answer cross-entropy. This project jointly
   optimises task CE and trajectory-segment alignment, exposing a
   defense surface that final-answer-only training cannot reach.
3. **Cross-architecture replication.** The defense is validated on
  three model families (Qwen2.5-VL-7B, LLaVA-v1.6-Mistral-7B,
   InternVL2-8B) with distinct LM backbones (Qwen2 / Mistral /
   InternLM-2) AND distinct vision encoders (Qwen-ViT / CLIP-ViT-L/14 /
   InternViT-300M), rather than a single architecture.
   Llama-3.2-Vision-11B is also wired (alias `llama`) for users with a
   Meta license grant.
4. **Statistically rigorous evaluation.** All robustness claims pass a
  paired Wilcoxon signed-rank test with Benjamini-Hochberg FDR
   control over four trajectory-aware metrics, not just headline
   numbers.

### 3.4 Out of scope (limitations stated up-front)

- **Defense-aware adaptive attacks.** All attack records use a fixed
PGD attacker; an attacker that knows the defense could in principle
do better.
- **Non-PGD attack families.** AutoAttack, BPDA, and gradient-free
query attacks are not evaluated.
- **Tasks beyond ProstateX.** Generalization to other modalities or
non-medical agentic tasks is left to future work.

---

## 4. Dataset

- **Name:** **BHI ProstateX** (medical prostate-MRI volumes, used here
in the diagnostic-workup task `prostate_mri_workup`).
- **Modality:** prostate MRI volumes. The threat model and the rule-
based oracle reference T2-weighted lookups explicitly; multi-sequence
support flows through the attacks-repo `load_task_sample` loader,
which consumes the on-disk preprocessed bundle.
- **Organisation:** three folds — `fold_1` / `fold_2` / `fold_3` —
mapped to `train` / `dev` / `test` by the `bhi_split_to_fold`
setting in the sibling repo's `configs/tasks.yaml`.
- **Train size:** **55 samples** (`fold_1`), declared in
`configs/data.yaml`. Dev and test fold sizes are resolved at load
time from the corresponding fold `.npy` arrays.
- **Patient count:** the repo reports samples-per-fold rather than
unique patient counts; for the patient-level breakdown see the
attacks-repo `configs/tasks.yaml` and the upstream BHI ProstateX
release. The training pipeline operates at the sample level.
- **Synthetic mode** (`configs/data.yaml: synthetic: true`): drives
sample iteration off the attacks-repo synthetic generator, so smoke
runs require no real DICOMs. T1 onward uses real BHI volumes.
- **Gold trajectories:** produced offline by `art-make-gold` (CLI
shim: `scripts/make_gold_trajectories.py`) from a rule-based oracle
that consumes ProstateX metadata, augmented with a **50-case
expert-reviewed probe set**. Each cached trajectory has the schema
`{tool_calls, final_answer, reasoning_trace}`. The runtime dataset
wrapper writes back any missing trajectories so the cache fills in
place.

---

## 5. The full pipeline + reproducible commands

All commands below are copied verbatim from
`[EXPERIMENT_RUNS.md](./EXPERIMENT_RUNS.md)`, the canonical runbook.
Console scripts are declared in `pyproject.toml` `[project.scripts]`:

- `art-train` → `adversarial_reasoning_training.cli.train:main`
- `art-eval-robust` → `adversarial_reasoning_training.cli.eval_robust:main`
- `art-make-gold` → `adversarial_reasoning_training.cli.make_gold:main`

### 5.0 Prerequisites (one-time)

```bash
# 1. Conda env
conda activate kosmasenv

# 2. Sibling deps first, then this repo, then optional flash-attn for Qwen
pip install -e ../adversarial-reasoning-attacks
pip install -e .[dev]
pip install -e .[flash]                       # optional, Qwen flash-attn

# 3. Verify console scripts
art-train       --help
art-eval-robust --help
art-make-gold   --help
```

Hardware: single H200 (≥ 120 GiB) per run; T0 enforces this ceiling.
Data: ProstateX preprocessed bundle at the path declared in
`configs/data.yaml`. Models: HF cache populated for `qwen2_5_vl_7b`,
`llava_v1_6_mistral_7b`, `internvl2_8b`; registry in
`../adversarial-reasoning-attacks/configs/models.yaml`.
`llama_3_2_vision_11b` and `llava_v1_6_vicuna_13b` are also defined for
opt-in via `--models qwen,llava,llama` or `--models qwen,llava,llava13b`.

### 5.1 Step 0 — Gold trajectories (one-time)

Materialize rule-based gold ReAct trajectories from ProstateX metadata
and the 50-case expert-reviewed probe. Re-run only when
`configs/gold.yaml` or upstream metadata changes.

```bash
art-make-gold \
  --config configs/gold.yaml \
  --data   configs/data.yaml \
  --split  train
# add --n N to cap samples for a quick smoke; add --overwrite to refresh cache.
```

### 5.2 Phase 0 — Pipeline unblock (single seed, single model)

Goal: one Qwen2.5-VL-7B run with all four gates green end-to-end.

```bash
cd /home/medadmin/kosmasapostolidis/adversarial-reasoning-training

# 0a. T0 environment gate.
python -m adversarial_reasoning_training.gates.T0_env \
  --model      qwen2_5_vl_7b \
  --defenses   configs/defenses.yaml \
  --data       configs/data.yaml \
  --gold       configs/gold.yaml \
  --full-ft    configs/full_ft.yaml \
  --device     cuda \
  --epsilon    0.01568627 \
  --pgd-steps  3 \
  --out        runs/t0_recheck/T0.json

# 0b. T1 clean-FT gate (no adversary, ≤200 steps).
python -m adversarial_reasoning_training.gates.T1_clean \
  --model              qwen2_5_vl_7b \
  --training           configs/training.yaml \
  --data               configs/data.yaml \
  --gold               configs/gold.yaml \
  --full-ft            configs/full_ft.yaml \
  --device             cuda \
  --max-steps          200 \
  --grad-accum         8 \
  --tool-name-acc-min  0.85 \
  --answer-em-min      0.70 \
  --out                runs/t1_qwen/gates/T1.json

# 0c. First end-to-end Qwen adv-FT, seed=0, full ε curriculum 2/255 → 8/255.
art-train \
  --config       configs/training.yaml \
  --defenses     configs/defenses.yaml \
  --data         configs/data.yaml \
  --gold         configs/gold.yaml \
  --full-ft      configs/full_ft.yaml \
  --model        qwen2_5_vl_7b \
  --run-dir      runs/qwen_full_seed0 \
  --device       cuda \
  --seed         0 \
  --models-yaml  ../adversarial-reasoning-attacks/configs/models.yaml

# 0d. T2 no-collapse gate against the resulting checkpoint.
python -m adversarial_reasoning_training.gates.T2_no_collapse \
  --model         qwen2_5_vl_7b \
  --ckpt          runs/qwen_full_seed0/ckpt/best.pt \
  --t1-result     runs/t1_qwen/gates/T1.json \
  --data          configs/data.yaml \
  --gold          configs/gold.yaml \
  --device        cuda \
  --tolerance-pp  3.0 \
  --out           runs/qwen_full_seed0/gates/T2.json

# 0e. Generate baseline (undefended) and defended records via attacks-repo runner.
cd ../adversarial-reasoning-attacks
python -m adversarial_reasoning.runner \
  --config         configs/experiments/baseline_qwen.yaml \
  --attacks-config configs/attacks.yaml \
  --split          dev \
  --max-steps      8 \
  --pgd-steps      20 \
  --target-tool    escalate_to_specialist \
  --target-step-k  0 \
  --out            ../adversarial-reasoning-training/runs/baseline_qwen/records.jsonl

python -m adversarial_reasoning.runner \
  --config         configs/experiments/defended_qwen.yaml \
  --attacks-config configs/attacks.yaml \
  --split          dev \
  --max-steps      8 \
  --pgd-steps      20 \
  --target-tool    escalate_to_specialist \
  --target-step-k  0 \
  --out            ../adversarial-reasoning-training/runs/qwen_full_seed0/records.jsonl
cd -

# 0f. T3 robust-eval gate (Wilcoxon + BH-FDR).
art-eval-robust \
  --baseline-records         runs/baseline_qwen/records.jsonl \
  --defended-records         runs/qwen_full_seed0/records.jsonl \
  --out-dir                  runs/qwen_full_seed0/gates/ \
  --alpha                    0.05 \
  --min-traj-edit-delta      0.10 \
  --min-significant-metrics  3
```

> ⚠️ Phase 0e expects
> `../adversarial-reasoning-attacks/configs/experiments/{baseline_qwen,defended_qwen}.yaml`
> to exist. `defended_qwen.yaml` should point its checkpoint at
> `runs/qwen_full_seed0/ckpt/best.pt`.

### 5.3 Phase 1 — Qwen 5-seed headline

```bash
cd /home/medadmin/kosmasapostolidis/adversarial-reasoning-training

for SEED in 0 1 2 3 4; do
  art-train \
    --config       configs/training.yaml \
    --defenses     configs/defenses.yaml \
    --data         configs/data.yaml \
    --gold         configs/gold.yaml \
    --full-ft      configs/full_ft.yaml \
    --model        qwen2_5_vl_7b \
    --run-dir      runs/qwen_main_seed${SEED} \
    --device       cuda \
    --seed         ${SEED} \
    --models-yaml  ../adversarial-reasoning-attacks/configs/models.yaml

  cd ../adversarial-reasoning-attacks
  python -m adversarial_reasoning.runner \
    --config         configs/experiments/defended_qwen.yaml \
    --attacks-config configs/attacks.yaml \
    --split          test \
    --max-steps      8 \
    --pgd-steps      20 \
    --target-tool    escalate_to_specialist \
    --target-step-k  0 \
    --out            ../adversarial-reasoning-training/runs/qwen_main_seed${SEED}/records.jsonl
  cd -

  art-eval-robust \
    --baseline-records         runs/baseline_qwen/records.jsonl \
    --defended-records         runs/qwen_main_seed${SEED}/records.jsonl \
    --out-dir                  runs/qwen_main_seed${SEED}/gates/ \
    --alpha                    0.05 \
    --min-traj-edit-delta      0.10 \
    --min-significant-metrics  3
done

# Aggregate after all 5 seeds finish.
python scripts/figures/aggregate_seeds.py \
  --inputs runs/qwen_main_seed{0,1,2,3,4}/gates/T3.json \
  --out    results/qwen_main/aggregate.json

python scripts/figures/make_figures.py \
  --aggregate results/qwen_main/aggregate.json \
  --out       figures/fig05_qwen_main_5seeds.png
```

**Exit criterion:** Wilcoxon q < 0.05 on ≥ 3 / 4 metrics
(`tool_acc`, `args_iou`, `answer_em`, `traj_edit`);
`aggregate.json` reports `n_seeds = 5`.

### 5.4 Phase 2 — Qwen ablation matrix

Four ablation axes × 5 seeds each. Reference cell is the Phase 1
default (TRADES, β = 6, full unfrozen, ε curriculum).

#### 2a. Loss family (PGD-AT, OAAT vs default TRADES)

```bash
for LOSS in pgd_at oaat; do
  for SEED in 0 1 2 3 4; do
    art-train \
      --config       configs/ablations/loss_${LOSS}.yaml \
      --defenses     configs/defenses.yaml \
      --data         configs/data.yaml \
      --gold         configs/gold.yaml \
      --full-ft      configs/full_ft.yaml \
      --model        qwen2_5_vl_7b \
      --run-dir      runs/qwen_abl_loss_${LOSS}_seed${SEED} \
      --device       cuda \
      --seed         ${SEED} \
      --models-yaml  ../adversarial-reasoning-attacks/configs/models.yaml
  done
done
```

#### 2b. Trajectory-KL weight β sweep (β ∈ {0, 1, 12}; default β = 6)

```bash
for BETA in 0 1 12; do
  for SEED in 0 1 2 3 4; do
    art-train \
      --config       configs/ablations/beta_${BETA}.yaml \
      --defenses     configs/defenses.yaml \
      --data         configs/data.yaml \
      --gold         configs/gold.yaml \
      --full-ft      configs/full_ft.yaml \
      --model        qwen2_5_vl_7b \
      --run-dir      runs/qwen_abl_beta_${BETA}_seed${SEED} \
      --device       cuda \
      --seed         ${SEED} \
      --models-yaml  ../adversarial-reasoning-attacks/configs/models.yaml
  done
done
```

#### 2c. Freeze strategy (`lm_only`, `vit_proj_frozen` vs default full unfreeze)

```bash
for FREEZE in lm_only vit_proj_frozen; do
  for SEED in 0 1 2 3 4; do
    art-train \
      --config       configs/training.yaml \
      --defenses     configs/defenses.yaml \
      --data         configs/data.yaml \
      --gold         configs/gold.yaml \
      --full-ft      configs/ablations/full_ft_${FREEZE}.yaml \
      --model        qwen2_5_vl_7b \
      --run-dir      runs/qwen_abl_freeze_${FREEZE}_seed${SEED} \
      --device       cuda \
      --seed         ${SEED} \
      --models-yaml  ../adversarial-reasoning-attacks/configs/models.yaml
  done
done
```

#### 2d. ε schedule (fixed 8/255 vs default curriculum 2 → 4 → 8/255)

```bash
for SEED in 0 1 2 3 4; do
  art-train \
    --config       configs/training.yaml \
    --defenses     configs/ablations/defenses_eps_fixed_8.yaml \
    --data         configs/data.yaml \
    --gold         configs/gold.yaml \
    --full-ft      configs/full_ft.yaml \
    --model        qwen2_5_vl_7b \
    --run-dir      runs/qwen_abl_eps_fixed_seed${SEED} \
    --device       cuda \
    --seed         ${SEED} \
    --models-yaml  ../adversarial-reasoning-attacks/configs/models.yaml
done
```

After every cell completes its 5 seeds, run the Phase 1 eval +
aggregate snippet, swapping `--inputs` and `--out`. Exit criterion per
axis: `results/qwen_abl_<axis>/aggregate.json` exists with paired
delta vs reference cell + significance markers.

### 5.5 Phase 3 — LLaVA-v1.6-Mistral-7B replication

```bash
# Headline 5-seed sweep.
for SEED in 0 1 2 3 4; do
  art-train \
    --config       configs/training.yaml \
    --defenses     configs/defenses.yaml \
    --data         configs/data.yaml \
    --gold         configs/gold.yaml \
    --full-ft      configs/full_ft.yaml \
    --model        llava_v1_6_mistral_7b \
    --run-dir      runs/llava_main_seed${SEED} \
    --device       cuda \
    --seed         ${SEED} \
    --models-yaml  ../adversarial-reasoning-attacks/configs/models.yaml
done

# Loss-family ablation only (3 cells × 5 seeds).
for LOSS in trades pgd_at oaat; do
  for SEED in 0 1 2 3 4; do
    art-train \
      --config       configs/ablations/loss_${LOSS}.yaml \
      --defenses     configs/defenses.yaml \
      --data         configs/data.yaml \
      --gold         configs/gold.yaml \
      --full-ft      configs/full_ft.yaml \
      --model        llava_v1_6_mistral_7b \
      --run-dir      runs/llava_abl_loss_${LOSS}_seed${SEED} \
      --device       cuda \
      --seed         ${SEED} \
      --models-yaml  ../adversarial-reasoning-attacks/configs/models.yaml
  done
done

# Plus baseline_llava records + per-seed defended records (same shape as
# Phase 0e and Phase 1 runner block; swap --config configs/experiments/{baseline,defended}_llava.yaml).
```

### 5.6 Phase 4 — InternVL2-8B replication

Same shape as Phase 3, swap `--model internvl2_8b`. Expect comparable
wall-time per run to the 7B llava (8B InternLM-2 LM backbone +
InternViT-300M vision encoder). Document any bf16-only / batch-size /
tile-budget (`max_tiles`) deviations in the paper appendix.

```bash
for SEED in 0 1 2 3 4; do
  art-train \
    --config       configs/training.yaml \
    --defenses     configs/defenses.yaml \
    --data         configs/data.yaml \
    --gold         configs/gold.yaml \
    --full-ft      configs/full_ft.yaml \
    --model        internvl2_8b \
    --run-dir      runs/internvl2_main_seed${SEED} \
    --device       cuda \
    --seed         ${SEED} \
    --models-yaml  ../adversarial-reasoning-attacks/configs/models.yaml
done
# Loss-family ablation block identical to Phase 3, --model swapped to internvl2_8b.
```

### 5.7 Phase 5 — Paper artifacts (no training)

```bash
# Headline 3-model figure (default lineup: qwen + llava-7b + internvl2-8b).
python scripts/figures/make_figures.py \
  --aggregate results/qwen_main/aggregate.json \
              results/llava_main/aggregate.json \
              results/internvl2_main/aggregate.json \
  --out       figures/fig_headline_3model.png

# Ablation tables (Qwen).
python scripts/figures/make_ablation_tables.py \
  --inputs  results/qwen_abl_loss/aggregate.json \
            results/qwen_abl_beta/aggregate.json \
            results/qwen_abl_freeze/aggregate.json \
            results/qwen_abl_eps/aggregate.json \
  --out     tables/qwen_ablations.tex

# Compute transparency report (H200 hours, peak GiB).
python scripts/figures/compute_summary.py \
  --runs runs/ \
  --out  tables/compute.tex
```

### 5.8 Run-directory layout

Every `art-train --run-dir runs/<id>` produces:

```
runs/<id>/
├── ckpt/             # CheckpointRegistry: best.pt + step-N.pt (ckpt_keep=2)
├── gates/            # T0.json … T3.json + figXX_*.png
├── records.jsonl     # per-sample defended records (from attacks-repo runner)
└── logs/             # train log + memory probe
```

`runs/baseline_<model>/records.jsonl` (the undefended reference) is
generated **once per model** then reused by every defended seed and
ablation cell — do not regenerate per seed.

### 5.9 Compute budget


| Phase     | Runs     | Notes                                                              |
| --------- | -------- | ------------------------------------------------------------------ |
| 0         | 1        | single Qwen, seed 0                                                |
| 1         | 5        | Qwen, headline                                                     |
| 2         | 60       | Qwen ablations: 10 + 15 + 10 + 5 cells × 5 seeds, minus shared ref |
| 3         | 20       | LLaVA: 5 headline + 15 loss-ablation                               |
| 4         | 20       | Llama: 5 headline + 15 loss-ablation                               |
| **Total** | **~105** | plus ~210 `art-eval-robust` invocations                            |


At ~8 h / run on a single H200, sequential ≈ 840 h.

---

## 6. Source-tree layout

```
src/adversarial_reasoning_training/
├── data/          # ProstateX dataset + collator + gold trajectory loader
├── gold/          # rule-based oracle + templates + expert probe
├── trajectory/    # teacher-forced linearization (load-bearing)
├── losses/        # task_ce, traj_kl, trades, pgd_at, oaat, selector
├── attacks/       # inner PGD wrapper over attacks-repo PGDAttack
├── trainer/       # outer loop, freeze strategy, optimizer, checkpointing
├── gates/         # T0–T3 phase gates
├── eval/          # bridge to attacks-repo robust-eval runner
└── utils/         # seed, hashing, memory probe
```

Reused from the sibling attacks repo (training depends on attacks; the
reverse is not true):

- `adversarial_reasoning.models.{VLMBase, QwenVL, LlavaNext}`
- `adversarial_reasoning.attacks.pgd.PGDAttack`
- `adversarial_reasoning.agents.base.{Trajectory, ToolCall}`
- `adversarial_reasoning.tasks.loader.load_task_sample`
- `adversarial_reasoning.gates.preprocessing_transfer`
- `adversarial_reasoning.metrics.*`

---

## 7. Pointers

- **Architecture deep-dive:** `[PROJECT_REPORT.md](./PROJECT_REPORT.md)`
— system architecture, training-loop internals, data pipeline,
gate-by-gate semantics, current results.
- **Canonical runbook:** `[EXPERIMENT_RUNS.md](./EXPERIMENT_RUNS.md)`
— every command in §5 above, with full CLI references and
publication-readiness notes.
- **Measurement code:** sibling repo
`[adversarial-reasoning-attacks](../../adversarial-reasoning-attacks)`
— VLM loaders, ReAct agent, PGD attacker, trajectory metrics.

