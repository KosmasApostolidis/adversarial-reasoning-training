# Experiment Runs — Publication Readiness

Runnable command sheet for the publication-readiness program described in
`.claude/plans/what-does-the-vectorized-codd.md`. **Do not auto-execute** —
each phase below is launched manually by the user.

Target venue: top-tier conference (NeurIPS / ICLR / CVPR-tier).
Model scope: `qwen2_5_vl_7b`, `llava_v1_6_mistral_7b`, `internvl2_8b`
(third-model slot; was `llama_3_2_vision_11b` — swapped to the open-access
13B Vicuna LLaVA-Next variant after Meta withheld the gated-repo grant.
Llama remains in `models.yaml` and `model_registry_id()` for users who
later acquire the license).
Seeds per cell: 5 (`SEED ∈ {0,1,2,3,4}`).

---

## Restoration plan (2026-04-28)

State on this branch (`fix/audit-trainer-pipeline-t3-losses`) before launching anything:

| Asset                                            | State                                          |
|--------------------------------------------------|------------------------------------------------|
| `runs/baseline_qwen/records.jsonl`               | 3.8 MB — intact                                |
| `runs/baseline_llava/records.jsonl`              | 8.4 MB — intact                                |
| `runs/baseline_internvl2/records.jsonl`          | **0 bytes** — poisoned stub (R1 below)         |
| `runs/{qwen,llava,internvl2}_main_seed{0..4}/`   | dirs exist, **0 files** each                   |
| `runs/adv1_{qwen,llava,internvl2}/ckpt/hf_dir/`  | **all missing** (Phase 3 of the standard sheet)|
| `results/qwen_main/aggregate.json`               | 1.9 KB — stale, regenerate after Phase 4       |

The `aggregate` phase pre-check (`scripts/run_pipeline.sh:183-191`, wired in
`run_one_model` around line 407) now reports any
model whose seed dirs are entirely empty as `aggregate/<alias> (training-not-run)`
in the pipeline summary, instead of the misleading `ERROR: missing seed dirs / rc=1`.
A clean run from the current state should be:

```bash
# Optional sanity dry-run — confirms WARN: skipping aggregate/<alias>
# — no training data lines for llava and internvl2.
bash scripts/run_pipeline.sh --models qwen,llava,internvl2 --phases aggregate
```

Required restoration steps, in order:

| Step | What                                                | Section                          | Wall-time est. |
|------|-----------------------------------------------------|----------------------------------|----------------|
| R1   | Re-baseline `internvl2` (regenerate 0-byte records) | "Phase 1 — Baseline records"     | ~2 h H200      |
| R2   | T0 + T1 gates per model (skip already-green ones)   | CLI Reference / Phase 1          | ~2 h × 3       |
| R3   | Adv-FT × 3 models × 5 seeds                         | Phase 2 / Phase 3                | ~6–8 h × 15    |
| R4   | HF export of seed-0 ckpts (one per model)           | Defended-model hf_dir conversion | ~5 min × 3     |
| R5   | Re-run pipeline end-to-end                          | "Pipeline driver"                | ~3 h × 3       |

After R5, expect:
- `results/qwen_main/aggregate.json` regenerated (5 seeds)
- `results/llava_main/aggregate.json` new
- `results/internvl2_main/aggregate.json` new
- `figures/fig_headline_3model.png` rendered with 3 model rows
- pipeline summary clean (no `FAIL:` block)

Two open issues to surface before the publication cut:

1. **Defended-attack seed degeneracy.** `adversarial_reasoning.runner` does not accept
   `--seed`; each `defended_<alias>.yaml` hardcodes `seeds: [0]`. The five "per-seed"
   defended records are degenerate copies. Tracked in
   `.claude/plans/idempotent-snuggling-quokka.md` Blocker section. Needs decision
   before treating the 5-seed defended variance as real.
2. **0-byte records on model-load failure.** `runner.py:378` opens `records.jsonl`
   *before* `load_hf_vlm`, leaving a 0-byte stub on any load failure. Cause of the
   `baseline_internvl2/records.jsonl` corruption above. Cross-repo fix; tracked in
   `.claude/plans/idempotent-snuggling-quokka.md` Fix 1.

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

**Llama gated-access prerequisite.** `meta-llama/Llama-3.2-11B-Vision-Instruct`
is a gated repo; **any** Phase 4 invocation (T0/T1/baseline/adv-FT/T3) fails
with `Cannot access gated repo for url …` until the running HF account has
the Meta license grant *and* a token with that grant in cache:

```bash
# 1. Visit the model page and request access (Meta gates this manually).
#    https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
# 2. Wait for the approval email (typically minutes-to-hours).
# 3. Login with a token that has read access to gated repos.
huggingface-cli login                 # paste a token from huggingface.co/settings/tokens
# 4. Verify the token resolves the gated config.json:
python -c "from huggingface_hub import HfApi; \
  print(HfApi().model_info('meta-llama/Llama-3.2-11B-Vision-Instruct').siblings[0])"
```

Until step 4 returns without HTTP 401/403, **skip Phase 4 entirely** — the
pipeline driver (`scripts/run_pipeline.sh --models qwen,llava`) is the
intended workaround.

**Already-trained ckpts are auto-skipped.** `scripts/run_pipeline.sh
--skip-existing` reads `runs/<model>_main_seed<N>/ckpt/index.json`
(`latest_path` / `best_path`) before re-launching `art-train`. A 0-byte
records.jsonl from a crashed prior run is treated as missing and forces a
regen — the size check uses `-s`, not `-e`.

**Defended-model hf_dir conversion (required before T3).** Phase T3 robust-eval
loads a defended model via the `defended_<model>_*` entry in
`../adversarial-reasoning-attacks/configs/models.yaml`, which points at a
self-contained HF dir at `runs/adv1_<alias>/ckpt/hf_dir`. After
`art-train` writes the seed ckpts, materialise that dir via the two-step
re-save:

```bash
# 1. Strip optimizer + EMA state; emit a weights-only .pt blob.
python scripts/resave_ckpt_weights_only.py \
  --src runs/qwen_main_seed0/ckpt/step0000030-ep05-*.pt \
  --dst runs/adv1_qwen/ckpt/weights_only.pt

# 2. Materialise the HF dir referenced by defended_qwen2_5_vl_7b.hf_id.
python scripts/ckpt_to_hf_dir.py \
  --base   Qwen/Qwen2.5-VL-7B-Instruct \
  --ckpt   runs/adv1_qwen/ckpt/weights_only.pt \
  --out-dir runs/adv1_qwen/ckpt/hf_dir

# Repeat with --base llava-hf/llava-v1.6-mistral-7b-hf and runs/adv1_llava/...
# Repeat with --base meta-llama/Llama-3.2-11B-Vision-Instruct and runs/adv1_llama/...
```

Until `runs/adv1_<alias>/ckpt/hf_dir/` exists, `defended_<alias>_*` entries
in `models.yaml` will fail to load with `HFValidationError` and the T3 phase
will WARN-skip for that model.

---

## Step 0 — Gold Trajectories (one-time)

Materialize rule-based gold ReAct trajectories from ProstateX metadata +
the 50-case expert-reviewed probe. Re-run only when `configs/gold.yaml`
or the upstream metadata changes.

```bash
art-make-gold \
  --config configs/gold.yaml \
  --data   configs/data.yaml \
  --split  train
# add --n N to cap samples for a quick smoke; add --overwrite to refresh cache.
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

Every command below lists **all** flags so phase commands can be copy-
pasted unedited. Optional flags retain their CLI defaults when given
explicitly — this keeps reproducibility intact.

`art-train` (`src/adversarial_reasoning_training/cli/train.py:45`):

```
art-train \
  --config   configs/training.yaml \
  --defenses configs/defenses.yaml \
  --data     configs/data.yaml \
  --gold     configs/gold.yaml \
  --full-ft  configs/full_ft.yaml \
  --model    <model_family_id> \
  --run-dir  runs/<id> \
  --device   cuda \
  --seed     0 \
  --models-yaml ../adversarial-reasoning-attacks/configs/models.yaml
```

`art-eval-robust` (`src/adversarial_reasoning_training/cli/eval_robust.py:34`):

```
art-eval-robust \
  --baseline-records       <records.jsonl> \
  --defended-records       <records.jsonl> \
  --out-dir                <dir> \
  --alpha                  0.05 \
  --min-traj-edit-delta    0.10 \
  --min-significant-metrics 3
```

`art-make-gold` (`src/adversarial_reasoning_training/cli/make_gold.py:30`):

```
art-make-gold \
  --config configs/gold.yaml \
  --data   configs/data.yaml \
  --split  train             # train | dev | test
  # --n N                    # optional sample cap; omit for full split
  # --overwrite               # destructive; only when refreshing cache
```

Per-sample `records.jsonl` from the attacks-repo runner
(`adversarial-reasoning-attacks/src/adversarial_reasoning/runner.py`):

```
python -m adversarial_reasoning.runner \
  --config         configs/experiments/<exp>.yaml \
  --attacks-config configs/attacks.yaml \
  --split          dev \
  --max-steps      8 \
  --pgd-steps      20 \
  --target-tool    escalate_to_specialist \
  --target-step-k  0 \
  --out            <records.jsonl>
```

Model registry IDs (`../adversarial-reasoning-attacks/configs/models.yaml`):
`qwen2_5_vl_7b`, `llava_v1_6_mistral_7b`, `llama_3_2_vision_11b`,
`defended_qwen2_5_vl_7b`.

Gates as standalone modules:

```
# T0: env + 1 fwd/bwd smoke
python -m adversarial_reasoning_training.gates.T0_env \
  --model    <id> \
  --defenses configs/defenses.yaml \
  --data     configs/data.yaml \
  --gold     configs/gold.yaml \
  --full-ft  configs/full_ft.yaml \
  --device   cuda \
  --epsilon  0.01568627  \
  --pgd-steps 3 \
  --out      runs/<id>/gates/T0.json

# T1: clean-only FT convergence
python -m adversarial_reasoning_training.gates.T1_clean \
  --model    <id> \
  --training configs/training.yaml \
  --data     configs/data.yaml \
  --gold     configs/gold.yaml \
  --full-ft  configs/full_ft.yaml \
  --device   cuda \
  --max-steps         200 \
  --grad-accum        8 \
  --tool-name-acc-min 0.85 \
  --answer-em-min     0.70 \
  --out      runs/<id>/gates/T1.json

# T2: no-collapse (requires T1 result + adv-FT checkpoint)
python -m adversarial_reasoning_training.gates.T2_no_collapse \
  --model        <id> \
  --ckpt         runs/<id>/ckpt/best.pt \
  --t1-result    runs/<id>/gates/T1.json \
  --data         configs/data.yaml \
  --gold         configs/gold.yaml \
  --device       cuda \
  --tolerance-pp 3.0 \
  --out          runs/<id>/gates/T2.json
```

T3 runs via `art-eval-robust` above (see `gates/T3_robust.py`).

---

## Phase 0 — Pipeline unblock (single seed, single model)

Goal: one Qwen2.5-VL-7B run with all four gates green end-to-end.

```bash
cd /home/medadmin/kosmasapostolidis/adversarial-reasoning-training

# 0a. Re-run T0 to confirm cosmetic NaN fix from cb575f8.
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
# > ⚠️ Not yet in repo: `../adversarial-reasoning-attacks/configs/experiments/{baseline_qwen,defended_qwen}.yaml`.
# > Author both before this step. `defended_qwen.yaml` should point its
# > checkpoint at `runs/qwen_full_seed0/ckpt/best.pt`.
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

# 0f. T3 robust-eval gate (Wilcoxon paired + BH-FDR).
art-eval-robust \
  --baseline-records         runs/baseline_qwen/records.jsonl \
  --defended-records         runs/qwen_full_seed0/records.jsonl \
  --out-dir                  runs/qwen_full_seed0/gates/ \
  --alpha                    0.05 \
  --min-traj-edit-delta      0.10 \
  --min-significant-metrics  3
```

**Exit criterion:** `runs/qwen_full_seed0/gates/{T0,T1,T2,T3}.json` all PASS;
`fig05_robust_comparison_*.png` rendered.

---

## Phase 1 — Qwen 5-seed headline (Qwen only)

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
# > ⚠️ Not yet in repo: `scripts/figures/aggregate_seeds.py`. Author with
# > interface `--inputs <T3.json>... --out <aggregate.json>` collecting
# > per-metric mean ± stderr + Wilcoxon survivors across seeds.
python scripts/figures/aggregate_seeds.py \
  --inputs runs/qwen_main_seed{0,1,2,3,4}/gates/T3.json \
  --out    results/qwen_main/aggregate.json

python scripts/figures/make_figures.py \
  --aggregate results/qwen_main/aggregate.json \
  --out       figures/fig05_qwen_main_5seeds.png
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

### 2b. Trajectory-KL weight β sweep

Apply same-step-count ceiling (cf. memory `feedback_t2_gate_budget_parity.md`).

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

### 2c. Freeze strategy

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

### 2d. ε schedule

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

# Plus baseline_llava records + per-seed defended records (same shape as Phase 0e
# and Phase 1 runner block; swap --config configs/experiments/{baseline,defended}_llava.yaml).
```

---

## Phase 4 — InternVL2-8B replication

Same shape as Phase 3, swap `--model internvl2_8b`. Expect comparable
wall-time per run to the 7B llava (8B InternLM-2 LM backbone +
InternViT-300M vision encoder). Document any
bf16-only / batch-size / tile-budget (`max_tiles`) deviations in the
paper appendix.

**Why this model in the third slot.**
`meta-llama/Llama-3.2-11B-Vision-Instruct` is a manually-gated repo and
the running HF account did not get the Meta license grant.
`llava_v1_6_vicuna_13b` shares the `llava_next` family wrapper with the
existing 7B `llava_v1_6_mistral_7b` — only the LM backbone differs, so
it gives no architectural diversity for cross-family robustness claims.
`OpenGVLab/InternVL2-8B` is open-access, ungated, and uses both a
distinct LM backbone (InternLM-2-7B) and a distinct vision encoder
(InternViT-300M) vs the two incumbents — true cross-family signal at
comparable parameter count. Lives behind a new `internvl2` family
wrapper at
`adversarial-reasoning-attacks/src/adversarial_reasoning/models/internvl2.py`.

The previous third slots are still wired:
- `llama` (alias in `model_registry_id()`) → opt in with the Meta grant.
- `llava13b` (alias) → opt in to reproduce earlier same-family runs.

**Prerequisite (one-time).** First run downloads ~16 GB to
`~/.cache/huggingface`. InternVL2 ships custom modeling code; load via
`AutoModel.from_pretrained(..., trust_remote_code=True)` is wired in
the family wrapper — no env-var or login step needed.

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

# Baseline + defended records (same shape as Phase 0e / Phase 3):
#   --config configs/experiments/{baseline,defended}_internvl2.yaml
```

---

## Phase 5 — Paper artifacts (no training)

```bash
# Headline 3-model figure (default lineup: qwen + llava-7b + internvl2-8b).
# Swap internvl2_main → llama_main if you instead trained the gated Llama variant,
# or → llava13b_main to reproduce the prior same-family LLaVA-Vicuna-13B run.
python scripts/figures/make_figures.py \
  --aggregate results/qwen_main/aggregate.json \
              results/llava_main/aggregate.json \
              results/internvl2_main/aggregate.json \
  --out       figures/fig_headline_3model.png

# Ablation tables (Qwen).
# > ⚠️ Not yet in repo: `scripts/figures/make_ablation_tables.py`.
# > Expected interface: `--inputs <aggregate.json>... --out <*.tex>`.
python scripts/figures/make_ablation_tables.py \
  --inputs  results/qwen_abl_loss/aggregate.json \
            results/qwen_abl_beta/aggregate.json \
            results/qwen_abl_freeze/aggregate.json \
            results/qwen_abl_eps/aggregate.json \
  --out     tables/qwen_ablations.tex

# Compute transparency report (H200 hours, peak GiB).
# > ⚠️ Not yet in repo: `scripts/figures/compute_summary.py`.
# > Expected interface: walk `runs/`, sum wall-time + peak memory per run,
# > emit a LaTeX table.
python scripts/figures/compute_summary.py \
  --runs runs/ \
  --out  tables/compute.tex
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

---

## Phase R — Restoration runs (llava + llava13b)

The last full pipeline run failed with three errors that all trace to two
root causes: (a) the manual `ckpt_to_hf_dir.py` export step was never run
for any of the three models, so `runs/adv1_<alias>/ckpt/hf_dir` is missing
and the defended-attack runner crashes inside transformers; and (b) T0/T1
gates plus adv-FT training were never completed for `llava` or `llava13b`.
This section documents the operational steps to restore both models.
`scripts/run_pipeline.sh` now calls `assert_hf_dir <alias>` before the
defended-attack runner; missing exports surface as a `PIPELINE_FAILURES`
entry instead of an OSError mid-load.

### ⚠ Decision required before running these (per-seed defended evals)

`adversarial_reasoning.runner` does **not** accept `--seed`; each
`configs/experiments/defended_<alias>.yaml` hardcodes `seeds: [0]`; and
`run_pipeline.sh` only overrides `--out`. Therefore launching the per-seed
defended attack five times (seed0…seed4) currently produces five identical
records.jsonl files. The aggregate would average degenerate copies and the
"5-seed defended variance" in the headline figure would be fake.

Pick one before re-running anything below:
- **Option 1:** add a `--seed` CLI flag to `runner.py` (attacks repo) and
  pass `--seed $seed` per loop iteration in the pipeline.
- **Option 2:** replace `seeds: [0]` in each `defended_<alias>.yaml` with
  `[0, 1, 2, 3, 4]` and let the runner iterate; drop the per-seed `--out`
  in favour of subdirs the runner creates.
- **Option 3:** declare the defended eval seed-invariant by design and
  average over data seeds only — but then drop "5-seed defended variance"
  language from the report and tables.

The training-side adv-FT seeds (R3 below) are already real per-seed runs;
this decision only affects the attack-side defended records.

### R1. T0 + T1 gates (llava, llava13b)

```bash
cd /home/medadmin/kosmasapostolidis/adversarial-reasoning-training

# llava (mistral-7b)
python -m adversarial_reasoning_training.gates.T0_env \
  --model      llava_v1_6_mistral_7b \
  --defenses   configs/defenses.yaml \
  --data       configs/data.yaml \
  --gold       configs/gold.yaml \
  --full-ft    configs/full_ft.yaml \
  --device     cuda \
  --epsilon    0.01568627 \
  --pgd-steps  3 \
  --out        runs/t0_llava/T0.json

python -m adversarial_reasoning_training.gates.T1_clean \
  --model              llava_v1_6_mistral_7b \
  --training           configs/training.yaml \
  --data               configs/data.yaml \
  --gold               configs/gold.yaml \
  --full-ft            configs/full_ft.yaml \
  --device             cuda \
  --max-steps          200 \
  --grad-accum         8 \
  --tool-name-acc-min  0.85 \
  --answer-em-min      0.70 \
  --out                runs/t1_llava/gates/T1.json

# llava13b (vicuna-13b)
python -m adversarial_reasoning_training.gates.T0_env \
  --model      llava_v1_6_vicuna_13b \
  --defenses   configs/defenses.yaml \
  --data       configs/data.yaml \
  --gold       configs/gold.yaml \
  --full-ft    configs/full_ft.yaml \
  --device     cuda \
  --epsilon    0.01568627 \
  --pgd-steps  3 \
  --out        runs/t0_llava13b/T0.json

python -m adversarial_reasoning_training.gates.T1_clean \
  --model              llava_v1_6_vicuna_13b \
  --training           configs/training.yaml \
  --data               configs/data.yaml \
  --gold               configs/gold.yaml \
  --full-ft            configs/full_ft.yaml \
  --device             cuda \
  --max-steps          200 \
  --grad-accum         8 \
  --tool-name-acc-min  0.85 \
  --answer-em-min      0.70 \
  --out                runs/t1_llava13b/gates/T1.json
```

### R2. Adv-FT (llava, llava13b — seeds 0..4)

```bash
# llava (mistral-7b) — repeat for SEED in 0 1 2 3 4
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
    --seed         "${SEED}" \
    --models-yaml  ../adversarial-reasoning-attacks/configs/models.yaml
done

# llava13b (vicuna-13b) — repeat for SEED in 0 1 2 3 4
for SEED in 0 1 2 3 4; do
  art-train \
    --config       configs/training.yaml \
    --defenses     configs/defenses.yaml \
    --data         configs/data.yaml \
    --gold         configs/gold.yaml \
    --full-ft      configs/full_ft.yaml \
    --model        llava_v1_6_vicuna_13b \
    --run-dir      runs/llava13b_main_seed${SEED} \
    --device       cuda \
    --seed         "${SEED}" \
    --models-yaml  ../adversarial-reasoning-attacks/configs/models.yaml
done
```

### R3. Export `adv1_<alias>/ckpt/hf_dir` (all three models)

`assert_hf_dir` in `scripts/run_pipeline.sh` checks for
`config.json` + `preprocessor_config.json` under each path. The defended
runner cannot proceed without these. Use the seed-0 checkpoint as the
canonical export.

```bash
cd /home/medadmin/kosmasapostolidis/adversarial-reasoning-training

# qwen
python scripts/ckpt_to_hf_dir.py \
  --src runs/qwen_main_seed0/ckpt \
  --dst runs/adv1_qwen/ckpt/hf_dir

# llava
python scripts/ckpt_to_hf_dir.py \
  --src runs/llava_main_seed0/ckpt \
  --dst runs/adv1_llava/ckpt/hf_dir

# llava13b
python scripts/ckpt_to_hf_dir.py \
  --src runs/llava13b_main_seed0/ckpt \
  --dst runs/adv1_llava13b/ckpt/hf_dir
```

After export, rerun `scripts/run_pipeline.sh` end-to-end. The seed-degeneracy
decision above must be resolved before treating the resulting "5-seed
defended" rows as real.
