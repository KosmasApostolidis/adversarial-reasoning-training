# adversarial-reasoning-training — Project Report

## 1. What this project is

`adversarial-reasoning-training` is the **defense-side** half of a
two-repo research stack on adversarial robustness of medical-imaging
**Vision-Language Model (VLM) agents**. Its sibling,
[`adversarial-reasoning-attacks`](../adversarial-reasoning-attacks),
provides the *measurement* tooling — VLM loaders, the ReAct agent,
the PGD attacker, and the trajectory metrics. This repository adds
the *training* loop that hardens the same models against those
attacks.

Concretely, the goal is to fine-tune two open-weight medical VLM
agents — **Qwen2.5-VL-7B** and **LLaVA-v1.6-Mistral-7B** — so that
their full ReAct reasoning trajectory (tool selection → tool
arguments → intermediate evidence → final diagnosis) remains correct
under **white-box L∞ PGD pixel perturbations** on prostate-MRI tasks
drawn from the BHI ProstateX dataset.

The core hypothesis is that defending only the final answer is not
enough for an *agentic* setting: an attacker can flip an early tool
call (for example, mis-routing a T2-weighted lookup) and propagate
the error all the way to the diagnosis without ever changing the
final-answer logits directly. The training loop therefore optimises
both a task-level cross-entropy *and* a trajectory-consistency loss,
and the evaluation gates the resulting model on per-step trajectory
metrics, not just answer accuracy.

## 2. System architecture

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

Key design decisions:

- **Reuse, don't re-implement.** The model wrappers
  (`VLMBase`, `QwenVL`, `LlavaNext`), the `PGDAttack`, the ReAct
  `Trajectory`/`ToolCall` primitives, the `load_task_sample` data
  loader, the preprocessing-transfer guards, and the metrics suite
  are **imported directly from the attacks repo**. Training depends
  on attacks; attacks does not depend on training. This keeps the
  measurement contract identical between the two halves.
- **Single-GPU, micro-batch 1 + gradient accumulation.** Memory
  budget for full fine-tuning of a 7B-param VLM on H200-class GPUs
  with `grad_accum=8` (effective batch 8). This is what makes
  unfrozen ViT + projector + LM fine-tuning feasible without LoRA.
- **Teacher-forced trajectory linearization** is load-bearing.
  Rather than rolling out the ReAct chain step-by-step (which would
  require multiple model calls per training example), the entire
  reasoning trace is linearized into a single token sequence with
  *segment masks* (`task_mask`, `traj_mask`) over the spans that
  correspond to tool names, tool args, thoughts, and the final
  answer. One forward pass, one backward pass, full chain coverage.
- **Phase gates instead of monolithic CI.** Training is gated by an
  ordered series of cheap, ratchet-style sanity checks
  (T0 → T1 → T2 → T3) that stop expensive runs early when something
  upstream is broken.

## 3. Training loop (`trainer/adv_trainer.AdvTrainer`)

Each *outer* step performs:

1. **Inner PGD.** A wrapper around `attacks.pgd.PGDAttack` crafts
   `x_adv` that maximises the teacher-forced cross-entropy on the
   gold trajectory. ε is resolved per epoch via
   `attacks.inner_pgd.epsilon_for_epoch` from a configurable
   `eps_schedule` (the trainer falls back to `default_epsilon`
   when no schedule is wired). The schedule defined in
   `configs/defenses.yaml` is `2/255` for epochs 1–2, `4/255` for
   epoch 3, `8/255` for epochs 4–5; smoke runs use a constant
   `2/255` (no schedule).
2. **Two forward passes.** One on the clean image, one on `x_adv`,
   both over the full teacher-forced sequence.
3. **Loss.** A configurable selector dispatches to one of:
   - **TRADES**:  `L = task_ce(clean) + β · KL(p_clean ‖ p_adv)`
   - **PGD-AT**:  `L = task_ce(adv)`
   - **OAAT**:    `L = α · task_ce(clean) + (1−α) · task_ce(adv)`
   plus an optional `traj_kl` term that aligns trajectory-segment
   distributions step-by-step.
4. **Backprop, accumulate, step.** AdamW (8-bit via bitsandbytes by
   default) with cosine LR schedule, per-component learning rates
   (`lm`, `projector`, `vit`), `grad_clip_norm=1.0`, bf16 AMP,
   and gradient checkpointing.
5. **Periodic checkpoint + dev eval.** A `CheckpointRegistry` keeps
   `best` (selected by `tool_name_acc`) and `latest`. Final-save
   has a knob `final_save_include_optimizer` to drop the optimizer
   state when disk is tight (e.g. smoke runs).

## 4. Data pipeline

- **Source:** BHI ProstateX volumes pre-organized into 3 folds
  (`fold_1` / `fold_2` / `fold_3`), wired by the
  `bhi_split_to_fold` mapping in the attacks-repo
  `configs/tasks.yaml`: `fold_1 → train`, `fold_2 → dev`,
  `fold_3 → test`. `configs/data.yaml` documents the train fold
  size (55 samples for `fold_1`); the dev/test fold sizes are
  whatever the corresponding fold `.npy` arrays actually contain
  on disk (resolved at load time by the attacks-repo
  `load_task_sample` loader).
- **Synthetic mode** (`synthetic: true`) drives sample iteration off
  the attacks-repo synthetic generator, so smoke runs require no
  real DICOMs. Smoke uses synthetic; T1 onward uses real BHI volumes.
- **Gold trajectories** are produced offline by
  `scripts/make_gold_trajectories.py` from a rule-based oracle that
  consumes ProstateX metadata, augmented with a 50-case
  expert-reviewed probe set. Each cached trajectory has the schema
  `{tool_calls, final_answer, reasoning_trace}`; the runtime dataset
  wrapper writes back any missing trajectories so the cache fills
  in-place.

## 5. Phase gates

Each gate is a self-contained CLI that emits a JSON verdict
(`gates/T*.json`) and an exit code; downstream gates consume the
upstream artifact and refuse to run on stale results.

| Gate | Module | Purpose | Pass criteria |
|------|--------|---------|---------------|
| **T0** | `gates.T0_env` | Environment + memory smoke. Single fwd/bwd on a synthetic batch. | Finite `loss_total`, all three grad-norm components present, `peak_memory_gb` under the configured ceiling, `duration_s` under cap. |
| **T1** | `gates.T1_clean` | Clean fine-tune sanity (no adversary). Up to 200 steps; teacher-forced eval over a held-out probe split. | `tool_name_acc ≥ 0.85`, `answer_em ≥ 0.7`, `train_loss_final` finite. |
| **T2** | `gates.T2_no_collapse` | Verifies adversarial training does not collapse trajectory diversity. | Per-segment entropy and tool-distribution checks. |
| **T3** | `gates.T3_robust` | Robust eval. Bridges to the attacks-repo robust-eval runner with BH-FDR over per-task PGD attack budgets. | Defended model strictly improves over undefended baseline at controlled FDR. |

## 6. Current results (smoke run, 2026-04-25)

The smoke configuration is the cheapest end-to-end exercise of the
loop: 5 outer steps, micro-batch 1, synthetic data, ε=0.0078,
TRADES with β=6.0, full ViT+projector+LM unfrozen.

| step | loss_total | loss_task | loss_kl | grad_norm | attack_loss_final |
|-----:|-----------:|----------:|--------:|----------:|------------------:|
| 1 | 4.344 | 4.263 | 0.0135 | 106.5 | −2.86 |
| 2 | 4.333 | 4.268 | 0.0108 |  93.0 | −2.88 |
| 3 | 3.511 | 3.452 | 0.0099 |  66.0 | −2.55 |
| 4 | 3.127 | 3.075 | 0.0086 |  40.0 | −2.36 |
| 5 | 2.773 | 2.737 | 0.0059 |  42.5 | −2.17 |

- `fit_done` after 5/5 outer steps, wall = 27.7 s.
- Loss drops monotonically from **4.34 → 2.77** over 5 steps; KL
  remains tiny but non-zero (β=6 keeps it weighted but not
  dominant).
- Adversary stays effective throughout (`attack_loss_final` < 0
  every step), confirming the inner loop is doing real work.
- Peak GPU: `peak_allocated ≈ 81.9 GiB`, `peak_reserved ≈ 88.7 GiB`,
  well under the 120 GiB H200 ceiling.
- Single weights-only checkpoint saved at step 5 (`include_optimizer=false`
  for smoke to avoid the 100% disk regression).

**Gate status:**

| Gate | Verdict | Headline |
|------|---------|----------|
| T0   | **PASS** | `loss_total = 4.327`, peak 44.4 / 120 GiB, 3.2 s |
| T1   | **PASS** | `tool_name_acc = 1.00`, `answer_em = 1.00`, `train_loss_final = 1.4 × 10⁻⁴` after 200 steps (10.3 min) |
| T2   | not yet exercised | — |
| T3   | not yet exercised | — |

T1 reaching 1.00/1.00 with `train_loss ≈ 0` is consistent with
memorization on a 55-sample fold and is exactly what the gate is
designed to detect — it answers the question *"can the loop fit?"*
not *"does the model generalise?"*. The T2/T3 gates are the ones
that interrogate generalisation and robustness.

A known artifact: T0's `loss_clean` field reports `NaN` because the
component-key extraction looked up `"task"` while the loss dict
exposes `"loss_task"`. The fix is committed (`cb575f8`); the gate
itself still passes because it pivots on `loss_total`, not
`loss_clean`. Worth re-running to clean up the JSON.

Figures rendered from these artifacts live in `figures/`:

- `fig01_smoke_loss.png` — loss curves
- `fig02_smoke_grad_mem.png` — grad-norm + GPU memory headroom
- `fig03_smoke_attack.png` — APGD inner-loop dynamics
- `fig04_gates_summary.png` — T0 grad-norm breakdown + T1 thresholds

## 7. What's next

In rough priority order:

1. **Re-run T0** to clear the cosmetic `loss_clean = NaN` after the
   `loss_task` key fix lands.
2. **Stand up T2** (`no_collapse`) so trajectory diversity is
   monitored before any long adversarial training run is committed
   to disk.
3. **First real adversarial training run** on Qwen2.5-VL-7B with the
   full ε curriculum from `configs/defenses.yaml` (wired through
   `configs/training.yaml` via the trainer's `eps_schedule`
   field). Target: complete one epoch and verify dev
   `tool_name_acc` stays within ≈ 5 pp of the T1 clean-FT
   baseline.
4. **Wire T3** to the attacks-repo robust-eval runner and produce
   the first defended-vs-undefended comparison plot (BH-FDR over
   PGD budgets).
5. **Repeat the pipeline for LLaVA-v1.6-Mistral-7B** once Qwen
   green-lights all four gates.

## 8. Status

Alpha. T0 + T1 green on synthetic / smoke-scale data; the
adversarial outer loop runs end-to-end without disk or memory
regressions; T2 / T3 are scaffolded but not yet exercised. The
training side is now ready for its first non-smoke run.
