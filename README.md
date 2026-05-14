# adversarial-reasoning-training

Fine-tune medical-imaging VLM agents (Qwen3-VL-8B, LLaVA-OneVision-7B, InternVL3-8B) so their ReAct reasoning trajectories — tool selection, tool args, intermediate evidence, final diagnosis — stay robust under white-box PGD-L∞ pixel perturbations on prostate-MRI tasks.

Sibling project to [adversarial-reasoning-attacks](../adversarial-reasoning-attacks) which provides the measurement tooling (VLM loaders, ReAct agent, PGD attack, trajectory metrics). This repo adds the training loop.

## What's here

- **TRADES / PGD-AT / OAAT** loss variants, selectable via config.
- **Teacher-forced trajectory linearization** — backprop through the whole "ReAct" chain collapses to one transformer forward with structured segment masks.
- **Inner PGD** wrapped around the teacher-forced CE; outer optimizer minimizes task loss + trajectory-consistency loss.
- **Full fine-tune** (ViT + projector + LM all unfrozen) with bnb 8-bit AdamW + gradient checkpointing.
- **ε curriculum** 2/255 → 4/255 → 8/255 across epochs.
- **Gated build**: T0 env, T1 clean-FT, T2 no-collapse, T3 robustness (BH-FDR).
- **Rule-based gold trajectories** from ProstateX metadata + 50-case expert-reviewed probe.

## Install

```bash
conda activate kosmasenv
pip install -e ../adversarial-reasoning-attacks     # sibling dep
pip install -e .[dev]
# optional: flash-attn for Qwen
pip install -e .[flash]
```

## Quickstart

```bash
# 1. Generate gold trajectories from ProstateX metadata
python scripts/make_gold_trajectories.py --config configs/gold.yaml

# 2. Gate T0 — env sanity + mem probe
python -m adversarial_reasoning_training.gates.T0_env --model qwen3_vl_8b

# 3. Full adversarial training on Qwen3
art-train --config configs/training.yaml --model qwen3_vl_8b

# 4. Robust eval vs undefended baseline (uses attacks repo runner)
python scripts/eval_robust.py --ckpt runs/<timestamp>/final/ckpt.pt
```

## Layout

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

## Reuse from attacks repo

- `adversarial_reasoning.models.{VLMBase, QwenVL, LlavaNext}`
- `adversarial_reasoning.attacks.pgd.PGDAttack`
- `adversarial_reasoning.agents.base.{Trajectory, ToolCall}`
- `adversarial_reasoning.tasks.loader.load_task_sample`
- `adversarial_reasoning.gates.preprocessing_transfer`
- `adversarial_reasoning.metrics.*`

## Status

Alpha. Three-model lineup: Qwen3-VL-8B, LLaVA-OneVision-7B, InternVL3-8B. Full pipeline via `scripts/run_pipeline.sh`.

## License

MIT. See `LICENSE`.
