# Gold trajectory cache

Content-addressed cache of oracle-generated reasoning trajectories produced by `scripts/make_gold_trajectories.py`.

## Layout

```
data/gold/
├── README.md                       # this file (tracked)
├── expert_probe.jsonl.example      # tracked example for the expert-probe schema
└── <sha256>.json                   # generated cache entries (gitignored)
```

The `<sha256>` filename is computed from `(oracle_version, prompt, image)` so re-running the generator over the same inputs is idempotent. Cache lookups go through `adversarial_reasoning_training.data.gold.gold_exists` / `save_gold`.

## Regeneration

```bash
python scripts/make_gold_trajectories.py \
    --config configs/gold.yaml \
    --data configs/data.yaml \
    --split train
```

## Why these files are gitignored

The generated cache scales with dataset size (tens of thousands of small JSONs). The `.gitignore` pattern `data/gold/*.json` keeps the working tree small while preserving `expert_probe.jsonl.example` via an explicit allow-list (`!data/gold/expert_probe.jsonl.example`). Any new tracked artifact in this directory must add its own `!` allow-list rule.
