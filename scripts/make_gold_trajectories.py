"""Populate the gold-trajectory cache.

Drives sample iteration via the attacks-repo task loader (real or
synthetic), looks up metadata in an optional CSV (keyed by ``sample_id``),
runs the rule-based oracle, and writes one JSON per sample to the cache
dir using the multi-arg ``save_gold(cache_dir, task_id, sample_id, prompt,
image, oracle_version, trajectory)`` API. Re-runs are idempotent — the
cache key is content-addressed (oracle_version + prompt + image hash).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from adversarial_reasoning_training.data.gold import gold_exists, save_gold
from adversarial_reasoning_training.gold.oracle import (
    OracleConfig,
    generate_trajectory,
    load_metadata_csv,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="configs/gold.yaml")
    parser.add_argument("--data", type=Path, required=True, help="configs/data.yaml")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n", type=int, default=None,
                        help="Cap samples; default = full split")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    gold_cfg = _load_yaml(args.config)
    data_cfg = _load_yaml(args.data)

    cache_dir = Path(gold_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    oracle_cfg = OracleConfig(version=gold_cfg["oracle_version"])

    task_id = data_cfg["task_id"]
    config_path = data_cfg.get(
        "config_path", "../adversarial-reasoning-attacks/configs/tasks.yaml"
    )
    synthetic = bool(data_cfg.get("synthetic", False))

    metadata_csv = data_cfg.get("metadata_csv")
    metadata_lookup: dict[str, dict[str, Any]] = (
        load_metadata_csv(metadata_csv) if metadata_csv else {}
    )

    from adversarial_reasoning.tasks.loader import load_task  # type: ignore

    samples = list(load_task(
        task_id, split=args.split, n=args.n, synthetic=synthetic,
        config_path=config_path,
    ))

    n_total = len(samples)
    n_written = 0
    n_skipped = 0
    for s in samples:
        sid = s.sample_id
        meta = metadata_lookup.get(sid, {})
        if gold_exists(cache_dir, task_id, sid, s.prompt, s.image, oracle_cfg.version) \
                and not args.overwrite:
            n_skipped += 1
            continue

        traj = generate_trajectory(
            task_id=task_id, sample_id=sid, metadata=meta, config=oracle_cfg,
        )
        save_gold(cache_dir, task_id, sid, s.prompt, s.image, oracle_cfg.version, traj)
        n_written += 1

    summary = {
        "total": n_total,
        "written": n_written,
        "skipped": n_skipped,
        "cache_dir": str(cache_dir),
        "oracle_version": oracle_cfg.version,
        "task_id": task_id,
        "split": args.split,
        "synthetic": synthetic,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
