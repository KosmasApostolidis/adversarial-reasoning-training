"""Populate the gold-trajectory cache.

Drives sample iteration via the attacks-repo task loader (real or
synthetic), looks up metadata in an optional CSV (keyed by ``sample_id``),
runs the rule-based oracle, and writes one JSON per sample to the cache
dir via ``save_gold(cache_dir, GoldKey(...), trajectory)``. Re-runs are
idempotent — the cache key is content-addressed (oracle_version + prompt
+ image hash).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ..data.gold import GoldKey, gold_exists, save_gold
from ..gold.oracle import (
    OracleConfig,
    generate_trajectory,
    load_metadata_csv,
)
from .config import load_yaml
from .runtime import setup_run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="art-make-gold",
        description="Populate the gold-trajectory cache from oracle templates.",
    )
    parser.add_argument("--config", type=Path, required=True, help="configs/gold.yaml")
    parser.add_argument("--data", type=Path, required=True, help="configs/data.yaml")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--n", type=int, default=None,
        help="Cap samples; default = full split",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _iter_and_write_gold(
    samples: list[Any],
    cache_dir: Path,
    oracle_cfg: OracleConfig,
    metadata_lookup: dict[str, dict[str, Any]],
    task_id: str,
    overwrite: bool,
) -> tuple[int, int]:
    """Iterate samples, generate trajectories, write to cache.

    Returns ``(n_written, n_skipped)``. Existing cache entries are skipped
    unless ``overwrite`` is true; the cache key is content-addressed so
    re-runs are idempotent.
    """
    n_written = 0
    n_skipped = 0
    for s in samples:
        sid = s.sample_id
        key = GoldKey(
            task_id=task_id,
            sample_id=sid,
            prompt=s.prompt,
            image=s.image,
            oracle_version=oracle_cfg.version,
        )
        if gold_exists(cache_dir, key) and not overwrite:
            n_skipped += 1
            continue
        traj = generate_trajectory(
            task_id=task_id,
            sample_id=sid,
            metadata=metadata_lookup.get(sid, {}),
            config=oracle_cfg,
        )
        save_gold(cache_dir, key, traj)
        n_written += 1
    return n_written, n_skipped


def _write_summary_sentinel(cache_dir: Path, split: str, summary: dict) -> None:
    """Atomically write a per-split summary file inside ``cache_dir``.

    External runners (e.g. scripts/run_pipeline.sh under ``--skip-existing``)
    rely on this sentinel to detect a completed split without re-iterating
    the entire dataset loader. SIGKILL/OOM/disk-full mid-write would
    otherwise leave a truncated JSON that the pipeline pre-check trusts as
    proof-of-completion; the tmp-then-rename pattern avoids that. The tmp
    filename includes the PID so concurrent runs on different splits cannot
    stomp each other's tmp file.
    """
    summary_path = Path(cache_dir) / f"_summary_{split}.json"
    tmp_path = summary_path.with_name(f"{summary_path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(summary, indent=2))
    os.replace(tmp_path, summary_path)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    gold_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data)

    cache_dir = setup_run_dir(gold_cfg["cache_dir"])
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

    n_written, n_skipped = _iter_and_write_gold(
        samples, cache_dir, oracle_cfg, metadata_lookup, task_id, args.overwrite,
    )

    summary = {
        "total": len(samples),
        "written": n_written,
        "skipped": n_skipped,
        "cache_dir": str(cache_dir),
        "oracle_version": oracle_cfg.version,
        "task_id": task_id,
        "split": args.split,
        "synthetic": synthetic,
    }
    _write_summary_sentinel(cache_dir, args.split, summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
