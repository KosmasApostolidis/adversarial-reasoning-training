"""Cache-backed gold-trajectory loader + writer."""

from __future__ import annotations

import json
from pathlib import Path

from adversarial_reasoning.agents.base import ToolCall, Trajectory
from PIL import Image

from ..utils.hashing import gold_cache_key


def _cache_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.json"


def gold_exists(
    cache_dir: str | Path,
    task_id: str,
    sample_id: str,
    prompt: str,
    image: Image.Image,
    oracle_version: str,
) -> bool:
    key = gold_cache_key(task_id, sample_id, prompt, image, oracle_version)
    return _cache_path(cache_dir, key).exists()


def save_gold(
    cache_dir: str | Path,
    task_id: str,
    sample_id: str,
    prompt: str,
    image: Image.Image,
    oracle_version: str,
    trajectory: Trajectory,
) -> Path:
    key = gold_cache_key(task_id, sample_id, prompt, image, oracle_version)
    path = _cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(trajectory.to_jsonl())
    return path


def load_gold(
    cache_dir: str | Path,
    task_id: str,
    sample_id: str,
    prompt: str,
    image: Image.Image,
    oracle_version: str,
) -> Trajectory:
    """Load a cached gold Trajectory or raise FileNotFoundError if absent."""
    key = gold_cache_key(task_id, sample_id, prompt, image, oracle_version)
    path = _cache_path(cache_dir, key)
    if not path.exists():
        raise FileNotFoundError(
            f"Gold trajectory not cached: {path}. "
            f"Run scripts/make_gold_trajectories.py to populate."
        )
    with path.open("r", encoding="utf-8") as f:
        d = json.loads(f.read())
    return Trajectory(
        task_id=d["task_id"],
        model_id=d["model_id"],
        seed=int(d["seed"]),
        tool_calls=[ToolCall(**c) for c in d["tool_calls"]],
        final_answer=d["final_answer"],
        reasoning_trace=d["reasoning_trace"],
        metadata=d.get("metadata", {}),
    )
