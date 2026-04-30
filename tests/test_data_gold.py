"""Unit tests for data/gold — cache I/O round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from adversarial_reasoning.agents.base import ToolCall, Trajectory  # type: ignore
from adversarial_reasoning_training.data.gold import (
    gold_exists,
    load_gold,
    save_gold,
)


def _img(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8))


def _traj() -> Trajectory:
    return Trajectory(
        task_id="t",
        model_id="oracle",
        seed=0,
        tool_calls=[ToolCall(step=1, name="calc_pi_rads", args={"lesion_id": "L0"}, result={"pi_rads": 4})],
        final_answer="ans",
        reasoning_trace="reason",
        metadata={"sample_id": "s"},
    )


def test_gold_exists_false_before_save(tmp_path: Path) -> None:
    img = _img()
    assert not gold_exists(tmp_path, "task", "s", "prompt", img, "v1")


def test_save_and_gold_exists_true(tmp_path: Path) -> None:
    img = _img()
    save_gold(tmp_path, "task", "s", "prompt", img, "v1", _traj())
    assert gold_exists(tmp_path, "task", "s", "prompt", img, "v1")


def test_save_and_load_roundtrip_preserves_fields(tmp_path: Path) -> None:
    img = _img()
    save_gold(tmp_path, "task", "s", "prompt", img, "v1", _traj())
    loaded = load_gold(tmp_path, "task", "s", "prompt", img, "v1")
    assert loaded.task_id == "t"
    assert loaded.model_id == "oracle"
    assert loaded.seed == 0
    assert loaded.final_answer == "ans"
    assert loaded.reasoning_trace == "reason"
    assert loaded.metadata == {"sample_id": "s"}
    assert len(loaded.tool_calls) == 1
    assert loaded.tool_calls[0].name == "calc_pi_rads"


def test_load_gold_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not cached"):
        load_gold(tmp_path, "task", "s", "prompt", _img(), "v1")


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "c"
    save_gold(deep, "task", "s", "prompt", _img(), "v1", _traj())
    assert deep.is_dir()


def test_save_returns_path_under_cache_dir(tmp_path: Path) -> None:
    out_path = save_gold(tmp_path, "task", "s", "prompt", _img(), "v1", _traj())
    assert out_path.is_file()
    assert out_path.parent == tmp_path
    assert out_path.suffix == ".json"


def test_different_oracle_versions_isolate_caches(tmp_path: Path) -> None:
    img = _img()
    save_gold(tmp_path, "task", "s", "prompt", img, "v1", _traj())
    assert not gold_exists(tmp_path, "task", "s", "prompt", img, "v2")


def test_different_prompts_isolate_caches(tmp_path: Path) -> None:
    img = _img()
    save_gold(tmp_path, "task", "s", "prompt-A", img, "v1", _traj())
    assert not gold_exists(tmp_path, "task", "s", "prompt-B", img, "v1")
