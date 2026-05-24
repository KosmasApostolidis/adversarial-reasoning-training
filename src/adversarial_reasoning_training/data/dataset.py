"""ProstateXTrainDS: yields (image, prompt, gold_trajectory) triples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adversarial_reasoning.agents.base import Trajectory
from adversarial_reasoning.tasks.loader import TaskSample, load_task
from PIL import Image
from torch.utils.data import Dataset

from ..gold.oracle import generate_trajectory
from .gold import GoldKey, gold_exists, load_gold, save_gold


@dataclass(frozen=True)
class TrainSample:
    task_id: str
    sample_id: str
    image: Image.Image
    prompt: str
    trajectory: Trajectory
    metadata: dict[str, Any]


class ProstateXTrainDS(Dataset):
    """Training dataset. Thin wrapper around attacks-repo `load_task`.

    On __getitem__, resolves the gold trajectory either from cache
    (`data/gold/<key>.json`) or via the rule-based oracle. If the cache
    miss uses the oracle, the result is written back so subsequent epochs
    skip the cost (oracle is cheap but determinism matters).
    """

    def __init__(
        self,
        task_id: str,
        split: str = "train",
        *,
        cache_dir: str | Path = "data/gold",
        oracle_version: str = "v1",
        metadata_lookup: dict[str, dict[str, Any]] | None = None,
        n: int | None = None,
        synthetic: bool = False,
        config_path: str | Path = "configs/tasks.yaml",
        write_back: bool = True,
    ) -> None:
        super().__init__()
        self.task_id = task_id
        self.split = split
        self.cache_dir = Path(cache_dir)
        self.oracle_version = oracle_version
        self.metadata_lookup = metadata_lookup or {}
        self.write_back = write_back
        self.samples: list[TaskSample] = list(
            load_task(
                task_id, split=split, n=n, synthetic=synthetic, config_path=config_path
            )
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TrainSample:
        s = self.samples[idx]
        md = self.metadata_lookup.get(s.sample_id, {})
        key = GoldKey(
            task_id=s.task_id,
            sample_id=s.sample_id,
            prompt=s.prompt,
            image=s.image,
            oracle_version=self.oracle_version,
        )
        if gold_exists(self.cache_dir, key):
            traj = load_gold(self.cache_dir, key)
        else:
            traj = generate_trajectory(s.task_id, s.sample_id, md)
            if self.write_back:
                save_gold(self.cache_dir, key, traj)
        return TrainSample(
            task_id=s.task_id,
            sample_id=s.sample_id,
            image=s.image,
            prompt=s.prompt,
            trajectory=traj,
            metadata=md,
        )
