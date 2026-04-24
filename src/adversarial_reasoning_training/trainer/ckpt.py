"""Checkpoint writer with rotation (keep best + latest)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointRegistry:
    """Track the best and latest checkpoints in a `runs/<id>/ckpt/` directory.

    `best_metric` is "higher is better" by default (e.g., tool-name acc on
    dev). Flip `higher_is_better=False` for loss-style metrics.
    """

    ckpt_dir: Path
    higher_is_better: bool = True
    keep: int = 2  # best + latest
    best_metric_value: float | None = None
    best_path: Path | None = None
    latest_path: Path | None = None

    def __post_init__(self) -> None:
        self.ckpt_dir = Path(self.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def is_better(self, value: float) -> bool:
        if self.best_metric_value is None:
            return True
        return (
            value > self.best_metric_value
            if self.higher_is_better
            else value < self.best_metric_value
        )

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        step: int,
        epoch: int,
        metric_value: float | None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save a new checkpoint as `latest` and — if metric improved — as `best`."""
        ts = time.strftime("%Y%m%d-%H%M%S")
        stem = f"step{step:07d}-ep{epoch:02d}-{ts}"
        latest = self.ckpt_dir / f"{stem}.pt"
        payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "step": step,
            "epoch": epoch,
            "metric_value": metric_value,
            "extra": extra or {},
        }
        if optimizer is not None:
            payload["optim_state_dict"] = optimizer.state_dict()
        torch.save(payload, latest)
        self._rm_old_latest(latest)
        self.latest_path = latest

        if metric_value is not None and self.is_better(metric_value):
            best = self.ckpt_dir / f"best-{stem}.pt"
            torch.save(payload, best)
            self._rm_old_best(best)
            self.best_metric_value = metric_value
            self.best_path = best

        self._write_index()
        return latest

    def _rm_old_latest(self, keep: Path) -> None:
        for p in sorted(self.ckpt_dir.glob("step*.pt")):
            if p != keep:
                p.unlink(missing_ok=True)

    def _rm_old_best(self, keep: Path) -> None:
        for p in sorted(self.ckpt_dir.glob("best-*.pt")):
            if p != keep:
                p.unlink(missing_ok=True)

    def _write_index(self) -> None:
        idx = {
            "best_metric_value": self.best_metric_value,
            "best_path": str(self.best_path) if self.best_path else None,
            "latest_path": str(self.latest_path) if self.latest_path else None,
        }
        with (self.ckpt_dir / "index.json").open("w", encoding="utf-8") as f:
            json.dump(idx, f, indent=2)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    if optimizer is not None and "optim_state_dict" in payload:
        optimizer.load_state_dict(payload["optim_state_dict"])
    return payload
