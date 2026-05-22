"""Checkpoint writer with rotation (keep best + latest)."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _atomic_torch_save(payload: dict[str, Any], target: Path) -> None:
    """Save ``payload`` to ``target`` atomically.

    Writes to a sibling tempfile, fsyncs, then ``os.replace`` to the
    final name. A crash mid-write therefore leaves the previous
    checkpoint intact instead of a half-written torch artifact that
    would crash ``torch.load`` on the next run.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        tmp_path.chmod(0o600)
        with tmp_path.open("rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)


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
        """Save a full-state checkpoint (model + optimizer)."""
        payload = self._build_payload(model, step, epoch, metric_value, extra)
        if optimizer is not None:
            payload["optim_state_dict"] = optimizer.state_dict()
        return self._write_and_rotate(payload, step, epoch, metric_value)

    def save_weights_only(
        self,
        *,
        model: torch.nn.Module,
        step: int,
        epoch: int,
        metric_value: float | None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save a weights-only checkpoint (model state_dict + metadata, no optimizer).

        Used by the periodic ``save_every`` cadence so frequent saves don't
        blow disk on long runs.
        """
        payload = self._build_payload(model, step, epoch, metric_value, extra)
        return self._write_and_rotate(payload, step, epoch, metric_value)

    def _build_payload(
        self,
        model: torch.nn.Module,
        step: int,
        epoch: int,
        metric_value: float | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "step": step,
            "epoch": epoch,
            "metric_value": metric_value,
            "extra": extra or {},
        }

    def _write_and_rotate(
        self,
        payload: dict[str, Any],
        step: int,
        epoch: int,
        metric_value: float | None,
    ) -> Path:
        ts = time.strftime("%Y%m%d-%H%M%S")
        stem = f"step{step:07d}-ep{epoch:02d}-{ts}"
        latest = self.ckpt_dir / f"{stem}.pt"
        _atomic_torch_save(payload, latest)
        self._rm_old_latest(latest)
        self.latest_path = latest

        if metric_value is not None and self.is_better(metric_value):
            best = self.ckpt_dir / f"best-{stem}.pt"
            _atomic_torch_save(payload, best)
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
    *,
    strict: bool = False,
    max_missing_frac: float = 0.05,
    max_unexpected_frac: float = 0.05,
) -> dict[str, Any]:
    """Load a checkpoint with the safe weights-only deserialiser.

    ``weights_only=True`` (torch >=2.4 default for trusted sources, but we
    set it explicitly so older releases pin the same behaviour) restricts
    the loader to plain tensors + a small allow-list of builtins, so a
    malicious or corrupt artifact cannot execute arbitrary code on load.
    The fallback re-raises with context — silent fallback to the unsafe
    loader would defeat the purpose.

    When ``strict=False`` the load is still audited: if more than
    ``max_missing_frac`` of expected keys are absent or more than
    ``max_unexpected_frac`` of payload keys are unknown, we raise.
    A genuine architecture mismatch should fail loudly, not silently
    initialise half the weights from the model's default state.
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    result = model.load_state_dict(payload["model_state_dict"], strict=strict)
    n_params = sum(1 for _ in model.state_dict())
    if n_params and not strict:
        missing = len(getattr(result, "missing_keys", []) or [])
        unexpected = len(getattr(result, "unexpected_keys", []) or [])
        if missing / n_params > max_missing_frac:
            raise RuntimeError(
                f"load_checkpoint: missing keys {missing}/{n_params} "
                f"({missing / n_params:.1%}) exceeds {max_missing_frac:.0%} "
                f"budget. First few: {result.missing_keys[:5]}"
            )
        if unexpected / n_params > max_unexpected_frac:
            raise RuntimeError(
                f"load_checkpoint: unexpected keys {unexpected}/{n_params} "
                f"({unexpected / n_params:.1%}) exceeds "
                f"{max_unexpected_frac:.0%} budget. First few: "
                f"{result.unexpected_keys[:5]}"
            )
    if optimizer is not None and "optim_state_dict" in payload:
        optimizer.load_state_dict(payload["optim_state_dict"])
    return payload
