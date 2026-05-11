"""Optimizer + LR scheduler factories.

bnb 8-bit AdamW is the default: under full fine-tune of a 7B VLM plus
adversarial-doubled forward memory, fp32 Adam states blow the H200
VRAM budget. 8-bit quantized optimizer states cut that ~4x with minor
quality impact and no training-loop changes.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .freeze import param_groups_by_role


@dataclass(frozen=True)
class OptimConfig:
    kind: str = "adamw8bit"  # adamw8bit | adamw | adamw_fused
    lr_lm: float = 5.0e-6
    lr_projector: float = 1.0e-5
    lr_vit: float = 1.0e-6
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)


def build_optimizer(model: torch.nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    pg = param_groups_by_role(
        model,
        lr_lm=cfg.lr_lm,
        lr_projector=cfg.lr_projector,
        lr_vit=cfg.lr_vit,
        weight_decay=cfg.weight_decay,
    )
    if not pg:
        raise ValueError("No trainable parameter groups; check freeze strategy.")
    if cfg.kind == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ImportError(
                "bitsandbytes not installed. Install with `pip install bitsandbytes` "
                "or set optim.kind=adamw in configs/training.yaml."
            ) from e
        return bnb.optim.AdamW8bit(pg, betas=cfg.betas)
    if cfg.kind == "adamw":
        return torch.optim.AdamW(pg, betas=cfg.betas)
    if cfg.kind == "adamw_fused":
        return torch.optim.AdamW(pg, betas=cfg.betas, fused=True)
    raise ValueError(f"Unknown optimizer kind: {cfg.kind}")


@dataclass(frozen=True)
class ScheduleConfig:
    total_steps: int
    warmup_pct: float = 0.03
    kind: str = "cosine"  # cosine | linear | constant


def _cosine_decay(progress: float) -> float:
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _linear_decay(progress: float) -> float:
    return max(0.0, 1.0 - progress)


def _constant_decay(progress: float) -> float:
    return 1.0


_DECAY_BY_KIND: dict[str, Callable[[float], float]] = {
    "cosine": _cosine_decay,
    "linear": _linear_decay,
    "constant": _constant_decay,
}


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: ScheduleConfig
) -> torch.optim.lr_scheduler._LRScheduler:
    warmup_steps = max(1, math.ceil(cfg.warmup_pct * cfg.total_steps))
    decay_fn = _DECAY_BY_KIND.get(cfg.kind, _constant_decay)

    def _lr(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, cfg.total_steps - warmup_steps)
        return decay_fn(progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr)
