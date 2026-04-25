"""Outer training loop: AMP, accumulation, checkpointing, periodic eval."""

from .adv_trainer import AdvTrainer, TrainerConfig
from .ckpt import CheckpointRegistry, load_checkpoint
from .freeze import FreezeConfig, apply_freeze, param_groups_by_role
from .optim import OptimConfig, ScheduleConfig, build_optimizer, build_scheduler

__all__ = [
    "AdvTrainer",
    "CheckpointRegistry",
    "FreezeConfig",
    "OptimConfig",
    "ScheduleConfig",
    "TrainerConfig",
    "apply_freeze",
    "build_optimizer",
    "build_scheduler",
    "load_checkpoint",
    "param_groups_by_role",
]
