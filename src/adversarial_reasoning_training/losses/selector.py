"""Factory: config-driven selection between TRADES / PGD-AT / OAAT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .oaat import OaatOutput, oaat_loss
from .pgd_at import PgdAtOutput, pgd_at_loss
from .trades import TradesOutput, trades_loss


@dataclass
class LossConfig:
    defense: str = "trades"  # trades | pgd_at | oaat
    beta: float = 6.0        # TRADES β; mutated in-place at epoch start when annealing
    beta_end: float = 3.0    # TRADES β target at final epoch (linear interpolation)
    temperature: float = 2.0  # TRADES
    alpha: float = 0.5       # OAAT


@dataclass(frozen=True)
class LossCallResult:
    total: torch.Tensor
    components: dict[str, float]


def build_loss(config: LossConfig):
    """Return a closure (logits_clean, logits_adv, input_ids, task_mask, traj_mask) -> LossCallResult."""
    defense = config.defense.lower()
    if defense == "trades":
        def _fn(
            logits_clean: torch.Tensor,
            logits_adv: torch.Tensor,
            input_ids: torch.Tensor,
            task_mask: torch.Tensor,
            traj_mask: torch.Tensor,
        ) -> LossCallResult:
            out: TradesOutput = trades_loss(
                logits_clean, logits_adv, input_ids,
                task_mask, traj_mask,
                beta=config.beta, temperature=config.temperature,
            )
            return LossCallResult(
                total=out.total,
                components={
                    "loss_total": float(out.total.detach()),
                    "loss_task": float(out.task.detach()),
                    "loss_kl": float(out.kl.detach()),
                    "beta": config.beta,
                },
            )
        _fn.config = config
        return _fn

    if defense == "pgd_at":
        def _fn(
            logits_clean: torch.Tensor,
            logits_adv: torch.Tensor,
            input_ids: torch.Tensor,
            task_mask: torch.Tensor,
            traj_mask: torch.Tensor,
        ) -> LossCallResult:
            out: PgdAtOutput = pgd_at_loss(logits_adv, input_ids, task_mask)
            return LossCallResult(
                total=out.total,
                components={
                    "loss_total": float(out.total.detach()),
                    "loss_task_adv": float(out.task_adv.detach()),
                },
            )
        _fn.config = config
        return _fn

    if defense == "oaat":
        def _fn(
            logits_clean: torch.Tensor,
            logits_adv: torch.Tensor,
            input_ids: torch.Tensor,
            task_mask: torch.Tensor,
            traj_mask: torch.Tensor,
        ) -> LossCallResult:
            out: OaatOutput = oaat_loss(
                logits_clean, logits_adv, input_ids, task_mask, alpha=config.alpha,
            )
            return LossCallResult(
                total=out.total,
                components={
                    "loss_total": float(out.total.detach()),
                    "loss_task_clean": float(out.task_clean.detach()),
                    "loss_task_adv": float(out.task_adv.detach()),
                    "alpha": config.alpha,
                },
            )
        _fn.config = config
        return _fn

    raise ValueError(f"Unknown defense: {config.defense!r}. Expected trades|pgd_at|oaat.")


def from_cfg_dict(d: dict[str, Any]) -> LossConfig:
    """Build a LossConfig from either a flat dict or a nested defenses.yaml.

    Flat shape:  ``{"defense": "trades", "beta": 6.0, "temperature": 2.0}``
    Nested:      ``{"defense": "trades", "trades": {"beta_start": 6.0, ...}}``
    """
    defense = str(d.get("defense", "trades"))
    trades_cfg = d.get("trades") or {}
    oaat_cfg = d.get("oaat") or {}
    beta = float(d.get("beta", trades_cfg.get("beta_start", 6.0)))
    beta_end = float(d.get("beta_end", trades_cfg.get("beta_end", 3.0)))
    temperature = float(d.get("temperature", trades_cfg.get("temperature", 2.0)))
    alpha = float(d.get("alpha", oaat_cfg.get("alpha", 0.5)))
    return LossConfig(defense=defense, beta=beta, beta_end=beta_end, temperature=temperature, alpha=alpha)
