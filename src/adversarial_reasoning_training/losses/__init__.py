"""Loss functions: task CE, trajectory KL, TRADES, PGD-AT, OAAT, and a selector."""

from .oaat import OaatOutput, oaat_loss
from .pgd_at import PgdAtOutput, pgd_at_loss
from .selector import LossCallResult, LossConfig, build_loss, from_cfg_dict
from .task_ce import task_ce
from .trades import TradesOutput, trades_loss
from .traj_kl import traj_kl

__all__ = [
    "LossCallResult",
    "LossConfig",
    "OaatOutput",
    "PgdAtOutput",
    "TradesOutput",
    "build_loss",
    "from_cfg_dict",
    "oaat_loss",
    "pgd_at_loss",
    "task_ce",
    "trades_loss",
    "traj_kl",
]
