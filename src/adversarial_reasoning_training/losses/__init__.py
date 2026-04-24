"""Loss functions: task CE, trajectory KL, TRADES, PGD-AT, OAAT, and a selector."""

from .selector import build_loss

__all__ = ["build_loss"]
