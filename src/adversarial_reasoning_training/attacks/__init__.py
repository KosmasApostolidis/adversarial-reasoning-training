"""Inner-loop attack wrappers. Thin bridge to attacks-repo `PGDAttack`."""

from .inner_pgd import InnerPgdConfig, epsilon_for_epoch, run_inner_pgd

__all__ = ["InnerPgdConfig", "epsilon_for_epoch", "run_inner_pgd"]
