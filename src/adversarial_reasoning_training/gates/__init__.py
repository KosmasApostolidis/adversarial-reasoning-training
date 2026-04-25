"""Phase gates T0 (env) / T1 (clean-FT) / T2 (no-collapse) / T3 (robustness)."""

from .T0_env import T0Result, run_t0
from .T1_clean import (
    T1Result,
    T1Thresholds,
    make_teacher_forced_evaluator,
    run_t1,
)
from .T2_no_collapse import T2Result, T2Thresholds, run_t2
from .T3_robust import T3Result, T3Thresholds, run_t3

__all__ = [
    "T0Result",
    "T1Result",
    "T1Thresholds",
    "T2Result",
    "T2Thresholds",
    "T3Result",
    "T3Thresholds",
    "make_teacher_forced_evaluator",
    "run_t0",
    "run_t1",
    "run_t2",
    "run_t3",
]
