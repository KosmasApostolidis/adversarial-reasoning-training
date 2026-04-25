"""Public-API contract tests.

Verifies two things for every refactored subpackage:

1. Every name in ``__all__`` resolves to a module attribute (no typos
   that survive because Python tolerates broken ``__all__`` until you
   ``from pkg import *``).
2. Legacy deep-import paths still work — the re-exports added during
   the clean-coding refactor must be additive only.
"""

from __future__ import annotations

import importlib

import pytest

PACKAGES_WITH_ALL = [
    "adversarial_reasoning_training.gates",
    "adversarial_reasoning_training.losses",
    "adversarial_reasoning_training.trajectory",
    "adversarial_reasoning_training.gold",
    "adversarial_reasoning_training.data",
    "adversarial_reasoning_training.attacks",
    "adversarial_reasoning_training.eval",
    "adversarial_reasoning_training.trainer",
    "adversarial_reasoning_training.utils",
    "adversarial_reasoning_training.cli",
]


@pytest.mark.parametrize("module_name", PACKAGES_WITH_ALL)
def test_all_symbols_resolve(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert hasattr(module, "__all__"), f"{module_name} missing __all__"
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert not missing, f"{module_name}.__all__ references unbound names: {missing}"


# Legacy deep-import paths — these must keep working after the package
# re-export refactor. Each tuple is ``(deep_module, attribute)``.
LEGACY_DEEP_IMPORTS = [
    ("adversarial_reasoning_training.losses.task_ce", "task_ce"),
    ("adversarial_reasoning_training.losses.traj_kl", "traj_kl"),
    ("adversarial_reasoning_training.losses.trades", "trades_loss"),
    ("adversarial_reasoning_training.losses.oaat", "oaat_loss"),
    ("adversarial_reasoning_training.losses.pgd_at", "pgd_at_loss"),
    ("adversarial_reasoning_training.losses.selector", "build_loss"),
    ("adversarial_reasoning_training.gates.T1_clean", "run_t1"),
    ("adversarial_reasoning_training.gates.T2_no_collapse", "run_t2"),
    ("adversarial_reasoning_training.gates.T3_robust", "run_t3"),
    ("adversarial_reasoning_training.gates.T0_env", "run_t0"),
    ("adversarial_reasoning_training.trajectory.mask", "build_masks"),
    ("adversarial_reasoning_training.trajectory.segments", "SegmentKind"),
    ("adversarial_reasoning_training.trajectory.teacher_force", "TeacherForcedBatch"),
    ("adversarial_reasoning_training.gold.oracle", "OracleConfig"),
    ("adversarial_reasoning_training.gold.templates", "ORACLE_VERSION"),
    ("adversarial_reasoning_training.data.collator", "TFCollator"),
    ("adversarial_reasoning_training.data.dataset", "ProstateXTrainDS"),
    ("adversarial_reasoning_training.data.gold", "save_gold"),
    ("adversarial_reasoning_training.attacks.inner_pgd", "run_inner_pgd"),
    ("adversarial_reasoning_training.eval.robust_eval", "RobustEvalConfig"),
    ("adversarial_reasoning_training.trainer.adv_trainer", "AdvTrainer"),
    ("adversarial_reasoning_training.trainer.optim", "build_optimizer"),
    ("adversarial_reasoning_training.trainer.freeze", "apply_freeze"),
    ("adversarial_reasoning_training.utils.seed", "seed_everything"),
    ("adversarial_reasoning_training.utils.hashing", "sha256_text"),
    ("adversarial_reasoning_training.utils.mem", "current_memory_stats"),
]


@pytest.mark.parametrize("deep_module,attr", LEGACY_DEEP_IMPORTS)
def test_legacy_deep_import_still_works(deep_module: str, attr: str) -> None:
    module = importlib.import_module(deep_module)
    assert hasattr(module, attr), f"deep import {deep_module}.{attr} broke"
