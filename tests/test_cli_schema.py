"""Schema-validator tests for cli/schema.py.

Production configs at configs/*.yaml must pass every validator and
known typos must raise ``ValueError`` with an actionable message.
This is the fail-fast layer that keeps art-train from spending 30s
loading a 7B VLM only to crash on a misspelled key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adversarial_reasoning_training.cli.config import load_yaml
from adversarial_reasoning_training.cli.schema import (
    validate_data,
    validate_defenses,
    validate_full_ft,
    validate_gold,
    validate_training,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_production_training_yaml_passes() -> None:
    validate_training(load_yaml(CONFIGS / "training.yaml"))


def test_production_training_1epoch_yaml_passes() -> None:
    validate_training(load_yaml(CONFIGS / "training_1epoch.yaml"))


def test_production_defenses_yaml_passes() -> None:
    validate_defenses(load_yaml(CONFIGS / "defenses.yaml"))


def test_production_data_yaml_passes() -> None:
    validate_data(load_yaml(CONFIGS / "data.yaml"))


def test_production_gold_yaml_passes() -> None:
    validate_gold(load_yaml(CONFIGS / "gold.yaml"))


def test_production_full_ft_yaml_passes() -> None:
    validate_full_ft(load_yaml(CONFIGS / "full_ft.yaml"))


def _undefended_training() -> dict:
    return load_yaml(CONFIGS / "training.yaml")


def _undefended_defenses() -> dict:
    return load_yaml(CONFIGS / "defenses.yaml")


def _undefended_full_ft() -> dict:
    return load_yaml(CONFIGS / "full_ft.yaml")


def test_training_rejects_unknown_defense() -> None:
    cfg = _undefended_training()
    cfg["defense"] = "tradse"  # typo
    with pytest.raises(ValueError, match="defense"):
        validate_training(cfg)


def test_training_rejects_unknown_optim() -> None:
    cfg = _undefended_training()
    cfg["optim"] = "adamw_8bit"  # underscore typo
    with pytest.raises(ValueError, match="optim"):
        validate_training(cfg)


def test_training_rejects_unknown_schedule() -> None:
    cfg = _undefended_training()
    cfg["schedule"] = "cosin"
    with pytest.raises(ValueError, match="schedule"):
        validate_training(cfg)


def test_training_rejects_missing_lr_role() -> None:
    cfg = _undefended_training()
    del cfg["lr"]["projector"]
    with pytest.raises(ValueError, match="projector"):
        validate_training(cfg)


def test_training_rejects_missing_top_level_key() -> None:
    cfg = _undefended_training()
    del cfg["grad_accum"]
    with pytest.raises(ValueError, match="grad_accum"):
        validate_training(cfg)


def test_defenses_rejects_missing_pgd_block() -> None:
    cfg = {"trades": {"beta_start": 6.0}}
    with pytest.raises(ValueError, match="pgd"):
        validate_defenses(cfg)


def test_defenses_rejects_missing_pgd_key() -> None:
    cfg = _undefended_defenses()
    del cfg["pgd"]["alpha_ratio"]
    with pytest.raises(ValueError, match="alpha_ratio"):
        validate_defenses(cfg)


def test_defenses_rejects_malformed_eps_schedule() -> None:
    cfg = _undefended_defenses()
    cfg["pgd"]["eps_schedule"] = [{"epoch_ranges": [1, 2], "eps": 0.0078}]  # plural typo
    with pytest.raises(ValueError, match="epoch_range"):
        validate_defenses(cfg)


def test_full_ft_rejects_unknown_freeze_strategy() -> None:
    cfg = _undefended_full_ft()
    cfg["freeze_strategy"] = "everything"
    with pytest.raises(ValueError, match="freeze_strategy"):
        validate_full_ft(cfg)
