"""Unit tests for helpers extracted from ``cli/train.main``.

The orchestrator was decomposed in the clean-code sweep so each helper
takes a narrow input/output contract that can be exercised without
loading a real VLM. End-to-end smoke is still covered by
``tests/test_cli_entry_points.py`` (the ``--help`` and missing-args
paths verify the parser + module imports).
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from adversarial_reasoning_training.cli.train import _build_trainer_config
from adversarial_reasoning_training.trainer.adv_trainer import TrainerConfig
from adversarial_reasoning_training.utils.constants import (
    DEFAULT_PGD_ALPHA_RATIO,
    EPS_4_255,
)


@pytest.mark.unit
def test_build_trainer_config_passes_through_pgd_block(tmp_path: Path) -> None:
    """defenses.yaml ``pgd`` block must be threaded into TrainerConfig."""
    args = Namespace(run_dir=tmp_path)
    train_cfg = {
        "epochs": 4,
        "grad_accum": 8,
        "log_every": 10,
        "eval_every": 50,
        "save_every": 0,
        "grad_clip_norm": 1.5,
        "amp": "bf16",
    }
    defense_cfg = {
        "pgd": {
            "eps_schedule": [{"epoch": 1, "epsilon": 0.01}],
            "default_eps": 0.05,
            "alpha_ratio": 0.4,
            "steps": 11,
        }
    }
    cfg = _build_trainer_config(args, train_cfg, defense_cfg)
    assert isinstance(cfg, TrainerConfig)
    assert cfg.epochs == 4
    assert cfg.grad_accum == 8
    assert cfg.log_every == 10
    assert cfg.grad_clip_norm == pytest.approx(1.5)
    assert cfg.eps_schedule == [{"epoch": 1, "epsilon": 0.01}]
    assert cfg.default_epsilon == pytest.approx(0.05)
    assert cfg.alpha_ratio == pytest.approx(0.4)
    assert cfg.pgd_steps == 11
    assert cfg.run_dir == tmp_path


@pytest.mark.unit
def test_build_trainer_config_defaults_when_pgd_missing(tmp_path: Path) -> None:
    """Empty defenses.yaml ``pgd`` falls back to module constants — the
    pre-refactor inline literal was ``EPS_4_255 / DEFAULT_PGD_ALPHA_RATIO``.
    """
    args = Namespace(run_dir=tmp_path)
    train_cfg = {"epochs": 1, "grad_accum": 1}
    defense_cfg = {"pgd": {}}
    cfg = _build_trainer_config(args, train_cfg, defense_cfg)
    assert cfg.eps_schedule is None
    assert cfg.default_epsilon == pytest.approx(EPS_4_255)
    assert cfg.alpha_ratio == pytest.approx(DEFAULT_PGD_ALPHA_RATIO)
    assert cfg.pgd_steps == 7
    # final_save_include_optimizer defaults to True
    assert cfg.final_save_include_optimizer is True


@pytest.mark.unit
def test_build_trainer_config_respects_final_save_optimizer_override(tmp_path: Path) -> None:
    args = Namespace(run_dir=tmp_path)
    train_cfg = {
        "epochs": 1,
        "grad_accum": 1,
        "final_save_include_optimizer": False,
    }
    defense_cfg = {"pgd": {}}
    cfg = _build_trainer_config(args, train_cfg, defense_cfg)
    assert cfg.final_save_include_optimizer is False
