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

from adversarial_reasoning_training.cli.train import (
    _build_optim_and_schedule,
    _build_trainer_config,
    _resolve_seed,
)
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


@pytest.mark.unit
def test_resolve_seed_uses_train_cfg_when_cli_unset() -> None:
    """No --seed on CLI ⇒ honour validated YAML ``seed:`` value."""
    args = Namespace(seed=None)
    assert _resolve_seed(args, {"seed": 7}) == 7


@pytest.mark.unit
def test_resolve_seed_cli_overrides_train_cfg() -> None:
    """Explicit --seed beats YAML so pipeline per-seed runs continue to work."""
    args = Namespace(seed=3)
    assert _resolve_seed(args, {"seed": 7}) == 3


@pytest.mark.unit
def test_resolve_seed_missing_train_cfg_key_raises() -> None:
    """Schema requires ``seed``; reaching the helper without it is a bug."""
    args = Namespace(seed=None)
    with pytest.raises(KeyError):
        _resolve_seed(args, {})


@pytest.mark.unit
def test_build_optim_and_schedule_threads_weight_decay_and_betas() -> None:
    """YAML ``weight_decay`` + ``betas`` must reach the optimizer.

    Pre-fix the ``OptimConfig`` constructor in
    ``_build_optim_and_schedule`` ignored both keys (defaulting to
    ``weight_decay=0.0`` and ``betas=(0.9, 0.999)``) while ``cli/schema.py``
    validated them as legal YAML keys — silent contract violation.
    """
    import torch

    model = torch.nn.Linear(4, 4)
    for name, _p in model.named_parameters():
        assert "weight" in name or "bias" in name  # smoke
    train_cfg = {
        "optim": "adamw",
        "lr": {"lm": 1e-4, "projector": 1e-4, "vit": 1e-4},
        "epochs": 1,
        "grad_accum": 1,
        "weight_decay": 0.07,
        "betas": [0.91, 0.95],
        "schedule": "constant",
    }
    optimizer, _ = _build_optim_and_schedule(model, train_cfg, train_ds_size=8)
    # weight_decay lives on every param-group built by ``param_groups_by_role``.
    for pg in optimizer.param_groups:
        assert pg["weight_decay"] == pytest.approx(0.07)
    assert optimizer.defaults["betas"] == (0.91, 0.95)


@pytest.mark.unit
def test_build_optim_and_schedule_keeps_optimconfig_defaults_when_absent() -> None:
    """No YAML keys ⇒ OptimConfig dataclass defaults stand (weight_decay=0,
    betas=(0.9, 0.999)). Guards against the inverse regression where adding
    the plumbing accidentally inverts the default.
    """
    import torch

    model = torch.nn.Linear(4, 4)
    train_cfg = {
        "optim": "adamw",
        "lr": {"lm": 1e-4, "projector": 1e-4, "vit": 1e-4},
        "epochs": 1,
        "grad_accum": 1,
        "schedule": "constant",
    }
    optimizer, _ = _build_optim_and_schedule(model, train_cfg, train_ds_size=8)
    for pg in optimizer.param_groups:
        assert pg["weight_decay"] == pytest.approx(0.0)
    assert optimizer.defaults["betas"] == (0.9, 0.999)
