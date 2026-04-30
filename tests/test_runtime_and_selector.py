"""Unit tests for cli/runtime + losses/selector — small factories."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from adversarial_reasoning_training.cli.runtime import (
    setup_device,
    setup_run_dir,
    setup_seed,
)
from adversarial_reasoning_training.losses.selector import (
    LossConfig,
    build_loss,
    from_cfg_dict,
)


def test_setup_run_dir_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "runs" / "exp42"
    out = setup_run_dir(target)
    assert out == target
    assert target.is_dir()


def test_setup_run_dir_idempotent_on_existing(tmp_path: Path) -> None:
    target = tmp_path / "already-here"
    target.mkdir()
    (target / "marker").write_text("keep")
    out = setup_run_dir(target)
    assert out == target
    assert (target / "marker").read_text() == "keep"


def test_setup_run_dir_accepts_string_path(tmp_path: Path) -> None:
    out = setup_run_dir(str(tmp_path / "stringy"))
    assert isinstance(out, Path)
    assert out.is_dir()


def test_setup_seed_delegates_without_raising() -> None:
    setup_seed(7, deterministic=False)
    a = float(torch.rand(1).item())
    setup_seed(7, deterministic=False)
    b = float(torch.rand(1).item())
    assert a == b


def test_setup_device_falls_back_to_cpu_when_cuda_missing() -> None:
    if not torch.cuda.is_available():
        dev = setup_device("cuda")
        assert dev.type == "cpu"
    else:
        dev = setup_device("cuda")
        assert dev.type == "cuda"


def test_setup_device_explicit_cpu() -> None:
    assert setup_device("cpu").type == "cpu"


# -------------------- losses/selector --------------------

# Keep dimensions small + deterministic.
_VOCAB = 5
_SEQ = 6
_BATCH = 1


def _logits() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(_BATCH, _SEQ, _VOCAB, requires_grad=True)


def _ids() -> torch.Tensor:
    return torch.randint(0, _VOCAB, (_BATCH, _SEQ))


def _mask_all_ones() -> torch.Tensor:
    return torch.ones(_BATCH, _SEQ, dtype=torch.bool)


def test_build_loss_trades_returns_callable_and_components() -> None:
    fn = build_loss(LossConfig(defense="trades", beta=4.0, temperature=2.0))
    out = fn(_logits(), _logits(), _ids(), _mask_all_ones(), _mask_all_ones())
    assert torch.is_tensor(out.total)
    assert {"loss_total", "loss_task", "loss_kl", "beta"} <= set(out.components)
    assert out.components["beta"] == 4.0


def test_build_loss_pgd_at_returns_callable_and_components() -> None:
    fn = build_loss(LossConfig(defense="pgd_at"))
    out = fn(_logits(), _logits(), _ids(), _mask_all_ones(), _mask_all_ones())
    assert torch.is_tensor(out.total)
    assert "loss_task_adv" in out.components


def test_build_loss_oaat_returns_callable_and_components() -> None:
    fn = build_loss(LossConfig(defense="oaat", alpha=0.25))
    out = fn(_logits(), _logits(), _ids(), _mask_all_ones(), _mask_all_ones())
    assert torch.is_tensor(out.total)
    assert out.components["alpha"] == 0.25


def test_build_loss_unknown_defense_raises() -> None:
    with pytest.raises(ValueError, match="Unknown defense"):
        build_loss(LossConfig(defense="madry"))


def test_build_loss_case_insensitive() -> None:
    fn = build_loss(LossConfig(defense="TRADES"))
    out = fn(_logits(), _logits(), _ids(), _mask_all_ones(), _mask_all_ones())
    assert "loss_kl" in out.components


def test_from_cfg_dict_flat_shape() -> None:
    cfg = from_cfg_dict({"defense": "oaat", "alpha": 0.7})
    assert cfg.defense == "oaat"
    assert cfg.alpha == 0.7


def test_from_cfg_dict_nested_trades_keys() -> None:
    cfg = from_cfg_dict({
        "defense": "trades",
        "trades": {"beta_start": 8.0, "temperature": 3.0},
    })
    assert cfg.defense == "trades"
    assert cfg.beta == 8.0
    assert cfg.temperature == 3.0


def test_from_cfg_dict_defaults_when_empty() -> None:
    cfg = from_cfg_dict({})
    assert cfg.defense == "trades"
    assert cfg.beta == 6.0
    assert cfg.temperature == 2.0
    assert cfg.alpha == 0.5
