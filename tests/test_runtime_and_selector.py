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
    assert cfg.task_weight == 1.0


def test_lossconfig_default_task_weight_is_one() -> None:
    """Default LossConfig() must keep canonical TRADES (task_weight=1.0).
    Anything else would silently mutate already-trained ckpts' loss math."""
    assert LossConfig().task_weight == 1.0


def test_from_cfg_dict_reads_nested_trades_task_weight() -> None:
    cfg = from_cfg_dict({
        "defense": "trades",
        "trades": {"beta_start": 6.0, "task_weight": 0.0},
    })
    assert cfg.task_weight == 0.0
    assert cfg.beta == 6.0


def test_from_cfg_dict_flat_task_weight_overrides_nested() -> None:
    """Flat-shape keys win over nested, matching how `beta` and
    `temperature` resolve in `from_cfg_dict`."""
    cfg = from_cfg_dict({
        "defense": "trades",
        "task_weight": 0.5,
        "trades": {"task_weight": 0.0},
    })
    assert cfg.task_weight == 0.5


def test_build_loss_trades_propagates_task_weight() -> None:
    """Setting task_weight=0 on the config must change the closure's
    output total — proving the value flows through `build_loss` to
    `trades_loss`, not just into `LossConfig`."""
    fn_one = build_loss(LossConfig(defense="trades", beta=4.0, task_weight=1.0))
    fn_zero = build_loss(LossConfig(defense="trades", beta=4.0, task_weight=0.0))
    lc, la, ids, tm = _logits(), _logits(), _ids(), _mask_all_ones()
    out_one = fn_one(lc, la, ids, tm, tm)
    out_zero = fn_zero(lc, la, ids, tm, tm)
    # canonical = task + 4·kl; pure-traj-KL = 4·kl. Difference must equal task.
    diff = float((out_one.total - out_zero.total).detach())
    assert diff == pytest.approx(out_one.components["loss_task"], rel=1e-5)
    assert out_zero.components["task_weight"] == 0.0
    assert out_one.components["task_weight"] == 1.0
