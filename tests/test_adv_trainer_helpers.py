"""Unit tests for the private helpers extracted from AdvTrainer.fit
and AdvTrainer._apply_optimizer_step in the clean-code sweep.

The end-to-end orchestration is already covered by:
    tests/test_grad_accum_window.py
    tests/test_nan_skip.py
    tests/test_amp_clip_order.py

This file targets the new helpers in isolation so regressions in any
single piece (beta annealing, end-of-training meta dump, clip-and-check
fork) surface with a precise failure rather than via an end-to-end
delta.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import Dataset

from adversarial_reasoning_training.trainer.adv_trainer import AdvTrainer, TrainerConfig
from adversarial_reasoning_training.trajectory.teacher_force import TeacherForcedBatch


@dataclass
class _StubLossCfg:
    beta: float = 6.0
    beta_end: float = 6.0


class _StubLossFn:
    def __init__(self, cfg: _StubLossCfg | None) -> None:
        self.config = cfg

    def __call__(self, *_a: object, **_kw: object) -> object:  # pragma: no cover
        raise AssertionError("not invoked in helper unit tests")


class _StubModel(torch.nn.Module):
    def __init__(self, vocab: int = 8, hidden: int = 4) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab)


class _StubVLM:
    family = "stub"

    def __init__(self, model: _StubModel) -> None:
        self.model = model

    def forward_with_logits(self, *_a: object, **_kw: object) -> torch.Tensor:  # pragma: no cover
        raise AssertionError("not invoked in helper unit tests")


def _make_trainer(
    tmp_path: Path,
    *,
    loss_cfg: _StubLossCfg | None = None,
    epochs: int = 1,
) -> AdvTrainer:
    model = _StubModel()
    vlm = _StubVLM(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    cfg = TrainerConfig(
        epochs=epochs,
        grad_accum=2,
        log_every=1,
        eval_every=0,
        save_every=0,
        grad_clip_norm=1.0,
        amp_dtype="fp32",
        run_dir=tmp_path,
        final_save_include_optimizer=False,
    )

    def _identity_collate(items: list) -> object:  # pragma: no cover
        return items[0]

    return AdvTrainer(
        vlm=vlm,
        model=model,
        collator=_identity_collate,  # type: ignore[arg-type]
        loss_fn=_StubLossFn(loss_cfg),  # type: ignore[arg-type]
        optimizer=optimizer,
        scheduler=None,
        config=cfg,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# _init_beta_schedule  /  _anneal_beta
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_init_beta_schedule_returns_constants_when_loss_cfg_missing(tmp_path: Path) -> None:
    """No loss_fn.config → schedule degenerates to (None, 6.0, 6.0).

    Captures the prior fit() default that fired when the loss closure
    has no exposed config (e.g. the gold/teacher-forced gates).
    """
    trainer = _make_trainer(tmp_path, loss_cfg=None)
    cfg, beta_start, beta_end = trainer._init_beta_schedule()
    assert cfg is None
    assert beta_start == pytest.approx(6.0)
    assert beta_end == pytest.approx(6.0)


@pytest.mark.unit
def test_init_beta_schedule_reads_from_loss_config(tmp_path: Path) -> None:
    loss_cfg = _StubLossCfg(beta=2.0, beta_end=8.0)
    trainer = _make_trainer(tmp_path, loss_cfg=loss_cfg)
    cfg, beta_start, beta_end = trainer._init_beta_schedule()
    assert cfg is loss_cfg
    assert beta_start == pytest.approx(2.0)
    assert beta_end == pytest.approx(8.0)


@pytest.mark.unit
def test_anneal_beta_linearly_interpolates_across_epochs(tmp_path: Path) -> None:
    """epoch=1 → beta_start; epoch=epochs → beta_end."""
    loss_cfg = _StubLossCfg(beta=2.0, beta_end=8.0)
    trainer = _make_trainer(tmp_path, loss_cfg=loss_cfg, epochs=4)
    cfg, beta_start, beta_end = trainer._init_beta_schedule()
    trainer._anneal_beta(cfg, beta_start, beta_end, epoch=1)
    assert loss_cfg.beta == pytest.approx(2.0)
    trainer._anneal_beta(cfg, beta_start, beta_end, epoch=4)
    assert loss_cfg.beta == pytest.approx(8.0)
    trainer._anneal_beta(cfg, beta_start, beta_end, epoch=2)
    # 1/3 of the way: 2.0 + (1/3)*(8 - 2) = 4.0
    assert loss_cfg.beta == pytest.approx(4.0)


@pytest.mark.unit
def test_anneal_beta_is_noop_when_cfg_is_none(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path, loss_cfg=None)
    # Must not raise.
    trainer._anneal_beta(None, 6.0, 6.0, epoch=3)


@pytest.mark.unit
def test_anneal_beta_is_noop_when_single_epoch(tmp_path: Path) -> None:
    """epochs == 1 ⇒ frac would divide by zero; helper must short-circuit."""
    loss_cfg = _StubLossCfg(beta=2.0, beta_end=8.0)
    trainer = _make_trainer(tmp_path, loss_cfg=loss_cfg, epochs=1)
    cfg, beta_start, beta_end = trainer._init_beta_schedule()
    trainer._anneal_beta(cfg, beta_start, beta_end, epoch=1)
    # beta untouched because there's no inner-loop interpolation.
    assert loss_cfg.beta == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _finalize_training: writes meta + fit_done event without re-running training
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_finalize_training_writes_meta_and_fit_done(tmp_path: Path) -> None:
    """Calling _finalize_training in isolation produces both the
    train_meta.json file and the fit_done log record. Used to live
    inline in fit(); now extracted so the meta-dump invariants
    (keys, shape) can be tested without spinning up DataLoader.
    """
    trainer = _make_trainer(tmp_path)
    # Simulate a completed run by setting global step manually.
    trainer._global_step = 42
    trainer._finalize_training(total_outer=84, start_time=0.0)

    meta_path = tmp_path / "train_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["global_step"] == 42
    assert meta["total_outer"] == 84
    assert meta["epochs"] == trainer.config.epochs
    assert "duration_s" in meta and meta["duration_s"] >= 0.0
    assert "peak_memory_gb" in meta

    log_path = tmp_path / "train_log.jsonl"
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    fit_done = [e for e in events if e["event"] == "fit_done"]
    assert len(fit_done) == 1
    assert fit_done[0]["global_step"] == 42
    assert fit_done[0]["total_outer"] == 84
