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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from adversarial_reasoning_training.trainer.adv_trainer import AdvTrainer, TrainerConfig


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


# ---------------------------------------------------------------------------
# _clip_grads / _handle_inf_grad / _maybe_periodic_*
# ---------------------------------------------------------------------------


def _seed_grads(model: torch.nn.Module, value: float) -> None:
    for p in model.parameters():
        p.grad = torch.full_like(p.data, value)


@pytest.mark.unit
def test_clip_grads_returns_zero_when_clip_norm_disabled(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    trainer.config.grad_clip_norm = 0.0
    _seed_grads(trainer.model, 2.0)
    total_norm = trainer._clip_grads()
    assert torch.isfinite(total_norm)
    assert float(total_norm) == 0.0


@pytest.mark.unit
def test_clip_grads_returns_finite_total_norm_when_enabled(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    trainer.config.grad_clip_norm = 1.0
    _seed_grads(trainer.model, 0.5)
    total_norm = trainer._clip_grads()
    assert torch.isfinite(total_norm)
    assert float(total_norm) > 0.0


@pytest.mark.unit
def test_handle_inf_grad_logs_skip_and_resets_accum(tmp_path: Path) -> None:
    """Inf-grad branch must (a) emit ``skipped_nan_grad``, (b) zero
    accumulators, and (c) call ``scaler.update`` so the AMP loss-scale
    halves on the next event. Pre-refactor this lived inline; the
    extracted helper preserves the exact bookkeeping order.
    """
    trainer = _make_trainer(tmp_path)
    trainer._accum_loss_acc = 1.5
    trainer._accum_count = 3
    trainer._handle_inf_grad(epoch=2, reason="window_full")

    assert trainer._accum_loss_acc == 0.0
    assert trainer._accum_count == 0
    log = (tmp_path / "train_log.jsonl").read_text()
    record = json.loads(log.strip())
    assert record["event"] == "skipped_nan_grad"
    assert record["epoch"] == 2
    assert record["reason"] == "window_full"
    assert record["accum_count"] == 3


@pytest.mark.unit
def test_maybe_save_periodic_skipped_when_disabled(tmp_path: Path) -> None:
    """save_every=0 ⇒ no checkpoint write. Calling the helper directly
    must not invoke ckpt.save."""
    trainer = _make_trainer(tmp_path)
    trainer.config.save_every = 0
    trainer._global_step = 10
    calls: list[dict[str, Any]] = []
    trainer.ckpt.save_weights_only = lambda **kw: calls.append(kw)  # type: ignore[assignment]
    trainer._maybe_save_periodic(epoch=1)
    assert calls == []


@pytest.mark.unit
def test_maybe_save_periodic_fires_when_step_aligns(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    trainer.config.save_every = 5
    trainer._global_step = 5
    calls: list[dict[str, Any]] = []
    trainer.ckpt.save_weights_only = lambda **kw: calls.append(kw)  # type: ignore[assignment]
    trainer._maybe_save_periodic(epoch=1)
    assert len(calls) == 1
    assert calls[0]["step"] == 5
    assert calls[0]["extra"] == {"reason": "save_every"}


@pytest.mark.unit
def test_maybe_eval_periodic_skipped_when_disabled(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    trainer.config.eval_every = 0
    trainer._global_step = 10
    trainer.evaluator = lambda *_a: {  # pragma: no cover
        "tool_name_acc": 1.0
    }
    calls: list[dict[str, Any]] = []
    trainer.ckpt.save = lambda **kw: calls.append(kw)  # type: ignore[assignment]
    trainer._maybe_eval_periodic(epoch=1)
    assert calls == []


@pytest.mark.unit
def test_maybe_eval_periodic_runs_evaluator_and_saves(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    trainer.config.eval_every = 4
    trainer._global_step = 4
    trainer.evaluator = lambda step, epoch: {"tool_name_acc": 0.7}
    save_calls: list[dict[str, Any]] = []
    trainer.ckpt.save = lambda **kw: save_calls.append(kw)  # type: ignore[assignment]

    trainer._maybe_eval_periodic(epoch=1)
    assert len(save_calls) == 1
    assert save_calls[0]["metric_value"] == pytest.approx(0.7)

    log = (tmp_path / "train_log.jsonl").read_text()
    eval_records = [
        json.loads(line) for line in log.splitlines()
        if line.strip() and json.loads(line).get("event") == "eval"
    ]
    assert len(eval_records) == 1
    assert eval_records[0]["metrics"] == {"tool_name_acc": 0.7}
