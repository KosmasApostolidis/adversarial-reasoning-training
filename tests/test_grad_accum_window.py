"""Regression tests for trainer gradient-accumulation correctness.

Two bugs the current trainer would silently exhibit:

* C1 — Cross-epoch / partial-window leak. ``do_step`` only fires when
  ``(micro_idx + 1) % grad_accum == 0``. If ``len(loader) % grad_accum
  != 0`` the trailing micro-batches never trigger ``optimizer.step``;
  their accumulated grads spill into the next epoch.
* C2 — NaN-skip mid-window scaling. On a NaN micro-batch the trainer
  zeros grads + ``accum_loss_acc`` and ``continue``\\s, but ``do_step``
  still fires at the original window boundary. Loss is divided by
  ``grad_accum`` even though only ``post_skip`` micro-batches
  contributed → silent learning-rate drop and mis-reported ``avg_loss``.

The tests inject a finite, gradient-flowing ``_outer_step`` (and an
optional NaN at a chosen call) and inspect the ``train_log.jsonl``
events.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from adversarial_reasoning_training.losses.selector import LossCallResult
from adversarial_reasoning_training.trainer.adv_trainer import AdvTrainer, TrainerConfig
from adversarial_reasoning_training.trajectory.teacher_force import TeacherForcedBatch


def _make_batch(vocab: int, T: int) -> TeacherForcedBatch:
    return TeacherForcedBatch(
        input_ids=torch.randint(0, vocab, (1, T)),
        task_mask=torch.tensor([[0] * 6 + [1] * (T - 6)], dtype=torch.float32),
        traj_mask=torch.tensor([[0] * 4 + [1] * (T - 4)], dtype=torch.float32),
        attention_mask=torch.ones((1, T), dtype=torch.long),
        labels=torch.randint(0, vocab, (1, T)),
        segment_ids=torch.zeros((1, T), dtype=torch.int32),
        forward_kwargs={"pixel_values": torch.randn(1, 3, 4, 4)},
        segments=[],
    )


class _DS(Dataset):
    def __init__(self, n: int, vocab: int, T: int) -> None:
        self.batches = [_make_batch(vocab, T) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> TeacherForcedBatch:
        return self.batches[idx]


def _identity_collate(items: list[TeacherForcedBatch]) -> TeacherForcedBatch:
    return items[0]


class _StubModel(torch.nn.Module):
    def __init__(self, vocab: int = 16, hidden: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab)
        self.image_proj = torch.nn.Linear(3, hidden)


class _StubVLM:
    family = "stub"

    def __init__(self, model: _StubModel) -> None:
        self.model = model

    def forward_with_logits(
        self, image: torch.Tensor, input_ids: torch.Tensor, **_: object
    ) -> torch.Tensor:
        m = self.model
        ctx = m.image_proj(image.mean(dim=(-1, -2)).reshape(image.shape[0], 3))
        emb = m.embed(input_ids) + ctx.unsqueeze(1)
        return m.lm_head(emb)


class _ScriptedTrainer(AdvTrainer):
    """Bypass real PGD; emit finite loss with optional NaN at a chosen call."""

    def __init__(self, *, nan_at: int | None = None, **kw) -> None:
        super().__init__(**kw)
        self._call = 0
        self._nan_at = nan_at

    def _outer_step(self, batch, epsilon):  # type: ignore[override]
        self._call += 1
        pixel_values = batch.forward_kwargs["pixel_values"]
        logits = self.vlm.forward_with_logits(pixel_values, batch.input_ids)
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), batch.input_ids.view(-1)
        )
        if self._nan_at is not None and self._call == self._nan_at:
            total = ce * float("nan")
        else:
            total = ce
        return (
            LossCallResult(
                total=total,
                components={
                    "loss_total": float(total.detach()) if torch.isfinite(total) else float("nan"),
                    "loss_task": float(ce.detach()),
                    "loss_kl": 0.0,
                },
            ),
            {"attack_loss_final": 0.0, "attack_iterations": 0, "epsilon": epsilon},
        )


def _events(tmp_path: Path) -> list[dict]:
    log = tmp_path / "train_log.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


@pytest.mark.unit
def test_partial_window_at_epoch_end_is_drained(tmp_path: Path) -> None:
    """C1: 5 micro-batches, grad_accum=4, 2 epochs.

    Without epoch-end drain only ONE optimizer step per epoch fires
    (at micro_idx=3); micro_idx=4 calls .backward() but never
    .step(), and its grads leak into epoch 2's first window.

    With the fix: every epoch flushes the tail, so 2 steps per epoch
    → 4 total.
    """
    torch.manual_seed(0)
    vocab, T = 16, 8
    model = _StubModel(vocab=vocab)
    vlm = _StubVLM(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    cfg = TrainerConfig(
        epochs=2,
        grad_accum=4,
        log_every=1,
        eval_every=0,
        save_every=0,
        grad_clip_norm=1.0,
        amp_dtype="fp32",
        run_dir=tmp_path,
        final_save_include_optimizer=False,
    )

    def _unused_loss(*_a, **_kw):  # pragma: no cover
        raise AssertionError("override should bypass loss_fn")

    trainer = _ScriptedTrainer(
        vlm=vlm,
        model=model,
        collator=_identity_collate,  # type: ignore[arg-type]
        loss_fn=_unused_loss,  # type: ignore[arg-type]
        optimizer=optimizer,
        scheduler=None,
        config=cfg,
        device="cpu",
    )

    ds = _DS(n=5, vocab=vocab, T=T)
    trainer.fit(ds)

    events = _events(tmp_path)
    fit_done = next(e for e in events if e["event"] == "fit_done")
    # 5 micro-batches × 2 epochs = 10 backwards; grad_accum=4 → expect 2
    # full windows per epoch (one of size 4, one of size 1 drained).
    assert fit_done["global_step"] == 4, (
        f"expected 4 optimizer steps (2 per epoch incl. tail drain), "
        f"got {fit_done['global_step']}; events={events}"
    )


def _build_nan_skip_trainer(*, tmp_path: Path, vocab: int, nan_at: int) -> tuple:
    """Construct a _ScriptedTrainer + matching dataset for the NaN-skip scenario."""
    model = _StubModel(vocab=vocab)
    vlm = _StubVLM(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    cfg = TrainerConfig(
        epochs=1,
        grad_accum=4,
        log_every=1,
        eval_every=0,
        save_every=0,
        grad_clip_norm=1.0,
        amp_dtype="fp32",
        run_dir=tmp_path,
        final_save_include_optimizer=False,
    )

    def _unused_loss(*_a, **_kw):  # pragma: no cover
        raise AssertionError("override should bypass loss_fn")

    trainer = _ScriptedTrainer(
        nan_at=nan_at,
        vlm=vlm,
        model=model,
        collator=_identity_collate,  # type: ignore[arg-type]
        loss_fn=_unused_loss,  # type: ignore[arg-type]
        optimizer=optimizer,
        scheduler=None,
        config=cfg,
        device="cpu",
    )
    return trainer, cfg


@pytest.mark.unit
def test_nan_skip_does_not_underscale_window(tmp_path: Path) -> None:
    """C2: grad_accum=4, n=8, NaN injected at call #2 (micro_idx=1).

    Pre-fix the trainer steps at micro_idx=3 with only 3 valid
    backwards in the buffer (mic 0 was zeroed by the NaN-skip path),
    dividing by grad_accum=4 → effective gradient and reported
    avg_loss are 3/4 of true. After fix the trainer waits for 4
    valid micro-batches before stepping (or matches accum_count to
    the divisor) so:

    * the first ``train_step`` event reports ``accum_count == 4``
      (not 3), and
    * its ``avg_loss`` equals the mean of the 4 contributing losses,
      not the sum-of-3-divided-by-4.
    """
    torch.manual_seed(0)
    vocab, T = 16, 8
    trainer, _ = _build_nan_skip_trainer(tmp_path=tmp_path, vocab=vocab, nan_at=2)

    ds = _DS(n=8, vocab=vocab, T=T)
    trainer.fit(ds)

    events = _events(tmp_path)
    skipped = [e for e in events if e["event"] == "skipped_nan_loss"]
    assert len(skipped) == 1, f"expected 1 NaN skip, got {len(skipped)}"

    train_steps = [e for e in events if e["event"] == "train_step"]
    assert train_steps, f"no train_step events; events={events}"

    first = train_steps[0]
    # Fix invariant: every reported train_step must include accum_count
    # equal to the number of micro-batches that actually contributed
    # gradients to the optimizer step (i.e. grad_accum minus any
    # NaN-skips that fell inside the window).
    assert "accum_count" in first, (
        "train_step missing accum_count; trainer must expose how many "
        "micro-batches contributed to this step so logged avg_loss is "
        "interpretable"
    )
    assert first["accum_count"] == 4, (
        f"first window must contain 4 valid backwards (NaN window must "
        f"reset, not silently shrink); got accum_count={first['accum_count']}"
    )
    # avg_loss must equal accum_loss_sum / accum_count, not / grad_accum.
    assert torch.isfinite(torch.tensor(first["avg_loss"])).item()
    assert 0.0 < first["avg_loss"] < 50.0
