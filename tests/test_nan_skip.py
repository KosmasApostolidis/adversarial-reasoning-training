"""Regression test: trainer must skip a NaN micro-batch and continue.

History (commit 6e98105 era): a single NaN loss at step 2 propagated
into optimizer state and poisoned every subsequent step. This test
injects NaN on call #2 of the loss path and verifies:

  - training continues past the NaN step
  - ``skipped_nan_loss`` event is logged
  - params are unchanged across the NaN step (zero_grad cleared grads)
  - final ckpt is written via save_every cadence
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
        task_mask=torch.tensor([[0] * 6 + [1] * 6], dtype=torch.float32),
        traj_mask=torch.tensor([[0] * 4 + [1] * 8], dtype=torch.float32),
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
    # micro_batch=1 only; just unwrap.
    return items[0]


class _StubModel(torch.nn.Module):
    def __init__(self, vocab: int = 16, hidden: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab)
        self.image_proj = torch.nn.Linear(3, hidden)


class _StubVLM:
    """Non-Module wrapper so assigning `vlm.model = stub` doesn't recursively
    register the stub as its own submodule (the recursion crashes
    state_dict serialization in ckpt save)."""

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


class _NanInjectingTrainer(AdvTrainer):
    """Override `_outer_step` to bypass PGD and inject NaN on call #nan_at."""

    def __init__(self, *, nan_at: int, **kw) -> None:
        super().__init__(**kw)
        self._call = 0
        self._nan_at = nan_at

    def _outer_step(self, batch, epsilon):  # type: ignore[override]
        self._call += 1
        # finite forward+loss using the stub model; gradient flows so
        # backward() is non-trivial on the non-NaN path.
        pixel_values = batch.forward_kwargs["pixel_values"]
        logits = self.vlm.forward_with_logits(pixel_values, batch.input_ids)
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), batch.input_ids.view(-1)
        )
        if self._call == self._nan_at:
            total = ce * float("nan")
        else:
            total = ce
        return (
            LossCallResult(
                total=total,
                components={"loss_total": float(total.detach()), "loss_task": float(ce.detach()), "loss_kl": 0.0},
            ),
            {"attack_loss_final": 0.0, "attack_iterations": 0, "epsilon": epsilon},
        )


def _build_nan_injecting_trainer(
    *, tmp_path: Path, vocab: int, nan_at: int,
) -> _NanInjectingTrainer:
    """Build a _NanInjectingTrainer with the trainer config shared by this test."""
    model = _StubModel(vocab=vocab)
    vlm = _StubVLM(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    cfg = TrainerConfig(
        epochs=1,
        grad_accum=1,
        log_every=1,
        eval_every=0,
        save_every=3,
        grad_clip_norm=1.0,
        amp_dtype="fp32",
        run_dir=tmp_path,
    )

    def _unused_loss(*_a, **_kw):  # pragma: no cover
        raise AssertionError("override should bypass loss_fn")

    return _NanInjectingTrainer(
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


@pytest.mark.unit
def test_trainer_skips_nan_and_continues(tmp_path: Path) -> None:
    torch.manual_seed(0)
    vocab, T = 16, 8
    trainer = _build_nan_injecting_trainer(tmp_path=tmp_path, vocab=vocab, nan_at=2)
    trainer.fit(_DS(n=4, vocab=vocab, T=T))

    # Inspect the JSONL training log for the skipped event.
    log_path = tmp_path / "train_log.jsonl"
    assert log_path.exists()
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    skipped = [e for e in events if e["event"] == "skipped_nan_loss"]
    assert len(skipped) == 1, f"expected 1 skipped_nan_loss event, got {len(skipped)}: {events}"
    assert skipped[0]["micro_idx"] == 1  # call #2 → 0-indexed micro_idx 1

    # Trainer must have completed (fit_done event present).
    fit_done = [e for e in events if e["event"] == "fit_done"]
    assert fit_done, f"trainer crashed; events={events}"
    # Three optimizer steps: calls 1, 3, 4 (call 2 was skipped).
    assert fit_done[0]["global_step"] == 3

    # Final ckpt written.
    ckpt_files = list((tmp_path / "ckpt").glob("step*.pt"))
    assert ckpt_files, f"no ckpt *.pt in {tmp_path / 'ckpt'}"
    # All saved params must be finite (sanity: NaN never reached optim).
    payload = torch.load(ckpt_files[0], map_location="cpu", weights_only=False)
    for k, v in payload["model_state_dict"].items():
        assert torch.isfinite(v).all(), f"NaN found in saved param {k}"
