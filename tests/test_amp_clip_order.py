"""Under fp16 AMP, the trainer must call ``scaler.unscale_`` BEFORE
``clip_grad_norm_``.

PyTorch's GradScaler scales gradients up so fp16 backward stays numerically
stable. Clipping before unscale operates on the scaled magnitudes — i.e.
``max_norm`` is silently inflated by the loss scale, miscalibrating the
configured ``grad_clip_norm`` by orders of magnitude. The PyTorch contract
is ``scale → backward → unscale_ → clip → step → update``.

We monkeypatch both calls to a shared ledger and assert the ordering on
one optimizer step under fp16. The same trainer must also call
``scaler.update()`` on the NaN-grad skip path so the loss scale halves on
inf instead of getting stuck at the pre-skip value.
"""

from __future__ import annotations

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
        task_mask=torch.tensor([[0] * 4 + [1] * 4], dtype=torch.float32),
        traj_mask=torch.tensor([[0] * 2 + [1] * 6], dtype=torch.float32),
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


def _identity_collate(items):
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

    def forward_with_logits(self, image, input_ids, **_):
        m = self.model
        ctx = m.image_proj(image.mean(dim=(-1, -2)).reshape(image.shape[0], 3))
        emb = m.embed(input_ids) + ctx.unsqueeze(1)
        return m.lm_head(emb)


class _OrderRecordingTrainer(AdvTrainer):
    """Bypass PGD; emit a finite loss so optimizer.step is reached."""

    def _outer_step(self, batch, epsilon):  # type: ignore[override]
        pixel_values = batch.forward_kwargs["pixel_values"]
        logits = self.vlm.forward_with_logits(pixel_values, batch.input_ids)
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), batch.input_ids.view(-1)
        )
        return (
            LossCallResult(
                total=ce,
                components={"loss_total": float(ce.detach()),
                            "loss_task": float(ce.detach()), "loss_kl": 0.0},
            ),
            {"attack_loss_final": 0.0, "attack_iterations": 0,
             "attack_diverged": 0, "epsilon": epsilon},
        )


@pytest.mark.unit
def test_unscale_called_before_clip_under_fp16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    vocab, T = 16, 8
    model = _StubModel(vocab=vocab)
    vlm = _StubVLM(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    cfg = TrainerConfig(
        epochs=1, grad_accum=1, log_every=1, eval_every=0, save_every=0,
        grad_clip_norm=1.0, amp_dtype="fp16", run_dir=tmp_path,
    )

    trainer = _OrderRecordingTrainer(
        vlm=vlm, model=model, collator=_identity_collate,  # type: ignore[arg-type]
        loss_fn=lambda *_a, **_kw: None,  # bypassed
        optimizer=optimizer, scheduler=None, config=cfg, device="cpu",
    )

    ledger: list[str] = []
    real_unscale = trainer.scaler.unscale_

    def _ledger_unscale(opt):
        ledger.append("unscale")
        return real_unscale(opt)

    real_clip = torch.nn.utils.clip_grad_norm_

    def _ledger_clip(*args, **kw):
        ledger.append("clip")
        return real_clip(*args, **kw)

    monkeypatch.setattr(trainer.scaler, "unscale_", _ledger_unscale)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _ledger_clip)

    ds = _DS(n=1, vocab=vocab, T=T)
    trainer.fit(ds)

    # Must have at least one unscale event and the first one must precede
    # the first clip event in the recorded order.
    assert "unscale" in ledger and "clip" in ledger, f"missing events: {ledger}"
    first_unscale = ledger.index("unscale")
    first_clip = ledger.index("clip")
    assert first_unscale < first_clip, (
        f"clip must follow unscale; got order={ledger}"
    )
