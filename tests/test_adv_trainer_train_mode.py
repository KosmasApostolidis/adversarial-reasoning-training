"""Regression test: AdvTrainer.outer step must train in ``model.train(True)``.

Pre-fix ``trainer/adv_trainer.py:_outer_step`` switched the model to
``train(False)`` for the PGD craft step (correct — PGD wants a
deterministic forward) but never restored the mode before the outer
forward+backward. ``fit()`` and ``_run_epoch`` never call ``.train(True)``
either, so the entire adv-FT run executed with dropout disabled and any
train-mode-only modules silently switched off.

This test asserts the model is in training mode at the moment ``loss_fn``
runs inside ``_outer_step``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from adversarial_reasoning_training.losses.selector import LossCallResult, LossConfig
from adversarial_reasoning_training.trainer.adv_trainer import AdvTrainer, TrainerConfig
from adversarial_reasoning_training.trajectory.teacher_force import TeacherForcedBatch


class _StubModel(torch.nn.Module):
    def __init__(self, vocab: int = 8, hidden: int = 4) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab)
        self.image_proj = torch.nn.Linear(3, hidden)


class _StubVLM:
    family = "stub"

    def __init__(self, model: _StubModel) -> None:
        self.model = model

    def forward_with_logits(
        self, image: torch.Tensor, input_ids: torch.Tensor, **_: Any
    ) -> torch.Tensor:
        ctx = self.model.image_proj(image.mean(dim=(-1, -2)).reshape(image.shape[0], 3))
        emb = self.model.embed(input_ids) + ctx.unsqueeze(1)
        return self.model.lm_head(emb)


def _make_batch(vocab: int, T: int) -> TeacherForcedBatch:
    return TeacherForcedBatch(
        input_ids=torch.randint(0, vocab, (1, T)),
        task_mask=torch.tensor([[0] * 4 + [1] * 4], dtype=torch.float32),
        traj_mask=torch.tensor([[0] * 4 + [1] * 4], dtype=torch.float32),
        attention_mask=torch.ones((1, T), dtype=torch.long),
        labels=torch.randint(0, vocab, (1, T)),
        segment_ids=torch.zeros((1, T), dtype=torch.int32),
        forward_kwargs={"pixel_values": torch.randn(1, 3, 4, 4)},
        segments=[],
    )


@pytest.mark.unit
def test_outer_step_runs_loss_fn_with_model_in_training_mode(tmp_path) -> None:
    """End-to-end assertion: drive the real ``_outer_step`` and verify
    ``model.training`` is True at the moment ``loss_fn`` is invoked.

    Approach: instrument ``loss_fn`` to capture the training-mode flag.
    Stub out the inner PGD via monkey-patching ``run_inner_pgd`` so the
    test runs in <1s on CPU.
    """
    import adversarial_reasoning_training.trainer.adv_trainer as adv_mod
    from adversarial_reasoning_training.attacks.inner_pgd import InnerPgdConfig

    vocab, T = 8, 8
    model = _StubModel(vocab=vocab)
    vlm = _StubVLM(model)
    batch = _make_batch(vocab=vocab, T=T)

    captured: dict[str, bool] = {}

    def _capturing_loss(logits_clean, logits_adv, input_ids, task_mask, traj_mask):
        captured["model_training"] = model.training
        total = logits_clean.sum() * 0.0 + logits_adv.sum() * 0.0  # finite, gradient flows
        return LossCallResult(total=total, components={"loss_total": 0.0})

    _capturing_loss.config = LossConfig(defense="trades")  # type: ignore[attr-defined]

    # Stub PGD to return the clean image immediately — keeps test CPU-cheap.
    from adversarial_reasoning.attacks.base import AttackResult

    def _stub_pgd(vlm_, image, batch_, cfg: InnerPgdConfig):
        img = image.detach().clone()
        return AttackResult(
            perturbed_image=img,
            delta=torch.zeros_like(img),
            loss_final=0.0,
            iterations=1,
        )

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    cfg = TrainerConfig(epochs=1, grad_accum=1, run_dir=tmp_path, amp_dtype="fp32")
    trainer = AdvTrainer(
        vlm=vlm, model=model, collator=lambda x: x[0],  # type: ignore[arg-type]
        loss_fn=_capturing_loss,  # type: ignore[arg-type]
        optimizer=optimizer, scheduler=None, config=cfg, device="cpu",
    )

    # Sanity: model starts in training mode by default.
    assert model.training is True

    # Patch the symbol the trainer module imported, not the one in attacks.
    monkey_target = adv_mod.run_inner_pgd
    adv_mod.run_inner_pgd = _stub_pgd
    try:
        trainer._outer_step(batch, epsilon=0.01)
    finally:
        adv_mod.run_inner_pgd = monkey_target

    assert captured.get("model_training") is True, (
        "model was in eval mode when loss_fn ran inside _outer_step; "
        "B01 fix did not restore train(True) after the PGD craft."
    )
    # And the model is still in training mode after the step returns —
    # the next call must not start in eval mode either.
    assert model.training is True
