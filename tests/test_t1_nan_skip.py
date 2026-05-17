"""Regression test: T1 clean-FT must skip a NaN micro-batch like AdvTrainer.

Pre-fix ``gates/T1_clean.py:_t1_train_step`` ran ``loss.backward()`` with
no ``torch.isfinite(loss)`` guard. A single degenerate batch (e.g.
all-zero ``task_mask`` → ``task_ce`` returns ``sum/0 = nan`` per its own
"deliberately NaN on degenerate batch" contract) propagated NaN through
gradients into optimizer state, then every subsequent step had NaN loss
and the gate failed with no obvious cause in the log.

This test exercises ``_t1_train_step`` directly with a degenerate
all-zero ``task_mask`` batch and asserts:

  * ``loss_val`` returned is NaN (signals the skip),
  * the model parameter is unchanged (no NaN write),
  * ``micro`` counter resets to 0 so the next valid micro-batches form
    a fresh accumulation window (matches AdvTrainer's drain semantics).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from adversarial_reasoning_training.gates.T1_clean import _t1_train_step
from adversarial_reasoning_training.trajectory.teacher_force import TeacherForcedBatch


def _make_degenerate_batch(vocab: int = 8, seq_len: int = 6) -> TeacherForcedBatch:
    """Build a batch whose ``task_mask`` sums to zero so ``task_ce`` returns NaN."""
    return TeacherForcedBatch(
        input_ids=torch.randint(0, vocab, (1, seq_len)),
        task_mask=torch.zeros((1, seq_len), dtype=torch.float32),
        traj_mask=torch.zeros((1, seq_len), dtype=torch.float32),
        attention_mask=torch.ones((1, seq_len), dtype=torch.long),
        labels=torch.full((1, seq_len), -100, dtype=torch.long),
        segment_ids=torch.zeros((1, seq_len), dtype=torch.int32),
        forward_kwargs={"pixel_values": torch.randn(1, 3, 4, 4)},
        segments=[],
    )


class _StubVLM:
    """Minimal VLM stub with a single trainable parameter so backward
    has somewhere to flow if the guard is missing."""

    def __init__(self, vocab: int) -> None:
        self.model = torch.nn.Linear(vocab, vocab, bias=False)

    def forward_with_logits(
        self, image: torch.Tensor, input_ids: torch.Tensor, **_: Any
    ) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=self.model.in_features).float()
        return self.model(one_hot)


@pytest.mark.unit
def test_t1_train_step_skips_nan_loss_and_preserves_weights() -> None:
    torch.manual_seed(0)
    vocab = 8
    vlm = _StubVLM(vocab=vocab)
    optimizer = torch.optim.SGD(vlm.model.parameters(), lr=1.0)  # large lr makes NaN obvious
    batch = _make_degenerate_batch(vocab=vocab, seq_len=6)

    before = vlm.model.weight.detach().clone()

    # grad_accum=1 means the step would normally fire on this micro-batch.
    step, micro, loss_val = _t1_train_step(
        vlm=vlm,
        batch=batch,
        model=vlm.model,
        optimizer=optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        amp_dtype=torch.float32,
        grad_accum=1,
        step=0,
        micro=0,
    )

    # Loss must surface as NaN so the gate operator sees the skip in
    # ``train_log.jsonl``; the value itself is the only signal.
    assert loss_val != loss_val, f"expected NaN loss_val, got {loss_val!r}"
    # Optimizer must NOT have stepped — weights unchanged. Pre-fix the
    # NaN flowed through backward+step and corrupted every weight.
    after = vlm.model.weight.detach()
    assert torch.equal(before, after), (
        "weights mutated by NaN backward; "
        "_t1_train_step did not skip the NaN micro-batch"
    )
    # No finite gradient should remain attached; the next call starts
    # with a clean accumulation window.
    for p in vlm.model.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
    # Step counter must not have advanced (no optimizer.step happened).
    assert step == 0
    # Micro counter must reset so the next valid micro-batches form a
    # full grad_accum window from scratch (matches AdvTrainer semantics).
    assert micro == 0
