"""End-to-end smoke test of one outer step using a CPU stub VLM.

This test must complete in <30s on CPU so the full pytest suite stays
under a minute. It does NOT exercise real PGD attack convergence; it
proves the loop wiring (collator -> inner attack -> losses -> backward
-> optimizer.step) is type-correct and gradient-flowing on a tiny VLM
substitute.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch


class _StubVLM(torch.nn.Module):
    """Tiny linear model that mimics ``forward_with_logits`` API."""

    def __init__(self, vocab: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab)
        self.image_proj = torch.nn.Linear(3, hidden)
        self.model = self  # AdvTrainer.optimizer expects vlm.model.parameters()
        self.family = "stub"

    def forward_with_logits(
        self, image: torch.Tensor, input_ids: torch.Tensor, **_: object
    ) -> torch.Tensor:
        # Squash image to a single hidden vector and add to embeddings
        ctx = self.image_proj(image.mean(dim=(-1, -2)).reshape(image.shape[0], 3))
        emb = self.embed(input_ids) + ctx.unsqueeze(1)
        return self.lm_head(emb)


@dataclass
class _Batch:
    input_ids: torch.Tensor
    task_mask: torch.Tensor
    traj_mask: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    segment_ids: torch.Tensor
    forward_kwargs: dict
    segments: list

    def to(self, device):
        return self


@pytest.mark.unit
def test_outer_step_on_cpu_stub() -> None:
    from adversarial_reasoning_training.losses.selector import build_loss, from_cfg_dict

    torch.manual_seed(0)
    vocab, T = 32, 12
    vlm = _StubVLM(vocab=vocab)
    batch = _Batch(
        input_ids=torch.randint(0, vocab, (1, T)),
        task_mask=torch.tensor([[0] * 6 + [1] * 6], dtype=torch.float32),
        traj_mask=torch.tensor([[0] * 4 + [1] * 8], dtype=torch.float32),
        attention_mask=torch.ones((1, T), dtype=torch.long),
        labels=torch.randint(0, vocab, (1, T)),
        segment_ids=torch.zeros((1, T), dtype=torch.int32),
        forward_kwargs={"pixel_values": torch.randn(1, 3, 4, 4)},
        segments=[],
    )

    loss_fn = build_loss(from_cfg_dict({
        "defense": "trades",
        "trades": {"beta_start": 4.0, "beta_end": 4.0, "temperature": 2.0},
    }))
    optimizer = torch.optim.SGD(vlm.parameters(), lr=1e-2)

    pixel_values = batch.forward_kwargs["pixel_values"]
    x_adv = pixel_values + 0.01 * torch.randn_like(pixel_values)

    optimizer.zero_grad()
    logits_clean = vlm.forward_with_logits(pixel_values, batch.input_ids)
    logits_adv = vlm.forward_with_logits(x_adv, batch.input_ids)
    out = loss_fn(logits_clean, logits_adv, batch.input_ids, batch.task_mask, batch.traj_mask)
    out.total.backward()
    optimizer.step()

    assert torch.isfinite(out.total)
    grad_norm = sum(
        p.grad.detach().pow(2).sum().item() for p in vlm.parameters() if p.grad is not None
    ) ** 0.5
    assert grad_norm > 0.0
