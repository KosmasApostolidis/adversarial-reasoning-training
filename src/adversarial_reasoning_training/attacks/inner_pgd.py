"""Inner PGD attack: thin wrapper over attacks-repo `PGDAttack`.

Crafts `x_adv` from a clean image tensor such that the teacher-forced
CE on the gold trajectory is MAXIMISED. The outer trainer then
minimises `task_ce(clean) + β·KL(clean‖adv)` (or the chosen defense
variant), closing the inner-max / outer-min TRADES loop.

The attacks-repo `PGDAttack._loss` already implements teacher-forced
CE against a `target` token sequence at the correct causal-LM offsets
(src/adversarial_reasoning/attacks/pgd.py:113). We reuse it as-is: our
`target` is the concatenation of all task-masked token positions, and
`prompt_tokens` is the prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from adversarial_reasoning.attacks.base import AttackResult
from adversarial_reasoning.attacks.pgd import PGDAttack

from ..trajectory.teacher_force import TeacherForcedBatch


@dataclass(frozen=True)
class InnerPgdConfig:
    epsilon: float = 4.0 / 255.0
    alpha_ratio: float = 0.25
    steps: int = 7
    random_restarts: int = 1


def _split_prompt_target(
    batch: TeacherForcedBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split input_ids into (prompt_tokens, target_tokens, target_mask).

    "Prompt" = everything up to the first task-masked position. This
    matches attacks-repo `PGDAttack._loss` teacher-forced convention.

    If no task-masked position exists (degenerate), raises.
    """
    task_mask = batch.task_mask[0]  # [T]
    input_ids = batch.input_ids[0]  # [T]
    nonzero = (task_mask > 0).nonzero(as_tuple=False).flatten()
    if nonzero.numel() == 0:
        raise ValueError("No task-masked positions in batch; cannot run inner PGD.")
    first = int(nonzero[0].item())
    prompt_tokens = input_ids[:first].unsqueeze(0)
    target_tokens = input_ids[first:].unsqueeze(0)
    target_mask = task_mask[first:].unsqueeze(0)
    return prompt_tokens, target_tokens, target_mask


def run_inner_pgd(
    vlm: Any,
    image_tensor: torch.Tensor,
    batch: TeacherForcedBatch,
    config: InnerPgdConfig,
) -> AttackResult:
    """Run PGD-L∞ on `image_tensor` using teacher-forced CE as objective.

    image_tensor must be in pixel-value domain [0, 1] and gradient-
    trackable with respect to `vlm.forward_with_logits`. The VLM's
    forward composes its own normalization, so grads propagate back to
    the pixel domain (attacks-repo design note).
    """
    prompt_tokens, target_tokens, _ = _split_prompt_target(batch)
    attack = PGDAttack(
        name="pgd_linf_inner",
        epsilon=config.epsilon,
        alpha=config.epsilon * config.alpha_ratio,
        steps=config.steps,
        random_restarts=config.random_restarts,
        targeted=False,
    )
    # Strip pixel_values from forward_kwargs: attacks-repo PGDAttack passes
    # `image` positionally to vlm.forward_with_logits, so including
    # pixel_values in kwargs would double-pass the visual input.
    fwd_kwargs = {
        k: v
        for k, v in batch.forward_kwargs.items()
        if k != "pixel_values" and v is not None
    }
    return attack.run(
        vlm=vlm,
        image=image_tensor,
        prompt_tokens=prompt_tokens,
        target=target_tokens,
        forward_kwargs=fwd_kwargs,
    )


def epsilon_for_epoch(
    epoch: int, schedule: list[dict[str, Any]], default_eps: float = 4.0 / 255.0
) -> float:
    """Resolve the ε value for the current epoch from the YAML schedule.

    Each schedule entry has ``{"epoch_range": [lo, hi], "eps": <float>}``.
    If no entry matches, returns `default_eps`.
    """
    for entry in schedule or []:
        lo, hi = entry["epoch_range"]
        if lo <= epoch <= hi:
            return float(entry["eps"])
    return default_eps
