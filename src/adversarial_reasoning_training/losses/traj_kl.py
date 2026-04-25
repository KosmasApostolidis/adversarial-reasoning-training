"""Per-position KL divergence between clean and adversarial output distributions.

Used as the trajectory-consistency term of TRADES:
    L_kl = KL( softmax(f(x)/T)[M_traj]  ‖  softmax(f(x_adv)/T)[M_traj] )

The standard distillation trick is to multiply the KL by T² so that
gradients scale comparably to the raw-temperature CE term regardless of
the temperature choice.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def traj_kl(
    logits_clean: torch.Tensor,
    logits_adv: torch.Tensor,
    traj_mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL(p_clean ‖ p_adv) averaged over positions weighted by `traj_mask`.

    Parameters
    ----------
    logits_clean, logits_adv : FloatTensor [B, T, V].
    traj_mask : FloatTensor [B, T]. Positions to score (shifted left by 1
        internally so that position i uses `traj_mask[:, i+1]`).
    temperature : softmax temperature.

    Returns
    -------
    scalar tensor.
    """
    if logits_clean.shape != logits_adv.shape:
        raise ValueError(
            f"logits shape mismatch: {logits_clean.shape} vs {logits_adv.shape}"
        )
    shift_clean = logits_clean[:, :-1, :].contiguous() / temperature
    shift_adv = logits_adv[:, :-1, :].contiguous() / temperature
    shift_mask = traj_mask[:, 1:].contiguous()

    log_p_clean = F.log_softmax(shift_clean, dim=-1)
    log_p_adv = F.log_softmax(shift_adv, dim=-1)
    p_clean = log_p_clean.exp()
    kl_per_pos = (p_clean * (log_p_clean - log_p_adv)).sum(dim=-1)
    # NaN-safe masking: NaN*0=NaN, so zero out non-mask positions via `where`,
    # then nan_to_num scrubs ±inf produced by log_softmax overflow under bf16.
    kl_per_pos = torch.nan_to_num(kl_per_pos, nan=0.0, posinf=0.0, neginf=0.0)
    mask_b = shift_mask.bool()
    kl_masked = torch.where(mask_b, kl_per_pos, kl_per_pos.new_zeros(()))
    weighted = kl_masked * (temperature * temperature)
    denom = shift_mask.sum().clamp_min(1.0)
    return weighted.sum() / denom
