"""Parameter-freeze helpers keyed on module-name patterns."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FreezeConfig:
    strategy: str = "none"  # none | vit_only | projector_only | lm_only


# Module-name substrings per component. Used with `any(p in name for p in patterns)`.
_VIT_PATTERNS = ("visual", "vision_model", "vision_tower", "image_encoder", "patch_embed")
_PROJECTOR_PATTERNS = ("mm_projector", "multi_modal_projector", "visual_projector", "merger")
_LM_PATTERNS = ("language_model", "model.layers", "lm_head", "model.embed_tokens")

# Strategy → patterns to freeze. Missing key (incl. "none") leaves all params trainable.
_FREEZE_PATTERNS_BY_STRATEGY: dict[str, tuple[str, ...]] = {
    "vit_only": _VIT_PATTERNS,
    "projector_only": _PROJECTOR_PATTERNS,
    "lm_only": _LM_PATTERNS,
}


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(p in name for p in patterns)


def apply_freeze(model: torch.nn.Module, config: FreezeConfig) -> dict[str, int]:
    """Apply the freeze strategy in-place. Returns a summary of param counts.

    strategy:
        - ``none``             : everything trainable (full FT).
        - ``vit_only``         : freeze only the vision backbone.
        - ``projector_only``   : freeze only the multimodal projector.
        - ``lm_only``          : freeze only the language model stack.
    """
    freeze_patterns = _FREEZE_PATTERNS_BY_STRATEGY.get(config.strategy, ())
    counts = {"trainable": 0, "frozen": 0}
    for name, p in model.named_parameters():
        should_freeze = bool(freeze_patterns) and _matches(name, freeze_patterns)
        p.requires_grad_(not should_freeze)
        bucket = "frozen" if should_freeze else "trainable"
        counts[bucket] += p.numel()
    return counts


def param_groups_by_role(
    model: torch.nn.Module,
    lr_lm: float,
    lr_projector: float,
    lr_vit: float,
    weight_decay: float = 0.0,
) -> list[dict]:
    """Build per-role param groups for the optimizer.

    Parameters whose names don't match any pattern fall into the "lm"
    group by default (model stacks often use generic names like
    `model.xxx` for the LM body).
    """
    groups = {"vit": [], "projector": [], "lm": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Test projector patterns BEFORE vit: Qwen2.5/3-VL names its
        # multimodal projector ``visual.merger.*``, which matches both the
        # projector pattern (``merger``) and the vit pattern (``visual``).
        # vit-first would mis-bucket the merger into the vit group — training
        # it at lr_vit (10x off) and leaving the projector group empty, which
        # also blanks the T0 gate's projector grad-norm signal. Projector
        # patterns are the more specific match, so they win. Disjoint for
        # LLaVA (multi_modal_projector / vision_tower) and InternVL, so this
        # ordering is a no-op there.
        if _matches(name, _PROJECTOR_PATTERNS):
            groups["projector"].append(p)
        elif _matches(name, _VIT_PATTERNS):
            groups["vit"].append(p)
        else:
            groups["lm"].append(p)
    pg: list[dict] = []
    if groups["vit"]:
        pg.append({"params": groups["vit"], "lr": lr_vit, "weight_decay": weight_decay})
    if groups["projector"]:
        pg.append({"params": groups["projector"], "lr": lr_projector, "weight_decay": weight_decay})
    if groups["lm"]:
        pg.append({"params": groups["lm"], "lr": lr_lm, "weight_decay": weight_decay})
    return pg
