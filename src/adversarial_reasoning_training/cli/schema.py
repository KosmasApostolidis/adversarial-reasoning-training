"""Fail-fast schema validation for top-level YAML configs.

Run at the start of every CLI entry-point so a typo in
``configs/training.yaml`` cannot survive past the first ``art-train``
invocation. Each ``validate_*`` raises ``ValueError`` with the offending
config label, the bad key, and the valid set so the failure message is
actionable without grepping the source.

Validators are intentionally dataclass-free: a ``ValueError`` with a
clear message is more useful than a ``ValidationError`` from a third-
party library, and avoiding pydantic keeps the production dependency
set minimal.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import NotRequired, TypedDict

from ..attacks.inner_pgd import validate_eps_schedule

VALID_DEFENSES = frozenset({"trades", "pgd_at", "oaat"})
VALID_OPTIMS = frozenset({"adamw8bit", "adamw", "adamw_fused"})
VALID_SCHEDULES = frozenset({"cosine", "linear", "constant"})
VALID_AMPS = frozenset({"bf16", "fp16", "fp32"})
VALID_FREEZE_STRATEGIES = frozenset(
    {"none", "vit_only", "projector_only", "lm_only"}
)

TRAINING_REQUIRED_KEYS = frozenset(
    {
        "defense", "seed", "epochs", "micro_batch", "grad_accum",
        "optim", "lr", "schedule", "amp",
    }
)
TRAINING_LR_REQUIRED_KEYS = frozenset({"lm", "projector", "vit"})

DEFENSES_PGD_REQUIRED_KEYS = frozenset(
    {"eps_schedule", "default_eps", "alpha_ratio", "steps"}
)

DATA_REQUIRED_KEYS = frozenset(
    {"task_id", "train_split", "dev_split", "test_split"}
)

GOLD_REQUIRED_KEYS = frozenset({"oracle_version", "cache_dir"})

FULL_FT_REQUIRED_KEYS = frozenset({"freeze_strategy", "memory"})


class _LRSpec(TypedDict):
    lm: float
    projector: float
    vit: float


class TrainingConfig(TypedDict):
    """Type-narrowed view of a validated ``training.yaml`` payload.

    Pilot of the broader TypedDict-config rollout flagged in the Phase 4
    plan. Only the keys ``validate_training`` actually checks are marked
    required; everything else (``warmup_pct``, ``log_every``, etc.) is
    ``NotRequired`` so that older configs lacking those knobs still
    type-narrow without runtime errors. Optional entries continue to
    carry their original ``train_cfg.get(...)`` defaults at the call
    sites in ``cli/train.py``.
    """

    defense: str
    seed: int
    epochs: int
    micro_batch: int
    grad_accum: int
    optim: str
    lr: _LRSpec
    schedule: str
    amp: str
    weight_decay: NotRequired[float]
    betas: NotRequired[list[float]]
    warmup_pct: NotRequired[float]
    grad_ckpt: NotRequired[bool]
    grad_clip_norm: NotRequired[float]
    log_every: NotRequired[int]
    eval_every: NotRequired[int]
    save_every: NotRequired[int]
    ckpt_keep: NotRequired[int]
    metric_for_best: NotRequired[str]
    final_save_include_optimizer: NotRequired[bool]


def _check_missing(label: str, cfg: dict[str, Any], required: frozenset[str]) -> None:
    missing = required - cfg.keys()
    if missing:
        raise ValueError(
            f"{label}: missing required keys {sorted(missing)} "
            f"(got: {sorted(cfg.keys())})"
        )


def _check_enum(label: str, key: str, value: Any, valid: frozenset[str]) -> None:
    if value not in valid:
        raise ValueError(
            f"{label}.{key}={value!r} not in valid set {sorted(valid)}"
        )


def validate_training(cfg: dict[str, Any]) -> TrainingConfig:
    """Validate a ``training.yaml`` payload and narrow its type.

    Catches typos like ``defense: tradse`` before they reach the loss
    selector, and ``optim: adamw_8bit`` before the optimizer factory
    silently falls back to a default kind.

    Returns the same ``cfg`` object (no copy) but annotated as
    :class:`TrainingConfig` so downstream code gets mypy-checked
    subscripts at sites like ``train_cfg["lr"]["lm"]``.
    """
    _check_missing("training", cfg, TRAINING_REQUIRED_KEYS)
    _check_enum("training", "defense", cfg["defense"], VALID_DEFENSES)
    _check_enum("training", "optim", cfg["optim"], VALID_OPTIMS)
    _check_enum("training", "schedule", cfg["schedule"], VALID_SCHEDULES)
    _check_enum("training", "amp", cfg["amp"], VALID_AMPS)
    lr = cfg["lr"]
    if not isinstance(lr, dict):
        raise ValueError(
            f"training.lr: expected dict, got {type(lr).__name__}"
        )
    _check_missing("training.lr", lr, TRAINING_LR_REQUIRED_KEYS)
    return cfg  # type: ignore[return-value]


def validate_defenses(cfg: dict[str, Any]) -> None:
    """Validate a ``defenses.yaml`` payload.

    Reuses :func:`validate_eps_schedule` from ``attacks/inner_pgd`` so
    schedule-typo detection lives in one place.
    """
    if "pgd" not in cfg:
        raise ValueError("defenses: missing 'pgd' block")
    pgd = cfg["pgd"]
    if not isinstance(pgd, dict):
        raise ValueError(
            f"defenses.pgd: expected dict, got {type(pgd).__name__}"
        )
    _check_missing("defenses.pgd", pgd, DEFENSES_PGD_REQUIRED_KEYS)
    validate_eps_schedule(pgd.get("eps_schedule"))


def validate_data(cfg: dict[str, Any]) -> None:
    """Validate a ``data.yaml`` payload."""
    _check_missing("data", cfg, DATA_REQUIRED_KEYS)


def validate_gold(cfg: dict[str, Any]) -> None:
    """Validate a ``gold.yaml`` payload."""
    _check_missing("gold", cfg, GOLD_REQUIRED_KEYS)


def validate_full_ft(cfg: dict[str, Any]) -> None:
    """Validate a ``full_ft.yaml`` payload."""
    _check_missing("full_ft", cfg, FULL_FT_REQUIRED_KEYS)
    _check_enum(
        "full_ft", "freeze_strategy", cfg["freeze_strategy"],
        VALID_FREEZE_STRATEGIES,
    )
