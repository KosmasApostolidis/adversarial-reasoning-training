"""Adversarial fine-tune entrypoint.

Usage:
    art-train --config configs/training.yaml \\
        --defenses configs/defenses.yaml --data configs/data.yaml \\
        --gold configs/gold.yaml --full-ft configs/full_ft.yaml \\
        --model qwen3_vl_8b --run-dir runs/<id>

Loads all YAMLs, constructs the attacks-repo VLM + trainer-repo
trainer, and runs ``AdvTrainer.fit``. Everything config-driven; no
hard-coded paths.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from ..gates._common import build_train_dataset, get_collator
from ..losses.selector import build_loss, from_cfg_dict
from ..trainer.adv_trainer import AdvTrainer, TrainerConfig
from ..trainer.freeze import FreezeConfig, apply_freeze
from ..trainer.optim import (
    OptimConfig,
    ScheduleConfig,
    build_optimizer,
    build_scheduler,
)
from ..utils.constants import DEFAULT_PGD_ALPHA_RATIO, EPS_4_255
from .config import load_yaml
from .runtime import setup_seed
from .schema import (
    validate_data,
    validate_defenses,
    validate_full_ft,
    validate_gold,
    validate_training,
)


def _build_vlm(model_family: str, models_yaml: Path) -> Any:
    from adversarial_reasoning.models.loader import load_hf_vlm  # type: ignore

    return load_hf_vlm(model_family, config_path=str(models_yaml))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="art-train",
        description="Adversarial fine-tune entrypoint for medical VLM agents.",
    )
    parser.add_argument("--config", type=Path, required=True, help="training.yaml")
    parser.add_argument("--defenses", type=Path, required=True, help="defenses.yaml")
    parser.add_argument("--data", type=Path, required=True, help="data.yaml")
    parser.add_argument("--gold", type=Path, required=True, help="gold.yaml")
    parser.add_argument("--full-ft", type=Path, required=True, help="full_ft.yaml")
    parser.add_argument("--model", type=str, required=True, help="model family id")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    # Default `None` (not 0) so the run honours the validated `training.seed`
    # YAML key when the operator does not pass --seed. The pipeline always
    # supplies --seed per-run, so its behaviour is unchanged. Pre-fix the CLI
    # default of 0 silently overrode `training.seed: N` from YAML.
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--models-yaml", type=Path, required=True,
        help="Path to attacks-repo models.yaml (no default; the cross-repo "
        "path is environment-dependent).",
    )
    return parser


def _resolve_seed(args: argparse.Namespace, train_cfg: dict[str, Any]) -> int:
    """Pick the seed for ``setup_seed``: CLI override beats YAML.

    ``--seed`` defaults to ``None``; when omitted, fall back to the
    schema-required ``training.seed`` YAML key. Pre-fix, the CLI default of
    0 silently overrode whatever ``training.seed`` declared.
    """
    if args.seed is not None:
        return int(args.seed)
    return int(train_cfg["seed"])


def _load_and_validate_configs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load all five YAMLs and run schema validation.

    Fail-fast pattern: every config validates before any model loads —
    a 7B VLM init is ~30s on H200 and ``adamw_8bit`` would otherwise
    silently fall back to the default optim kind (see
    ``optim.build_optimizer``).
    """
    defense_cfg = load_yaml(args.defenses)
    data_cfg = load_yaml(args.data)
    gold_cfg = load_yaml(args.gold)
    ft_cfg = load_yaml(args.full_ft)
    train_cfg = validate_training(load_yaml(args.config))
    validate_defenses(defense_cfg)
    validate_data(data_cfg)
    validate_gold(gold_cfg)
    validate_full_ft(ft_cfg)
    return train_cfg, defense_cfg, data_cfg, gold_cfg, ft_cfg


def _build_optim_and_schedule(
    model: torch.nn.Module,
    train_cfg: dict[str, Any],
    train_ds_size: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler | None]:
    # ``weight_decay`` and ``betas`` are validated as legal optional YAML
    # keys (cli/schema.py:79-80) but pre-fix this builder ignored both,
    # silently using the OptimConfig dataclass defaults. Thread them so a
    # YAML-declared regularization regime is honoured. Mirror the pattern
    # already used by gates/T1_clean.py:_build_t1_optimization.
    betas = train_cfg.get("betas", list(OptimConfig.betas))
    optim_cfg = OptimConfig(
        kind=train_cfg.get("optim", "adamw8bit"),
        lr_lm=train_cfg["lr"]["lm"],
        lr_projector=train_cfg["lr"]["projector"],
        lr_vit=train_cfg["lr"]["vit"],
        weight_decay=float(train_cfg.get("weight_decay", OptimConfig.weight_decay)),
        betas=(float(betas[0]), float(betas[1])),
    )
    optimizer = build_optimizer(model, optim_cfg)
    total_steps = train_cfg["epochs"] * max(
        1, math.ceil(train_ds_size / train_cfg["grad_accum"])
    )
    scheduler = build_scheduler(
        optimizer,
        ScheduleConfig(
            total_steps=total_steps,
            warmup_pct=train_cfg.get("warmup_pct", 0.03),
            kind=train_cfg.get("schedule", "cosine"),
        ),
    )
    return optimizer, scheduler


def _build_trainer_config(
    args: argparse.Namespace,
    train_cfg: dict[str, Any],
    defense_cfg: dict[str, Any],
) -> TrainerConfig:
    pgd_cfg = defense_cfg.get("pgd", {})
    return TrainerConfig(
        epochs=train_cfg["epochs"],
        grad_accum=train_cfg["grad_accum"],
        log_every=train_cfg.get("log_every", 20),
        eval_every=train_cfg.get("eval_every", 200),
        save_every=train_cfg.get("save_every", 0),
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        amp_dtype=train_cfg.get("amp", "bf16"),
        eps_schedule=pgd_cfg.get("eps_schedule"),
        default_epsilon=pgd_cfg.get("default_eps", EPS_4_255),
        alpha_ratio=pgd_cfg.get("alpha_ratio", DEFAULT_PGD_ALPHA_RATIO),
        pgd_steps=pgd_cfg.get("steps", 7),
        pgd_random_restarts=int(pgd_cfg.get("random_restarts", 1)),
        loader_seed=int(args.seed),
        run_dir=args.run_dir,
        final_save_include_optimizer=train_cfg.get(
            "final_save_include_optimizer", True
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    train_cfg, defense_cfg, data_cfg, gold_cfg, ft_cfg = _load_and_validate_configs(args)
    # Seed only AFTER validation so YAML's ``training.seed`` is reachable;
    # pre-fix this ran ``setup_seed(args.seed)`` before validation, which
    # silently used the CLI default of 0 whenever the operator omitted --seed.
    setup_seed(_resolve_seed(args, train_cfg))

    vlm = _build_vlm(args.model, args.models_yaml)
    model = vlm.model
    # Cast to bf16: attacks-repo models.yaml loads fp16 by default but fp16
    # AdamW optimizer state overflows/underflows in full-FT; bf16 has the same
    # exponent range as fp32 and avoids NaN cascades without a GradScaler.
    model.to(torch.bfloat16)

    apply_freeze(model, FreezeConfig(strategy=ft_cfg.get("freeze_strategy", "none")))

    # InternVL2 ships no formal HF processor — pass the wrapper itself so the
    # teacher-force assembler can reach .preprocess_image / .tokenizer /
    # ._num_image_token. Qwen + LLaVA-NeXT use their AutoProcessor as before.
    collator = get_collator(vlm)
    train_ds = build_train_dataset(
        data_cfg,
        gold_cfg,
        split=data_cfg.get("train_split", "train"),
        n=data_cfg.get("n_train"),
    )

    optimizer, scheduler = _build_optim_and_schedule(model, train_cfg, len(train_ds))
    loss_fn = build_loss(
        from_cfg_dict({**defense_cfg, "defense": train_cfg.get("defense", "trades")})
    )
    trainer = AdvTrainer(
        vlm=vlm,
        model=model,
        collator=collator,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=_build_trainer_config(args, train_cfg, defense_cfg),
        device=args.device,
    )
    trainer.fit(train_ds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
