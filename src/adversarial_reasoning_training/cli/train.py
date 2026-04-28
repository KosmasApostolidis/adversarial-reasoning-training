"""Adversarial fine-tune entrypoint.

Usage:
    art-train --config configs/training.yaml \\
        --defenses configs/defenses.yaml --data configs/data.yaml \\
        --gold configs/gold.yaml --full-ft configs/full_ft.yaml \\
        --model qwen2_5_vl_7b --run-dir runs/<id>

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

from ..data.collator import TFCollator
from ..data.dataset import ProstateXTrainDS
from ..gold.oracle import load_metadata_csv
from ..losses.selector import build_loss, from_cfg_dict
from ..trainer.adv_trainer import AdvTrainer, TrainerConfig
from ..trainer.freeze import FreezeConfig, apply_freeze
from ..trainer.optim import (
    OptimConfig,
    ScheduleConfig,
    build_optimizer,
    build_scheduler,
)
from .config import load_yaml
from .runtime import setup_seed


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--models-yaml", type=Path,
        default=Path("../adversarial-reasoning-attacks/configs/models.yaml"),
        help="Path to attacks-repo models.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    setup_seed(args.seed)

    train_cfg = load_yaml(args.config)
    defense_cfg = load_yaml(args.defenses)
    data_cfg = load_yaml(args.data)
    gold_cfg = load_yaml(args.gold)
    ft_cfg = load_yaml(args.full_ft)

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
    if vlm.family == "internvl2":
        proc_arg: Any = vlm
    else:
        proc_arg = getattr(vlm, "processor", None) or vlm.tokenizer
    collator = TFCollator(
        family=vlm.family,
        processor=proc_arg,
    )
    metadata_csv = data_cfg.get("metadata_csv")
    metadata_lookup = (
        load_metadata_csv(metadata_csv) if metadata_csv else {}
    )
    n_train = data_cfg.get("n_train")
    train_ds = ProstateXTrainDS(
        task_id=data_cfg["task_id"],
        split=data_cfg.get("train_split", "train"),
        cache_dir=Path(gold_cfg["cache_dir"]),
        oracle_version=gold_cfg["oracle_version"],
        metadata_lookup=metadata_lookup,
        n=n_train,
        synthetic=bool(data_cfg.get("synthetic", False)),
        config_path=data_cfg.get(
            "config_path", "../adversarial-reasoning-attacks/configs/tasks.yaml"
        ),
    )

    optim_cfg = OptimConfig(
        kind=train_cfg.get("optim", "adamw8bit"),
        lr_lm=train_cfg["lr"]["lm"],
        lr_projector=train_cfg["lr"]["projector"],
        lr_vit=train_cfg["lr"]["vit"],
    )
    optimizer = build_optimizer(model, optim_cfg)
    total_steps = train_cfg["epochs"] * max(1, math.ceil(len(train_ds) / train_cfg["grad_accum"]))
    scheduler = build_scheduler(
        optimizer,
        ScheduleConfig(
            total_steps=total_steps,
            warmup_pct=train_cfg.get("warmup_pct", 0.03),
            kind=train_cfg.get("schedule", "cosine"),
        ),
    )

    loss_fn = build_loss(from_cfg_dict({**defense_cfg, "defense": train_cfg.get("defense", "trades")}))

    trainer_cfg = TrainerConfig(
        epochs=train_cfg["epochs"],
        grad_accum=train_cfg["grad_accum"],
        log_every=train_cfg.get("log_every", 20),
        eval_every=train_cfg.get("eval_every", 200),
        save_every=train_cfg.get("save_every", 0),
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        amp_dtype=train_cfg.get("amp", "bf16"),
        eps_schedule=defense_cfg["pgd"].get("eps_schedule"),
        default_epsilon=defense_cfg["pgd"].get("default_eps", 4.0 / 255.0),
        alpha_ratio=defense_cfg["pgd"].get("alpha_ratio", 0.25),
        pgd_steps=defense_cfg["pgd"].get("steps", 7),
        run_dir=args.run_dir,
        final_save_include_optimizer=train_cfg.get(
            "final_save_include_optimizer", True
        ),
    )
    trainer = AdvTrainer(
        vlm=vlm,
        model=model,
        collator=collator,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=trainer_cfg,
        device=args.device,
    )
    trainer.fit(train_ds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
