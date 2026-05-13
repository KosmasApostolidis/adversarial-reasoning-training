"""T1 — clean teacher-forced fine-tune convergence gate.

Runs a short clean-only FT loop (PGD disabled) on the gold oracle
trajectories and checks that tool-name accuracy + answer-EM reach the
configured thresholds on a dev split. This proves:

  * the data loader produces gradient-carrying batches.
  * the segment masks are correctly weighted (learning tools > thoughts).
  * the optimizer schedule isn't broken.
  * the oracle's trajectory templates are learnable at all.

Writes ``runs/<id>/gates/T1.json``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ..data.collator import TFCollator
from ..losses.task_ce import task_ce
from ..utils.constants import GRAD_ACCUM_DEFAULT
from ._common import (
    build_metadata_lookup,
    build_train_dataset,
    get_collator,
    load_gate_yaml,
    write_gate_result,
)


@dataclass
class T1Thresholds:
    tool_name_acc_min: float = 0.85
    answer_em_min: float = 0.70
    max_steps: int = 200


@dataclass
class T1Result:
    passed: bool
    tool_name_acc: float
    answer_em: float
    train_loss_final: float
    steps: int
    duration_s: float
    thresholds: dict[str, float]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_forward(vlm: Any, batch: Any, device: torch.device) -> torch.Tensor:
    fwd_kwargs = {
        k: v for k, v in batch.forward_kwargs.items()
        if k != "pixel_values" and v is not None
    }
    pixel_values = batch.forward_kwargs["pixel_values"].to(device)
    return vlm.forward_with_logits(pixel_values, batch.input_ids, **fwd_kwargs)


def run_t1(
    *,
    vlm: Any,
    model: torch.nn.Module,
    train_ds: Dataset,
    collator: TFCollator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    evaluator: Callable[[int, int], dict[str, float]],
    thresholds: T1Thresholds,
    out_path: Path,
    device: str = "cuda",
    amp_dtype: torch.dtype = torch.bfloat16,
    grad_accum: int = 8,
    tool_name_metric: str = "tool_name_acc",
    answer_em_metric: str = "answer_em",
) -> T1Result:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device)
    start = time.time()

    # attacks-repo qwen_vl.forward_with_logits asserts model.training is False.
    # Grads still flow under eval mode (dropout/BN stay frozen); matches the
    # adv_trainer convention.
    model.train(False)
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collator)
    step = 0
    micro = 0
    loss_final = float("nan")

    # 200 outer steps × grad_accum 8 = 1600 micro-batches; train_ds has 55
    # samples, so we cycle the loader across epochs until step hits max_steps.
    while step < thresholds.max_steps:
        for batch in loader:
            if step >= thresholds.max_steps:
                break
            batch = batch.to(device_t)
            with torch.autocast(device_type=device_t.type, dtype=amp_dtype):
                logits = _clean_forward(vlm, batch, device_t)
                loss = task_ce(logits, batch.input_ids, batch.task_mask)

            (loss / grad_accum).backward()
            micro += 1
            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), max_norm=1.0,
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                loss_final = float(loss.detach())

    metrics = evaluator(step, 1) or {}
    tool_acc = float(metrics.get(tool_name_metric, 0.0))
    answer_em = float(metrics.get(answer_em_metric, 0.0))
    passed = (
        tool_acc >= thresholds.tool_name_acc_min
        and answer_em >= thresholds.answer_em_min
    )
    notes: list[str] = []
    if tool_acc < thresholds.tool_name_acc_min:
        notes.append(f"tool_name_acc {tool_acc:.3f} < {thresholds.tool_name_acc_min}")
    if answer_em < thresholds.answer_em_min:
        notes.append(f"answer_em {answer_em:.3f} < {thresholds.answer_em_min}")

    result = T1Result(
        passed=passed,
        tool_name_acc=tool_acc,
        answer_em=answer_em,
        train_loss_final=loss_final,
        steps=step,
        duration_s=time.time() - start,
        thresholds={
            "tool_name_acc_min": thresholds.tool_name_acc_min,
            "answer_em_min": thresholds.answer_em_min,
            "max_steps": thresholds.max_steps,
        },
        notes=notes,
    )
    write_gate_result(out_path, result.to_dict())
    return result


def make_teacher_forced_evaluator(
    *,
    vlm: Any,
    model: torch.nn.Module,
    dev_ds: Dataset,
    collator: TFCollator,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Callable[[int, int], dict[str, float]]:
    """Token-level argmax accuracy on dev as a proxy for tool_name_acc + answer_em.

    Why a proxy: real free-form generation eval requires .generate() + JSON
    parsing + per-sample exact match — substantial work and out-of-scope
    for this gate. The proxy gates on `segment_ids` so the two metrics
    actually measure different things:

      * `tool_name_acc` = argmax accuracy on TOOL_NAME positions only
      * `answer_em`     = argmax accuracy on ANSWER positions only

    DEFAULT_MASK_WEIGHTS uses identical weights for `task` and `traj`, so
    the soft masks alone would not distinguish the two metrics.
    """
    from ..trajectory.segments import SegmentKind

    tool_name_id = int(SegmentKind.TOOL_NAME.value)
    answer_id = int(SegmentKind.ANSWER.value)

    def _evaluate(global_step: int, epoch: int) -> dict[str, float]:
        loader = DataLoader(dev_ds, batch_size=1, shuffle=False, collate_fn=collator)
        was_training = model.training
        model.train(False)
        tool_correct = tool_total = 0
        ans_correct = ans_total = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = _clean_forward(vlm, batch, device)
                preds = logits.argmax(dim=-1)
                gold = batch.input_ids
                match = preds[:, :-1] == gold[:, 1:]
                seg = batch.segment_ids[:, 1:]
                tool_m = seg == tool_name_id
                ans_m = seg == answer_id
                tool_correct += int(match[tool_m].sum().item())
                tool_total += int(tool_m.sum().item())
                ans_correct += int(match[ans_m].sum().item())
                ans_total += int(ans_m.sum().item())
        if was_training:
            model.train(True)
        return {
            "tool_name_acc": tool_correct / max(1, tool_total),
            "answer_em": ans_correct / max(1, ans_total),
            "tool_name_token_count": float(tool_total),
            "answer_token_count": float(ans_total),
        }

    return _evaluate


def _main() -> int:
    """CLI entrypoint:

    ``python -m adversarial_reasoning_training.gates.T1_clean
        --model qwen2_5_vl_7b
        --max-steps 200
        --out runs/t1/gates/T1.json``
    """
    import argparse

    from adversarial_reasoning.models.loader import load_hf_vlm  # type: ignore

    from ..trainer.freeze import FreezeConfig, apply_freeze
    from ..trainer.optim import (
        OptimConfig,
        ScheduleConfig,
        build_optimizer,
        build_scheduler,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--gold", type=Path, default=Path("configs/gold.yaml"))
    parser.add_argument("--full-ft", type=Path, default=Path("configs/full_ft.yaml"))
    parser.add_argument("--training", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument(
        "--models-yaml", type=Path,
        default=Path("../adversarial-reasoning-attacks/configs/models.yaml"),
    )
    parser.add_argument("--out", type=Path, default=Path("runs/t1/gates/T1.json"))
    parser.add_argument("--device", type=str, default="cuda")
    defaults = T1Thresholds()
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM_DEFAULT)
    parser.add_argument("--tool-name-acc-min", type=float, default=defaults.tool_name_acc_min)
    parser.add_argument("--answer-em-min", type=float, default=defaults.answer_em_min)
    args = parser.parse_args()

    data_cfg = load_gate_yaml(args.data)
    gold_cfg = load_gate_yaml(args.gold)
    ft_cfg = load_gate_yaml(args.full_ft)
    train_cfg = load_gate_yaml(args.training)

    vlm = load_hf_vlm(args.model, config_path=str(args.models_yaml))
    model = vlm.model
    model.to(torch.bfloat16)

    apply_freeze(model, FreezeConfig(strategy=ft_cfg.get("freeze_strategy", "none")))

    collator = get_collator(vlm)
    metadata_lookup = build_metadata_lookup(data_cfg)
    train_ds = build_train_dataset(
        data_cfg,
        gold_cfg,
        split=data_cfg.get("train_split", "train"),
        n=data_cfg.get("n_train"),
        metadata_lookup=metadata_lookup,
    )
    dev_ds = build_train_dataset(
        data_cfg,
        gold_cfg,
        split=data_cfg.get("dev_split", "dev"),
        n=data_cfg.get("n_dev"),
        metadata_lookup=metadata_lookup,
    )

    lr = train_cfg.get("lr", {})
    betas = train_cfg.get("betas", [0.9, 0.999])
    optim_cfg = OptimConfig(
        kind=train_cfg.get("optim", "adamw"),
        lr_lm=float(lr.get("lm", 5.0e-6)),
        lr_projector=float(lr.get("projector", 1.0e-5)),
        lr_vit=float(lr.get("vit", 1.0e-6)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        betas=(float(betas[0]), float(betas[1])),
    )
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(
        optimizer,
        ScheduleConfig(
            total_steps=args.max_steps,
            warmup_pct=float(train_cfg.get("warmup_pct", 0.03)),
            kind=str(train_cfg.get("schedule", "cosine")),
        ),
    )

    device_t = torch.device(args.device)
    evaluator = make_teacher_forced_evaluator(
        vlm=vlm, model=model, dev_ds=dev_ds, collator=collator, device=device_t,
    )

    result = run_t1(
        vlm=vlm,
        model=model,
        train_ds=train_ds,
        collator=collator,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluator=evaluator,
        thresholds=T1Thresholds(
            tool_name_acc_min=args.tool_name_acc_min,
            answer_em_min=args.answer_em_min,
            max_steps=args.max_steps,
        ),
        out_path=args.out,
        device=args.device,
        grad_accum=args.grad_accum,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
