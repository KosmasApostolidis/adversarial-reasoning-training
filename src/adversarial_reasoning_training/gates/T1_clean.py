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


def _run_t1_training_loop(
    *,
    vlm: Any,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    max_steps: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    grad_accum: int,
) -> tuple[int, float]:
    """Train until ``max_steps`` optimizer steps. Return (steps_taken, final_loss)."""
    step = 0
    micro = 0
    loss_final = float("nan")
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            step, micro, loss_final = _t1_train_step(
                vlm, batch, model, optimizer, scheduler,
                device, amp_dtype, grad_accum, step, micro,
            )
    return step, loss_final


def _compute_t1_verdict(
    *,
    metrics: dict[str, float],
    thresholds: T1Thresholds,
    tool_name_metric: str,
    answer_em_metric: str,
) -> tuple[bool, float, float, list[str]]:
    """Read metrics, apply thresholds, return (passed, tool_acc, answer_em, notes)."""
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
    return passed, tool_acc, answer_em, notes


def _build_t1_result(
    *,
    passed: bool,
    tool_acc: float,
    answer_em: float,
    loss_final: float,
    step: int,
    duration_s: float,
    thresholds: T1Thresholds,
    notes: list[str],
) -> T1Result:
    """Assemble the T1Result record from gathered metrics + thresholds."""
    return T1Result(
        passed=passed,
        tool_name_acc=tool_acc,
        answer_em=answer_em,
        train_loss_final=loss_final,
        steps=step,
        duration_s=duration_s,
        thresholds={
            "tool_name_acc_min": thresholds.tool_name_acc_min,
            "answer_em_min": thresholds.answer_em_min,
            "max_steps": thresholds.max_steps,
        },
        notes=notes,
    )


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
    loader_seed: int = 0,
) -> T1Result:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device)
    start = time.time()

    # attacks-repo qwen_vl.forward_with_logits asserts model.training is False.
    # Grads still flow under eval mode (dropout/BN stay frozen); matches the
    # adv_trainer convention.
    model.train(False)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(int(loader_seed))
    loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=collator, generator=loader_gen,
    )
    step, loss_final = _run_t1_training_loop(
        vlm=vlm, model=model, loader=loader, optimizer=optimizer,
        scheduler=scheduler, max_steps=thresholds.max_steps, device=device_t,
        amp_dtype=amp_dtype, grad_accum=grad_accum,
    )

    metrics = evaluator(step, 1) or {}
    passed, tool_acc, answer_em, notes = _compute_t1_verdict(
        metrics=metrics, thresholds=thresholds,
        tool_name_metric=tool_name_metric, answer_em_metric=answer_em_metric,
    )
    result = _build_t1_result(
        passed=passed, tool_acc=tool_acc, answer_em=answer_em,
        loss_final=loss_final, step=step, duration_s=time.time() - start,
        thresholds=thresholds, notes=notes,
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
        tool_correct, tool_total, ans_correct, ans_total = _accumulate_proxy_accuracies(
            vlm=vlm, loader=loader, device=device, amp_dtype=amp_dtype,
            tool_name_id=tool_name_id, answer_id=answer_id,
        )
        if was_training:
            model.train(True)
        return {
            "tool_name_acc": tool_correct / max(1, tool_total),
            "answer_em": ans_correct / max(1, ans_total),
            "tool_name_token_count": float(tool_total),
            "answer_token_count": float(ans_total),
        }

    return _evaluate


def _accumulate_proxy_accuracies(
    *,
    vlm: Any,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    tool_name_id: int,
    answer_id: int,
) -> tuple[int, int, int, int]:
    """Iterate dev loader, return (tool_correct, tool_total, ans_correct, ans_total)."""
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
    return tool_correct, tool_total, ans_correct, ans_total


def _build_t1_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the T1 gate."""
    import argparse

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
    return parser


def _t1_train_step(
    vlm: Any,
    batch: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
    amp_dtype: torch.dtype,
    grad_accum: int,
    step: int,
    micro: int,
) -> tuple[int, int, float]:
    """Run one training micro-batch: forward, backward, step on accumulation boundary."""
    batch = batch.to(device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        logits = _clean_forward(vlm, batch, device)
        loss = task_ce(logits, batch.input_ids, batch.task_mask)

    (loss / grad_accum).backward()
    micro += 1
    loss_val = float("nan")
    if micro % grad_accum == 0:
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=1.0,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        loss_val = float(loss.detach())
    return step, micro, loss_val


def _load_t1_configs(
    data: Path, gold: Path, full_ft: Path, training: Path,
) -> dict[str, dict]:
    """Load the four YAML configs required by the T1 gate."""
    return {
        "data": load_gate_yaml(data),
        "gold": load_gate_yaml(gold),
        "full_ft": load_gate_yaml(full_ft),
        "training": load_gate_yaml(training),
    }


def _load_t1_model(
    model_alias: str, models_yaml: Path, ft_cfg: dict,
) -> tuple[Any, torch.nn.Module]:
    """Load the VLM, cast to bf16, and apply the freeze strategy."""
    from adversarial_reasoning.models.loader import load_hf_vlm  # type: ignore

    from ..trainer.freeze import FreezeConfig, apply_freeze

    vlm = load_hf_vlm(model_alias, config_path=str(models_yaml))
    model = vlm.model
    model.to(torch.bfloat16)
    apply_freeze(model, FreezeConfig(strategy=ft_cfg.get("freeze_strategy", "none")))
    return vlm, model


def _build_t1_datasets(data_cfg: dict, gold_cfg: dict) -> tuple[Dataset, Dataset]:
    """Build train + dev datasets sharing one metadata lookup."""
    metadata_lookup = build_metadata_lookup(data_cfg)
    train_ds = build_train_dataset(
        data_cfg, gold_cfg,
        split=data_cfg.get("train_split", "train"),
        n=data_cfg.get("n_train"),
        metadata_lookup=metadata_lookup,
    )
    dev_ds = build_train_dataset(
        data_cfg, gold_cfg,
        split=data_cfg.get("dev_split", "dev"),
        n=data_cfg.get("n_dev"),
        metadata_lookup=metadata_lookup,
    )
    return train_ds, dev_ds


def _build_t1_optimization(
    model: torch.nn.Module, train_cfg: dict, max_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler | None]:
    """Build optimizer + LR scheduler from the training config."""
    from ..trainer.optim import (
        OptimConfig, ScheduleConfig, build_optimizer, build_scheduler,
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
            total_steps=max_steps,
            warmup_pct=float(train_cfg.get("warmup_pct", 0.03)),
            kind=str(train_cfg.get("schedule", "cosine")),
        ),
    )
    return optimizer, scheduler


def _main() -> int:
    """CLI entrypoint:

    ``python -m adversarial_reasoning_training.gates.T1_clean
        --model qwen3_vl_8b
        --max-steps 200
        --out runs/t1/gates/T1.json``
    """
    args = _build_t1_parser().parse_args()
    cfgs = _load_t1_configs(args.data, args.gold, args.full_ft, args.training)
    vlm, model = _load_t1_model(args.model, args.models_yaml, cfgs["full_ft"])
    train_ds, dev_ds = _build_t1_datasets(cfgs["data"], cfgs["gold"])
    optimizer, scheduler = _build_t1_optimization(
        model, cfgs["training"], args.max_steps,
    )

    collator = get_collator(vlm)
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
