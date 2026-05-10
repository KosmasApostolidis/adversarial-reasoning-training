"""T2 — no-collapse gate.

After adversarial fine-tune, clean-input metrics must remain within a
small tolerance of the T1 (clean-only FT) ceiling. This protects
against the classic adversarial-training failure mode where robustness
is bought by completely sacrificing clean accuracy.

Pass criterion per metric:
    adv_ft_clean >= clean_ft_ceiling - tolerance_pp / 100

Writes ``runs/<id>/gates/T2.json``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ._common import (
    build_train_dataset,
    get_collator,
    load_gate_yaml,
    write_gate_result,
)


@dataclass
class T2Thresholds:
    tolerance_pp: float = 3.0  # allowed absolute drop in percentage points
    metrics: tuple[str, ...] = ("tool_name_acc", "answer_em", "args_iou")


@dataclass
class T2Result:
    passed: bool
    per_metric: dict[str, dict[str, float]]
    duration_s: float
    thresholds: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_metrics(path: Path, metric_keys: tuple[str, ...]) -> dict[str, float]:
    """Read metrics from a gate JSON payload.

    Accepts either the T1 result layout (flat keys) or an arbitrary
    evaluator payload stored under a ``metrics`` key.
    """
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    flat: dict[str, float] = {}
    source = payload.get("metrics", payload)
    for key in metric_keys:
        if key in payload:
            flat[key] = float(payload[key])
        elif key in source:
            flat[key] = float(source[key])
    return flat


def run_t2(
    *,
    adv_clean_evaluator: Callable[[], dict[str, float]],
    t1_result_path: Path,
    out_path: Path,
    thresholds: T2Thresholds = T2Thresholds(),
) -> T2Result:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    ceiling = _read_metrics(t1_result_path, thresholds.metrics)
    current = adv_clean_evaluator() or {}

    per_metric: dict[str, dict[str, float]] = {}
    notes: list[str] = []
    passed = True
    tol = thresholds.tolerance_pp / 100.0

    for key in thresholds.metrics:
        ceil = ceiling.get(key)
        current_value = float(current.get(key, 0.0))
        if ceil is None:
            notes.append(f"metric {key} missing from T1 result")
            continue
        drop = ceil - current_value
        per_metric[key] = {
            "ceiling": ceil,
            "current": current_value,
            "drop": drop,
            "tolerance": tol,
            "ok": drop <= tol,
        }
        if drop > tol:
            passed = False
            notes.append(
                f"{key}: drop {drop * 100:.1f} pp exceeds tolerance {thresholds.tolerance_pp:.1f} pp"
            )

    result = T2Result(
        passed=passed,
        per_metric=per_metric,
        duration_s=time.time() - start,
        thresholds={
            "tolerance_pp": thresholds.tolerance_pp,
            "metrics": list(thresholds.metrics),
        },
        notes=notes,
    )
    write_gate_result(out_path, result.to_dict())
    return result


def _main() -> int:
    """CLI entrypoint:

    ``python -m adversarial_reasoning_training.gates.T2_no_collapse
        --model qwen2_5_vl_7b
        --ckpt runs/adv1_qwen/ckpt/best.pt
        --t1-result runs/t1_v2/gates/T1.json
        --tolerance-pp 5.0
        --out runs/adv1_qwen/gates/T2.json``

    Loads the adversarially-FT checkpoint, runs the same teacher-forced
    proxy evaluator from T1 on the dev split, and gates clean-input
    metrics against the T1 ceiling within ``--tolerance-pp`` percentage
    points.
    """
    import argparse

    import torch
    from adversarial_reasoning.models.loader import load_hf_vlm  # type: ignore

    from ..trainer.ckpt import load_checkpoint
    from .T1_clean import make_teacher_forced_evaluator

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="adversarially-FT checkpoint; if omitted, evaluates HF-init weights")
    parser.add_argument("--t1-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--gold", type=Path, default=Path("configs/gold.yaml"))
    parser.add_argument(
        "--models-yaml", type=Path,
        default=Path("../adversarial-reasoning-attacks/configs/models.yaml"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tolerance-pp", type=float, default=3.0)
    parser.add_argument("--max-eval-samples", type=int, default=None,
                        help="cap dev_ds size; falls back to data.yaml n_dev")
    parser.add_argument(
        "--metrics", type=str, nargs="+",
        default=["tool_name_acc", "answer_em"],
        help="subset of T1 metrics to gate; args_iou skipped by default since "
             "the teacher-forced proxy does not emit it",
    )
    args = parser.parse_args()

    data_cfg = load_gate_yaml(args.data)
    gold_cfg = load_gate_yaml(args.gold)

    vlm = load_hf_vlm(args.model, config_path=str(args.models_yaml))
    model = vlm.model
    model.to(torch.bfloat16)

    if args.ckpt is not None:
        load_checkpoint(args.ckpt, model, optimizer=None, map_location=args.device)

    collator = get_collator(vlm)
    n_dev = args.max_eval_samples or data_cfg.get("n_dev")
    dev_ds = build_train_dataset(
        data_cfg,
        gold_cfg,
        split=data_cfg.get("dev_split", "dev"),
        n=n_dev,
    )

    device_t = torch.device(args.device)
    eval_fn = make_teacher_forced_evaluator(
        vlm=vlm, model=model, dev_ds=dev_ds, collator=collator, device=device_t,
    )

    def adv_clean_evaluator() -> dict[str, float]:
        return eval_fn(0, 0)

    result = run_t2(
        adv_clean_evaluator=adv_clean_evaluator,
        t1_result_path=args.t1_result,
        out_path=args.out,
        thresholds=T2Thresholds(
            tolerance_pp=args.tolerance_pp,
            metrics=tuple(args.metrics),
        ),
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
