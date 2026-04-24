"""Robust evaluation entrypoint.

Loads an adv-trained checkpoint, runs the attacks-repo PGD eval suite,
feeds the per-sample arrays to T3, and writes the T3 JSON verdict.

Usage:
    python scripts/eval_robust.py --ckpt runs/<id>/ckpt/best.pt \
        --attacks-config ../adversarial-reasoning-attacks/configs/attacks.yaml \
        --model qwen2_5_vl_7b --baseline runs/<id>/baseline_per_sample.json \
        --out-dir runs/<id>/gates/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adversarial_reasoning_training.eval.robust_eval import (
    RobustEvalConfig,
    run_robust_suite,
    save_per_sample,
)
from adversarial_reasoning_training.gates.T3_robust import T3Thresholds, run_t3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--attacks-config", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="per-sample baseline metrics JSON from undefended run")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min-traj-edit-delta", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    defended = run_robust_suite(RobustEvalConfig(
        ckpt_path=args.ckpt,
        attacks_config=args.attacks_config,
        output_dir=args.out_dir / "defended",
        model_family=args.model,
        device=args.device,
    ))
    save_per_sample(args.out_dir / "defended_per_sample.json", defended)

    with args.baseline.open("r", encoding="utf-8") as f:
        baseline = json.load(f)

    result = run_t3(
        baseline_per_sample=baseline,
        defended_per_sample=defended,
        out_path=args.out_dir / "T3.json",
        thresholds=T3Thresholds(
            min_traj_edit_delta=args.min_traj_edit_delta,
            alpha=args.alpha,
        ),
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
