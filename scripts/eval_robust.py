"""Robust-eval entrypoint.

Reads two ``records.jsonl`` files emitted by the attacks-repo runner
(``python -m adversarial_reasoning.runner --mode pgd ...``) — one
from the undefended baseline model, one from the adversarially
fine-tuned model — converts them to T3-compatible per-sample dicts
under the *vs-benign-on-same-model* semantic, and runs the T3
robustness gate.

The runner itself is invoked separately upstream of this script. We
do not load checkpoints here; the runner has its own model loader.

Usage:
    python scripts/eval_robust.py \\
        --baseline-records runs/baseline_qwen/records.jsonl \\
        --defended-records runs/adv1_qwen/records.jsonl \\
        --out-dir runs/adv1_qwen/gates/

Writes:
    <out-dir>/baseline_per_sample.json
    <out-dir>/defended_per_sample.json
    <out-dir>/T3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adversarial_reasoning_training.eval.robust_eval import (
    align_per_sample,
    save_per_sample,
)
from adversarial_reasoning_training.gates.T3_robust import T3Thresholds, run_t3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-records",
        type=Path,
        required=True,
        help="records.jsonl emitted by the attacks runner against the undefended model",
    )
    parser.add_argument(
        "--defended-records",
        type=Path,
        required=True,
        help="records.jsonl emitted by the attacks runner against the adv-FT model",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="directory to write per_sample dicts and T3.json into",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-traj-edit-delta", type=float, default=0.10)
    parser.add_argument("--min-significant-metrics", type=int, default=3)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline, defended, shared = align_per_sample(
        args.baseline_records, args.defended_records
    )
    if not shared:
        print(
            "ERROR: no overlapping (sample_id, epsilon, attack_mode, seed) keys "
            "between baseline and defended records.",
            file=sys.stderr,
        )
        return 2

    baseline_path = args.out_dir / "baseline_per_sample.json"
    defended_path = args.out_dir / "defended_per_sample.json"
    save_per_sample(baseline_path, baseline)
    save_per_sample(defended_path, defended)

    result = run_t3(
        baseline_per_sample=baseline,
        defended_per_sample=defended,
        out_path=args.out_dir / "T3.json",
        thresholds=T3Thresholds(
            min_traj_edit_delta=args.min_traj_edit_delta,
            alpha=args.alpha,
            min_significant_metrics=args.min_significant_metrics,
        ),
    )
    payload = result.to_dict()
    payload["n_paired"] = len(shared)
    print(json.dumps(payload, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
