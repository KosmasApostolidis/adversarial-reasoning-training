"""Re-save a full training checkpoint as a weights-only .pt to free disk.

The training loop writes ``{model_state_dict, optim_state_dict, step,
epoch, metric_value, extra}`` so that we can resume mid-curriculum.
Once a run is no longer a resume target — and we only need the
weights for downstream eval — the optimizer state is dead weight on
disk. This script extracts the model + scalar bookkeeping fields and
writes a compact checkpoint.

Usage:
    python scripts/resave_ckpt_weights_only.py \\
        --src runs/adv1_qwen/ckpt/step0000006-ep01-XXX.pt \\
        --dst runs/adv1_qwen/ckpt/weights_only.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.src.exists():
        print(f"ERROR: source not found: {args.src}", file=sys.stderr)
        return 1

    src_size_gib = args.src.stat().st_size / (1024 ** 3)
    print(f"loading {args.src} ({src_size_gib:.1f} GiB)")
    payload = torch.load(args.src, map_location="cpu", weights_only=True)
    if "model_state_dict" not in payload:
        print("ERROR: payload has no model_state_dict", file=sys.stderr)
        return 2

    keep = {k: payload[k] for k in payload if k != "optim_state_dict"}
    n_tensors = len(payload["model_state_dict"])
    print(f"keeping {sorted(keep.keys())} ({n_tensors} tensors)")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(keep, args.dst)
    dst_size_gib = args.dst.stat().st_size / (1024 ** 3)
    print(f"wrote {args.dst} ({dst_size_gib:.1f} GiB)")

    print("verifying weights-only payload reloads with same tensor count")
    reloaded = torch.load(args.dst, map_location="cpu", weights_only=True)
    n_reloaded = len(reloaded["model_state_dict"])
    if n_reloaded != n_tensors:
        raise RuntimeError(
            f"tensor count mismatch: {n_reloaded} reloaded vs {n_tensors} saved"
        )
    print(f"verify OK — {n_tensors} tensors round-tripped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
