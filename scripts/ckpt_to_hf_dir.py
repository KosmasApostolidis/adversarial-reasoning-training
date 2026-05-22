"""Materialise a defended .pt checkpoint into a self-contained HF model dir.

The training loop saves checkpoints as torch ``.pt`` files containing a
``model_state_dict``. The attacks-repo runner, by contrast, calls
``load_hf_vlm(model_key)`` which reads a model_key from
``configs/models.yaml`` and instantiates the model from its
``hf_id`` (a HF Hub or local path). To run the runner against an
adversarially fine-tuned model without modifying the runner, we
materialise the defended weights as a local HF directory that
``models.yaml`` can point at.

The script:
  1. Loads the base Qwen2.5-VL-7B (or any registered model_key) via
     the attacks-repo loader.
  2. Overlays the defended ``model_state_dict`` from the .pt file with
     ``strict=False`` so any optimiser-only keys are silently dropped.
  3. Saves both the model and the processor into ``--out-dir`` via
     ``save_pretrained`` with ``safe_serialization=True``, producing a
     directory that ``load_hf_vlm`` can ingest.

Usage:
    python scripts/ckpt_to_hf_dir.py \\
        --base-model qwen3_vl_8b \\
        --ckpt runs/adv1_qwen/ckpt/weights_only.pt \\
        --out-dir runs/adv1_qwen/ckpt/hf_dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="model_key in attacks-repo models.yaml")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "adversarial-reasoning-attacks" / "configs" / "models.yaml",
    )
    args = parser.parse_args(argv)

    if not args.ckpt.exists():
        print(f"ERROR: ckpt not found: {args.ckpt}", file=sys.stderr)
        return 1

    from adversarial_reasoning.models.loader import load_hf_vlm

    print(f"loading base model {args.base_model}")
    vlm = load_hf_vlm(args.base_model, config_path=str(args.config_path))

    print(f"loading defended state_dict from {args.ckpt}")
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    state_dict = payload["model_state_dict"]

    missing, unexpected = vlm.model.load_state_dict(state_dict, strict=False)
    print(f"overlay applied — missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print(f"  first 3 unexpected keys: {unexpected[:3]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving model to {args.out_dir}")
    vlm.model.save_pretrained(args.out_dir, safe_serialization=True)
    processor = getattr(vlm, "processor", None)
    tokenizer = getattr(vlm, "tokenizer", None)
    tok_like = processor if processor is not None else tokenizer
    if tok_like is None:
        raise RuntimeError(
            f"VLM wrapper {type(vlm).__name__} exposes neither `processor` nor `tokenizer`"
        )
    kind = "processor" if processor is not None else "tokenizer"
    print(f"saving {kind} to {args.out_dir}")
    tok_like.save_pretrained(args.out_dir)

    print(f"done — listing {args.out_dir}:")
    for p in sorted(args.out_dir.iterdir()):
        size_mib = p.stat().st_size / (1024 ** 2) if p.is_file() else 0
        print(f"  {p.name}  ({size_mib:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
