"""Robust evaluation bridge to the attacks-repo runner.

Instead of re-implementing the PGD eval harness, we load an adv-trained
checkpoint into the same VLM wrapper, inject it into the attacks-repo
``Runner``, and let it run the existing ``configs/attacks.yaml``
robustness suite (PGD-L inf at eps in {4/255, 8/255}, steps=40, etc.). The
result JSON is then passed to ``gates/T3_robust.run_t3``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..trainer.ckpt import load_checkpoint


@dataclass(frozen=True)
class RobustEvalConfig:
    ckpt_path: Path
    attacks_config: Path
    output_dir: Path
    model_family: str
    device: str = "cuda"


def _load_runner(attacks_config: Path) -> Any:
    """Import and instantiate attacks-repo Runner.

    The attacks repo must be importable as ``adversarial_reasoning``
    (installed as editable dep via pyproject.toml).
    """
    from adversarial_reasoning.runner import Runner  # type: ignore

    return Runner(config_path=str(attacks_config))


def load_defended_vlm(
    ckpt_path: Path, model_family: str, device: str = "cuda"
) -> Any:
    """Load an adv-trained checkpoint back into the attacks-repo VLM.

    We re-construct the VLM via the attacks-repo loader, then overlay
    the state_dict from our checkpoint. The VLM exposes ``.model``
    (the HF transformer); load_state_dict is done on that handle.
    """
    from adversarial_reasoning.models.loader import load_vlm  # type: ignore

    vlm = load_vlm(model_family, device=device)
    load_checkpoint(ckpt_path, vlm.model, map_location=device)
    return vlm


def run_robust_suite(cfg: RobustEvalConfig) -> dict[str, Any]:
    """Run the attacks-repo robust-eval suite using the defended VLM.

    Writes the full runner output JSON under ``cfg.output_dir``.
    Returns the metrics dictionary structured for ``run_t3``:
        {metric_name: [per-sample float, ...]}
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    vlm = load_defended_vlm(cfg.ckpt_path, cfg.model_family, device=cfg.device)
    runner = _load_runner(cfg.attacks_config)
    # Runner API in attacks repo: pass the pre-loaded VLM and write
    # per-sample metric arrays. We keep the attribute name loose so
    # the bridge works with minor runner-signature changes.
    if hasattr(runner, "set_vlm"):
        runner.set_vlm(vlm)
    else:
        runner.vlm = vlm

    output_path = cfg.output_dir / "robust_eval_output.json"
    runner.run(output_path=str(output_path))

    with output_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("per_sample_metrics", payload)


def save_per_sample(path: Path, per_sample: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(per_sample, f, indent=2)
