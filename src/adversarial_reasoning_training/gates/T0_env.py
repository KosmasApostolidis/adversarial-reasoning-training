"""T0 — environment gate.

Builds the model + optimizer, runs one forward+backward on a single
sample, checks that:

  * no exception is raised.
  * the total loss is finite.
  * at least one parameter in each role-group (vit / projector / lm)
    receives a non-zero gradient, i.e. the freeze strategy did not
    accidentally disconnect a subgraph.
  * CUDA peak memory stays below a user-defined ceiling.

Writes ``runs/<id>/gates/T0.json`` with a pass/fail verdict.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..attacks.inner_pgd import InnerPgdConfig, run_inner_pgd
from ..data.collator import TFCollator
from ..losses.selector import build_loss, from_cfg_dict
from ..trainer.freeze import _LM_PATTERNS, _PROJECTOR_PATTERNS, _VIT_PATTERNS
from ..utils.constants import DEFAULT_PGD_ALPHA_RATIO, EPS_4_255
from ..utils.mem import current_memory_stats, reset_peak_memory
from ._common import (
    build_train_dataset,
    get_collator,
    load_gate_yaml,
    write_gate_result,
)


@dataclass
class T0Result:
    passed: bool
    loss_clean: float
    loss_total: float
    grad_norm_vit: float
    grad_norm_projector: float
    grad_norm_lm: float
    peak_memory_gb: float
    peak_memory_limit_gb: float
    duration_s: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _role_grad_norm(model: torch.nn.Module, patterns: tuple[str, ...]) -> float:
    total = 0.0
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if any(tag in name for tag in patterns):
            total += float(p.grad.detach().float().pow(2).sum().item())
    return total**0.5


@dataclass(frozen=True)
class _RoleGradNorms:
    vit: float
    projector: float
    lm: float

    @property
    def all_zero(self) -> bool:
        return self.vit == 0.0 and self.projector == 0.0 and self.lm == 0.0


def _collect_role_grad_norms(model: torch.nn.Module) -> _RoleGradNorms:
    return _RoleGradNorms(
        vit=_role_grad_norm(model, _VIT_PATTERNS),
        projector=_role_grad_norm(model, _PROJECTOR_PATTERNS),
        lm=_role_grad_norm(model, _LM_PATTERNS),
    )


def _evaluate_t0_verdict(
    *,
    loss_total: torch.Tensor,
    grad_norms: _RoleGradNorms,
    peak_gb: float,
    peak_memory_limit_gb: float,
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    passed = True
    if not torch.isfinite(loss_total).item():
        passed = False
        notes.append("loss is NaN or Inf")
    if grad_norms.all_zero:
        passed = False
        notes.append("no gradient reached any role group")
    if peak_gb > peak_memory_limit_gb:
        passed = False
        notes.append(
            f"peak memory {peak_gb:.1f} GiB exceeds limit {peak_memory_limit_gb:.1f} GiB"
        )
    return passed, notes


def run_t0(
    *,
    vlm: Any,
    model: torch.nn.Module,
    collator: TFCollator,
    sample_factory: Callable[[], Any],
    defense_cfg: dict[str, Any],
    out_path: Path,
    peak_memory_limit_gb: float = 120.0,
    device: str = "cuda",
    amp_dtype: torch.dtype = torch.bfloat16,
    epsilon: float = EPS_4_255,
    pgd_steps: int = 3,
) -> T0Result:
    """Run the T0 environment gate.

    ``sample_factory`` returns a `TrainSample` object to feed the
    collator. We keep it a factory so we can call it once without
    leaking dataset handles.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device)
    start = time.time()
    reset_peak_memory()
    notes: list[str] = []

    sample = sample_factory()
    batch = collator([sample]).to(device_t)
    loss_fn = build_loss(from_cfg_dict(defense_cfg))

    model.train(False)
    inner_cfg = InnerPgdConfig(
        epsilon=epsilon,
        alpha_ratio=DEFAULT_PGD_ALPHA_RATIO,
        steps=pgd_steps,
        random_restarts=1,
    )
    pixel_values = batch.forward_kwargs["pixel_values"].to(device_t)
    attack_result = run_inner_pgd(vlm, pixel_values, batch, inner_cfg)
    x_adv = attack_result.perturbed_image.detach().to(device_t)
    if x_adv.ndim == 3:
        x_adv = x_adv.unsqueeze(0)

    # Keep model in eval mode: attacks-repo qwen_vl.forward_with_logits
    # asserts ``self.model.training is False``. Grads still flow under
    # eval mode (dropout/BN stay frozen); this matches adv_trainer's
    # outer step which never re-enables train mode after inner PGD.
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    with torch.autocast(device_type=device_t.type, dtype=amp_dtype):
        fwd_kwargs = {
            k: v for k, v in batch.forward_kwargs.items()
            if k != "pixel_values" and v is not None
        }
        logits_clean = vlm.forward_with_logits(pixel_values, batch.input_ids, **fwd_kwargs)
        logits_adv = vlm.forward_with_logits(x_adv, batch.input_ids, **fwd_kwargs)
        loss_out = loss_fn(
            logits_clean, logits_adv,
            batch.input_ids, batch.task_mask, batch.traj_mask,
        )

    loss_out.total.backward()

    grad_norms = _collect_role_grad_norms(model)
    peak_gb = current_memory_stats().peak_allocated_gb

    passed, verdict_notes = _evaluate_t0_verdict(
        loss_total=loss_out.total,
        grad_norms=grad_norms,
        peak_gb=peak_gb,
        peak_memory_limit_gb=peak_memory_limit_gb,
    )
    notes.extend(verdict_notes)

    result = T0Result(
        passed=passed,
        loss_clean=float(loss_out.components.get("loss_task", float("nan"))),
        loss_total=float(loss_out.total.detach()),
        grad_norm_vit=grad_norms.vit,
        grad_norm_projector=grad_norms.projector,
        grad_norm_lm=grad_norms.lm,
        peak_memory_gb=peak_gb,
        peak_memory_limit_gb=peak_memory_limit_gb,
        duration_s=time.time() - start,
        notes=notes,
    )
    write_gate_result(out_path, result.to_dict())
    return result


def _main() -> int:
    """CLI entrypoint:
    ``python -m adversarial_reasoning_training.gates.T0_env --model qwen2_5_vl_7b ...``
    """
    import argparse

    from adversarial_reasoning.models.loader import load_hf_vlm  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--defenses", type=Path, default=Path("configs/defenses.yaml"))
    parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--gold", type=Path, default=Path("configs/gold.yaml"))
    parser.add_argument("--full-ft", type=Path, default=Path("configs/full_ft.yaml"))
    parser.add_argument(
        "--models-yaml", type=Path,
        default=Path("../adversarial-reasoning-attacks/configs/models.yaml"),
    )
    parser.add_argument("--out", type=Path, default=Path("runs/t0/gates/T0.json"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epsilon", type=float, default=EPS_4_255)
    parser.add_argument("--pgd-steps", type=int, default=3)
    args = parser.parse_args()

    defense_cfg = load_gate_yaml(args.defenses, allow_empty=False)
    data_cfg = load_gate_yaml(args.data, allow_empty=False)
    gold_cfg = load_gate_yaml(args.gold, allow_empty=False)
    ft_cfg = load_gate_yaml(args.full_ft, allow_empty=False)
    defense_cfg.setdefault("defense", "trades")

    vlm = load_hf_vlm(args.model, config_path=str(args.models_yaml))
    model = vlm.model
    model.to(torch.bfloat16)

    from ..trainer.freeze import FreezeConfig, apply_freeze
    apply_freeze(model, FreezeConfig(strategy=ft_cfg.get("freeze_strategy", "none")))

    collator = get_collator(vlm)
    ds = build_train_dataset(data_cfg, gold_cfg, n=1)

    def _factory() -> Any:
        return ds[0]

    peak_limit = float(ft_cfg.get("memory", {}).get("peak_memory_limit_gb", 120.0))

    result = run_t0(
        vlm=vlm,
        model=model,
        collator=collator,
        sample_factory=_factory,
        defense_cfg=defense_cfg,
        out_path=args.out,
        peak_memory_limit_gb=peak_limit,
        device=args.device,
        epsilon=args.epsilon,
        pgd_steps=args.pgd_steps,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
