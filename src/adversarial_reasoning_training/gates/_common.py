"""Shared helpers for gate scripts (T0/T1/T2/T3).

Each gate's CLI surface is legitimately distinct (different required configs,
different outputs), so this module deliberately does **not** wrap the entire
``_main()`` shape in a base class. It targets the pieces that were copy-pasted
verbatim across the gates:

* the inline YAML loader (3 of 4 gates: T0, T1, T2)
* the inline ``out_path.open("w") + json.dump(...)`` writer (4 of 4 gates)
* the family-aware processor + collator wiring (T0, T1, T2, cli/train)
* the metadata CSV → ``ProstateXTrainDS`` plumbing (T0, T1, T2, cli/train)

By centralising both, the gates fail fast on misshapen configs and write
their result JSON atomically (tempfile + rename), which prevents a partial
file from being read by a downstream gate or aggregation script if the
process is killed mid-write.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from ..data.collator import TFCollator
from ..data.dataset import ProstateXTrainDS
from ..gold.oracle import load_metadata_csv
from ..utils.constants import VLMFamily
from ..utils.paths import ensure_parent

__all__ = [
    "build_metadata_lookup",
    "build_train_dataset",
    "get_collator",
    "get_processor",
    "load_gate_yaml",
    "write_gate_result",
]


def load_gate_yaml(
    path: str | Path,
    *,
    required_keys: Iterable[str] = (),
    allow_empty: bool = True,
) -> dict[str, Any]:
    """Load a gate YAML config and validate its top-level shape.

    Parameters
    ----------
    path
        Path to the YAML file. Resolved before opening.
    required_keys
        Keys that must be present in the loaded mapping. Raises
        ``ValueError`` listing all missing keys (one shot — no partial
        diagnosis) so operators see the full gap on the first run.
    allow_empty
        If True, an empty document parses to ``{}`` (matches the old
        ``yaml.safe_load(f) or {}`` shape used by T1 and T2). If False,
        an empty document raises ``ValueError`` (matches the old T0
        shape, which subscripted the result immediately).

    Returns
    -------
    dict[str, Any]
        The parsed mapping. Type is ``dict[str, Any]`` rather than a
        TypedDict because each gate has its own schema; PR3 pilots a
        TypedDict for the training config specifically.
    """
    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        if allow_empty:
            return {}
        raise ValueError(f"{yaml_path}: YAML document is empty")
    if not isinstance(data, dict):
        raise ValueError(
            f"{yaml_path}: expected top-level YAML mapping, got {type(data).__name__}"
        )
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"{yaml_path}: missing required keys: {sorted(missing)}")
    return data


def write_gate_result(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write ``payload`` as JSON to ``path``.

    Uses a temp file in the same directory + ``os.replace`` so that a
    half-written file is never visible to a concurrent reader (e.g. an
    orchestrator polling for ``T1.json``). Indent matches the prior
    ``json.dump(..., indent=2)`` shape so checked-in golden files stay
    diffable.
    """
    target = ensure_parent(path)
    fd, tmp_path = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dict(payload), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp_path).chmod(0o600)
        os.replace(tmp_path, target)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return target


def get_processor(vlm: Any) -> Any:
    """Return the assembler-ready processor object for ``vlm``.

    InternVL2 ships no first-class HF processor — its assembler reaches
    ``preprocess_image`` / ``tokenizer`` / ``_num_image_token`` on the
    wrapper itself, so the wrapper is returned as-is. Qwen and LLaVA-NeXT
    expose an ``AutoProcessor`` (preferred) and fall back to the bare
    tokenizer when one was not attached.
    """
    if vlm.family == VLMFamily.INTERNVL2:
        return vlm
    return getattr(vlm, "processor", None) or vlm.tokenizer


def get_collator(vlm: Any) -> TFCollator:
    """Build the family-aware ``TFCollator`` for ``vlm``."""
    return TFCollator(family=vlm.family, processor=get_processor(vlm))


def build_metadata_lookup(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Materialise the optional metadata CSV referenced by ``data_cfg``.

    Returns an empty dict when the config does not name a CSV path; this
    matches the prior ``load_metadata_csv(metadata_csv) if metadata_csv else {}``
    idiom used by every gate and the CLI trainer.
    """
    metadata_csv = data_cfg.get("metadata_csv")
    if not metadata_csv:
        return {}
    return load_metadata_csv(metadata_csv)


def build_train_dataset(
    data_cfg: Mapping[str, Any],
    gold_cfg: Mapping[str, Any],
    *,
    split: str | None = None,
    n: int | None = None,
    metadata_lookup: Mapping[str, Any] | None = None,
) -> ProstateXTrainDS:
    """Construct a ``ProstateXTrainDS`` from the gate config trio.

    The factory mirrors the call shape duplicated across T0, T1, T2 and the
    training CLI. ``split`` defaults to ``data_cfg["train_split"]`` (or
    ``"train"``) so callers wanting a dev split must pass it explicitly.
    """
    if metadata_lookup is None:
        metadata_lookup = build_metadata_lookup(data_cfg)
    resolved_split = split if split is not None else data_cfg.get("train_split", "train")
    return ProstateXTrainDS(
        task_id=data_cfg["task_id"],
        split=resolved_split,
        cache_dir=Path(gold_cfg["cache_dir"]),
        oracle_version=gold_cfg["oracle_version"],
        metadata_lookup=metadata_lookup,
        n=n,
        synthetic=bool(data_cfg.get("synthetic", False)),
        config_path=data_cfg.get("config_path") or str(
            Path(__file__).resolve().parent.parent.parent.parent
            / ".." / "adversarial-reasoning-attacks" / "configs" / "tasks.yaml"
        ),
    )
