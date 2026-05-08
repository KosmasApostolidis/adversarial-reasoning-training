"""Shared helpers for gate scripts (T0/T1/T2/T3).

Each gate's CLI surface is legitimately distinct (different required configs,
different outputs), so this module deliberately does **not** wrap the entire
``_main()`` shape in a base class. It targets the two pieces of code that
were copy-pasted verbatim across all four gates:

* the inline YAML loader (3 of 4 gates: T0, T1, T2)
* the inline ``out_path.open("w") + json.dump(...)`` writer (4 of 4 gates)

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

from ..utils.paths import ensure_parent

__all__ = ["load_gate_yaml", "write_gate_result"]


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
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        if allow_empty:
            return {}
        raise ValueError(f"{p}: YAML document is empty")
    if not isinstance(data, dict):
        raise ValueError(
            f"{p}: expected top-level YAML mapping, got {type(data).__name__}"
        )
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"{p}: missing required keys: {sorted(missing)}")
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
    fd, tmp = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dict(payload), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target
