"""YAML loading for CLI scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return its top-level mapping.

    The file must parse to a mapping (``dict``); anything else (a list,
    a scalar, an empty document) raises ``ValueError`` so that scripts
    fail fast with a useful message instead of an obscure ``AttributeError``
    later.

    A top-level ``_extends: <relative-path>`` key, if present, names a
    parent YAML to deep-merge under the current document. The caller's
    keys win on scalar conflicts; nested dicts merge recursively; lists
    are replaced (not concatenated) since list-merge semantics are
    rarely the intent. This lets ablation configs share the production
    baseline (e.g. ``configs/defenses.yaml``) without duplicating its
    pgd/trades subtrees in each ablation cell. PyYAML's ``safe_load``
    only resolves anchors intra-file, so cross-file composition has to
    happen at the loader layer.
    """
    return _load_yaml(Path(path), _seen=frozenset())


_MAX_EXTENDS_DEPTH = 8


def _load_yaml(path: Path, *, _seen: frozenset[Path]) -> dict[str, Any]:
    p = path.resolve()
    if p in _seen:
        raise ValueError(
            f"_extends cycle detected at {p}; visited chain: "
            f"{[str(s) for s in _seen]}"
        )
    if len(_seen) >= _MAX_EXTENDS_DEPTH:
        raise ValueError(
            f"_extends chain exceeds depth {_MAX_EXTENDS_DEPTH} at {p}"
        )
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"{p}: expected top-level YAML mapping, got {type(data).__name__}"
        )
    parent_ref = data.pop("_extends", None)
    if parent_ref is None:
        return data
    if not isinstance(parent_ref, str):
        raise ValueError(
            f"{p}: _extends must be a string path, got {type(parent_ref).__name__}"
        )
    parent_path = (p.parent / parent_ref).resolve()
    parent = _load_yaml(parent_path, _seen=_seen | {p})
    return _deep_merge(parent, data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` over ``base``; child wins on scalar
    conflicts, dicts merge recursively, lists are replaced wholesale.
    """
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
