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
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"{p}: expected top-level YAML mapping, got {type(data).__name__}"
        )
    return data
