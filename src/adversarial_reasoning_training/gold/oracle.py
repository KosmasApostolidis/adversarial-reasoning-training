"""Rule-based oracle: deterministic Trajectory from ProstateX metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adversarial_reasoning.agents.base import Trajectory

from .templates import ORACLE_VERSION, TEMPLATES, pick_template_name


@dataclass(frozen=True)
class OracleConfig:
    version: str = ORACLE_VERSION
    metadata_default: dict[str, Any] | None = None


def generate_trajectory(
    task_id: str,
    sample_id: str,
    metadata: dict[str, Any] | None = None,
    config: OracleConfig = OracleConfig(),
) -> Trajectory:
    """Return a gold Trajectory for one (task, sample) given metadata.

    Metadata is typically pulled from ProstateX CSV + PI-RADS spreadsheet
    joined on ProxID + fid. If metadata is None, falls back to an
    equivocal PI-RADS 3 template with default PSA/density values — useful
    for smoke tests and sample IDs missing from the metadata table.
    """
    md = dict(config.metadata_default or {})
    if metadata:
        md.update(metadata)
    if "pi_rads" not in md:
        md["pi_rads"] = 3
    template_key = pick_template_name(md)
    template_fn = TEMPLATES[template_key]
    traj = template_fn(md, task_id=task_id, sample_id=sample_id)
    traj.metadata["template"] = template_key
    traj.metadata["oracle_version"] = config.version
    return traj


def load_metadata_csv(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load ProstateX metadata keyed by sample_id. CSV schema is flexible —
    we only read columns that map onto template inputs. Unknown columns
    are preserved in the per-sample dict under their original name.
    """
    import csv

    out: dict[str, dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("sample_id") or row.get("ProxID") or row.get("fid")
            if not sid:
                continue
            out[sid] = _coerce_metadata(row)
    return out


def _coerce_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Best-effort numeric coercion for known-numeric fields.

    ``pi_rads`` is a discrete clinical score (1-5) and must canonicalise
    to ``int``. Pre-fix a ternary keyed on the literal ``"." in str(...)``
    silently produced ``3.0`` for CSV values like ``pi_rads=3.0``, which
    poisoned JSON serialisation and any equality check expecting an int.
    """
    md = dict(row)
    for k in ("pi_rads", "psa", "psa_density", "lesion_size_mm", "volume_cc"):
        if k in md and md[k] not in (None, ""):
            try:
                if k == "pi_rads":
                    md[k] = int(float(md[k]))
                else:
                    md[k] = float(md[k])
            except ValueError:
                pass
    return md
