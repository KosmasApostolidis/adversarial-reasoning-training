"""Robust-eval bridge: read attacks-repo records.jsonl, align as paired per-sample."""

from .robust_eval import (
    align_per_sample,
    load_records,
    records_to_per_sample,
    save_per_sample,
)

__all__ = [
    "align_per_sample",
    "load_records",
    "records_to_per_sample",
    "save_per_sample",
]
