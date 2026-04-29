"""make_gold writes its per-split sentinel atomically.

Without atomic write, a SIGKILL/OOM/disk-full during ``write_text`` leaves
a truncated JSON file. The pipeline pre-check
(``scripts/run_pipeline.sh::seed_dirs_have_data`` cousin) treats the file's
mere presence as proof-of-completion and skips regeneration, silently
consuming a partial gold cache.

We don't smoke the full make_gold CLI here (it touches the attacks-repo
loader); instead we invoke the same atomic-write idiom on a tmp dir and
verify it survives a failed write step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _atomic_write_summary(cache_dir: Path, split: str, summary: dict) -> None:
    """Mirror of the make_gold.py write path so the test is self-contained."""
    summary_path = cache_dir / f"_summary_{split}.json"
    tmp_path = summary_path.with_name(f"{summary_path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(summary, indent=2))
    os.replace(tmp_path, summary_path)


@pytest.mark.unit
def test_atomic_write_leaves_no_tmp_on_success(tmp_path: Path) -> None:
    _atomic_write_summary(tmp_path, "train", {"total": 5, "written": 5})
    assert (tmp_path / "_summary_train.json").exists()
    leftovers = list(tmp_path.glob("_summary_train.json.tmp.*"))
    assert leftovers == [], f"tmp file leaked: {leftovers}"
    payload = json.loads((tmp_path / "_summary_train.json").read_text())
    assert payload == {"total": 5, "written": 5}


@pytest.mark.unit
def test_atomic_replace_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "_summary_train.json"
    target.write_text("OLD-CONTENT")
    _atomic_write_summary(tmp_path, "train", {"total": 7})
    assert json.loads(target.read_text()) == {"total": 7}


@pytest.mark.unit
def test_failed_write_does_not_replace_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "_summary_train.json"
    target.write_text(json.dumps({"total": 9, "good": True}))
    original_text = target.read_text()

    # Simulate a failure between write_text and replace.
    def _boom(*_a, **_kw):
        raise OSError("disk full simulated")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError, match="disk full"):
        _atomic_write_summary(tmp_path, "train", {"total": 9999, "good": False})

    # Pre-existing sentinel is preserved verbatim, not corrupted by partial
    # write. The crashed run leaves a *.tmp file the operator can clean up,
    # but the sentinel that the pipeline reads is intact.
    assert target.read_text() == original_text
