"""Regression tests for figure-script tolerance to truncated / corrupt JSON.

Background: a trainer (or attacks-runner) crash mid-write of a gate
JSON leaves a truncated file on disk. Subsequent pipeline reruns under
``--skip-existing`` would crash the figure / aggregate scripts with
``json.JSONDecodeError``, dropping all downstream artifacts. Each
script must SOFT-FAIL on a corrupt file (skip-with-warn) the same way
it already soft-fails on a missing file.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.unit
def test_aggregate_seeds_load_returns_none_on_truncated_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = importlib.import_module("scripts.figures.aggregate_seeds")
    truncated = tmp_path / "T2.json"
    truncated.write_text('{"passed": tru')  # mid-write truncation
    result = mod._load(truncated)
    assert result is None, (
        "aggregate_seeds._load must return None on truncated JSON, "
        "not raise json.JSONDecodeError"
    )
    err = capsys.readouterr().err
    assert "T2.json" in err, "operator must see the corrupt path on stderr"


@pytest.mark.unit
def test_aggregate_seeds_load_still_returns_none_on_missing(tmp_path: Path) -> None:
    mod = importlib.import_module("scripts.figures.aggregate_seeds")
    missing = tmp_path / "does_not_exist.json"
    assert mod._load(missing) is None


@pytest.mark.unit
def test_aggregate_seeds_load_parses_valid_json(tmp_path: Path) -> None:
    mod = importlib.import_module("scripts.figures.aggregate_seeds")
    good = tmp_path / "T1.json"
    good.write_text(json.dumps({"passed": True, "tool_name_acc": 0.9}))
    result = mod._load(good)
    assert result == {"passed": True, "tool_name_acc": 0.9}


@pytest.mark.unit
def test_make_ablation_tables_load_aggregate_handles_truncated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = importlib.import_module("scripts.figures.make_ablation_tables")
    truncated = tmp_path / "aggregate.json"
    truncated.write_text('{"summary": ')
    result = mod._load_aggregate(truncated)
    assert result is None, (
        "make_ablation_tables._load_aggregate must soft-fail on corrupt "
        "JSON; the caller already filters None inputs"
    )
    err = capsys.readouterr().err
    assert "aggregate.json" in err


@pytest.mark.unit
def test_make_figures_load_json_lenient_handles_truncated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = importlib.import_module("scripts.figures.make_figures")
    truncated = tmp_path / "aggregate.json"
    truncated.write_text("{")
    result = mod.load_json_lenient(truncated)
    assert result is None, (
        "make_figures.load_json_lenient must soft-fail on corrupt JSON"
    )
    err = capsys.readouterr().err
    assert "aggregate.json" in err
