"""Unit tests for gates/_common.py — shared YAML loader + atomic JSON writer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from adversarial_reasoning_training.gates._common import (
    load_gate_yaml,
    write_gate_result,
)


def test_load_gate_yaml_returns_mapping(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("a: 1\nb: two\n")
    out = load_gate_yaml(p)
    assert out == {"a": 1, "b": "two"}


def test_load_gate_yaml_empty_allow_returns_empty_dict(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert load_gate_yaml(p, allow_empty=True) == {}


def test_load_gate_yaml_empty_disallow_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_gate_yaml(p, allow_empty=False)


def test_load_gate_yaml_non_mapping_raises(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="expected top-level YAML mapping"):
        load_gate_yaml(p)


def test_load_gate_yaml_required_keys_missing_lists_all(tmp_path: Path) -> None:
    p = tmp_path / "partial.yaml"
    p.write_text("present: 1\n")
    with pytest.raises(ValueError, match=r"missing required keys.*alpha.*beta"):
        load_gate_yaml(p, required_keys=("alpha", "beta", "present"))


def test_load_gate_yaml_required_keys_present(tmp_path: Path) -> None:
    p = tmp_path / "ok.yaml"
    p.write_text("alpha: 1\nbeta: 2\n")
    out = load_gate_yaml(p, required_keys=("alpha", "beta"))
    assert out == {"alpha": 1, "beta": 2}


def test_write_gate_result_atomic_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "T1.json"
    payload = {"passed": True, "metrics": {"acc": 0.9}}
    written = write_gate_result(target, payload)
    assert written == target.resolve()
    assert json.loads(target.read_text()) == payload


def test_write_gate_result_no_tempfile_left_on_disk(tmp_path: Path) -> None:
    target = tmp_path / "T0.json"
    write_gate_result(target, {"x": 1})
    siblings = [p.name for p in tmp_path.iterdir()]
    # Only the final file; no leftover ".tmp" tempfiles.
    assert siblings == ["T0.json"]


def test_write_gate_result_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "T2.json"
    target.write_text('{"old": true}')
    write_gate_result(target, {"new": True})
    assert json.loads(target.read_text()) == {"new": True}


def test_write_gate_result_keeps_indent_format(tmp_path: Path) -> None:
    """Downstream tooling expects ``indent=2`` JSON; preserve diff-friendliness."""
    target = tmp_path / "T3.json"
    write_gate_result(target, {"k": [1, 2]})
    text = target.read_text()
    # indent=2 puts each list item on its own line with two-space prefix.
    assert "  " in text
    assert text.count("\n") >= 4


def test_write_gate_result_cleans_tmp_on_serialize_error(tmp_path: Path) -> None:
    """If json.dump raises (e.g. non-serialisable obj), no tempfile remains."""
    target = tmp_path / "T0.json"

    class _NotJSON:
        pass

    with pytest.raises(TypeError):
        write_gate_result(target, {"bad": _NotJSON()})  # type: ignore[dict-item]
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith("T0.json.")]
    assert leftovers == []
    assert not target.exists()


def test_load_gate_yaml_accepts_str_path(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("k: v\n")
    out = load_gate_yaml(str(p))
    assert out == {"k": "v"}


def test_write_gate_result_accepts_str_path(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    write_gate_result(str(target), {"a": 1})
    assert target.exists()


def test_write_gate_result_resolves_user_expansion(monkeypatch, tmp_path: Path) -> None:
    """``~`` in path is expanded so callers can pass ``~/runs/...`` strings."""
    monkeypatch.setenv("HOME", str(tmp_path))
    write_gate_result("~/T0.json", {"ok": True})
    expanded = (tmp_path / "T0.json").resolve()
    assert expanded.exists()
    assert json.loads(expanded.read_text()) == {"ok": True}
    # Cleanup: avoid bleeding into other tests via lingering file in tmp HOME.
    os.unlink(expanded)
