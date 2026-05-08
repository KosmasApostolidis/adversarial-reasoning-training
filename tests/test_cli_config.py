"""Loader tests for cli/config.py — covers _extends error paths.

The happy path (deep-merge of ablation YAMLs over defenses.yaml) is
exercised in test_ablation_schemas.py. This file targets the failure
modes that should fail fast at config-load time:

* cycle detection (parent extends back to child)
* depth limit (chain longer than _MAX_EXTENDS_DEPTH)
* non-string _extends value
* non-dict YAML body
* missing parent file
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adversarial_reasoning_training.cli.config import load_yaml


def _w(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_load_yaml_rejects_non_dict_top_level(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    _w(p, "- one\n- two\n")
    with pytest.raises(ValueError, match="expected top-level YAML mapping"):
        load_yaml(p)


def test_load_yaml_extends_simple_merge(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    _w(parent, "a: 1\nb:\n  x: 10\n  y: 20\n")
    _w(child, "_extends: parent.yaml\nb:\n  y: 99\nc: 3\n")
    cfg = load_yaml(child)
    assert cfg == {"a": 1, "b": {"x": 10, "y": 99}, "c": 3}


def test_load_yaml_extends_lists_replaced_not_merged(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    _w(parent, "items: [1, 2, 3]\n")
    _w(child, "_extends: parent.yaml\nitems: [9]\n")
    cfg = load_yaml(child)
    assert cfg == {"items": [9]}


def test_load_yaml_rejects_extends_cycle(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    _w(a, "_extends: b.yaml\nfrom_a: 1\n")
    _w(b, "_extends: a.yaml\nfrom_b: 2\n")
    with pytest.raises(ValueError, match="cycle detected"):
        load_yaml(a)


def test_load_yaml_rejects_extends_self_cycle(tmp_path: Path) -> None:
    p = tmp_path / "self.yaml"
    _w(p, "_extends: self.yaml\nk: 1\n")
    with pytest.raises(ValueError, match="cycle detected"):
        load_yaml(p)


def test_load_yaml_rejects_excessive_extends_depth(tmp_path: Path) -> None:
    """Linear chain longer than the depth ceiling fails fast."""
    n_links = 10  # > _MAX_EXTENDS_DEPTH (8)
    for i in range(n_links):
        body = f"k_{i}: {i}\n"
        if i + 1 < n_links:
            body = f"_extends: link_{i + 1}.yaml\n" + body
        _w(tmp_path / f"link_{i}.yaml", body)
    with pytest.raises(ValueError, match="exceeds depth"):
        load_yaml(tmp_path / "link_0.yaml")


def test_load_yaml_rejects_non_string_extends(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    _w(p, "_extends: [a, b]\nk: 1\n")
    with pytest.raises(ValueError, match="_extends must be a string"):
        load_yaml(p)


def test_load_yaml_rejects_missing_parent(tmp_path: Path) -> None:
    p = tmp_path / "child.yaml"
    _w(p, "_extends: nonexistent.yaml\nk: 1\n")
    with pytest.raises(FileNotFoundError):
        load_yaml(p)


def test_load_yaml_extends_caller_overrides_parent_scalar(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    _w(parent, "k: parent_value\n")
    _w(child, "_extends: parent.yaml\nk: child_value\n")
    cfg = load_yaml(child)
    assert cfg["k"] == "child_value"


def test_load_yaml_extends_chain_deep_merge(tmp_path: Path) -> None:
    """Three-level chain: grandparent → parent → child."""
    gp = tmp_path / "gp.yaml"
    p = tmp_path / "p.yaml"
    c = tmp_path / "c.yaml"
    _w(gp, "block:\n  a: 1\n  b: 2\n  c: 3\n")
    _w(p, "_extends: gp.yaml\nblock:\n  b: 20\n")
    _w(c, "_extends: p.yaml\nblock:\n  c: 300\nextra: 'yes'\n")
    cfg = load_yaml(c)
    assert cfg == {"block": {"a": 1, "b": 20, "c": 300}, "extra": "yes"}
