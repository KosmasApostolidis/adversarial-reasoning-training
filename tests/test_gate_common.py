"""Unit tests for gates/_common.py — shared YAML loader + atomic JSON writer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from adversarial_reasoning_training.gates._common import (
    build_metadata_lookup,
    build_train_dataset,
    get_collator,
    get_processor,
    load_gate_yaml,
    write_gate_result,
)
from adversarial_reasoning_training.utils.constants import VLMFamily


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


# --- get_processor / get_collator -----------------------------------------


def test_get_processor_internvl2_returns_wrapper() -> None:
    """InternVL2 has no first-class HF processor; the wrapper itself is passed."""
    vlm = SimpleNamespace(
        family=VLMFamily.INTERNVL2,
        processor=object(),
        tokenizer=object(),
    )
    assert get_processor(vlm) is vlm


def test_get_processor_qwen_returns_processor_attr() -> None:
    proc = object()
    vlm = SimpleNamespace(family=VLMFamily.QWEN_VL, processor=proc, tokenizer=object())
    assert get_processor(vlm) is proc


def test_get_processor_falls_back_to_tokenizer_when_processor_missing() -> None:
    tok = object()
    vlm = SimpleNamespace(family=VLMFamily.LLAVA_NEXT, processor=None, tokenizer=tok)
    assert get_processor(vlm) is tok


def test_get_processor_uses_string_family_for_compat() -> None:
    """Pre-enum gate configs may carry a raw string family — still routed."""
    vlm = SimpleNamespace(family="internvl2", processor=object(), tokenizer=object())
    assert get_processor(vlm) is vlm


def test_get_collator_constructs_tfcollator(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _StubCollator:
        def __init__(self, *, family: str, processor: object) -> None:
            captured["family"] = family
            captured["processor"] = processor

    monkeypatch.setattr(
        "adversarial_reasoning_training.gates._common.TFCollator", _StubCollator
    )
    proc = object()
    vlm = SimpleNamespace(family=VLMFamily.QWEN_VL, processor=proc, tokenizer=object())
    coll = get_collator(vlm)
    assert isinstance(coll, _StubCollator)
    assert captured["family"] == VLMFamily.QWEN_VL
    assert captured["processor"] is proc


# --- build_metadata_lookup ------------------------------------------------


def test_build_metadata_lookup_empty_when_csv_missing() -> None:
    assert build_metadata_lookup({}) == {}
    assert build_metadata_lookup({"metadata_csv": ""}) == {}
    assert build_metadata_lookup({"metadata_csv": None}) == {}


def test_build_metadata_lookup_calls_load_metadata_csv(monkeypatch) -> None:
    seen: list[str] = []

    def _fake_load(csv_path: str) -> dict[str, str]:
        seen.append(csv_path)
        return {"id-1": "row-1"}

    monkeypatch.setattr(
        "adversarial_reasoning_training.gates._common.load_metadata_csv", _fake_load
    )
    out = build_metadata_lookup({"metadata_csv": "/tmp/m.csv"})
    assert out == {"id-1": "row-1"}
    assert seen == ["/tmp/m.csv"]


# --- build_train_dataset --------------------------------------------------


def test_build_train_dataset_passes_through_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _StubDS:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "adversarial_reasoning_training.gates._common.ProstateXTrainDS", _StubDS
    )
    monkeypatch.setattr(
        "adversarial_reasoning_training.gates._common.load_metadata_csv",
        lambda _: {"k": "v"},
    )
    data_cfg = {
        "task_id": "tA",
        "metadata_csv": "/tmp/x.csv",
        "synthetic": True,
        "config_path": "/cfg/tasks.yaml",
    }
    gold_cfg = {"cache_dir": "/cache", "oracle_version": "v3"}

    ds = build_train_dataset(data_cfg, gold_cfg, split="dev", n=4)
    assert isinstance(ds, _StubDS)
    assert captured["task_id"] == "tA"
    assert captured["split"] == "dev"
    assert str(captured["cache_dir"]) == "/cache"
    assert captured["oracle_version"] == "v3"
    assert captured["metadata_lookup"] == {"k": "v"}
    assert captured["n"] == 4
    assert captured["synthetic"] is True
    assert captured["config_path"] == "/cfg/tasks.yaml"


def test_build_train_dataset_default_split_uses_train_split(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _StubDS:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "adversarial_reasoning_training.gates._common.ProstateXTrainDS", _StubDS
    )
    build_train_dataset(
        {"task_id": "t", "train_split": "train_v2"},
        {"cache_dir": "/c", "oracle_version": "v1"},
    )
    assert captured["split"] == "train_v2"
    # No metadata_csv → empty lookup, not a KeyError.
    assert captured["metadata_lookup"] == {}


def test_build_train_dataset_accepts_explicit_metadata_lookup(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _StubDS:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "adversarial_reasoning_training.gates._common.ProstateXTrainDS", _StubDS
    )
    explicit = {"forced": "yes"}
    build_train_dataset(
        {"task_id": "t"},
        {"cache_dir": "/c", "oracle_version": "v1"},
        metadata_lookup=explicit,
    )
    # Explicit lookup takes precedence; load_metadata_csv must not run.
    assert captured["metadata_lookup"] is explicit
