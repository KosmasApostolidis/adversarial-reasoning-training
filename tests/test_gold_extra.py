"""Extended unit tests for gold/{templates,oracle,expert_probe}.

Companion to test_oracle.py — that file covers the happy paths; this one
covers the branching, edge cases, and persistence helpers needed to lift
coverage on the gold/ package.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from adversarial_reasoning.agents.base import ToolCall, Trajectory  # type: ignore

from adversarial_reasoning_training.gold.expert_probe import (
    load_expert_probe,
    save_expert_probe,
)
from adversarial_reasoning_training.gold.oracle import (
    OracleConfig,
    generate_trajectory,
    load_metadata_csv,
)
from adversarial_reasoning_training.gold.templates import (
    ORACLE_VERSION,
    TEMPLATES,
    pick_template_name,
    template_pirads_3,
    template_pirads_4,
    template_pirads_5,
    template_post_treatment,
)

# -------------------- pick_template_name branches --------------------


def test_pick_template_name_post_treatment_branch() -> None:
    md = {"context": "post_treatment", "psa": 0.4}
    assert pick_template_name(md) == "post_treatment"


def test_pick_template_name_default_when_pirads_absent() -> None:
    # Default is 3 when neither pirads nor pi_rads is present.
    assert pick_template_name({}) == "pirads_3"


def test_pick_template_name_tolerates_pi_rads_alias() -> None:
    assert pick_template_name({"pi_rads": 5}) == "pirads_5"


def test_pick_template_name_post_treatment_overrides_pirads() -> None:
    md = {"context": "post_treatment", "pirads": 5}
    assert pick_template_name(md) == "post_treatment"


def test_templates_dict_exposes_all_five_keys() -> None:
    assert set(TEMPLATES.keys()) == {
        "pirads_1_2",
        "pirads_3",
        "pirads_4",
        "pirads_5",
        "post_treatment",
    }


def test_oracle_version_constant_is_string() -> None:
    assert isinstance(ORACLE_VERSION, str) and ORACLE_VERSION


# -------------------- template body branches --------------------


def test_template_pirads_3_biopsy_yes_branch_high_psad() -> None:
    md = {"psa": 12.0, "psa_density": 0.20, "lesion_size_mm": 8.0, "volume_cc": 40.0}
    traj = template_pirads_3(md, task_id="t", sample_id="s")
    biopsy_call = next(c for c in traj.tool_calls if c.name == "recommend_biopsy")
    assert biopsy_call.result["biopsy"] is True
    assert "MRI-targeted biopsy" in traj.final_answer


def test_template_pirads_3_biopsy_yes_branch_large_lesion() -> None:
    md = {"psa": 6.0, "psa_density": 0.05, "lesion_size_mm": 11.0, "volume_cc": 60.0}
    traj = template_pirads_3(md, task_id="t", sample_id="s")
    biopsy_call = next(c for c in traj.tool_calls if c.name == "recommend_biopsy")
    assert biopsy_call.result["biopsy"] is True


def test_template_pirads_3_biopsy_no_branch() -> None:
    md = {"psa": 4.0, "psa_density": 0.05, "lesion_size_mm": 5.0, "volume_cc": 60.0}
    traj = template_pirads_3(md, task_id="t", sample_id="s")
    biopsy_call = next(c for c in traj.tool_calls if c.name == "recommend_biopsy")
    assert biopsy_call.result["biopsy"] is False
    assert "repeat mpMRI" in traj.final_answer


def test_template_pirads_4_emits_targeted_biopsy() -> None:
    traj = template_pirads_4({"lesion_size_mm": 14.0}, task_id="t", sample_id="s")
    names = [c.name for c in traj.tool_calls]
    assert names == ["calc_pi_rads", "measure_lesion", "lookup_guideline", "recommend_biopsy"]
    biopsy = traj.tool_calls[-1].result
    assert biopsy["biopsy"] is True
    assert biopsy["approach"] == "mri_targeted"


def test_template_pirads_5_flags_ece_when_size_threshold_hit() -> None:
    traj = template_pirads_5({"lesion_size_mm": 16.0}, task_id="t", sample_id="s")
    measure_call = traj.tool_calls[1]
    assert measure_call.name == "measure_lesion"
    assert measure_call.result["ece_suspected"] is True


def test_template_pirads_5_no_ece_below_size_threshold() -> None:
    traj = template_pirads_5({"lesion_size_mm": 12.0}, task_id="t", sample_id="s")
    measure_call = traj.tool_calls[1]
    assert measure_call.result["ece_suspected"] is False


def test_template_post_treatment_psa_rising_branch() -> None:
    traj = template_post_treatment({"psa": 0.5, "psa_rising": True}, task_id="t", sample_id="s")
    assert "rising PSA" in traj.final_answer
    biopsy = traj.tool_calls[-1].result
    assert biopsy["biopsy"] is True
    assert biopsy["reason"] == "rising_psa_post_treatment"


def test_template_post_treatment_psa_stable_branch() -> None:
    traj = template_post_treatment({"psa": 0.05, "psa_rising": False}, task_id="t", sample_id="s")
    assert "stable" in traj.final_answer
    biopsy = traj.tool_calls[-1].result
    assert biopsy["biopsy"] is False


def test_template_post_treatment_infers_rising_from_psa_threshold() -> None:
    # No explicit psa_rising → inferred when psa > 0.2.
    traj = template_post_treatment({"psa": 0.4}, task_id="t", sample_id="s")
    biopsy = traj.tool_calls[-1].result
    assert biopsy["biopsy"] is True


# -------------------- oracle.generate_trajectory --------------------


def test_generate_trajectory_post_treatment_routing() -> None:
    cfg = OracleConfig(version="v1")
    traj = generate_trajectory(
        task_id="t", sample_id="s",
        metadata={"context": "post_treatment", "psa": 0.5, "psa_rising": True},
        config=cfg,
    )
    assert traj.metadata["template"] == "post_treatment"
    assert traj.metadata["oracle_version"] == "v1"


def test_generate_trajectory_default_pi_rads_when_missing() -> None:
    cfg = OracleConfig(version="v1")
    traj = generate_trajectory(task_id="t", sample_id="s", metadata=None, config=cfg)
    # Falls through to PI-RADS 3 default when metadata is None.
    assert traj.metadata["template"] == "pirads_3"


def test_generate_trajectory_metadata_default_merged_under_explicit() -> None:
    cfg = OracleConfig(
        version="v1",
        metadata_default={"pi_rads": 5, "psa": 12.0},
    )
    traj = generate_trajectory(
        task_id="t", sample_id="s",
        metadata={"pi_rads": 1},
        config=cfg,
    )
    # Explicit metadata overrides the default.
    assert traj.metadata["template"] == "pirads_1_2"


# -------------------- oracle.load_metadata_csv --------------------


def test_load_metadata_csv_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_metadata_csv(tmp_path / "nope.csv") == {}


def test_load_metadata_csv_parses_and_coerces(tmp_path: Path) -> None:
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "sample_id,pi_rads,psa,lesion_size_mm,custom\n"
        "s1,4,8.5,12.3,extra\n"
        "s2,2,3.0,5.0,other\n",
        encoding="utf-8",
    )
    md = load_metadata_csv(csv_path)
    assert set(md.keys()) == {"s1", "s2"}
    assert md["s1"]["pi_rads"] == 4
    assert md["s1"]["psa"] == 8.5
    assert md["s1"]["lesion_size_mm"] == 12.3
    assert md["s1"]["custom"] == "extra"


def test_load_metadata_csv_uses_proxid_as_fallback_key(tmp_path: Path) -> None:
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("ProxID,pi_rads\nProx-001,3\n", encoding="utf-8")
    md = load_metadata_csv(csv_path)
    assert "Prox-001" in md


def test_load_metadata_csv_skips_rows_without_id(tmp_path: Path) -> None:
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("sample_id,pi_rads\n,3\ns1,4\n", encoding="utf-8")
    md = load_metadata_csv(csv_path)
    assert list(md.keys()) == ["s1"]


def test_load_metadata_csv_tolerates_unparseable_numeric(tmp_path: Path) -> None:
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("sample_id,pi_rads\ns1,not-a-number\n", encoding="utf-8")
    md = load_metadata_csv(csv_path)
    # Coercion fails silently — original string preserved.
    assert md["s1"]["pi_rads"] == "not-a-number"


# -------------------- expert_probe round-trip --------------------


def _probe_traj(task_id: str = "tool_select") -> Trajectory:
    return Trajectory(
        task_id=task_id,
        model_id="expert",
        seed=0,
        tool_calls=[ToolCall(step=1, name="calc_pi_rads", args={}, result={"pi_rads": 4})],
        final_answer="targeted biopsy",
        reasoning_trace="...",
        metadata={"sample_id": "case_001"},
    )


def test_load_expert_probe_missing_path_returns_empty(tmp_path: Path) -> None:
    assert load_expert_probe(tmp_path / "absent.jsonl") == []


def test_save_and_load_expert_probe_roundtrip(tmp_path: Path) -> None:
    pairs = [("case_001", _probe_traj()), ("case_002", _probe_traj())]
    out = tmp_path / "probe.jsonl"
    save_expert_probe(pairs, out)
    loaded = load_expert_probe(out)
    assert [sid for sid, _ in loaded] == ["case_001", "case_002"]
    assert loaded[0][1].final_answer == "targeted biopsy"
    assert loaded[0][1].tool_calls[0].name == "calc_pi_rads"


def test_save_expert_probe_creates_parent(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "probe.jsonl"
    save_expert_probe([("c1", _probe_traj())], out)
    assert out.is_file()


def test_load_expert_probe_skips_blank_lines(tmp_path: Path) -> None:
    out = tmp_path / "probe.jsonl"
    save_expert_probe([("c1", _probe_traj())], out)
    # Append blank lines and re-load.
    with out.open("a", encoding="utf-8") as f:
        f.write("\n\n")
    loaded = load_expert_probe(out)
    assert len(loaded) == 1


def test_load_expert_probe_falls_back_to_metadata_sample_id(tmp_path: Path) -> None:
    """When top-level sample_id is missing, the loader should use
    metadata.sample_id as the key."""
    out = tmp_path / "probe.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "task_id": "t",
        "model_id": "expert",
        "seed": 0,
        "tool_calls": [],
        "final_answer": "x",
        "reasoning_trace": "y",
        "metadata": {"sample_id": "case_999"},
    }
    import json
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps(blob) + "\n")
    loaded = load_expert_probe(out)
    assert loaded[0][0] == "case_999"


def test_load_expert_probe_uses_question_mark_when_no_id(tmp_path: Path) -> None:
    """No sample_id anywhere → defaults to '?'."""
    out = tmp_path / "probe.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "task_id": "t",
        "model_id": "expert",
        "seed": 0,
        "tool_calls": [],
        "final_answer": "x",
        "reasoning_trace": "y",
        "metadata": {},
    }
    import json
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps(blob) + "\n")
    loaded = load_expert_probe(out)
    assert loaded[0][0] == "?"


_ = pytest  # keep linter quiet on unused import in test files
