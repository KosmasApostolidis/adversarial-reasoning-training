"""Rule-based oracle: deterministic + template coverage."""

from __future__ import annotations

import pytest

from adversarial_reasoning_training.gold.oracle import (
    OracleConfig,
    generate_trajectory,
)
from adversarial_reasoning_training.gold.templates import pick_template_name


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"pirads": 1, "psa": 4.0, "lesion_size_mm": 5.0}, "pirads_1_2"),
        ({"pirads": 2, "psa": 5.0, "lesion_size_mm": 6.0}, "pirads_1_2"),
        ({"pirads": 3, "psa": 6.0, "lesion_size_mm": 8.0}, "pirads_3"),
        ({"pirads": 4, "psa": 8.0, "lesion_size_mm": 12.0}, "pirads_4"),
        ({"pirads": 5, "psa": 12.0, "lesion_size_mm": 18.0}, "pirads_5"),
    ],
)
def test_pick_template_by_pirads(metadata: dict, expected: str) -> None:
    assert pick_template_name(metadata) == expected


def test_oracle_deterministic() -> None:
    cfg = OracleConfig(version="v1")
    metadata = {"pirads": 4, "psa": 8.0, "lesion_size_mm": 12.0}
    a = generate_trajectory("prostate_tool_select", "case_001", metadata, cfg)
    b = generate_trajectory("prostate_tool_select", "case_001", metadata, cfg)
    assert a == b, "oracle must be deterministic given identical inputs"


def test_oracle_emits_at_least_one_tool_call() -> None:
    cfg = OracleConfig(version="v1")
    metadata = {"pirads": 4, "psa": 8.0, "lesion_size_mm": 12.0}
    traj = generate_trajectory("prostate_tool_select", "case_001", metadata, cfg)
    assert len(getattr(traj, "tool_calls", [])) >= 1
