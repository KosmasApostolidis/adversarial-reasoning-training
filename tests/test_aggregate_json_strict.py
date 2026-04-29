"""aggregate_seeds output must parse under strict JSON.

A metric backed by fewer than two finite seeds becomes ``float('nan')``
in ``_summarise``. ``json.dumps`` writes the bare token ``NaN`` by default,
which is valid JS but illegal RFC 7159 — strict downstream consumers
(``jq -e``, R's ``jsonlite``, ``pandas.read_json(strict=True)``) reject it.
The aggregator must convert NaN to ``null`` before serialization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make ``scripts/`` importable so the aggregator script's helpers load
# without packaging.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.figures.aggregate_seeds import (  # noqa: E402
    _nan_to_none,
    aggregate,
    main,
)


def _write_t1(seed_dir: Path, *, tool_name_acc: float, passed: bool) -> None:
    gates = seed_dir / "gates"
    gates.mkdir(parents=True, exist_ok=True)
    (gates / "T1.json").write_text(
        json.dumps({"passed": passed, "tool_name_acc": tool_name_acc})
    )


@pytest.mark.unit
def test_nan_to_none_descends_recursively() -> None:
    payload = {
        "a": float("nan"),
        "b": [1.0, float("nan"), {"c": float("nan")}],
        "d": "ok",
    }
    cleaned = _nan_to_none(payload)
    assert cleaned["a"] is None
    assert cleaned["b"][1] is None
    assert cleaned["b"][2]["c"] is None
    assert cleaned["d"] == "ok"
    # original payload untouched
    assert payload["a"] != payload["a"]  # still NaN


@pytest.mark.unit
def test_aggregate_emits_strict_json_for_single_seed(tmp_path: Path) -> None:
    # One seed → bootstrap CI returns NaN (n<2). Output must still parse
    # under strict JSON.
    seed = tmp_path / "seed0"
    _write_t1(seed, tool_name_acc=0.7, passed=True)
    out = tmp_path / "agg.json"
    rc = main([
        "--seeds", str(seed),
        "--out", str(out),
        "--min-seeds", "1",  # silence n<min warning
    ])
    assert rc == 0
    text = out.read_text()
    assert "NaN" not in text, "NaN literal leaked into aggregate.json"
    parsed = json.loads(text)
    # ci_lo/ci_hi should be null (was NaN), not the JS literal NaN.
    summary = parsed["summary"]
    assert "T1.tool_name_acc" in summary
    assert summary["T1.tool_name_acc"]["n"] == 1
    assert summary["T1.tool_name_acc"]["ci_lo"] is None
    assert summary["T1.tool_name_acc"]["ci_hi"] is None


@pytest.mark.unit
def test_aggregate_two_seeds_keeps_finite_ci(tmp_path: Path) -> None:
    s0 = tmp_path / "seed0"
    s1 = tmp_path / "seed1"
    _write_t1(s0, tool_name_acc=0.6, passed=True)
    _write_t1(s1, tool_name_acc=0.8, passed=True)
    payload = aggregate([s0, s1])
    summary = payload["summary"]["T1.tool_name_acc"]
    assert summary["n"] == 2
    assert summary["ci_lo"] is not None and not (summary["ci_lo"] != summary["ci_lo"])
    assert summary["ci_hi"] is not None and not (summary["ci_hi"] != summary["ci_hi"])
