"""Unit coverage for trajectory.teacher_force shared helpers.

P2.2 split surfaced ``_split_thoughts`` and ``_format_observation`` as
shared utilities consumed by every per-family assembler in
``trajectory.templates``. Both were previously private to a single
file and untested standalone — drift in either silently corrupts
teacher-forced sequences (one wrong observation = one wrong
``OBSERVATION``-segment loss mask = one wrong gradient).
"""

from __future__ import annotations

from adversarial_reasoning.agents.base import ToolCall

from adversarial_reasoning_training.trajectory.teacher_force import (
    _format_observation,
    _split_thoughts,
)


# --- _split_thoughts --------------------------------------------------------


def test_split_thoughts_zero_steps_returns_empty() -> None:
    assert _split_thoughts("any text", n_steps=0) == []


def test_split_thoughts_negative_steps_returns_empty() -> None:
    assert _split_thoughts("any text", n_steps=-1) == []


def test_split_thoughts_empty_trace_pads_with_empty_strings() -> None:
    assert _split_thoughts("", n_steps=3) == ["", "", ""]


def test_split_thoughts_single_block_pads_remainder() -> None:
    """Single thought block → first slot, rest empty."""
    result = _split_thoughts("only thought", n_steps=3)
    assert result == ["only thought", "", ""]


def test_split_thoughts_delimiter_split() -> None:
    trace = "first\n---\nsecond\n---\nthird"
    assert _split_thoughts(trace, n_steps=3) == ["first", "second", "third"]


def test_split_thoughts_strips_whitespace_per_segment() -> None:
    trace = "  first  \n---\n  second  "
    assert _split_thoughts(trace, n_steps=2) == ["first", "second"]


def test_split_thoughts_truncates_oversized() -> None:
    trace = "a\n---\nb\n---\nc\n---\nd"
    assert _split_thoughts(trace, n_steps=2) == ["a", "b"]


def test_split_thoughts_undersized_pads_to_steps() -> None:
    trace = "a\n---\nb"
    assert _split_thoughts(trace, n_steps=4) == ["a", "b", "", ""]


# --- _format_observation ----------------------------------------------------


def test_format_observation_error_takes_precedence() -> None:
    call = ToolCall(step=0, name="t", args={}, result="ignored", error="boom")
    assert _format_observation(call) == "ERROR: boom"


def test_format_observation_none_result_emits_sentinel() -> None:
    call = ToolCall(step=0, name="t", args={}, result=None, error=None)
    assert _format_observation(call) == "(no result)"


def test_format_observation_dict_result_serialises_json() -> None:
    call = ToolCall(step=0, name="t", args={}, result={"a": 1, "b": "x"}, error=None)
    out = _format_observation(call)
    # JSON keys preserved; non-ASCII allowed.
    assert '"a": 1' in out
    assert '"b": "x"' in out


def test_format_observation_list_result_serialises_json() -> None:
    call = ToolCall(step=0, name="t", args={}, result=[1, 2, 3], error=None)
    assert _format_observation(call) == "[1, 2, 3]"


def test_format_observation_scalar_result_stringifies() -> None:
    call = ToolCall(step=0, name="t", args={}, result=42, error=None)
    assert _format_observation(call) == "42"


def test_format_observation_string_result_passes_through() -> None:
    call = ToolCall(step=0, name="t", args={}, result="plain text", error=None)
    assert _format_observation(call) == "plain text"


def test_format_observation_non_ascii_unicode() -> None:
    """ensure_ascii=False: keep medical Greek / unicode glyphs verbatim."""
    call = ToolCall(step=0, name="t", args={}, result={"prostate": "α"}, error=None)
    assert "α" in _format_observation(call)
