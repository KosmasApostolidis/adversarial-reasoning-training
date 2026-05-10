"""Tests for logging on silent-fallback paths.

Two paths historically swallowed exceptions and returned a sentinel
without telling the operator:

* ``gates.T3_robust._wilcoxon_signed_rank`` — falls back to ``(nan, 1.0)``
  when scipy is missing or ``wilcoxon`` rejects the input.
* ``trajectory.templates.<llava image token resolver>`` — falls back to
  ``num_image_tokens = 1`` when the HF processor does not pre-expand the
  image placeholder.

Both are legitimate fallbacks but they should leave a log record so a
distressed pipeline can be diagnosed without re-running with extra
print() instrumentation.
"""

from __future__ import annotations

import builtins
import logging

from adversarial_reasoning_training.gates import T3_robust


def _force_scipy_import_error(monkeypatch) -> None:
    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):
        if name.startswith("scipy"):
            raise ImportError("forced scipy missing for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def test_wilcoxon_logs_warning_when_scipy_missing(
    monkeypatch, caplog
) -> None:
    _force_scipy_import_error(monkeypatch)
    caplog.set_level(logging.WARNING, logger=T3_robust.__name__)
    stat, p = T3_robust._wilcoxon_signed_rank([1.0, 2.0], [3.0, 4.0])
    assert stat != stat  # NaN
    assert p == 1.0
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("scipy" in m.lower() for m in messages), (
        f"expected scipy-fallback warning, got: {messages}"
    )


def test_wilcoxon_logs_warning_on_value_error(monkeypatch, caplog) -> None:
    """When wilcoxon rejects input (e.g. constant deltas), warn the operator."""
    import scipy.stats  # type: ignore  # noqa: F401

    def _raises(*_args: object, **_kwargs: object):
        raise ValueError("forced wilcoxon failure for test")

    # Monkeypatch the symbol that the function imports inside its body.
    monkeypatch.setattr("scipy.stats.wilcoxon", _raises)
    caplog.set_level(logging.WARNING, logger=T3_robust.__name__)
    stat, p = T3_robust._wilcoxon_signed_rank([1.0, 2.0, 3.0], [2.0, 4.0, 5.0])
    assert stat != stat  # NaN
    assert p == 1.0
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("wilcoxon" in m.lower() for m in messages), (
        f"expected wilcoxon-fallback warning, got: {messages}"
    )
