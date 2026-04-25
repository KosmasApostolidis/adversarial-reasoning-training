"""Smoke tests for the console-script entry points.

We invoke each ``main(argv)`` in-process with ``["--help"]`` and assert
``SystemExit(0)``. Subprocess invocation would couple the test to
``pip install -e .`` which CI doesn't enforce.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from adversarial_reasoning_training.cli.eval_robust import main as eval_robust_main
from adversarial_reasoning_training.cli.make_gold import main as make_gold_main
from adversarial_reasoning_training.cli.train import main as train_main


@pytest.mark.parametrize(
    "main,prog",
    [
        (train_main, "art-train"),
        (eval_robust_main, "art-eval-robust"),
        (make_gold_main, "art-make-gold"),
    ],
)
def test_help_exits_zero_and_prints_prog(
    main: Callable[[list[str] | None], int],
    prog: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert prog in out, f"expected --help text to mention {prog!r}, got: {out[:200]!r}"


@pytest.mark.parametrize(
    "main",
    [train_main, eval_robust_main, make_gold_main],
)
def test_missing_required_args_exits_nonzero(
    main: Callable[[list[str] | None], int],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code != 0
