"""Robust evaluation entrypoint shim.

The implementation lives in
:mod:`adversarial_reasoning_training.cli.eval_robust`. This shim
preserves the legacy ``python scripts/eval_robust.py ...`` invocation;
the same ``main()`` powers the ``art-eval-robust`` console script.
"""

from __future__ import annotations

from adversarial_reasoning_training.cli.eval_robust import main

if __name__ == "__main__":
    raise SystemExit(main())
