"""Adversarial fine-tune entrypoint shim.

The implementation lives in :mod:`adversarial_reasoning_training.cli.train`
so the same ``main()`` is callable as the ``art-train`` console script
defined in ``pyproject.toml``. This shim preserves the legacy
``python scripts/train.py ...`` invocation.
"""

from __future__ import annotations

from adversarial_reasoning_training.cli.train import main

if __name__ == "__main__":
    raise SystemExit(main())
