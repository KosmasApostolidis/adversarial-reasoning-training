"""Gold-trajectory cache populator shim.

The implementation lives in
:mod:`adversarial_reasoning_training.cli.make_gold`. This shim preserves
the legacy ``python scripts/make_gold_trajectories.py ...`` invocation;
the same ``main()`` powers the ``art-make-gold`` console script.
"""

from __future__ import annotations

from adversarial_reasoning_training.cli.make_gold import main

if __name__ == "__main__":
    raise SystemExit(main())
