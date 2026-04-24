"""Pytest fixtures + path configuration.

Adds the repo's ``src/`` to ``sys.path`` so tests can import
``adversarial_reasoning_training.*`` even when the package isn't
installed in editable mode (useful in CI before the install step).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
