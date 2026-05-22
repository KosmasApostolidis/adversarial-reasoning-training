"""Adversarial reasoning training: robust fine-tuning of VLM medical agents.

Sibling to `adversarial_reasoning` (attacks repo). This package adds the
training loop; attack + eval machinery is imported from that package.
"""

from __future__ import annotations

import os as _os
import warnings as _warnings

# -- suppress third-party diagnostic noise (TF, XLA, Transformers) ----------
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# HuggingFace deprecation + dtype warnings
_warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers"
)
_warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
# runpy RuntimeWarning about prior-to-execution module
_warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")

__version__ = "0.1.0"
