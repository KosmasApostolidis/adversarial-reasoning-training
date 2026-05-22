"""Path-handling helpers shared by CLI entrypoints and gate scripts.

Centralises the ``str | Path`` → resolved ``Path`` boundary so that callers
stop mixing raw strings and ``Path`` objects in the same call chain.
"""

from __future__ import annotations

from pathlib import Path


def normalize_run_dir(path: str | Path) -> Path:
    """Resolve ``path`` to an absolute ``Path`` and create it (with parents).

    Idempotent: repeated calls on the same path are a no-op after the first.
    Use at any boundary that previously called ``Path(path).mkdir(parents=True,
    exist_ok=True)`` ad hoc.
    """
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    resolved.chmod(0o700)
    return resolved


def ensure_parent(path: str | Path) -> Path:
    """Resolve ``path`` and ensure its parent directory exists.

    Returns the resolved file path itself (not the parent) so callers can
    pass it straight into open / write_text.
    """
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.parent.chmod(0o700)
    return resolved
