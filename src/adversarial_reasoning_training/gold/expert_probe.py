"""Expert-reviewed probe set: hold-out of ~50 clinician-audited trajectories.

Probe trajectories are stored as JSON Lines with the same shape as
`Trajectory.to_jsonl()`. Use `load_expert_probe` to pull them back for
gate T1 evaluation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from adversarial_reasoning.agents.base import ToolCall, Trajectory

logger = logging.getLogger(__name__)


def _trajectory_from_dict(d: dict) -> Trajectory:
    calls = [ToolCall(**c) for c in d.get("tool_calls", [])]
    return Trajectory(
        task_id=d["task_id"],
        model_id=d.get("model_id", "expert"),
        seed=int(d.get("seed", 0)),
        tool_calls=calls,
        final_answer=d.get("final_answer", ""),
        reasoning_trace=d.get("reasoning_trace", ""),
        metadata=d.get("metadata", {}),
    )


def load_expert_probe(path: str | Path) -> list[tuple[str, Trajectory]]:
    """Load (sample_id, Trajectory) pairs from a probe JSONL file.

    Expected per-line fields: sample_id, task_id, tool_calls, final_answer,
    reasoning_trace, metadata. Missing files return an empty list.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("expert probe file not found at %s — returning empty list", p)
        return []
    import logging

    _log = logging.getLogger(__name__)
    out: list[tuple[str, Trajectory]] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                _log.warning(
                    "Skipping malformed JSON in expert probe %s:%d — %s",
                    p, lineno, line[:80],
                )
                continue
            sid = d.get("sample_id") or d.get("metadata", {}).get("sample_id") or "?"
            out.append((sid, _trajectory_from_dict(d)))
    return out


def save_expert_probe(
    pairs: list[tuple[str, Trajectory]],
    path: str | Path,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for sid, traj in pairs:
            blob = json.loads(traj.to_jsonl())
            blob["sample_id"] = sid
            f.write(json.dumps(blob, ensure_ascii=False) + "\n")
