"""CheckpointRegistry rotation + weights-only save.

Two invariants this test pins down:

  - Saving twice in the same dir leaves at most one ``step*.pt`` (the
    most recent), even when both saves happen at the same global_step
    with different timestamps. This is what kept us from accumulating
    multiple 47 GB ckpts during the smoke run.
  - ``save_weights_only`` writes a weights-only payload — used by the
    ``save_every`` periodic cadence so disk doesn't explode on long
    training runs.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from adversarial_reasoning_training.trainer.ckpt import CheckpointRegistry


@pytest.mark.unit
def test_rotation_keeps_only_latest_step_ckpt(tmp_path: Path) -> None:
    reg = CheckpointRegistry(tmp_path)
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    reg.save(model=model, optimizer=optimizer, step=5, epoch=1, metric_value=None)
    time.sleep(1.1)  # ensure a distinct %H%M%S in the second filename
    reg.save(model=model, optimizer=optimizer, step=5, epoch=1, metric_value=None)

    step_ckpts = sorted(tmp_path.glob("step*.pt"))
    assert len(step_ckpts) == 1, f"expected 1 step ckpt after rotation, got {step_ckpts}"


@pytest.mark.unit
def test_save_weights_only_omits_optim_state(tmp_path: Path) -> None:
    reg = CheckpointRegistry(tmp_path)
    model = torch.nn.Linear(4, 4)

    path = reg.save_weights_only(
        model=model, step=1, epoch=1, metric_value=None,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in payload
    assert "optim_state_dict" not in payload, (
        "save_weights_only must omit optimizer state to keep periodic "
        "saves disk-cheap on long training runs"
    )


@pytest.mark.unit
def test_save_includes_optim_state(tmp_path: Path) -> None:
    reg = CheckpointRegistry(tmp_path)
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer.zero_grad()
    # Take one step so the optimizer state is non-empty.
    out = model(torch.randn(2, 4)).sum()
    out.backward()
    optimizer.step()

    path = reg.save(model=model, optimizer=optimizer, step=1, epoch=1, metric_value=None)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "optim_state_dict" in payload
