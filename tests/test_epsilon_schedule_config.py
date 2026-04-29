"""Validate eps_schedule shape at startup, not mid-epoch.

A typo (``epoch_ranges`` vs ``epoch_range``) used to crash the inner-PGD
attack mid-epoch and waste hours of H200 wall time. ``validate_eps_schedule``
fails fast with a key-listing error so the bug surfaces before any training.
"""

from __future__ import annotations

import pytest

from adversarial_reasoning_training.attacks.inner_pgd import (
    epsilon_for_epoch,
    validate_eps_schedule,
)


@pytest.mark.unit
def test_validate_accepts_well_formed_schedule() -> None:
    schedule = [
        {"epoch_range": [1, 2], "eps": 0.0078},
        {"epoch_range": [3, 5], "eps": 0.0314},
    ]
    validate_eps_schedule(schedule)  # no raise


@pytest.mark.unit
def test_validate_accepts_none_or_empty() -> None:
    validate_eps_schedule(None)
    validate_eps_schedule([])


@pytest.mark.unit
def test_validate_rejects_typo_in_epoch_range_key() -> None:
    schedule = [{"epoch_ranges": [1, 2], "eps": 0.0078}]
    with pytest.raises(ValueError, match="epoch_range"):
        validate_eps_schedule(schedule)


@pytest.mark.unit
def test_validate_rejects_missing_eps_key() -> None:
    schedule = [{"epoch_range": [1, 2]}]
    with pytest.raises(ValueError, match="eps"):
        validate_eps_schedule(schedule)


@pytest.mark.unit
def test_validate_rejects_bad_range_shape() -> None:
    schedule = [{"epoch_range": [1, 2, 3], "eps": 0.0078}]
    with pytest.raises(ValueError, match="2-element"):
        validate_eps_schedule(schedule)


@pytest.mark.unit
def test_epsilon_for_epoch_skips_malformed_entry_safely() -> None:
    # If a malformed entry slips past validation (e.g. user constructs the
    # schedule programmatically), ``epsilon_for_epoch`` must skip rather
    # than KeyError mid-attack.
    schedule = [
        {"foo": "bar"},  # bogus
        {"epoch_range": [1, 5], "eps": 0.05},
    ]
    assert epsilon_for_epoch(3, schedule, default_eps=0.01) == pytest.approx(0.05)


@pytest.mark.unit
def test_epsilon_for_epoch_default_when_no_match() -> None:
    schedule = [{"epoch_range": [1, 2], "eps": 0.05}]
    assert epsilon_for_epoch(99, schedule, default_eps=0.01) == pytest.approx(0.01)
