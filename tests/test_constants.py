"""Tests for the shared constants module.

Pins the public surface used by gates, trainer, and CLI so that drifting
literals (e.g. a stray ``"internvl2"`` in a fresh module) is caught quickly.
"""

from __future__ import annotations

from adversarial_reasoning_training.utils import constants


def test_vlm_family_values_match_canonical_dispatch_keys() -> None:
    assert constants.VLMFamily.QWEN_VL.value == "qwen_vl"
    assert constants.VLMFamily.LLAVA_NEXT.value == "llava_next"
    assert constants.VLMFamily.INTERNVL2.value == "internvl2"


def test_vlm_family_is_str_enum_for_backwards_compatibility() -> None:
    family = constants.VLMFamily.INTERNVL2
    assert isinstance(family, str)
    assert family == "internvl2"
    assert f"{family}" == "internvl2"


def test_vlm_family_membership_check_supports_raw_strings() -> None:
    valid = {f.value for f in constants.VLMFamily}
    assert "qwen_vl" in valid
    assert "llava_next" in valid
    assert "internvl2" in valid
    assert "qwen" not in valid


def test_grad_clip_norm_default() -> None:
    assert constants.GRAD_CLIP_NORM == 1.0


def test_default_freeze_strategy() -> None:
    assert constants.DEFAULT_FREEZE_STRATEGY == "none"


def test_default_gate_config_paths_keys() -> None:
    paths = constants.DEFAULT_GATE_CONFIG_PATHS
    assert paths["defenses"] == "configs/defenses.yaml"
    assert paths["data"] == "configs/data.yaml"
    assert paths["gold"] == "configs/gold.yaml"


def test_existing_constants_unchanged() -> None:
    assert constants.BYTE_SCALE == 1.0 / 255.0
    assert constants.EPS_2_255 == 2.0 / 255.0
    assert constants.DEFAULT_PGD_ALPHA_RATIO == 0.25
