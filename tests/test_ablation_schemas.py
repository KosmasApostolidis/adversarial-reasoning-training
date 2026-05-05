"""Schema validation for configs/ablations/*.yaml.

Each ablation YAML must be loadable by the same paths that production
training uses (``art-train --config | --defenses | --full-ft``) so a
silent typo in an ablation cell cannot reach an H200 run before the
first epoch.

Tests assert:

* every YAML parses,
* loss-axis configs match the training.yaml schema,
* defenses-axis configs match the defenses.yaml schema and reference an
  ε baseline that does not silently confound the ablation,
* freeze-axis configs match the full_ft.yaml schema and respect the T0
  peak-memory ceiling.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from adversarial_reasoning_training.cli.config import load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"
ABLATIONS = CONFIGS / "ablations"

LOSS_AXIS_FILES = ("loss_oaat.yaml", "loss_pgd_at.yaml")
BETA_AXIS_FILES = ("beta_0.yaml", "beta_1.yaml", "beta_12.yaml")
EPS_AXIS_FILES = (
    "defenses_eps_fixed_8.yaml",
    "defenses_eps_reverse.yaml",
    "defenses_eps_mid_only.yaml",
)
# Phase 6 Tier C #9 — pure-trajectory-KL novelty isolation.
TRAJ_ONLY_FILE = "loss_traj_only.yaml"
DEFENSES_AXIS_FILES = BETA_AXIS_FILES + EPS_AXIS_FILES + (TRAJ_ONLY_FILE,)
FREEZE_AXIS_FILES = ("full_ft_lm_only.yaml", "full_ft_vit_proj_frozen.yaml")

TRAINING_REQUIRED_KEYS = {
    "defense", "seed", "epochs", "micro_batch", "grad_accum",
    "optim", "lr", "schedule", "amp",
}
DEFENSES_REQUIRED_KEYS = {"pgd", "trades"}
FULL_FT_REQUIRED_KEYS = {"freeze_strategy", "memory"}

VALID_DEFENSE_VALUES = {"trades", "pgd_at", "oaat"}
VALID_FREEZE_STRATEGIES = {"none", "vit_only", "projector_only", "lm_only"}


@pytest.fixture(scope="module")
def baseline_training() -> dict:
    return load_yaml(CONFIGS / "training.yaml")


@pytest.fixture(scope="module")
def baseline_defenses() -> dict:
    return load_yaml(CONFIGS / "defenses.yaml")


@pytest.fixture(scope="module")
def baseline_full_ft() -> dict:
    return load_yaml(CONFIGS / "full_ft.yaml")


@pytest.mark.parametrize("name", LOSS_AXIS_FILES)
def test_loss_axis_yaml_matches_training_schema(name: str) -> None:
    cfg = load_yaml(ABLATIONS / name)
    missing = TRAINING_REQUIRED_KEYS - cfg.keys()
    assert not missing, f"{name} missing training keys: {missing}"
    assert cfg["defense"] in VALID_DEFENSE_VALUES, (
        f"{name}: invalid defense {cfg['defense']!r}; expected one of {VALID_DEFENSE_VALUES}"
    )
    assert {"lm", "projector", "vit"} <= cfg["lr"].keys(), (
        f"{name}: lr block must define lm/projector/vit"
    )


@pytest.mark.parametrize("name", LOSS_AXIS_FILES)
def test_loss_axis_epochs_anchored_to_curriculum(
    name: str, baseline_defenses: dict,
) -> None:
    """Adv-FT epochs must not exceed the eps_schedule end so the
    curriculum's strongest ε is actually reached."""
    cfg = load_yaml(ABLATIONS / name)
    schedule_max = max(
        entry["epoch_range"][1] for entry in baseline_defenses["pgd"]["eps_schedule"]
    )
    assert cfg["epochs"] <= schedule_max, (
        f"{name}: epochs={cfg['epochs']} exceeds eps_schedule max={schedule_max}; "
        "curriculum will fall back to default_eps and confound the ablation"
    )


@pytest.mark.parametrize("name", DEFENSES_AXIS_FILES)
def test_defenses_axis_yaml_matches_defenses_schema(name: str) -> None:
    cfg = load_yaml(ABLATIONS / name)
    missing = DEFENSES_REQUIRED_KEYS - cfg.keys()
    assert not missing, f"{name} missing defenses keys: {missing}"
    pgd = cfg["pgd"]
    for k in ("eps_schedule", "default_eps", "alpha_ratio", "steps", "eval_eps", "eval_steps"):
        assert k in pgd, f"{name}: pgd block missing {k}"


@pytest.mark.parametrize("name", BETA_AXIS_FILES)
def test_beta_files_share_eps_with_baseline(
    name: str, baseline_defenses: dict,
) -> None:
    """The β sweep must vary only TRADES.β. Verify the PGD block matches
    the baseline so β cannot be confounded with an ε change."""
    cfg = load_yaml(ABLATIONS / name)
    base_pgd = baseline_defenses["pgd"]
    assert cfg["pgd"]["default_eps"] == base_pgd["default_eps"], (
        f"{name}: default_eps drift confounds the β ablation"
    )
    assert cfg["pgd"]["eps_schedule"] == base_pgd["eps_schedule"], (
        f"{name}: eps_schedule drift confounds the β ablation"
    )
    assert cfg["pgd"]["steps"] == base_pgd["steps"]
    assert cfg["pgd"]["eval_eps"] == base_pgd["eval_eps"]


def test_eps_fixed_8_collapses_to_single_step() -> None:
    cfg = load_yaml(ABLATIONS / "defenses_eps_fixed_8.yaml")
    schedule = cfg["pgd"]["eps_schedule"]
    assert len(schedule) == 1, "fixed-8 ablation must use a single schedule entry"
    only = schedule[0]
    assert only["eps"] == pytest.approx(0.0314)
    assert cfg["pgd"]["default_eps"] == pytest.approx(0.0314)


def test_eps_mid_only_collapses_to_single_step_at_mid_budget() -> None:
    """Phase 6 Tier C #11 — constant 4/255 across all epochs."""
    cfg = load_yaml(ABLATIONS / "defenses_eps_mid_only.yaml")
    schedule = cfg["pgd"]["eps_schedule"]
    assert len(schedule) == 1, "mid-only ablation must use a single schedule entry"
    only = schedule[0]
    assert only["eps"] == pytest.approx(0.0157)
    assert cfg["pgd"]["default_eps"] == pytest.approx(0.0157)


def test_eps_reverse_inverts_baseline_curriculum(baseline_defenses: dict) -> None:
    """Phase 6 Tier C #11 — reverse curriculum must visit the same eps
    values as the baseline forward curriculum but in inverted order so
    epoch 1 uses the strongest budget instead of the weakest."""
    cfg = load_yaml(ABLATIONS / "defenses_eps_reverse.yaml")
    sched = cfg["pgd"]["eps_schedule"]
    base_eps_in_order = [e["eps"] for e in baseline_defenses["pgd"]["eps_schedule"]]
    reverse_eps_in_order = [e["eps"] for e in sched]
    assert reverse_eps_in_order == list(reversed(base_eps_in_order)), (
        "defenses_eps_reverse.yaml must mirror baseline eps values in reverse "
        "(strict 8/255 → 4/255 → 2/255)"
    )
    # Epoch ranges must still cover [1, training.epochs] without gaps.
    covered = set()
    for entry in sched:
        lo, hi = entry["epoch_range"]
        covered.update(range(lo, hi + 1))
    base_covered = set()
    for entry in baseline_defenses["pgd"]["eps_schedule"]:
        lo, hi = entry["epoch_range"]
        base_covered.update(range(lo, hi + 1))
    assert covered == base_covered, (
        f"defenses_eps_reverse.yaml epoch coverage {covered} differs from baseline {base_covered}; "
        "reverse curriculum would silently fall back to default_eps for uncovered epochs"
    )


def test_loss_traj_only_zeros_trades_task_weight() -> None:
    """Phase 6 Tier C #9 — the novelty-isolation cell must declare
    `trades.task_weight = 0`. Anything else (including default 1.0)
    would silently re-introduce the clean-CE term and confound the
    "pure trajectory-KL" claim."""
    cfg = load_yaml(ABLATIONS / TRAJ_ONLY_FILE)
    trades_block = cfg.get("trades") or {}
    assert "task_weight" in trades_block, (
        f"{TRAJ_ONLY_FILE}: trades.task_weight must be set explicitly to 0.0; "
        "missing key would default to 1.0 and reproduce canonical TRADES"
    )
    assert trades_block["task_weight"] == pytest.approx(0.0)


def test_loss_traj_only_keeps_canonical_eps_curriculum(
    baseline_defenses: dict,
) -> None:
    """The traj-only cell varies only the loss term — its ε schedule
    must match the baseline forward curriculum so the ablation isolates
    `task_weight` rather than confounding it with an ε change."""
    cfg = load_yaml(ABLATIONS / TRAJ_ONLY_FILE)
    base_pgd = baseline_defenses["pgd"]
    assert cfg["pgd"]["eps_schedule"] == base_pgd["eps_schedule"]
    assert cfg["pgd"]["default_eps"] == base_pgd["default_eps"]
    assert cfg["pgd"]["steps"] == base_pgd["steps"]


@pytest.mark.parametrize("name", FREEZE_AXIS_FILES)
def test_freeze_axis_yaml_matches_full_ft_schema(
    name: str, baseline_full_ft: dict,
) -> None:
    cfg = load_yaml(ABLATIONS / name)
    missing = FULL_FT_REQUIRED_KEYS - cfg.keys()
    assert not missing, f"{name} missing full_ft keys: {missing}"
    assert cfg["freeze_strategy"] in VALID_FREEZE_STRATEGIES, (
        f"{name}: invalid freeze_strategy {cfg['freeze_strategy']!r}"
    )
    # Memory ceiling must match the baseline so peak-mem comparisons
    # across freeze cells are apples-to-apples.
    assert cfg["memory"]["peak_memory_limit_gb"] == baseline_full_ft["memory"][
        "peak_memory_limit_gb"
    ], f"{name}: peak_memory_limit_gb drift confounds the freeze comparison"


def test_freeze_axis_actually_changes_strategy() -> None:
    """Each freeze ablation must differ from the baseline freeze_strategy."""
    base = load_yaml(CONFIGS / "full_ft.yaml")["freeze_strategy"]
    for name in FREEZE_AXIS_FILES:
        cfg = load_yaml(ABLATIONS / name)
        assert cfg["freeze_strategy"] != base, (
            f"{name}: freeze_strategy unchanged from baseline ({base}); "
            "the ablation toggles nothing"
        )


def test_loss_axis_actually_changes_defense() -> None:
    """Each loss ablation must differ from the baseline defense selector."""
    base = load_yaml(CONFIGS / "training.yaml")["defense"]
    for name in LOSS_AXIS_FILES:
        cfg = load_yaml(ABLATIONS / name)
        assert cfg["defense"] != base, (
            f"{name}: defense unchanged from baseline ({base}); "
            "the ablation toggles nothing"
        )


def test_baseline_training_curriculum_alignment(
    baseline_training: dict, baseline_defenses: dict,
) -> None:
    """Regression guard for the budget-parity bug fixed in this branch:
    bumping epochs past the curriculum endpoint silently makes adv-FT
    train at default_eps for the tail epochs."""
    schedule_max = max(
        entry["epoch_range"][1] for entry in baseline_defenses["pgd"]["eps_schedule"]
    )
    assert baseline_training["epochs"] <= schedule_max, (
        f"training.yaml epochs={baseline_training['epochs']} > eps_schedule max={schedule_max}; "
        "extend defenses.yaml::pgd.eps_schedule before raising epochs"
    )
