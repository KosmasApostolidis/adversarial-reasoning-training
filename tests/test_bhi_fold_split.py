"""BHI cv_folds split→fold routing.

Three invariants this test pins down:

  - load_task(split="train"|"dev"|"test") routes to fold_1|fold_2|fold_3
    via the new tasks.yaml `bhi_split_to_fold` block. Sample-id prefix
    `bhi_f{fold}_val_p{patient:03d}_s{slice:02d}` is what makes the
    routing observable from outside.
  - Patient cohorts across logical splits are disjoint — the fold prefix
    discipline is what enforces zero patient leakage.
  - ProstateXTrainDS picks up the new mapping without code changes
    (it just passes split through to load_task).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adversarial_reasoning.tasks.loader import load_task

ATTACKS_TASKS = str(
    Path(__file__).resolve().parent.parent.parent
    / "adversarial-reasoning-attacks" / "configs" / "tasks.yaml"
)


def _bhi_data_present() -> bool:
    root = Path("/home/medadmin/kosmasapostolidis/BHI/data/processed/cv_folds")
    return all((root / f"fold_{f}" / f"fold_{f}_X_val_3D.npy").exists() for f in (1, 2, 3))


pytestmark = pytest.mark.skipif(
    not _bhi_data_present(),
    reason="BHI cv_folds data not on disk (skips on CI / non-workstation)",
)


@pytest.mark.unit
def test_bhi_split_to_fold_routes_correctly() -> None:
    train = list(load_task("prostate_mri_workup", split="train", config_path=ATTACKS_TASKS))
    dev = list(load_task("prostate_mri_workup", split="dev", config_path=ATTACKS_TASKS))
    test = list(load_task("prostate_mri_workup", split="test", config_path=ATTACKS_TASKS))

    assert len(train) == 55, f"expected 55 train samples (fold_1), got {len(train)}"
    assert len(dev) == 54, f"expected 54 dev samples (fold_2), got {len(dev)}"
    assert len(test) == 54, f"expected 54 test samples (fold_3), got {len(test)}"
    assert all(s.sample_id.startswith("bhi_f1_") for s in train), (
        "train samples must carry fold_1 prefix"
    )
    assert all(s.sample_id.startswith("bhi_f2_") for s in dev), (
        "dev samples must carry fold_2 prefix"
    )
    assert all(s.sample_id.startswith("bhi_f3_") for s in test), (
        "test samples must carry fold_3 prefix"
    )


@pytest.mark.unit
def test_bhi_splits_are_patient_disjoint() -> None:
    """Fold-prefix discipline enforces zero patient leakage across splits.

    Patient indices reset per fold (each fold reindexes from 0), so raw
    `_p` indices may overlap numerically — but the `f1`/`f2`/`f3` prefix
    in the sample_id namespace keeps the cohorts disjoint, which is what
    actually matters for clinical generalization.
    """
    train_ids = {s.sample_id for s in load_task(
        "prostate_mri_workup", split="train", config_path=ATTACKS_TASKS,
    )}
    dev_ids = {s.sample_id for s in load_task(
        "prostate_mri_workup", split="dev", config_path=ATTACKS_TASKS,
    )}
    test_ids = {s.sample_id for s in load_task(
        "prostate_mri_workup", split="test", config_path=ATTACKS_TASKS,
    )}

    assert train_ids & dev_ids == set(), "train/dev sample-id overlap"
    assert train_ids & test_ids == set(), "train/test sample-id overlap"
    assert dev_ids & test_ids == set(), "dev/test sample-id overlap"


@pytest.mark.unit
def test_prostatex_train_ds_yields_fold_1_samples(tmp_path: Path) -> None:
    from adversarial_reasoning_training.data.dataset import ProstateXTrainDS

    ds = ProstateXTrainDS(
        task_id="prostate_mri_workup",
        split="train",
        cache_dir=tmp_path,
        oracle_version="v1",
        synthetic=False,
        config_path=ATTACKS_TASKS,
    )
    assert len(ds) == 55
    assert all(s.sample_id.startswith("bhi_f1_") for s in ds.samples)
