"""Validation of the released input data, annual splits and model configuration."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crop_type.data import (CLASSES, CUTOFFS, ID_COLS, LABELS, read_study_year,
                            heldout_partition, signature)
from crop_type.modeling import RunConfig, build_pipeline, make_search, scores

DATA = Path(__file__).resolve().parents[1] / "data" / "Data.zip"
PARTITIONS = DATA.with_name("reference_partitions.csv")


@pytest.mark.parametrize("year,n_fields,n_train,n_test", [
    (2023, 1016, 812, 204), (2024, 1181, 944, 237),
])
def test_annual_inputs_and_fixed_partition(year, n_fields, n_train, n_test):
    study = read_study_year(DATA, year)
    assert len(study.frame) == n_fields
    assert set(study.frame.Crop.unique()) == set(LABELS)
    assert set(study.frame.target.unique()) == set(CLASSES)
    assert study.frame[list(ID_COLS)].notna().all().all()
    assert not study.frame.duplicated(list(ID_COLS)).any()
    train, test = heldout_partition(study, reference=PARTITIONS)
    assert len(train) == n_train and len(test) == n_test
    assert not set(study.groups[train]).intersection(study.groups[test])
    assert len(signature(study.frame, test)) == 64
    assert not study.features(CUTOFFS[0]).empty
    assert study.features(CUTOFFS[-1]).shape[1] >= study.features(CUTOFFS[0]).shape[1]
    assert all(c.startswith(tuple(f"{b}__" for b in (
        "Blue", "Green", "Red", "RedEdge1", "RedEdge2", "SWIR1", "SWIR2"
    ))) for c in study.features(CUTOFFS[0]))


def test_model_configuration():
    for model in ("RF", "LGBM"):
        cfg = RunConfig(model, 2023, DATA, Path("/tmp"))
        assert cfg.n_search == 12
        assert cfg.outer_folds == 5 and cfg.inner_folds == 3
        assert cfg.n_permutations == 20
        search = make_search(cfg)
        assert search.scoring == "f1_macro"
        assert search.n_iter == 12
        assert list(build_pipeline(model).named_steps) == ["imputer", "variance", "model"]
    assert scores(np.array([0, 1, 2]), np.array([0, 1, 2]))["accuracy"] == 1


def test_main_outputs_use_numerical_groups_only():
    released = pd.read_csv(PARTITIONS)
    assert not released.duplicated(["year", *ID_COLS]).any()
    assert set(released.partition) == {"TRAIN", "TEST"}
    assert set(released.Crop) == set(LABELS)
