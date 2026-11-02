from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crop_type.data import CUTOFFS, read_study_year
from crop_type.modeling import build_pipeline
from crop_type.robustness import (RobustnessConfig, normalize_county,
                                  repeated_partition_indices)
from crop_type.shap_analysis import _contributions, _model_features

ROOT = Path(__file__).resolve().parents[1]


def test_public_study_data_and_cutoffs():
    assert CUTOFFS == ('06-01','06-15','07-01','08-01','09-01','10-01','11-01')
    for year,count in ((2023,1016),(2024,1181)):
        study=read_study_year(ROOT/'data/Data.zip',year)
        assert len(study.frame)==count
        assert set(study.frame.Crop)=={'Corn 1','Corn 2','Cotton 1','Cotton 2','Soy 1','Soy 2'}
        assert not any('Crop' in c or 'County' in c for c in study.features('11-01').columns)


def test_phase4_reference_memberships_and_normalization():
    assert normalize_county('Gobson6')=='Gibson'
    assert normalize_county('Fayettee4')=='Fayette'
    with pytest.raises(ValueError):
        normalize_county('NotACounty6')
    for year in (2023,2024):
        study=read_study_year(ROOT/'data/Data.zip',year)
        pairs=repeated_partition_indices(study,ROOT/'data/reference_phase4_resplits.csv',20)
        assert len(pairs)==20
        for train,test in pairs:
            assert len(train)+len(test)==len(study.frame)
            assert not set(study.groups[train])&set(study.groups[test])


def test_robustness_settings():
    config=RobustnessConfig('RF',2023,ROOT/'data/Data.zip',ROOT/'results',
                            ROOT/'data/reference_partitions.csv',
                            ROOT/'data/reference_phase4_resplits.csv')
    assert (config.n_search,config.n_repeats,config.bootstrap_repeats,config.county_folds)==(12,20,2000,5)
    assert len(config.cutoffs)==7


@pytest.mark.parametrize('algorithm',['RF','LGBM'])
def test_multiclass_shap_additivity(algorithm):
    rng=np.random.default_rng(42)
    X=pd.DataFrame(rng.normal(size=(75,5)),columns=[f'Blue__2023-0{i+4}-10' for i in range(5)])
    y=np.arange(len(X))%3
    fitted=build_pipeline(algorithm)
    fitted.set_params(model__n_estimators=10)
    if algorithm=='LGBM':
        fitted.set_params(model__min_child_samples=5)
    fitted.fit(X,y)
    transformed=_model_features(fitted,X.iloc[:5])
    values,expected,error=_contributions(algorithm,fitted,transformed)
    assert values.shape==(5,5,3)
    assert expected.shape==(3,)
    assert error<1e-4
