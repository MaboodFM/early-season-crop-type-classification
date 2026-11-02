"""Tests for completeness and strict numerical comparison of archived summaries."""
from pathlib import Path
import importlib.util

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_reference_results_complete():
    primary = pd.read_csv(ROOT / 'docs/reference_metrics.csv', dtype={'cutoff':str})
    assert len(primary) == 28
    assert len(primary[['year','algorithm','cutoff']].drop_duplicates()) == 28
    phase4 = pd.read_csv(ROOT / 'docs/reference_phase4_metrics.csv', dtype={'cutoff':str})
    assert len(phase4) == 1288
    assert not phase4.duplicated(['year','algorithm','cutoff','stage','metric']).any()
    assert set(phase4.stage) == {'bootstrap','repeated','county'}
    for algorithm in ('RF','LGBM'):
        for year in (2023,2024):
            selection = phase4.loc[phase4.algorithm.eq(algorithm) & phase4.year.eq(year)]
            assert len(selection) == 322


def test_primary_comparison_rejects_difference_and_missing_cutoff():
    check = module('compare_metrics')
    ref = pd.read_csv(ROOT / 'docs/reference_metrics.csv', dtype={'cutoff':str})
    actual = ref.drop(columns='reference_source').copy()
    assert check.compare(ref, actual, 1e-6).empty
    actual.loc[0,'test_accuracy'] += 0.01
    difference = check.compare(ref, actual, 1e-6)
    assert len(difference)==1 and difference.iloc[0]['metric']=='test_accuracy'
    with pytest.raises(ValueError, match='Missing reference rows'):
        check.compare(ref, actual.iloc[1:], 1e-6)


def test_robustness_comparison_rejects_difference_and_missing_stage():
    check = module('compare_robustness')
    ref = pd.read_csv(ROOT / 'docs/reference_phase4_metrics.csv',dtype={'cutoff':str})
    assert check.compare(ref, ref.copy(), 1e-6).empty
    actual=ref.copy()
    actual.loc[0,'value'] += 0.01
    assert len(check.compare(ref,actual,1e-6))==1
    with pytest.raises(ValueError,match='incomplete'):
        check.compare(ref,actual.loc[actual.stage.ne('county')],1e-6)


def test_original_fitting_thread_settings():
    from crop_type.modeling import build_pipeline
    assert build_pipeline('RF').named_steps['model'].n_jobs == -1
    assert build_pipeline('LGBM').named_steps['model'].n_jobs == -1
    assert build_pipeline('RF',n_jobs=1).named_steps['model'].n_jobs == 1
    assert build_pipeline('LGBM',n_jobs=1).named_steps['model'].n_jobs == 1
