"""Supplementary robustness evaluation of the two crop-type classifiers.

The resampling and tuning schedules follow the original Phase 4 notebooks.
Each search is checkpointed so the computationally intensive runs can resume.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from lightgbm import __version__ as lightgbm_version
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, StratifiedGroupKFold

from .data import CLASSES, CUTOFFS, heldout_partition, read_study_year, signature
from .modeling import RF_SEARCH, LGBM_SEARCH, SEED, build_pipeline, scores

LOG = logging.getLogger(__name__)
COUNTIES = frozenset(('Carroll','Crockett','Dyer','Fayette','Gibson','Hardeman',
                      'Haywood','Henry','Lake','Lauderdale','Madison','Obion',
                      'Tipton','Weakley'))
COUNTY_CORRECTIONS = {'Fayettee':'Fayette', 'Gobson':'Gibson'}
METRICS = ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'kappa')


@dataclass(frozen=True)
class RobustnessConfig:
    algorithm: str
    year: int
    data: Path
    output_root: Path
    partition_file: Path
    repeated_memberships: Path
    pilot: bool = False
    workers: int = 1

    @property
    def n_search(self): return 2 if self.pilot else 12

    @property
    def n_repeats(self): return 2 if self.pilot else 20

    @property
    def bootstrap_repeats(self): return 100 if self.pilot else 2000

    @property
    def county_folds(self): return 3 if self.pilot else 5

    @property
    def cutoffs(self): return CUTOFFS[:1] if self.pilot else CUTOFFS


def normalize_county(label: str) -> str:
    match = re.match(r'^([A-Za-z]+)', str(label).strip())
    if match is None:
        raise ValueError(f'Unrecognized county identifier: {label!r}')
    county = COUNTY_CORRECTIONS.get(match.group(1).title(), match.group(1).title())
    if county not in COUNTIES:
        raise ValueError(f'Unrecognized county: {label!r}')
    return county


def _search(config: RobustnessConfig, seed: int) -> RandomizedSearchCV:
    space = RF_SEARCH if config.algorithm == 'RF' else LGBM_SEARCH
    if config.pilot:
        space = {**space, 'model__n_estimators': [20, 40]}
    return RandomizedSearchCV(
        build_pipeline(config.algorithm, n_jobs=1), space, n_iter=config.n_search,
        scoring='f1_macro', refit=True,
        cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED+17),
        random_state=seed, n_jobs=config.workers,
        pre_dispatch=config.workers, error_score='raise', verbose=0,
    )


def repeated_partition_indices(study, reference_path: Path, n_repeats: int):
    """Use saved test membership; it is not affected by source-label spelling."""
    reference = pd.read_csv(reference_path, dtype={'field_key':str})
    reference = reference.loc[reference['year'].astype(int).eq(study.year)]
    keys = pd.Index(study.groups)
    assert keys.is_unique
    pairs = []
    for rep in range(1, n_repeats+1):
        subset = reference.loc[reference['repeat'].astype(int).eq(rep), 'field_key']
        if subset.empty or subset.duplicated().any() or not set(subset).issubset(set(keys)):
            raise ValueError(f'Invalid reference repeated split: {study.year}, {rep}')
        mask = keys.isin(subset)
        train = np.flatnonzero(~mask)
        test = np.flatnonzero(mask)
        if not set(np.unique(study.targets[train])) == {0, 1, 2}:
            raise ValueError('Repeated split missing a training crop class')
        pairs.append((train, test))
    return pairs


def _fingerprint(config, study, train_signature, test_signature, main_manifest):
    record = {
        'stage':'phase4-v2', 'year':config.year, 'algorithm':config.algorithm,
        'source_sha256':study.hashes, 'train_signature':train_signature,
        'test_signature':test_signature, 'full_main_cutoffs':main_manifest['cutoffs'],
        'n_search':config.n_search, 'n_repeats':config.n_repeats,
        'bootstrap_repeats':config.bootstrap_repeats,
        'county_folds':config.county_folds, 'cutoffs':config.cutoffs,
        'search_space':RF_SEARCH if config.algorithm=='RF' else LGBM_SEARCH,
        'seed':SEED, 'sklearn':sklearn.__version__, 'lightgbm':lightgbm_version,
        'repeated_membership_sha256': hashlib.sha256(config.repeated_memberships.read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(),record


class Checkpoints:
    def __init__(self, folder: Path, fingerprint: str, details: dict):
        folder.mkdir(parents=True,exist_ok=True)
        self.folder, self.fingerprint = folder, fingerprint
        config = folder/'configuration.json'
        if config.exists():
            if json.loads(config.read_text())['fingerprint'] != fingerprint:
                raise ValueError(f'Checkpoint configuration mismatch: {folder}')
        else:
            config.write_text(json.dumps({'fingerprint':fingerprint,'settings':details},indent=2))

    def task(self, stage: str, cutoff: str, number: int, compute):
        file = self.folder/f'{stage}_{cutoff}_{number:02d}.json'
        if file.exists():
            saved = json.loads(file.read_text())
            if saved['fingerprint'] != self.fingerprint:
                raise ValueError(f'Checkpoint fingerprint mismatch: {file}')
            return saved['result']
        result=compute()
        tmp=file.with_suffix('.json.tmp')
        tmp.write_text(json.dumps({'fingerprint':self.fingerprint,'result':result},allow_nan=False))
        os.replace(tmp,file)
        return result


def _bootstrap(config, study, test_idx, cutoff, predictions, summary):
    frame=predictions.loc[predictions.cutoff.eq(cutoff)].copy()
    keys=frame[['Crop','aid','County']].astype(str).agg('|'.join,axis=1)
    if len(frame)!=len(test_idx) or not keys.is_unique or set(keys)!=set(study.groups[test_idx]):
        raise ValueError(f'Invalid held-out prediction membership: {cutoff}')
    actual=study.frame.set_index('group_id')['target'].to_dict()
    if any(actual[k]!=v for k,v in zip(keys,frame.true_class)):
        raise ValueError('Saved true crop labels do not match the input data')
    y=np.asarray([CLASSES.index(v) for v in frame.true_class])
    pred=np.asarray([CLASSES.index(v) for v in frame.predicted_class])
    original=scores(y,pred)
    row=summary.loc[summary.cutoff.eq(cutoff)]
    if len(row)!=1 or any(not np.isclose(original[k],row.iloc[0][f'test_{k}'],atol=1e-10)
                          for k in METRICS):
        raise ValueError(f'Primary summary does not match saved predictions for {cutoff}')
    rng=np.random.default_rng(SEED+800000+CUTOFFS.index(cutoff))
    within=[np.flatnonzero(y==i) for i in range(3)]
    replicates={k:[] for k in METRICS}
    for _ in range(config.bootstrap_repeats):
        sampled=np.concatenate([rng.choice(ix,size=len(ix),replace=True) for ix in within])
        values=scores(y[sampled],pred[sampled])
        for key in METRICS:
            replicates[key].append(values[key])
    output=[]
    for key in METRICS:
        lo,hi=np.nanpercentile(replicates[key],[2.5,97.5])
        output.append({'year':config.year,'algorithm':config.algorithm,'cutoff':cutoff,
                       'metric':key,'n_test':len(y),'n_bootstrap':config.bootstrap_repeats,
                       'original_score':original[key], 'ci95_lower':float(lo),'ci95_upper':float(hi),
                       'ci_method':'class-stratified percentile; fixed test predictions; no refit'})
    return {'rows':output}


def _run_search(config, features, targets, train, test, cv_groups, tuning_seed):
    search=_search(config,tuning_seed)
    start=time.perf_counter()
    search.fit(features.iloc[train],targets[train],groups=cv_groups)
    elapsed=time.perf_counter()-start
    predictions=search.predict(features.iloc[test])
    return search,predictions,scores(targets[test],predictions),elapsed


def run(config: RobustnessConfig) -> Path:
    if config.algorithm not in ('RF','LGBM') or config.year not in (2023,2024):
        raise ValueError('Invalid model/year')
    study=read_study_year(config.data,config.year)
    train,test=heldout_partition(study,reference=config.partition_file)
    name=f'{config.algorithm}_{config.year}_revised_results'+('_PILOT' if config.pilot else '')
    parent=config.output_root/name
    if not (parent/'run_manifest.json').is_file():
        raise FileNotFoundError(f'Run the primary model before robustness: {parent}')
    main=json.loads((parent/'run_manifest.json').read_text())
    if (main['algorithm']!=config.algorithm or int(main['year'])!=config.year
        or main['band_file_sha256']!=study.hashes or main['cutoffs']!=list(config.cutoffs)
        or main['train_signature']!=signature(study.frame,train)
        or main['test_signature']!=signature(study.frame,test)):
        raise ValueError('The primary results do not match the current data and field membership')
    summary=pd.read_csv(parent/'test_and_nested_cv_summary.csv')
    predictions=pd.read_csv(parent/'heldout_field_predictions.csv',dtype={'Crop':str,'aid':str,'County':str,'cutoff':str})
    out=config.output_root/(f'{config.algorithm}_{config.year}_Phase4_ONE_RUN'+('_PILOT' if config.pilot else ''))
    out.mkdir(parents=True,exist_ok=True)
    fprint,details=_fingerprint(config,study,main['train_signature'],main['test_signature'],main)
    checkpoints=Checkpoints(out/'checkpoints',fprint,details)
    repeated=repeated_partition_indices(study,config.repeated_memberships,config.n_repeats)
    county=study.frame.County.map(normalize_county).to_numpy()
    table=pd.crosstab(county,study.frame.target).reindex(columns=CLASSES,fill_value=0)
    table['Total']=table.sum(axis=1)
    table.to_csv(out/'county_class_distribution.csv')
    county_splits=list(GroupKFold(n_splits=config.county_folds).split(study.frame,study.targets,county))
    bootstrap, repeat_rows, repeat_parameters, repeat_members=[],[],[],[]
    county_rows,county_parameters,county_predictions=[],[],[]
    timings=[]
    for cutoff in config.cutoffs:
        feature=study.features(cutoff)
        LOG.info('Phase 4 %s %d cutoff %s (%d features)',config.algorithm,config.year,cutoff,len(feature.columns))
        boot=checkpoints.task('BOOTSTRAP',cutoff,1,
            lambda:_bootstrap(config,study,test,cutoff,predictions,summary))
        bootstrap.extend(boot['rows'])
        for number,(tr,te) in enumerate(repeated,start=1):
            def compute_repeat(tr=tr,te=te,number=number):
                tune_seed=SEED+300000+(number-1)
                search,pred,metrics,elapsed=_run_search(config,feature,study.targets,tr,te,
                                             study.groups[tr],tune_seed)
                row={'year':config.year,'algorithm':config.algorithm,'cutoff':cutoff,
                     'repeat':number,'split_seed':SEED+100000+(number-1),
                     'tuning_seed':tune_seed,'n_train':len(tr),'n_test':len(te),
                     **{f'test_{c}':int(sum(study.targets[te]==i)) for i,c in enumerate(CLASSES)},
                     'train_signature':signature(study.frame,tr),'test_signature':signature(study.frame,te),
                     'inner_best_f1_macro':float(search.best_score_),**metrics}
                selected={'year':config.year,'algorithm':config.algorithm,'cutoff':cutoff,
                          'repeat':number,'parameters':json.dumps(search.best_params_,sort_keys=True),
                          'inner_best_f1_macro':float(search.best_score_)}
                members=[{'cutoff':cutoff,'repeat':number,'field_key':str(study.groups[i]),
                          'true_crop':CLASSES[study.targets[i]]} for i in te]
                return {'metric_row':row,'selected_row':selected,'members':members,'seconds':elapsed}
            result=checkpoints.task('REPEAT',cutoff,number,compute_repeat)
            if result['metric_row']['test_signature']!=signature(study.frame,te):
                raise ValueError('Reference repeated membership disagrees with checkpoint')
            repeat_rows.append(result['metric_row']);repeat_parameters.append(result['selected_row'])
            repeat_members.extend(result['members'])
            timings.append({'algorithm':config.algorithm,'year':config.year,'stage':'REPEAT',
                            'cutoff':cutoff,'iteration':number,'elapsed_seconds':result['seconds']})
        for fold,(tr,te) in enumerate(county_splits,start=1):
            held=set(county[te]); training=set(county[tr])
            if held & training:
                raise ValueError('County holdout split is not geographically disjoint')
            def compute_county(tr=tr,te=te,fold=fold,held=held,training=training):
                search,pred,metrics,elapsed=_run_search(config,feature,study.targets,tr,te,
                                              county[tr],SEED+400000+fold)
                row={'year':config.year,'algorithm':config.algorithm,'cutoff':cutoff,
                     'fold':fold,'heldout_counties':';'.join(sorted(held)),
                     'n_heldout_counties':len(held),'n_train':len(tr),'n_test':len(te),
                     'n_train_counties':len(training),
                     **{f'test_{c}':int(sum(study.targets[te]==i)) for i,c in enumerate(CLASSES)},
                     'all_three_classes_in_test':bool(len(np.unique(study.targets[te]))==3),
                     'inner_best_f1_macro':float(search.best_score_),**metrics}
                selected={'year':config.year,'algorithm':config.algorithm,'cutoff':cutoff,
                          'fold':fold,'parameters':json.dumps(search.best_params_,sort_keys=True),
                          'inner_best_f1_macro':float(search.best_score_)}
                pred_rows=[{'year':config.year,'algorithm':config.algorithm,'cutoff':cutoff,
                            'fold':fold,'county_normalized':str(county[i]),
                            'field_key':str(study.groups[i]),'true_class':CLASSES[study.targets[i]],
                            'predicted_class':CLASSES[int(p)]} for i,p in zip(te,pred)]
                return {'metric_row':row,'selected_row':selected,'predictions':pred_rows,'seconds':elapsed}
            result=checkpoints.task('COUNTY',cutoff,fold,compute_county)
            if result['metric_row']['heldout_counties']!=';'.join(sorted(held)):
                raise ValueError('County membership disagrees with checkpoint')
            county_rows.append(result['metric_row']);county_parameters.append(result['selected_row'])
            county_predictions.extend(result['predictions'])
            timings.append({'algorithm':config.algorithm,'year':config.year,'stage':'COUNTY',
                            'cutoff':cutoff,'iteration':fold,'elapsed_seconds':result['seconds']})
    exports={'heldout_bootstrap_95ci.csv':bootstrap,
             'repeated_splits_20_metrics.csv':repeat_rows,
             'repeated_splits_selected_parameters.csv':repeat_parameters,
             'repeated_splits_test_membership.csv':repeat_members,
             'county_holdout_fold_metrics.csv':county_rows,
             'county_holdout_selected_parameters.csv':county_parameters,
             'county_holdout_field_predictions.csv':county_predictions,
             'search_timing_log.csv':timings}
    for filename,rows in exports.items():
        pd.DataFrame(rows).to_csv(out/filename,index=False)
    repeat_df=pd.DataFrame(repeat_rows)
    repeat_summary=repeat_df.groupby(['year','algorithm','cutoff'],as_index=False).agg(
        **{k+'_mean':(k,'mean') for k in METRICS},
        **{k+'_sd':(k,'std') for k in METRICS},
        n_repeats=('repeat','size'),mean_n_test=('n_test','mean'))
    if not (repeat_summary.n_repeats==config.n_repeats).all():
        raise RuntimeError('Incomplete repeated splits')
    repeat_summary.to_csv(out/'repeated_splits_summary.csv',index=False)
    pooled=[]
    county_df=pd.DataFrame(county_rows)
    for cutoff in config.cutoffs:
        blocks=[row for row in county_predictions if row['cutoff']==cutoff]
        if len(blocks)!=len(study.frame) or len(set(x['field_key'] for x in blocks))!=len(study.frame):
            raise RuntimeError('County holdout does not cover every field exactly once')
        truth=np.asarray([CLASSES.index(x['true_class']) for x in blocks])
        predicted=np.asarray([CLASSES.index(x['predicted_class']) for x in blocks])
        pooled.append({'year':config.year,'algorithm':config.algorithm,'cutoff':cutoff,
                       'n_counties':len(set(county)),'n_folds':config.county_folds,
                       'n_oof_fields':len(blocks),'n_folds_with_all_3_test_classes':int(
                           county_df.loc[county_df.cutoff.eq(cutoff),'all_three_classes_in_test'].sum()),
                       **{'pooled_county_oof_'+k:v for k,v in scores(truth,predicted).items()}})
    pd.DataFrame(pooled).to_csv(out/'county_holdout_pooled_summary.csv',index=False)
    status=[{'cutoff':c,'bootstrap_complete':True,
             'repeated_splits_completed':config.n_repeats,'repeated_splits_required':config.n_repeats,
             'repeated_splits_complete':True,'county_folds_completed':config.county_folds,
             'county_folds_required':config.county_folds,'county_validation_complete':True}
            for c in config.cutoffs]
    pd.DataFrame(status).to_csv(out/'stage_completion_status.csv',index=False)
    manifest={'stage':'Phase4_robustness','algorithm':config.algorithm,'year':config.year,
              'pilot':config.pilot,'cutoffs':config.cutoffs,'source_sha256':study.hashes,
              'main_train_signature':main['train_signature'],'main_test_signature':main['test_signature'],
              'bootstrap_repeats':config.bootstrap_repeats,
              'repeated_splits_required':config.n_repeats,'county_holdout_folds_required':config.county_folds,
              'search_candidates':config.n_search,'inner_folds':3,'checkpoint_fingerprint':fprint,
              'stage_completion':status}
    (out/'phase4_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    LOG.info('Phase 4 finished: %s',out)
    return out
