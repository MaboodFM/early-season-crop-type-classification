"""End-of-season, class-specific Tree SHAP attribution for fitted models."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score

from .data import BANDS, CLASSES, ID_COLS, heldout_partition, read_study_year, signature
from .modeling import SEED

LOG=logging.getLogger(__name__)
CUTOFF='11-01'


def _model_features(pipeline, raw_features):
    imputer=pipeline.named_steps['imputer']
    variance=pipeline.named_steps['variance']
    names=np.asarray(imputer.get_feature_names_out(raw_features.columns))
    names=names[variance.get_support()]
    values=pipeline[:-1].transform(raw_features)
    if values.shape[1]!=len(names):
        raise RuntimeError('Preprocessed feature dimensions do not match feature names')
    return pd.DataFrame(values,columns=names,index=raw_features.index)


def _contributions(algorithm, pipeline, transformed):
    estimator=pipeline.named_steps['model']
    explainer=shap.TreeExplainer(estimator,feature_perturbation='tree_path_dependent',
                                model_output='raw')
    result=explainer.shap_values(transformed,check_additivity=False)
    values=np.stack(result,axis=-1) if isinstance(result,list) else np.asarray(result)
    if values.shape!=(len(transformed),transformed.shape[1],len(CLASSES)):
        raise RuntimeError(f'Unexpected multiclass SHAP tensor shape: {values.shape}')
    expected=np.asarray(explainer.expected_value,dtype=float).reshape(-1)
    if expected.shape!=(len(CLASSES),):
        raise RuntimeError(f'Unexpected SHAP baseline shape: {expected.shape}')
    reconstructed=values.sum(axis=1)+expected[None,:]
    predicted=(estimator.predict_proba(transformed) if algorithm=='RF'
               else estimator.predict(transformed,raw_score=True))
    if reconstructed.shape!=predicted.shape:
        raise RuntimeError('SHAP output dimensions do not match model outputs')
    error=float(np.max(np.abs(reconstructed-predicted)))
    if error>1e-4:
        raise RuntimeError(f'SHAP additivity check failed: maximum absolute error {error:.4g}')
    return values,expected,error


def _explain_one(algorithm: str, year: int, data: Path, output_root: Path,
                 partition_file: Path, output: Path) -> dict:
    study=read_study_year(data,year)
    train,test=heldout_partition(study,reference=partition_file)
    source=output_root/f'{algorithm}_{year}_revised_results'
    manifest_path=source/'run_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Full primary model run required: {manifest_path}')
    main=json.loads(manifest_path.read_text())
    if main.get('pilot') or CUTOFF not in main['cutoffs']:
        raise ValueError('SHAP requires the full November 1 model run, not pilot outputs')
    if (main['algorithm']!=algorithm or int(main['year'])!=year
        or main['band_file_sha256']!=study.hashes
        or main['train_signature']!=signature(study.frame,train)
        or main['test_signature']!=signature(study.frame,test)):
        raise ValueError('Main model manifest and released data/partitions differ')
    model_file=source/'model_1101.joblib'
    if not model_file.exists():
        raise FileNotFoundError(f'Missing fitted November model: {model_file}')
    # Model artifacts are loaded only from the local trusted training stage.
    pipeline=joblib.load(model_file)
    raw=study.features(CUTOFF)
    actual=study.targets[test]
    predict=np.asarray(pipeline.predict(raw.iloc[test]),dtype=int)
    predictions=pd.read_csv(source/'heldout_field_predictions.csv',
                            dtype={'Crop':str,'aid':str,'County':str,'cutoff':str})
    saved=predictions.loc[predictions.cutoff.eq(CUTOFF)]
    if len(saved)!=len(test):
        raise ValueError('Held-out November predictions missing or incomplete')
    calculated=study.frame.iloc[test][list(ID_COLS)].reset_index(drop=True)
    calculated['true_class']=[CLASSES[i] for i in actual]
    calculated['predicted_class']=[CLASSES[i] for i in predict]
    verified=calculated.merge(saved[list(ID_COLS)+['true_class','predicted_class']],
                              on=list(ID_COLS),how='inner',validate='one_to_one',suffixes=('_new','_saved'))
    if len(verified)!=len(test) or any(not verified[f'{key}_new'].equals(verified[f'{key}_saved'])
                                     for key in ('true_class','predicted_class')):
        raise RuntimeError('Saved held-out predictions disagree with the fitted model')
    transformed=_model_features(pipeline,raw.iloc[test])
    values,expected,error=_contributions(algorithm,pipeline,transformed)
    columns=transformed.columns.to_list()
    by_band={band:np.flatnonzero([s.startswith(band+'__') for s in columns]) for band in BANDS}
    if any(len(ids)==0 for ids in by_band.values()):
        raise ValueError('Missing a required spectral band after feature filtering')
    grouped=[];individual=[]
    for index,crop in enumerate(CLASSES):
        for feature_idx,feature in enumerate(columns):
            col=values[:,feature_idx,index]
            individual.append({'year':year,'algorithm':algorithm,'cutoff':CUTOFF,
                               'crop':crop,'feature':feature,'band':feature.split('__',1)[0],
                               'mean_abs_shap':float(np.abs(col).mean()),
                               'mean_signed_shap':float(col.mean()),
                               'pct_positive':float(100*np.mean(col>0)),
                               'pct_negative':float(100*np.mean(col<0))})
        for band in BANDS:
            ids=by_band[band]
            signed=values[:,ids,index].sum(axis=1)
            absolute=np.abs(signed)
            grouped.append({'year':year,'algorithm':algorithm,'cutoff':CUTOFF,
                            'crop':crop,'band':band,'n_date_features':len(ids),
                            'mean_abs_group_shap':float(absolute.mean()),
                            'sd_abs_group_shap':float(absolute.std(ddof=1)),
                            'mean_signed_group_shap':float(signed.mean()),
                            'pct_group_positive':float(100*np.mean(signed>0)),
                            'pct_group_negative':float(100*np.mean(signed<0)),
                            'sum_feature_mean_abs':float(np.abs(values[:,ids,index]).mean(axis=0).sum())})
    grouped=pd.DataFrame(grouped)
    total=grouped.groupby('crop')['mean_abs_group_shap'].transform('sum')
    grouped['normalized_share_pct']=np.where(total>0,
        100*grouped['mean_abs_group_shap']/total,np.nan)
    grouped['shap_rank']=grouped.groupby('crop')['mean_abs_group_shap'].rank(
        ascending=False,method='min').astype(int)
    details=pd.DataFrame(individual)
    permutation=pd.read_csv(source/'permutation_mean_sd_by_band_crop.csv')
    permutation=permutation.loc[permutation.cutoff.eq(CUTOFF),
          ['crop','band','mean_f1_drop','sd_f1_drop','positive_share_pct']].rename(
          columns={'positive_share_pct':'permutation_positive_share_pct'})
    native=pd.read_csv(source/'native_global_band_importance.csv')
    native=native.loc[native.cutoff.eq(CUTOFF)].drop(columns=['year','cutoff'])
    native_key='gini_impurity_decrease' if algorithm=='RF' else 'total_split_gain'
    comparison=grouped.merge(permutation,on=['crop','band'],how='left',validate='one_to_one')
    comparison=comparison.merge(native[['band',native_key]],on='band',how='left',validate='many_to_one')
    if comparison[[native_key,'mean_f1_drop']].isna().any().any():
        raise RuntimeError('Incomplete native or permutation importance outputs')
    comparison['permutation_rank']=comparison.groupby('crop')['mean_f1_drop'].rank(
        ascending=False,method='min').astype(int)
    native_rank=native.set_index('band')[native_key].rank(ascending=False,method='min').astype(int)
    comparison['native_global_rank']=comparison['band'].map(native_rank).astype(int)
    grouped.sort_values(['crop','shap_rank','band']).to_csv(
        output/f'{algorithm}_{year}_Nov01_SHAP_band_by_crop.csv',index=False)
    details.sort_values(['crop','mean_abs_shap'],ascending=[True,False]).to_csv(
        output/f'{algorithm}_{year}_Nov01_SHAP_date_feature_by_crop.csv',index=False)
    comparison.sort_values(['crop','shap_rank']).to_csv(
        output/f'{algorithm}_{year}_Nov01_SHAP_permutation_native_comparison.csv',index=False)
    table=grouped.pivot(index='band',columns='crop',values='normalized_share_pct').reindex(
        index=BANDS,columns=CLASSES).reset_index()
    table.to_csv(output/f'{algorithm}_{year}_Nov01_SHAP_class_percent_table.csv',index=False)
    np.savez_compressed(output/f'{algorithm}_{year}_Nov01_SHAP_raw_arrays.npz',
        shap_values=values.astype(np.float32),
        transformed_feature_values=transformed.to_numpy(dtype=np.float32),
        feature_names=np.asarray(columns),crop_names=np.asarray(CLASSES),
        expected_values=expected,
        test_record_keys=np.asarray(study.groups[test]))
    for idx,crop in enumerate(CLASSES):
        plt.close('all')
        try:
            shap.summary_plot(values[:,:,idx],features=transformed,
                              feature_names=columns,max_display=15,show=False,plot_type='dot')
            fig=plt.gcf()
            fig.set_size_inches(11,7)
            fig.suptitle(f'{algorithm} {year} — {crop} (Nov 1)',y=0.995)
            fig.savefig(output/f'{algorithm}_{year}_{crop}_SHAP_beeswarm.png',
                        dpi=190,bbox_inches='tight')
        finally:
            plt.close('all')
    fig,axis=plt.subplots(figsize=(10,5))
    table.set_index('band')[list(CLASSES)].plot(kind='bar',ax=axis,rot=25)
    axis.set_xlabel('Spectral band')
    axis.set_ylabel('Mean |group SHAP| share (%)')
    axis.set_title(f'{algorithm} {year}: end-of-season spectral-band attribution')
    axis.legend(title='Crop')
    fig.tight_layout()
    fig.savefig(output/f'{algorithm}_{year}_Nov01_SHAP_band_summary.png',
                dpi=190,bbox_inches='tight')
    plt.close(fig)
    LOG.info('%s %s SHAP complete; max additivity error %.3g',algorithm,year,error)
    return {'year':year,'algorithm':algorithm,'cutoff':CUTOFF,
            'n_train':len(train),'n_test':len(test),
            'n_raw_features':len(raw.columns),'n_retained_features':len(columns),
            'reconstructed_accuracy':float(accuracy_score(actual,predict)),
            'original_test_signature':main['test_signature'],
            'shap_max_additivity_error':error,
            'output_space':'probability' if algorithm=='RF' else 'raw_class_margin',
            'shap_perturbation':'tree_path_dependent'}


def run(algorithm: str, years: tuple[int, ...], data: Path,
        output_root: Path, partitions: Path) -> Path:
    if algorithm not in ('RF','LGBM'):
        raise ValueError(algorithm)
    output=output_root/f'{algorithm}_November_SHAP_2023_2024'
    output.mkdir(parents=True,exist_ok=True)
    reports=[_explain_one(algorithm,year,data,output_root,partitions,output) for year in years]
    pd.DataFrame(reports).to_csv(output/f'{algorithm}_Nov01_SHAP_verification.csv',index=False)
    (output/'README.txt').write_text(
        'End-of-season (November 1) SHAP attribution on held-out fields.\n'
        'The analysis loads and validates the fitted models from the primary runs.\n'
        'TreeExplainer uses tree_path_dependent, model_output=raw. Random Forest\n'
        'explains class probabilities; LightGBM explains class raw margins.\n'
        'Within each crop and record, signed attributions across dates in a band\n'
        'are summed before taking the mean absolute value across test records.\n'
        'Magnitudes are not directly comparable across algorithms and are not causal.\n',encoding='utf-8')
    return output
