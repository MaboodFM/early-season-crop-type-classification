"""Spectral profile diagnostics of November 1 held-out classification errors.

Profiles use 14-day observed-reflectance medians; no interpolation is applied.
The comparisons are descriptive and are not causal explanations.
"""
import io
import json
import re
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .data import read_study_year, heldout_partition, signature

BANDS = [('Blue','B02-'), ('Green','B03-'), ('Red','B04-'),
         ('Red-edge 1','B05-'), ('Red-edge 2','B06-'),
         ('SWIR1','B11-'), ('SWIR2','B12-')]
YEARS = (2023, 2024)
KEYS = ['Crop','aid','County']
CROPS = ['Corn','Cotton','Soy']


def mkkey(df):
    if df[KEYS].isna().any().any():
        raise RuntimeError('Missing values in Crop/aid/County identifiers')
    return df[KEYS].astype(str).agg('|'.join, axis=1)


def canon(s):
    x = str(s).strip()
    if x.lower().startswith('soy'): return 'Soy'
    if x.lower().startswith('corn'): return 'Corn'
    if x.lower().startswith('cotton'): return 'Cotton'
    raise RuntimeError(f'Unknown crop class: {s!r}')


def load_predictions(input_dir, model):
    """Load saved held-out predictions for ONE model only, both years."""
    records = []
    for year in YEARS:
        path = input_dir / f'{model}_{year}_revised_results' / 'heldout_field_predictions.csv'
        if not path.is_file():
            raise FileNotFoundError(path)
        p = pd.read_csv(path, dtype={'Crop':str,'aid':str,'County':str,'cutoff':str})
        p = p.loc[p['cutoff'].astype(str).str.strip().eq('11-01')].copy()
        if p.empty:
            raise RuntimeError(f'No November 1 predictions: {path}')
        p['field_key'] = mkkey(p)
        p['true_class'] = p['true_class'].map(canon)
        p['predicted_class'] = p['predicted_class'].map(canon)
        p['correct'] = (p.true_class == p.predicted_class)
        if p.field_key.duplicated().any():
            raise RuntimeError(f'Duplicate field predictions: {path}')
        p['year'] = year
        p['model'] = model
        records.append(p[['year','model'] + KEYS +
                         ['field_key','true_class','predicted_class','correct']])
    preds = pd.concat(records, ignore_index=True)
    meta = preds[['year','field_key','true_class','correct']].copy()
    return preds, meta


def load_hls_year(data_zip,year):
    y2=str(year)[-2:]
    pieces=[]; all_fields=None
    with zipfile.ZipFile(data_zip) as z:
        for band,prefix in BANDS:
            matches=[n for n in z.namelist() if f'/{y2}/' in '/'+n and
                     Path(n).name.startswith(prefix) and n.lower().endswith('.csv')]
            if len(matches)!=1: raise RuntimeError(f'Missing/ambiguous {year} {band}: {matches}')
            df=pd.read_csv(io.BytesIO(z.read(matches[0])),low_memory=False)
            if df.duplicated(KEYS).any(): raise RuntimeError(f'Duplicate keys in {matches[0]}')
            df['field_key']=mkkey(df)
            if all_fields is None:
                all_fields=df[KEYS+['field_key']].copy()
            date_cols=[col for col in df.columns if re.fullmatch(r'20\d\d-\d\d-\d\d',str(col))]
            if not date_cols: raise RuntimeError(f'No dated HLS columns in {matches[0]}')
            long=df.melt(id_vars=['field_key'],value_vars=date_cols,
                         var_name='date',value_name='reflectance')
            long['reflectance']=pd.to_numeric(long['reflectance'],errors='coerce')
            long=long.dropna(subset=['reflectance'])
            # No interpolation or imputation is used for diagnostic plots.
            long['date']=pd.to_datetime(long['date'])
            long=long.loc[(long.date>=pd.Timestamp(f'{year}-04-01')) &
                          (long.date<=pd.Timestamp(f'{year}-11-01'))]
            long['band']=band
            pieces.append(long)
    all_obs=pd.concat(pieces,ignore_index=True)
    anchor=pd.Timestamp(f'{year}-04-01')
    all_obs['bin_start']=anchor+pd.to_timedelta(
        ((all_obs['date']-anchor).dt.days//14)*14,unit='D')
    all_obs['year']=year
    return all_obs,all_fields


def prepare_binned(obs, meta, year):
    """Reference profiles from correctly classified held-out fields of THIS model.

    Using only test fields as references prevents contaminating the descriptive
    comparison with in-sample predictions. This comparison is still exploratory.
    """
    fieldbins = (obs.groupby(['field_key','band','bin_start'], as_index=False)
                 .agg(value=('reflectance','median'), n_observations=('reflectance','size')))
    mymeta = meta.loc[meta.year == year, ['field_key','true_class','correct']]
    refs = fieldbins.merge(mymeta, on='field_key', how='inner', validate='many_to_one')
    refs = refs[refs.correct].copy()
    profiles = (refs.groupby(['true_class','band','bin_start'], as_index=False)
                .agg(median_ref=('value','median'),
                     q25_ref=('value',lambda s:s.quantile(.25)),
                     q75_ref=('value',lambda s:s.quantile(.75)),
                     n_reference_fields=('field_key','nunique')))
    profiles = profiles.rename(columns={'true_class':'reference_class'})
    profiles['year'] = year
    return fieldbins, profiles


def comparison_for_error(predrow, binned, prof):
    # Get band-bin measurements of an incorrect field and compare to both
    # reference profiles. These distances are DESCRIPTIVE, not causal.
    k=predrow.field_key; true=predrow.true_class; proposed=predrow.predicted_class
    p=binned[binned.field_key==k].copy()
    t=prof[prof.reference_class==true][['band','bin_start','median_ref','q25_ref','q75_ref','n_reference_fields']]
    u=prof[prof.reference_class==proposed][['band','bin_start','median_ref','q25_ref','q75_ref','n_reference_fields']]
    a=p.merge(t,on=['band','bin_start'],how='left').rename(columns={
        'median_ref':'true_median','q25_ref':'true_q25','q75_ref':'true_q75','n_reference_fields':'n_true_ref'})
    a=a.merge(u,on=['band','bin_start'],how='left').rename(columns={
        'median_ref':'pred_median','q25_ref':'pred_q25','q75_ref':'pred_q75','n_reference_fields':'n_pred_ref'})
    a['abs_deviation_true']=(a.value-a.true_median).abs()
    a['abs_deviation_predicted']=(a.value-a.pred_median).abs()
    # A band-specific assessment only where both class reference profiles exist.
    a['comparable']=a.true_median.notna()&a.pred_median.notna()
    a['season_part']=pd.cut(a.bin_start.dt.month,
        bins=[0,5,7,9,12],labels=['Apr-May','Jun-Jul','Aug-Sep','Oct-Nov'])
    a['year']=predrow.year; a['model']=predrow.model
    a['true_class']=true; a['predicted_class']=proposed
    a['Crop']=predrow.Crop; a['aid']=predrow.aid; a['County']=predrow.County
    a['field_key']=k
    return a


def get_count(s):
    return int(s.sum()) if len(s) else 0


def one_error_summary(row,detail,obs):
    comparable=detail[detail.comparable]
    dtrue=float(comparable.abs_deviation_true.mean()) if len(comparable) else np.nan
    dpred=float(comparable.abs_deviation_predicted.mean()) if len(comparable) else np.nan
    this=obs[obs.field_key==row.field_key]
    result=dict(year=row.year,model=row.model,**{x:getattr(row,x) for x in KEYS},
        field_key=row.field_key,true_class=row.true_class,predicted_class=row.predicted_class,
        original_observations=int(len(this)),unique_valid_dates=int(this.date.nunique()),
        bands_with_observations=int(this.band.nunique()),
        valid_14day_band_bins=int(len(detail)),
        comparison_bins_both_classes=int(len(comparable)),
        mean_abs_deviation_true=dtrue,mean_abs_deviation_predicted=dpred,
        closer_profile=('predicted' if dpred<dtrue else 'true') if np.isfinite(dtrue) and np.isfinite(dpred) else 'insufficient data',
        profile_distance_difference_true_minus_pred=(dtrue-dpred) if np.isfinite(dtrue) and np.isfinite(dpred) else np.nan)
    for band,_ in BANDS:
        m=detail[detail.band==band]
        n=this[this.band==band]
        result['n_obs_'+band.replace(' ','_')]=int(len(n))
        result['n_bins_'+band.replace(' ','_')]=int(len(m))
    return result


def safe_id(v):
    return re.sub(r'[^0-9A-Za-z_-]+','_',str(v)).strip('_')


def draw_error_chart(p, binned, profiles, out, models_wrong):
    # One chart per FIELD x MODEL where model misclassified it.
    t=profiles[profiles.reference_class==p.true_class]
    r=profiles[profiles.reference_class==p.predicted_class]
    f=binned[binned.field_key==p.field_key]
    fig, axes=plt.subplots(4,2,figsize=(13.0,13.6),sharex=True)
    axes=axes.ravel()
    for i,(band,_) in enumerate(BANDS):
        ax=axes[i]
        tb=t[t.band==band].sort_values('bin_start')
        rb=r[r.band==band].sort_values('bin_start')
        fb=f[f.band==band].sort_values('bin_start')
        if len(tb):
            line=ax.plot(tb.bin_start,tb.median_ref,linewidth=1.7,label=f'True {p.true_class}: reference median')[0]
            ax.fill_between(tb.bin_start,tb.q25_ref.to_numpy(float),tb.q75_ref.to_numpy(float),
                            alpha=.14,color=line.get_color(),linewidth=0)
        if len(rb):
            line=ax.plot(rb.bin_start,rb.median_ref,linewidth=1.7,ls='--',label=f'Predicted {p.predicted_class}: reference median')[0]
            ax.fill_between(rb.bin_start,rb.q25_ref.to_numpy(float),rb.q75_ref.to_numpy(float),
                            alpha=.12,color=line.get_color(),linewidth=0)
        if len(fb):
            ax.plot(fb.bin_start,fb.value,linewidth=1.3,marker='o',markersize=3.1,
                    label='Misclassified field (14-day median)')
        else:
            ax.text(.5,.5,'No valid observations',transform=ax.transAxes,
                    ha='center',va='center')
        ax.set_title(band,fontsize=11,loc='left',weight='bold')
        ax.set_ylabel('HLS reflectance')
        ax.grid(alpha=.15)
        ax.tick_params(labelsize=8)
        ax.set_xlim(pd.Timestamp(f'{p.year}-04-01'),pd.Timestamp(f'{p.year}-11-06'))
    axes[7].axis('off')
    ls,labels=axes[0].get_legend_handles_labels()
    if ls:
        axes[7].legend(ls,labels,loc='center left',fontsize=9,frameon=False)
    fig.suptitle(f'{p.year} | Misclassified by {models_wrong} | {p.Crop} | {p.aid} | {p.County}\n'
                 f'True: {p.true_class}  |  Predicted: {p.predicted_class}',
                 fontsize=14,weight='bold',y=.99)
    fig.text(.51,.025,'Date (14-day median bins; shaded = reference interquartile range)',
             ha='center',fontsize=10)
    fig.text(.04,.006,f'Reference: fields correctly classified by {models_wrong} in the held-out test set. '
             'Patterns are descriptive, not established causes of model errors.',fontsize=8)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=[0,.045,1,.947])
    fig.savefig(out,dpi=125,bbox_inches='tight')
    plt.close(fig)


def plot_confusion_bar(preds,year,model,out):
    q=preds[(preds.year==year)&(preds.model==model)&(~preds.correct)]
    cats=[f'{a} → {b}' for a in CROPS for b in CROPS if a!=b]
    counts=(q.groupby(['true_class','predicted_class']).size()
             .reindex(pd.MultiIndex.from_tuples([(a,b) for a in CROPS for b in CROPS if a!=b]),fill_value=0))
    fig,ax=plt.subplots(figsize=(9,4.2))
    bars=ax.barh(cats,counts.to_numpy())
    ax.invert_yaxis()
    for bar,v in zip(bars,counts.to_numpy()):
        ax.text(float(v)+.1,bar.get_y()+bar.get_height()/2,str(int(v)),va='center')
    ax.set_xlim(0,max(int(counts.max())+1,2))
    ax.set_xlabel('Misclassified held-out fields')
    ax.set_title(f'{year} | {model} | November 1 errors (n={len(q)})')
    fig.tight_layout()
    fig.savefig(out,dpi=125,bbox_inches='tight')
    plt.close(fig)


def make_zip(folder,zip_path):
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file(): z.write(p,arcname=p.relative_to(folder))


def run(data_zip, results_root, model, output_dir, partitions, generate_figures=True):
    """Describe November 1 errors for a selected classifier and both years."""
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figs = output_dir / 'Figures'
    if generate_figures:
        figs.mkdir(exist_ok=True)
    data_zip = Path(data_zip)
    if not data_zip.is_file():
        raise FileNotFoundError(f'Missing {data_zip}')
    preds, meta = load_predictions(results_root, model)
    expected = {}
    for year in YEARS:
        result = results_root / f'{model}_{year}_revised_results'
        manifest = json.loads((result / 'run_manifest.json').read_text())
        if (manifest.get('pilot') or '11-01' not in manifest['cutoffs']
                or manifest.get('algorithm') != model or int(manifest.get('year')) != year):
            raise RuntimeError('Full primary model runs are required for diagnostics')
        study=read_study_year(data_zip,year)
        train,test=heldout_partition(study,reference=Path(partitions))
        if (manifest.get('band_file_sha256')!=study.hashes
            or manifest.get('test_signature')!=signature(study.frame,test)
            or manifest.get('train_signature')!=signature(study.frame,train)):
            raise ValueError('Primary-run source hashes and partitions do not match current data')
        reported=preds.loc[preds.year.eq(year)]
        if set(reported.field_key)!=set(study.groups[test]) or len(reported)!=len(test):
            raise ValueError('Predicted held-out field membership differs from fixed partition')
        crops=study.frame.set_index('group_id')['target'].to_dict()
        if any(crops[k]!=v for k,v in zip(reported.field_key,reported.true_class)):
            raise ValueError('Saved true classes differ from the released data')
        expected[year] = int(((preds.year == year) & (~preds.correct)).sum())
        print(f'Validated {model} {year}: {expected[year]} misclassified held-out fields')
    preds.to_csv(output_dir / 'All_original_heldout_predictions_Nov01.csv', index=False)

    all_profile=[]; all_detail=[]; all_summary=[]; all_raw=[]; all_coverage=[]
    for year in YEARS:
        obs, fieldmap = load_hls_year(data_zip, year)
        mypred = preds[preds.year == year]
        missing = set(mypred.field_key) - set(fieldmap.field_key)
        if missing:
            raise RuntimeError(f'{len(missing)} original held-out fields absent in HLS data for {year}')
        fieldbins, profiles = prepare_binned(obs, meta, year)
        correctly = meta[(meta.year == year) & (meta.correct)]
        print(f'{year}: {len(obs):,} valid band-date-field readings; '
              f'{len(correctly)} correctly classified {model} held-out reference fields')
        if len(correctly) < 50:
            raise RuntimeError('Too few reference fields for class profiles')
        all_profile.append(profiles)
        error_pred = mypred[~mypred.correct]
        error_obs = obs[obs.field_key.isin(error_pred.field_key)].copy()
        error_obs = error_obs.merge(fieldmap, on='field_key', how='left', validate='many_to_one')
        all_raw.append(error_obs)
        for row in error_pred.itertuples(index=False):
            detail = comparison_for_error(row, fieldbins, profiles)
            if detail.empty:
                raise RuntimeError(f'No spectral observations for {row.field_key}')
            all_detail.append(detail)
            all_summary.append(one_error_summary(row, detail, obs))
            if generate_figures:
                fname = (f'{year}_{model}_ERRORFIELD_{safe_id(row.Crop)}_'
                         f'{safe_id(row.aid)}_{safe_id(row.County)}.png')
                draw_error_chart(row, fieldbins, profiles, figs / fname, model)
        if generate_figures:
            plot_confusion_bar(preds, year, model, figs / f'{year}_{model}_Error_types.png')
        counts = (obs[obs.field_key.isin(mypred.field_key)]
                  .groupby(['field_key','band'], as_index=False).size()
                  .rename(columns={'size':'valid_observation_count'}))
        expected_index = pd.MultiIndex.from_product(
            [mypred.field_key.unique(), [b for b,_ in BANDS]], names=['field_key','band'])
        counts = (counts.set_index(['field_key','band'])
                  .reindex(expected_index, fill_value=0).reset_index())
        q = meta.loc[meta.year == year, ['field_key','true_class','correct']]
        counts = counts.merge(q, on='field_key', how='left', validate='many_to_one')
        counts['year'] = year
        counts['model'] = model
        all_coverage.append(counts)

    summary = pd.DataFrame(all_summary).sort_values(['year','field_key'])
    summary.to_csv(output_dir / 'Misclassified_field_diagnostic_summary.csv', index=False)
    detail = pd.concat(all_detail, ignore_index=True)
    detail.to_csv(output_dir / 'Misclassified_field_14day_deviations.csv', index=False)
    pd.concat(all_profile, ignore_index=True).to_csv(
        output_dir / 'Reference_class_profiles_14day.csv', index=False)
    pd.concat(all_raw, ignore_index=True).to_csv(
        output_dir / 'Misclassified_fields_original_HLS_observations.csv', index=False)
    pd.concat(all_coverage, ignore_index=True).to_csv(
        output_dir / 'Heldout_fields_valid_observation_counts.csv', index=False)

    by_band = (detail[detail.comparable]
               .groupby(['year','model','field_key','true_class','predicted_class','band'], as_index=False)
               .agg(n_comparable_bins=('value','size'),
                    mean_abs_deviation_from_true=('abs_deviation_true','mean'),
                    mean_abs_deviation_from_predicted=('abs_deviation_predicted','mean')))
    by_band['difference_true_minus_predicted'] = (by_band.mean_abs_deviation_from_true -
                                                   by_band.mean_abs_deviation_from_predicted)
    by_band.to_csv(output_dir / 'Error_by_band_profile_distances.csv', index=False)

    season = (detail[detail.comparable]
              .groupby(['year','model','field_key','true_class','predicted_class','season_part'],
                       observed=True, as_index=False)
              .agg(n_comparable_bins=('value','size'),
                   mae_to_true=('abs_deviation_true','mean'),
                   mae_to_predicted=('abs_deviation_predicted','mean')))
    season['distance_true_minus_predicted'] = season.mae_to_true - season.mae_to_predicted
    season.to_csv(output_dir / 'Error_by_season_profile_distances.csv', index=False)

    lines = [
        f'{model} NOVEMBER 1 MISCLASSIFICATION SPECTRAL DIAGNOSTICS',
        'Separate analyses for 2023 and 2024, original held-out fields only.',
        'No model training or replacement predictions.',
        f'Validated error totals: {year_counts_for_readme(expected)}.',
        '', 'METHOD:',
        '1. Read revised HLS date-specific reflectance from Data.zip through November 1.',
        '2. Bin each field-band series in 14-day bins from April 1, using valid medians.',
        f'3. Reference groups: fields correctly classified by {model} in the SAME',
        '   year and original held-out test partition (not training fields).',
        '   Plot their classwise per-bin median and 25th-75th percentiles.',
        '4. For each error, compare its reflectance with the reference distributions',
        '   of its true and incorrectly predicted classes.',
        '5. Report mean absolute differences across bins where both class',
        '   profiles are observed. Plots are DESCRIPTIVE, not causal.',
        '', 'LIMITATIONS:',
        '- These binned trajectories do not exactly reproduce the full-feature',
        '  multivariate classification decision.',
        '- Missing observations and crop-profile resemblance may contribute',
        '  to ambiguous classification but do not establish its cause.',
        '- Field area, mixed crops, planting date, and soil properties',
        '  were not supplied for independent verification.',
        '- Record identifiers use the revised Crop/aid/County fields; no coordinates.',
        '', 'OUTPUT FILES:',
        f'{len(summary)} error rows, one for each {model} error across both years.',
        'Misclassified_field_diagnostic_summary.csv: per-error summary.',
        'Misclassified_field_14day_deviations.csv: per-error/band/bin comparisons.',
        'Error_by_band_profile_distances.csv: comparisons by spectral band.',
        'Error_by_season_profile_distances.csv: comparisons by season segment.',
        'Reference_class_profiles_14day.csv: reference medians/IQR and counts.',
        'Heldout_fields_valid_observation_counts.csv: test data availability.',
        'Misclassified_fields_original_HLS_observations.csv: original readings.',
        'All_original_heldout_predictions_Nov01.csv: original predictions.',
        'Figures/*.png: one seven-band profile per error + error-type bars.',
    ]
    (output_dir / 'README_METHODS_AND_LIMITATIONS.txt').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print('\nErrors by year:')
    print(summary.groupby('year').size().to_string())
    zip_path = output_dir.parent / (output_dir.name + '.zip')
    make_zip(output_dir, zip_path)
    print('\nFinished. Output ZIP:', zip_path,
          f'({zip_path.stat().st_size / 1e6:.1f} MB)')
    return zip_path


def year_counts_for_readme(expected):
    return ', '.join(f'{year}={expected[year]}' for year in YEARS)

