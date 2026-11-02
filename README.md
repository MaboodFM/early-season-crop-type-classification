# Early-season crop-type classification

This repository provides the code and data used for early-season classification of corn, cotton, and soybean fields in West Tennessee using multi-temporal Harmonized Landsat and Sentinel-2 (HLS) surface reflectance data from the 2023 and 2024 growing seasons. The analysis employs Random Forest (RF) and Light Gradient Boosting Machine (LightGBM) models to evaluate crop classification performance at different stages of the growing season.

The repository accompanies the manuscript Early-Season Identification of Crop Types Using Satellite Remote Sensing and Machine Learning. It includes the primary classification models, robustness analyses, SHAP-based model interpretation, and misclassification diagnostics, along with the datasets and instructions required to reproduce the results.

## Reproduction verification

The complete Random Forest and LightGBM analyses were executed for both study years (2023 and 2024) in Google Colab. The primary classification metrics matched the reference results, and all 1,288 summary values from the robustness analyses agreed within an absolute tolerance of 1 × 10⁻⁶.

SHAP analyses were completed for both models and years, with numerical consistency confirmed through additivity checks. The end-of-season spectral diagnostic outputs also matched the reference results after accounting for changes to field identifiers.

Detailed verification procedures, results, and limitations are provided in docs/VERIFICATION_REPORT.md.

## Data

The repository includes data/Data.zip, containing 14 CSV files organized by study year (2023 and 2024). Each year includes seven wide-format tables representing the Harmonized Landsat and Sentinel-2 (HLS) spectral bands.

| Study year | Fields | Spectral bands |
| --- | ---: | --- |
| 2023 | 1,016 | Blue, Green, Red, RedEdge1, RedEdge2, SWIR1, SWIR2 |
| 2024 | 1,181 | Blue, Green, Red, RedEdge1, RedEdge2, SWIR1, SWIR2 |

Each CSV file contains three identification columns (Crop, aid, and County), followed by spectral reflectance observations organized by acquisition date. Invalid or nonnumeric reflectance values are treated as missing observations during preprocessing.

Field records are uniquely identified using the composite key (Crop, aid, County), which is used to merge the spectral datasets and maintain consistent records across analyses. This composite key distinguishes records in the supplied datasets but does not independently establish geographic non-overlap.

The classification target consists of three crop classes: corn, cotton, and soybean. Numeric suffixes (1 and 2) in the Crop column are retained for record identification. Both suffixes are mapped to their respective crop class and are excluded from the model predictors.

The file data/reference_partitions.csv contains the fixed training and test field assignments used to reproduce the reference results. These assignments are applied consistently to Random Forest and LightGBM. Alternatively, the --new-split option generates a new training/test partition, although results obtained using this option are not directly comparable to those based on the reference held-out sample.

## Installation

The reference archives report Python 3.13.16. Use Python 3.13 with the pinned dependencies for closest environment matching, and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The original calculations used scikit-learn 1.7.2, LightGBM 4.6.0, pandas 2.3.3 and NumPy 2.1.3. Exact computations may vary with library versions, operating systems and parallel execution.

## Run the classifiers

Run one algorithm and year from the repository root:

```bash
python scripts/run_model.py --algorithm RF --year 2023
python scripts/run_model.py --algorithm LGBM --year 2023
python scripts/run_model.py --algorithm RF --year 2024
python scripts/run_model.py --algorithm LGBM --year 2024
```

Alternatively, run all four analyses:

```bash
python scripts/run_all.py
```

For a **nonpublication** smoke test, run a single cutoff with fewer tuning configurations and fitted trees:

```bash
python scripts/run_model.py --algorithm RF --year 2023 --pilot
python scripts/run_model.py --algorithm LGBM --year 2023 --pilot
```

**Pilot outputs must not be used in the manuscript.** Complete runs require considerable computation. Set `--workers` according to available memory and CPU capacity. The primary models use the original estimator threading setting (`n_jobs=-1`); Phase 4 uses one estimator thread per randomized-search worker. Set `--workers 1` for the primary runs when prioritizing fidelity to the archived notebook configuration.

## Analytical design

- For each year, select the fixed development/test field membership. The original test partition originated from the first fold of a 5-fold stratified group-aware split with random seed 1337.
- Evaluate seven cumulative observation windows beginning April 1, ending June 1, June 15, July 1, August 1, September 1, October 1 and November 1.
- Within the development set, perform 5-fold outer and 3-fold inner stratified group-aware cross-validation, with the same group identifier (`Crop|aid|County`) used throughout. The nested CV scores summarize the outer validation folds.
- Tune algorithm-specific hyperparameters using 12 random configurations per search, optimizing macro-averaged F1. The held-out test data are not used in tuning.
- Fit median imputation and zero-variance filtering **within each model-fitting fold**, followed by a class-weighted estimator. No synthetic oversampling is used.
- Evaluate the final development-fitted model once on the fixed held-out fields for each cutoff, reporting overall accuracy, macro-precision, macro-recall, macro-F1, Cohen's kappa and a confusion matrix.
- Evaluate spectral-band permutation importance by jointly shuffling all acquisition-date columns within each of the seven bands. For each crop class, calculate signed per-class F1 decreases across 20 permutations; normalized positive shares are supplementary descriptive statistics, not causal contributions.

### Core parameters

| Setting | Value |
| --- | --- |
| Random state | 1337 |
| Held-out split | Fixed, as supplied in `reference_partitions.csv` |
| Outer/inner CV | 5 / 3 folds |
| Random search | 12 configurations per search |
| Model selection metric | Macro F1 |
| Permutation repetitions | 20 per band |
| Class weighting | `balanced` |

Model-specific parameter distributions are declared in `src/crop_type/modeling.py` and match the primary second-revision model notebooks.

## Results

Each run writes `results/RF_YEAR_revised_results/` or `results/LGBM_YEAR_revised_results/` with:

- `test_and_nested_cv_summary.csv`: held-out and nested-CV metrics for each cutoff.
- `nested_cv_outer_folds.csv`: outer-fold evaluation metrics.
- `hyperparameter_sensitivity.csv`: randomized-search candidate scores.
- `heldout_field_predictions.csv`: predicted crop classes for the fixed test fields.
- `field_partition_ids.csv`: exact development/test membership.
- `permutation_repeats_signed_f1.csv`, `permutation_mean_sd_by_band_crop.csv`, and `permutation_positive_shares_pct.csv`.
- `native_global_band_importance.csv`: model-specific Gini impurity decrease or LightGBM gain.
- `run_manifest.json`: parameters, source checksums, software versions and partition signatures.
- `model_MMDD.joblib`: final fitted pipeline for each cutoff (local output only).
- `RF_YEAR_report.docx` or `LGBM_YEAR_report.docx`: tabulated results.

The saved model files are generated locally and ignored by Git. Only load model files from a trusted source.

## Reference checks

`docs/reference_metrics.csv` contains numeric summary metrics from the earlier model-result archives. It does not contain the earlier source labels or individual field records. After all four **full** model runs complete, compare numeric outputs:

```bash
python scripts/compare_metrics.py
```

**Do not assume exact numerical agreement.** The source record identifiers have been revised, and the fixed held-out membership does not by itself guarantee identical inner/outer validation folds or parameter-search outcomes. The new code must be executed and any differences investigated before using its output for final manuscript tables. In particular, archived outputs and figures containing previous identifiers must not be uploaded to this repository.

Run input and configuration tests with `python -m pytest tests -q` after installing the optional test dependencies (`python -m pip install -e '.[test]'`).

## Supplementary analyses and complete reproduction

The project includes the seven-cutoff primary modeling, Phase 4 robustness analyses, November 1 SHAP explanations, and the descriptive spectral diagnostics of misclassified fields for both algorithms and both study years. There is no USDA CDL benchmark.

**Full workflow** (computationally intensive, including retuning under resampling):

```bash
python scripts/run_full_workflow.py --workers 1
```

This sequence runs RF and LightGBM separately for 2023 and 2024, then performs their robustness checks, November 1 SHAP explanations, and spectral diagnostics. All stages use the same `data/Data.zip` and save their outputs under `results/`. No manual transfer of intermediate ZIP files is required. To test the program without generating publication results, use `--pilot`: this runs a single cutoff with reduced tuning, bootstrap, resampling, and county-validation settings; SHAP and diagnostics are omitted in pilot mode.

The stages can also be run individually. First generate the **full** primary model results:

```bash
python scripts/run_model.py --algorithm RF --year 2023
python scripts/run_model.py --algorithm RF --year 2024
python scripts/run_model.py --algorithm LGBM --year 2023
python scripts/run_model.py --algorithm LGBM --year 2024
```

Then run Phase 4 for each algorithm/year pair:

```bash
python scripts/run_robustness.py --algorithm RF --year 2023
python scripts/run_robustness.py --algorithm RF --year 2024
python scripts/run_robustness.py --algorithm LGBM --year 2023
python scripts/run_robustness.py --algorithm LGBM --year 2024
```

Phase 4 follows the supplied supplementary analysis: **2,000 class-stratified bootstrap resamples** of fixed test predictions per cutoff, **20 retuned field-level repeated partitions** per cutoff, and **five retuned county-grouped validation folds** per cutoff. A 3-fold stratified group-aware search with 12 random parameter configurations is conducted inside each repeated partition or county training fold. The full analysis covers **all seven cutoff dates** for both algorithms. Phase 4 writes `heldout_bootstrap_95ci.csv`, `repeated_splits_summary.csv`, `county_holdout_pooled_summary.csv`, and the corresponding per-split/fold metrics and predictions.

`data/reference_phase4_resplits.csv` contains the original repeated test memberships translated to the released numeric record identifiers, with no original group descriptions. These memberships are fixed to avoid resampling different fields after the identifiers were changed. RF and LightGBM share the same membership lists. Only the training examples for a specific split are used to tune that split's model. Phase 4 checkpoints individual searches under `results/<algorithm>_<year>_Phase4_ONE_RUN/checkpoints/`; rerunning the same command resumes completed tasks. Checkpoints are invalidated when the data, model settings, or fixed primary split changes.

After the four full primary runs, compute November 1 class-specific Tree SHAP values:

```bash
python scripts/run_shap.py --algorithm RF
python scripts/run_shap.py --algorithm LGBM
```

SHAP uses the **saved November 1 fitted model** and original held-out field membership for each study year. The code checks the data hashes, split signatures, saved test predictions, and SHAP additivity before exporting results. The TreeExplainer configuration is `tree_path_dependent` with raw model output. Random Forest explains class probabilities; LightGBM explains unnormalized class margins. Within each class and held-out record, signed SHAP values across all dates in a band are summed before averaging their absolute magnitude across records. The normalized seven-band shares are calculated separately for each crop. Neither the absolute magnitudes across model types nor SHAP importance and permutation importance are interchangeable.

To produce the descriptive November 1 misclassification diagnostics:

```bash
python scripts/run_diagnostics.py --algorithm RF
python scripts/run_diagnostics.py --algorithm LGBM
```

The diagnostic stage compares 14-day observed reflectance profiles of misclassified held-out fields against correctly classified held-out crop-class profiles. Its figures and deviations are descriptive, not tests of causation. Use `--no-figures` to export only the tables.

### Publication verification

The full workflow has been coded from the original research notebooks, and the fixed primary/repeated field memberships have been translated to the revised identifiers. However, **completion of a pilot run is not evidence that the full numerical results exactly match the previous archives**. A complete run under the pinned environment, followed by `python scripts/compare_metrics.py` and review of Phase 4/SHAP outputs, is required before reporting manuscript values. The revised identifiers can affect order-dependent cross-validation allocations and fitted results even when fixed held-out memberships are identical.

The supplied reference metrics are for numerical comparison only. The historical result archives, which contain different field labels, are **not** part of this public repository and should not be uploaded unchanged.

## Limitations

The unique source key identifies a CSV record, not necessarily a physically independent geographic field. The published numeric group codes are a naming convention, not a guarantee of irreversible anonymization. The classifiers predict crop type, not these numeric groups. The data and code should be reviewed for release permission and residual re-identification risk before making the repository public.


## Numerical audit

Before citing this repository as a reproduction of the manuscript, run the full workflow
and check both sets of saved results:

```bash
python scripts/compare_metrics.py
python scripts/compare_robustness.py
```

The checks require all seven cutoff dates for both algorithms and both years, compare
all archived numeric summary fields, and return a nonzero exit status when there is
a discrepancy. The reference tables contain only aggregate metrics, not historical
field-status labels. Review `docs/IMPLEMENTATION_AUDIT.md` for the methods audit and
remaining validation requirements.
