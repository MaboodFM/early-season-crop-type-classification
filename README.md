# Early-season crop-type classification

This repository provides the code and data used for early-season classification of corn, cotton, and soybean fields in West Tennessee using multi-temporal Harmonized Landsat and Sentinel-2 (HLS) surface reflectance data from the 2023 and 2024 growing seasons. The analysis employs Random Forest (RF) and Light Gradient Boosting Machine (LightGBM) models to evaluate crop classification performance at different stages of the growing season.

The repository accompanies the manuscript Early-Season Identification of Crop Types Using Satellite Remote Sensing and Machine Learning. It includes the primary classification models, robustness analyses, SHAP-based model interpretation, and misclassification diagnostics, along with the datasets and instructions required to reproduce the results.

## Reproduction verification

The complete Random Forest and LightGBM analyses were executed for both study years (2023 and 2024) in Google Colab. The primary classification metrics matched the reference results, and all 1,288 summary values from the robustness analyses agreed within an absolute tolerance of 1 × 10⁻⁶.

SHAP analyses were completed for both models and years, with numerical consistency confirmed through additivity checks. The end-of-season spectral diagnostic outputs also matched the reference results after accounting for changes to field identifiers.

Detailed verification procedures, results, and limitations are provided in docs/VERIFICATION_REPORT.md.

## Data

The repository includes data/Data.zip, containing 14 CSV files organized by study year (2023 and 2024). Each year includes seven wide-format tables representing the Harmonized Landsat and Sentinel-2 (HLS) spectral bands.

| Study year | Number of Fields | Spectral bands |
| --- | ---: | --- |
| 2023 | 1,016 | Blue, Green, Red, RedEdge1, RedEdge2, SWIR1, SWIR2 |
| 2024 | 1,181 | Blue, Green, Red, RedEdge1, RedEdge2, SWIR1, SWIR2 |

Each CSV file contains three identification columns (Crop, aid, and County), followed by spectral reflectance observations organized by acquisition date. Invalid or nonnumeric reflectance values are treated as missing observations during preprocessing.

Field records are uniquely identified using the composite key (Crop, aid, County), which is used to merge the spectral datasets and maintain consistent records across analyses. This composite key distinguishes records in the supplied datasets but does not independently establish geographic non-overlap.

The classification target consists of three crop classes: corn, cotton, and soybean. Numeric suffixes (1 and 2) in the Crop column are retained for record identification. Both suffixes are mapped to their respective crop class and are excluded from the model predictors.

The file data/reference_partitions.csv contains the fixed training and test field assignments used to reproduce the reference results. These assignments are applied consistently to Random Forest and LightGBM. Alternatively, the --new-split option generates a new training/test partition, although results obtained using this option are not directly comparable to those based on the reference held-out sample.

## Installation

The analyses were reproduced using Python 3.13.16. Python 3.13 and the specified package versions are recommended to maintain consistency with the reference computational environment.

Clone the repository and navigate to its root directory:

```bash
git clone https://github.com/MaboodFM/early-season-crop-type-classification.git
cd early-season-crop-type-classification
```

Create and activate a Python virtual environment.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Install the project and its dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The required dependencies are specified in `pyproject.toml` and `requirements.txt`. Key versions include NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.7.2, LightGBM 4.6.0, and SHAP 0.50.0.

Numerical results may vary with differences in software versions, operating systems, and parallel execution settings. Using the specified environment is recommended for reproducibility.



## Run the classifiers

The Random Forest (RF) and LightGBM models are trained and evaluated independently for the 2023 and 2024 growing seasons. Each complete run evaluates seven cumulative observation windows, ending on June 1, June 15, July 1, August 1, September 1, October 1, and November 1.

All commands should be executed from the repository's root directory after installing the required dependencies.

**Run all four classification analyses:**

```bash
python scripts/run_all.py
```

Alternatively, run each model and study year separately:

```bash
python scripts/run_model.py --algorithm RF --year 2023 --workers 1
python scripts/run_model.py --algorithm RF --year 2024 --workers 1
python scripts/run_model.py --algorithm LGBM --year 2023 --workers 1
python scripts/run_model.py --algorithm LGBM --year 2024 --workers 1
```

By default, the scripts use `data/Data.zip` and the fixed development/test partitions provided in `data/reference_partitions.csv`. The same field partitions are used for both algorithms to ensure consistent model comparisons.

Model outputs are saved to the `results/` directory, with separate subdirectories for each algorithm and study year. These include classification metrics, confusion matrices, field-level predictions, hyperparameter tuning results, feature importance, and fitted models.

**Optional pilot execution**

A reduced pilot run can be used to verify that the code and dependencies function correctly before executing the complete analysis:

```bash
python scripts/run_model.py --algorithm RF --year 2023 --pilot
python scripts/run_model.py --algorithm LGBM --year 2023 --pilot
```

Pilot mode uses a single cutoff date and reduced computational settings. **Pilot outputs are intended only for code testing and should not be used to reproduce the manuscript results.**

The complete analyses are computationally intensive. The default setting of `--workers 1` is recommended for consistency with the reference model configuration.


## Analytical design

The Random Forest and LightGBM classification workflows follow the same data partitioning, preprocessing, cross-validation, and evaluation procedures.

1. Data partitioning: For each study year, the dataset is divided into development and held-out test sets. The reference test partition was originally selected using the first fold of a five-fold stratified group-aware split with a random seed of 1337. Fixed field assignments are provided in data/reference_partitions.csv and are shared by both algorithms.

2. Temporal evaluation: Classification performance is evaluated using seven cumulative observation windows beginning April 1 and ending on June 1, June 15, July 1, August 1, September 1, October 1, and November 1.

3. Nested cross-validation: Within the development set, five-fold outer and three-fold inner stratified group-aware cross-validation are performed. The inner folds are used for hyperparameter optimization, while the outer folds provide performance estimates. The composite identifier (Crop|aid|County) is used to maintain group separation.

4. Hyperparameter optimization: Model-specific hyperparameters are optimized using randomized search with 12 configurations per search. Macro-averaged F1-score is the optimization criterion. The held-out test set is excluded from model selection and hyperparameter tuning.

5. Preprocessing and class balancing: Missing spectral observations are handled using median imputation, followed by zero-variance feature filtering. Both operations are fitted within the training folds to prevent information leakage. Class imbalance is addressed using balanced class weights, without synthetic oversampling.

6. Model evaluation: Following hyperparameter selection, the final model is fitted on the development set and evaluated on the fixed held-out test set for each cutoff date. Performance metrics include overall accuracy, macro-averaged precision, recall, and F1-score, Cohen's kappa, and class-specific confusion matrices.

7. Spectral-band importance: Permutation importance is evaluated by jointly shuffling all acquisition-date features belonging to each spectral band, with 20 repetitions per band. Changes in class-specific F1-scores are used to quantify the effect of permutation. Positive mean F1 reductions are also normalized to obtain relative band-importance shares for each crop class. These measures describe model sensitivity to individual spectral bands and should not be interpreted as causal effects.

### Core parameters

The following settings are applied consistently to the full Random Forest and LightGBM classification analyses.

| Setting | Value |
| --- | --- |
| Random state | 1337 |
| Held-out split | Fixed, as supplied in `reference_partitions.csv` |
| Outer/inner cross-validation | 5 / 3 folds |
| Random search | 12 random configurations per search |
| Model selection metric | Macro-averaged F1-score |
| Permutation repetitions | 20 per spectral band |
| Class weighting | `balanced` |

Algorithm-specific hyperparameter search spaces and model configurations are defined in `src/crop_type/modeling.py`.


## Results

Each classification run generates a separate output directory for the corresponding algorithm and study year:

- `results/RF_2023_revised_results/`
- `results/RF_2024_revised_results/`
- `results/LGBM_2023_revised_results/`
- `results/LGBM_2024_revised_results/`

The following files are generated for each analysis:

- **`test_and_nested_cv_summary.csv`**: Overall accuracy, macro-averaged precision, recall, F1-score, and Cohen's kappa for the held-out test set and nested cross-validation across all seven cutoff dates.
- **`nested_cv_outer_folds.csv`**: Classification performance metrics for individual outer cross-validation folds.
- **`hyperparameter_sensitivity.csv`**: Hyperparameter configurations and their associated cross-validation scores from randomized search.
- **`heldout_field_predictions.csv`**: Observed and predicted crop classes for individual fields in the held-out test set at each cutoff date.
- **`field_partition_ids.csv`**: Field identifiers and their corresponding development or test set assignments.
- **`permutation_repeats_signed_f1.csv`**: Class-specific F1-score changes obtained from individual spectral-band permutation repetitions.
- **`permutation_mean_sd_by_band_crop.csv`**: Mean and standard deviation of permutation-based F1-score changes for each spectral band and crop class.
- **`permutation_positive_shares_pct.csv`**: Normalized positive permutation-importance shares expressed as percentages for each crop class.
- **`native_global_band_importance.csv`**: Model-specific spectral-band importance based on Gini impurity decrease for Random Forest and split gain for LightGBM.
- **`run_manifest.json`**: Model settings, selected hyperparameters, software versions, input data checksums, and training/test partition signatures.
- **`model_MMDD.joblib`**: Fitted model pipeline for each cutoff date, where `MMDD` represents the month and day.
- **`RF_YEAR_report.docx`** or **`LGBM_YEAR_report.docx`**: Summary tables, confusion matrices, class-specific classification metrics, and spectral-band importance results.

All output files are saved locally in the `results/` directory. Generated results and fitted model files are excluded from version control by default and can be reproduced using the provided scripts and input datasets.

For security, serialized model files (`.joblib`) should only be loaded from trusted sources.




## Reference checks

The repository includes reference summary metrics for verifying the reproducibility of the classification and robustness analyses:

- `docs/reference_metrics.csv`: Reference performance metrics from the primary Random Forest and LightGBM analyses.
- `docs/reference_phase4_metrics.csv`: Reference statistics from the bootstrap, repeated-split, and county-based robustness analyses.

These files contain numerical summary results without individual field records or the original field-group labels.

After completing the four primary classification runs, verify the generated metrics against the reference values:

```bash
python scripts/compare_metrics.py
```

After completing the four robustness analyses, verify their results using:

```bash
python scripts/compare_robustness.py
```

Both scripts compare the generated outputs with the reference results using an absolute numerical tolerance of `1e-6`. A `PASS` message indicates agreement across the checked summary values. Any discrepancies outside the specified tolerance are reported for further examination.

The full Random Forest and LightGBM analyses for 2023 and 2024 have been executed and verified against the reference results. All 1,288 numerical summary values from the robustness analyses matched within the specified tolerance.

These comparisons assess the reported numerical summaries and do not independently verify every intermediate calculation or SHAP value. Additional verification procedures and limitations are documented in [docs/VERIFICATION_REPORT.md](docs/VERIFICATION_REPORT.md).

### Automated tests

The repository also includes automated tests for data processing, configuration, and reproducibility checks. Install the optional test dependencies and execute the test suite from the repository root:

```bash
python -m pip install -e ".[test]"
python -m pytest tests -q
```


### Automated tests

The repository also includes automated tests for data processing, configuration, and reproducibility checks. Install the optional test dependencies and execute the test suite from the repository root:

```bash
python -m pip install -e ".[test]"
python -m pytest tests -q
```

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
