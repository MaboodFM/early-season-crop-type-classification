
# Early-season crop-type classification

This repository provides the code and data used for early-season classification of corn, cotton, and soybean fields in West Tennessee using multi-temporal Harmonized Landsat and Sentinel-2 (HLS) surface reflectance data from the 2023 and 2024 growing seasons. Random Forest (RF) and Light Gradient Boosting Machine (LightGBM) models are used to evaluate crop classification performance at different stages of the growing season.

The repository accompanies the manuscript *Early-Season Identification of Crop Types Using Satellite Remote Sensing and Machine Learning*. It includes the primary classification models, robustness analyses, SHAP-based model interpretation, misclassification diagnostics, and the datasets and instructions required to reproduce the results.



## Reproduction verification

The complete Random Forest and LightGBM analyses were executed for both study years (2023 and 2024) in Google Colab. The primary classification metrics matched the reference results, and all 1,288 summary values from the robustness analyses agreed within an absolute tolerance of `1e-6`.

SHAP analyses were completed for both models and years, with numerical consistency confirmed through additivity checks. The end-of-season spectral diagnostic outputs also matched the reference results after accounting for changes to field identifiers.

Detailed verification procedures, results, and limitations are provided in [docs/VERIFICATION_REPORT.md](docs/VERIFICATION_REPORT.md).




## Data

The repository includes `data/Data.zip`, containing 14 CSV files organized by study year (2023 and 2024). Each year includes seven wide-format tables representing the Harmonized Landsat and Sentinel-2 (HLS) spectral bands.

| Study year | Number of fields | Spectral bands |
|---|---:|---|
| 2023 | 1,016 | Blue, Green, Red, RedEdge1, RedEdge2, SWIR1, SWIR2 |
| 2024 | 1,181 | Blue, Green, Red, RedEdge1, RedEdge2, SWIR1, SWIR2 |

Each CSV file contains three identification columns (`Crop`, `aid`, and `County`), followed by spectral reflectance observations organized by acquisition date. Invalid or nonnumeric reflectance values are treated as missing observations during preprocessing.

Field records are uniquely identified using the composite key (`Crop`, `aid`, `County`), which is used to merge the spectral datasets and maintain consistent records across analyses. This composite key distinguishes records in the supplied datasets but does not independently establish geographic non-overlap.

The classification target consists of three crop classes: corn, cotton, and soybean. Numeric suffixes (`1` and `2`) in the `Crop` column are retained for record identification. Both suffixes are mapped to their respective crop class and are excluded from the model predictors.

The file `data/reference_partitions.csv` contains the fixed training and test field assignments used to reproduce the reference results. These assignments are applied consistently to Random Forest and LightGBM. Alternatively, the `--new-split` option generates a new training/test partition, although results obtained using this option are not directly comparable to those based on the reference held-out sample.


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

6. Model evaluation: Following hyperparameter selection, the final model is fitted on the development set and evaluated on the fixed held-out test set for each cutoff date. Performance metrics include overall accuracy, macro-averaged precision, recall, and F1-score, Cohen's kappa, and a multiclass confusion matrix.

7. Spectral-band importance: Permutation importance is evaluated by jointly shuffling all acquisition-date features belonging to each spectral band, with 20 repetitions per band. Changes in class-specific F1-scores are used to quantify the effect of permutation. Positive mean F1 reductions are also normalized to obtain relative band-importance shares for each crop class. These measures describe model sensitivity to individual spectral bands and should not be interpreted as causal effects.

### Core parameters

The following settings are applied consistently to the full Random Forest and LightGBM classification analyses.

| Setting | Value |
| --- | --- |
| Random state | 1337 |
| Held-out split | Fixed, as supplied in `data/reference_partitions.csv` |
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





## Supplementary analyses and complete reproduction

The repository includes robustness assessments, SHAP-based model interpretation, and spectral diagnostics of misclassified fields for Random Forest and LightGBM across the 2023 and 2024 growing seasons. These analyses complement the primary classification models and provide additional evaluation of model reliability, interpretability, and classification errors.

### Complete workflow

The complete analysis can be executed from the repository root using:

```bash
python scripts/run_full_workflow.py --workers 1
```

This command sequentially executes the primary classification models for both study years, followed by robustness analyses, SHAP explanations, and misclassification diagnostics. All stages use the dataset provided in `data/Data.zip`, and outputs are saved in the `results/` directory. Intermediate results are transferred automatically between analytical stages.

The complete workflow is computationally intensive. Individual analyses may also be executed separately using the commands described below.

### Robustness analyses

Model robustness is assessed using three complementary procedures:

1. **Bootstrap confidence intervals:** For each cutoff date, 2,000 class-stratified bootstrap resamples of the fixed held-out test predictions are used to estimate 95% confidence intervals for classification metrics.

2. **Repeated field-level partitions:** Twenty additional train/test partitions are evaluated for each cutoff date to examine sensitivity to field selection. Hyperparameters are independently optimized within the training portion of each partition.

3. **County-based cross-validation:** Five county-grouped validation folds are used to assess classification performance when geographically grouped observations are withheld from model training.

For the repeated-partition and county-based analyses, hyperparameter optimization uses three-fold stratified group-aware cross-validation with 12 randomized parameter configurations per search.

All three procedures are applied to the seven growing-season cutoff dates for both algorithms and both study years.

To execute the robustness analyses separately:

```bash
python scripts/run_robustness.py --algorithm RF --year 2023 --workers 1
python scripts/run_robustness.py --algorithm RF --year 2024 --workers 1
python scripts/run_robustness.py --algorithm LGBM --year 2023 --workers 1
python scripts/run_robustness.py --algorithm LGBM --year 2024 --workers 1
```

The file `data/reference_phase4_resplits.csv` provides fixed field memberships for the repeated-partition analyses, ensuring consistent evaluation samples across the two algorithms.

The principal robustness outputs include:

- `heldout_bootstrap_95ci.csv`: Bootstrap-based confidence intervals for classification metrics.
- `repeated_splits_summary.csv`: Summary statistics across repeated train/test partitions.
- `county_holdout_pooled_summary.csv`: Pooled classification performance from county-based validation.

Additional files contain individual partition and fold metrics, predictions, and selected hyperparameters.

To accommodate computationally intensive execution, intermediate results are saved under `results/<algorithm>_<year>_Phase4_ONE_RUN/checkpoints/`. Interrupted analyses can be resumed by rerunning the corresponding command, provided that the input data and model configurations remain unchanged.

### SHAP analysis

SHapley Additive exPlanations (SHAP) are used to interpret the Random Forest and LightGBM models at the final November 1 cutoff date.

SHAP analyses are performed using the fitted end-of-season models and their corresponding held-out test fields for 2023 and 2024.

Run the analyses using:

```bash
python scripts/run_shap.py --algorithm RF
python scripts/run_shap.py --algorithm LGBM
```

The implementation uses TreeExplainer with `tree_path_dependent` feature perturbation and raw model output. Random Forest SHAP values explain class probabilities, whereas LightGBM SHAP values explain unnormalized class margins.

For each crop class, SHAP values corresponding to observation dates within the same spectral band are summed at the field level. The mean absolute magnitude of these aggregated values is then used to calculate normalized spectral-band importance shares.

The analysis generates class-specific SHAP summaries, spectral-band importance tables, feature-level attribution results, and visualization figures.

SHAP additivity is checked to verify numerical consistency. Because the two algorithms use different output scales, their absolute SHAP magnitudes should not be directly compared. SHAP attribution and permutation importance also represent different measures of feature importance.

### Misclassification diagnostics

End-of-season spectral diagnostics are conducted to examine the reflectance characteristics of fields incorrectly classified by the models.

The analysis compares observed spectral reflectance profiles of misclassified held-out fields with those of correctly classified fields belonging to the corresponding crop classes. Spectral observations are summarized using 14-day intervals.

Run the diagnostic analyses using:

```bash
python scripts/run_diagnostics.py --algorithm RF
python scripts/run_diagnostics.py --algorithm LGBM
```

The scripts generate spectral comparison tables, summary statistics, and diagnostic figures for both study years. Results are saved in the `results/` directory.

To generate only the tabular outputs without figures, add the `--no-figures` argument to the corresponding command.

These diagnostics provide descriptive information about classification errors and spectral variability and should not be interpreted as establishing causal relationships.




## Limitations

The classification models were developed and evaluated using field observations from West Tennessee during the 2023 and 2024 growing seasons. Their performance may differ under other geographic, environmental, or agricultural conditions. Application to other regions, growing seasons, or crop types requires additional validation.

Field records are identified using the composite key (`Crop`, `aid`, `County`). Although this key uniquely identifies records within the supplied datasets, it does not independently verify the geographic separation of individual fields.

The numeric suffixes in the source crop labels are used solely for record identification and are not included as model predictors or classification targets. These labels do not, by themselves, guarantee the anonymity of individual field records. Appropriate data-sharing permissions and potential re-identification risks should be considered before public distribution.

