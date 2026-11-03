# Reproduction verification record

**Verification date:** 2 November 2026  
**Scope:** Random Forest and LightGBM, 2023 and 2024, using the supplied public-coded spectral data and fixed reference field memberships.


## Model analyses

The four complete seven-cutoff Random Forest and LightGBM classification analyses were executed in Google Colab for the 2023 and 2024 growing seasons. The generated classification summary metrics matched the supplied reference results within the comparison tolerance of `1e-6`.

All four complete robustness analyses were executed, including bootstrap confidence intervals, repeated field-level partitions, and county-based validation. The numerical comparison confirmed agreement across all 1,288 reference summary values within an absolute tolerance of `1e-6`.

SHAP analyses were completed for both algorithms and study years at the November 1 cutoff date. Numerical consistency was confirmed through additivity checks, with the following maximum absolute errors:

| Model | Year | Maximum additivity error |
|---|---|---|
| Random Forest | 2023 | 2.22e-15 |
| Random Forest | 2024 | 1.78e-15 |
| LightGBM | 2023 | 8.44e-15 |
| LightGBM | 2024 | 7.21e-12 |

These additivity checks confirm numerical consistency of the SHAP calculations. Independent numerical comparison against historical SHAP attribution tables was not performed.



## Spectral misclassification diagnostics

The end-of-season spectral diagnostic analyses were executed for Random Forest and LightGBM using the released dataset. The generated outputs were compared against the reference results for both study years.

All eight CSV tables generated for each algorithm matched the corresponding reference tables after accounting for the revised field identifiers. No numerical differences were identified in the compared CSV values, using an absolute tolerance of `1e-12`.

| Analysis | Random Forest | LightGBM |
|---|---:|---:|
| Misclassified held-out fields, 2023 | 8 | 5 |
| Misclassified held-out fields, 2024 | 9 | 7 |
| Held-out predictions, both years | 441 | 441 |
| Misclassification summary records | 17 | 12 |
| Spectral profile CSV tables compared | 8 | 8 |

The generated ZIP archives passed integrity checks. Filenames, CSV contents, and PNG metadata were examined for previous field-group terminology, with no instances identified. Sample diagnostic figures were also reviewed and confirmed to use the revised numeric identifiers. Figures were not compared pixel by pixel with the original versions.

## What this establishes

The complete Random Forest and LightGBM analyses reproduced the checked primary-model summary metrics. All 1,288 Phase 4 robustness summary values matched the reference results within an absolute tolerance of `1e-6`.

The spectral diagnostic CSV outputs reproduced the reference calculations after accounting for revised field identifiers. SHAP calculations were successfully completed and passed numerical additivity checks; however, direct equivalence with historical SHAP attribution values was not independently established.

The full computational outputs are generated locally and are not included in the source repository. The provided scripts, input datasets, reference metrics, and documentation support reproduction of the analyses.

These verification procedures establish numerical agreement for the evaluated outputs but do not independently verify every intermediate calculation or guarantee field-level anonymity. Authorization to distribute the underlying research data must be established separately.


## Reproduction commands

From the repository root after installation:

```bash
python scripts/run_all.py
python scripts/compare_metrics.py
python scripts/run_robustness.py --algorithm RF --year 2023 --workers 1
python scripts/run_robustness.py --algorithm RF --year 2024 --workers 1
python scripts/run_robustness.py --algorithm LGBM --year 2023 --workers 1
python scripts/run_robustness.py --algorithm LGBM --year 2024 --workers 1
python scripts/compare_robustness.py
python scripts/run_shap.py --algorithm RF
python scripts/run_shap.py --algorithm LGBM
python scripts/run_diagnostics.py --algorithm RF
python scripts/run_diagnostics.py --algorithm LGBM
```

The full workflow can instead be run with `python scripts/run_full_workflow.py --workers 1`. The full analyses can be computationally expensive. Checkpoint support is provided for Phase 4 runs. The `--pilot` option is exclusively for nonpublication smoke testing.
