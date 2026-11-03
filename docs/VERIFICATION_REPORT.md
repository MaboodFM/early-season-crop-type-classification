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

The two diagnostics were rerun using the public-coded data and their new ZIP outputs were provided for direct comparison against the originally generated archives. All eight CSV tables for each algorithm match the earlier values row-for-row after translating the identifiers to the public-coded representation. Comparison used the same row order and column names and a maximum absolute numeric tolerance of 1e-12; actual differences in all parsed numeric cells were zero.

| Analysis | RF | LightGBM |
| --- | ---: | ---: |
| Misclassified held-out fields, 2023 | 8 | 5 |
| Misclassified held-out fields, 2024 | 9 | 7 |
| Held-out predictions (both years) | 441 | 441 |
| Misclassification summary records | 17 | 12 |
| Spectral profile CSV tables compared | 8 | 8 |

New diagnostic ZIP files passed integrity testing. Their filenames, CSVs, documentation and PNG metadata were checked for outdated field-group terminology; none was found. Sample generated diagnostic plots were reviewed and display the numbered group notation. **Figures were not compared pixel-by-pixel** against the original plots. Model abbreviation `RF` remains in method and figure names and does not denote a field group.

## What this establishes

The checked primary and Phase 4 summary results reproduce the supplied numerical references in the user-run environment; the diagnostic CSV data exactly reproduce prior diagnostic calculations subject to the intended record-identifier changes. SHAP outputs were successfully computed and passed additivity checks. The user-generated full training and robustness outputs remain in their Google Drive and are not bundled into this source repository.

These results do not independently guarantee field confidentiality, establish equality of unexamined intermediate data, or constitute a standalone external reproducibility exercise. Users should still ensure the data may legally be released, review field identifiers and associated county metadata, and select an appropriate code and data license.

## Reproduction commands

From the repository root after installation:

```bash
python scripts/run_all.py --workers 1
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
