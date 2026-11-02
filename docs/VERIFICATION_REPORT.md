# Reproduction verification record

**Verification date:** 2 November 2026  
**Scope:** Random Forest and LightGBM, 2023 and 2024, using the supplied public-coded spectral data and fixed reference field memberships.

## Model analyses

- The research user executed the four full seven-cutoff primary model runs in Google Colab, then reported a `PASS` from `scripts/compare_metrics.py`. The output supports equality of the checked summary metrics against the reference tables at the comparator's configured tolerance.
- The user executed all four full Phase 4 robustness jobs. The `scripts/compare_robustness.py` output reported: `PASS: 1288 Phase 4 summary values match (atol=1e-06).`
- The user ran the November 1 SHAP analysis for RF and LightGBM, both years. Their logs confirmed completion and maximum additivity errors of RF 2023 **2.22e-15**, RF 2024 **1.78e-15**, LightGBM 2023 **8.44e-15**, LightGBM 2024 **7.21e-12**. These test SHAP additivity; independent equality to historical SHAP tables was not assessed.

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
