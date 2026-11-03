# Implementation audit: Colab notebooks and released Python code

**Audit date:** 29 October 2026  
**Subsequent validation:** See `VERIFICATION_REPORT.md` for the completed November 2026 reproduction checks. This file records the earlier pre-run implementation audit.  
**Scope:** Primary Random Forest and LightGBM models, Phase 4 robustness, November 1 SHAP, and end-of-season reflectance error diagnostics for 2023 and 2024.

## Source material

The primary-model source of record is the *IDFIX* second-revision notebook for each algorithm. Those notebooks supersede the earlier `SecondRevision(1)` notebooks. Phase 4 was checked against the `SecondRevision1` notebooks for both algorithms. SHAP and spectral diagnostics were checked against the respective `SHAP` and `Misclassified_diagnosis` notebooks. The four primary and four Phase 4 output archives provide numeric references. No previous SHAP output archive was supplied.

## Verified equivalence of inputs and field membership

- The public `data/Data.zip` contains all **14 annual spectral-band CSVs** and all **2,197 records** (2023: 1,016; 2024: 1,181).
- A file-by-file comparison with the source archive confirmed that **all dated spectral observations** are unchanged. Source label codes and some descriptive field identifiers were changed; these identifiers are not predictors.
- The field identifiers map **one-to-one** between the source and the public data, without duplicates.
- Original held-out memberships, translated to released identifiers, match `reference_partitions.csv` for RF and LightGBM: 2023: **812 development / 204 test**; 2024: **944 development / 237 test**.
- All **20 originally selected Phase 4 repeated-test memberships** in each year/model match `reference_phase4_resplits.csv` when translated to the released identifiers.

These tests establish matching *membership*, not matching fitted predictions. They do not establish equivalence of every inner and outer validation fold.

## Primary modeling methods

| Design element | IDFIX notebooks | Released code |
| --- | --- | --- |
| Class target | Corn, Cotton, Soy | Same |
| Predictors | Seven bands, dated features starting April 1 | Same |
| Cutoffs | June 1, June 15, July 1, August 1, September 1, October 1, November 1 | Same |
| Holdout | Original first 5-fold stratified grouped split | Saved exact membership |
| Nested CV | 5 outer / 3 inner, grouped | Same |
| Hyperparameter search | 12 randomized candidates, macro F1 | Same |
| Data preprocessing | Median imputation and zero-variance filtering inside fitting folds | Same |
| Class weighting | Balanced | Same |
| Permutation importance | 20 joint within-band permutations, signed class-F1 drop | Same |
| Seed | 1337 plus documented offsets | Same |
| Primary estimator threads | `n_jobs=-1` | Restored to `n_jobs=-1` in this audit |

The model parameter distributions for Random Forest and LightGBM match the IDFIX notebooks. The released implementation also saves final fitted models for downstream SHAP; this is an additional artifact, not a new model-training procedure.

## Phase 4 methods

The robustness code retains the original **2,000 class-stratified bootstrap resamples** per cutoff, **20 retuned repeated splits** per cutoff, **five county-based folds** per cutoff, and **12 randomized tuning candidates** using the three-fold grouped inner CV. Original repeated-test membership is supplied explicitly to preserve it across the revised field identifiers. Phase 4 estimator threading is `n_jobs=1`, as in the original Phase 4 notebooks.

The original reference outputs contain seven cutoff dates for all four model/year combinations. The comparison scripts evaluate 28 primary-model summary rows and 1,288 Phase 4 aggregate values. During the initial implementation audit, these scripts were tested against the reference archives to verify their correctness and completeness. Full numerical reproduction was subsequently established through execution of the complete workflows, as documented in [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md).



## SHAP and diagnostic methods

The TreeExplainer configuration, November 1 cutoff date, multiclass output space, preprocessing procedures, additivity verification, and spectral-band aggregation follow the original SHAP analyses. The original implementation rebuilt the November models using selected parameters, whereas the released code loads the fitted models saved during the primary classification stage. This avoids redundant model fitting while preserving the analytical workflow. The subsequent SHAP analyses passed numerical additivity checks, although direct comparison against historical SHAP attribution tables was not performed.

The spectral diagnostic procedures, including temporal aggregation, reference spectral profiles, profile comparisons, and figure generation, were implemented in the connected workflow. The regenerated diagnostic outputs were subsequently compared with the reference results. All numerical values in the corresponding CSV tables matched after accounting for the revised field identifiers. Detailed results are provided in [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md).


## Corrections made in this audit

1. Restored original primary estimators' `n_jobs=-1` setting while preserving `n_jobs=1` in Phase 4.
2. Added a strict **primary numerical comparator** that checks all 20 available numeric reference columns for each of 28 run/cutoff combinations, tests completeness, and returns a failure exit status for mismatches.
3. Added a strict **Phase 4 comparator** covering bootstrap, repeated-split, and county-held-out summaries, together with a label-safe aggregate reference table.
4. Updated the reproduction documentation to report the reference Python version, **3.13.16**, and recommend `--workers 1` for the primary analyses.
5. Added automated tests of reference completeness, comparison failures, and original estimator threading.


## Subsequent reproducibility verification

Following the initial implementation audit, the complete Random Forest and LightGBM workflows were executed in Google Colab for the 2023 and 2024 growing seasons.

The primary classification metrics matched the reference results. All 1,288 summary values from the robustness analyses also matched within an absolute tolerance of `1e-6`.

The SHAP analyses were successfully completed for both algorithms and years, with numerical consistency confirmed through additivity checks. The end-of-season spectral diagnostic tables reproduced the reference calculations after accounting for the revised field identifiers.

These results establish numerical agreement for the evaluated outputs but do not independently verify every intermediate calculation or establish exact equivalence of historical SHAP attributions.

The complete verification procedures, numerical results, and limitations are documented in [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md).

## Remaining publication considerations

Before public release, the repository maintainers should confirm authorization to distribute the field-level datasets and associated metadata, assess potential re-identification risks, and establish appropriate software and data licensing terms.

The manuscript tables and figures should also be checked against the verified computational outputs before final publication.

**Audit conclusion:** The released implementation preserves the principal modeling procedures, configuration settings, and reference field memberships. Subsequent full executions confirmed agreement with the checked numerical reference results. The repository provides the code, input data, and instructions required to reproduce the reported analyses, subject to the documented limitations.

