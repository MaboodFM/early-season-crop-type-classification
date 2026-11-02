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

The original reference outputs contain seven cutoffs for all four model/year combinations. The accompanying checkers compare **28 complete primary summary rows** and **1,288 Phase 4 aggregate values**. When tested against the actual uploaded historical summaries, both comparison tools passed with no differences. This checks the comparison tools and the completeness of the references; it **does not** constitute a full new training run.

## SHAP and diagnostic methods

The TreeExplainer configuration, November 1 cutoff, multiclass output space, preprocessing, additivity verification, and band grouping follow the original SHAP notebooks. The original SHAP notebooks rebuilt the fitted November model using saved parameters; the GitHub code loads the fitted November model saved directly by the primary training stage. This change eliminates redundant training, but exact equivalence requires successful primary-run verification. No original SHAP numeric output was supplied for an independent values-based comparison.

The reflectance diagnostic helper functions, seasonal binning, reference profiles, profile distances, and figure-generation procedures were compared with the diagnostic notebooks; the input and output adapters were adjusted for the connected workflow. The existing integration test exercises this stage; a full reference-result numerical comparison has not yet been performed.

## Corrections made in this audit

1. Restored original primary estimators' `n_jobs=-1` setting while preserving `n_jobs=1` in Phase 4.
2. Added a strict **primary numerical comparator** that checks all 20 available numeric reference columns for each of 28 run/cutoff combinations, tests completeness, and returns a failure exit status for mismatches.
3. Added a strict **Phase 4 comparator** covering bootstrap, repeated-split, and county-held-out summaries, together with a label-safe aggregate reference table.
4. Updated the reproduction documentation to report the reference Python version, **3.13.16**, and recommend `--workers 1` for the primary analyses.
5. Added automated tests of reference completeness, comparison failures, and original estimator threading.

## Outstanding publication checks

- **Full numeric reproduction is not yet certified.** The available test runtime has `scikit-learn 1.8.0`, whereas the archived output reports **1.7.2**. When a test split is regenerated under the current local version, its membership does not match the archived split. The pinned repository version and fixed original held-out/repeated memberships reduce this risk, but the inner and outer CV outcomes still require an execution under the archived dependency versions.
- The original primary models ran with Python 3.13.16, NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.7.2, and LightGBM 4.6.0. Use these versions for the final reproduction wherever possible.
- Run the **full**, nonpilot, seven-cutoff main models and Phase 4 jobs, then execute both comparison scripts. Inspect every discrepancy before publication.
- Run SHAP and reflectance diagnostics using the newly fitted model outputs and inspect their verification reports, figures and summary tables.
- Review the field records and release permission. Numeric group names alone do not guarantee irreversible confidentiality.

**Audit conclusion:** The released package contains the required analysis stages and preserves the principal configurations and reference memberships. The source-code audit and unit tests pass, but full numerical reproduction has **not** yet been established; do not describe it as verified until the full comparison checks pass.
