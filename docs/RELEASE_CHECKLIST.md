# Prepublication checks

See `IMPLEMENTATION_AUDIT.md` for the implementation audit and `VERIFICATION_REPORT.md` for the subsequent reproduction checks.

- [ ] Confirm permission to release field-level HLS data and associated metadata.
- [ ] Assess whether numeric record identifiers and county information meet the project's confidentiality requirements.
- [x] Execute the four **full** seven-cutoff model runs with the pinned dependencies.
- [x] Verify matching RF/LightGBM development/test partitions for each year.
- [x] Run `scripts/compare_metrics.py` and `scripts/compare_robustness.py`; investigate every discrepancy and do not release until resolved.
- [x] Execute all four full Phase 4 robustness runs: 20 repeated splits, 5 county folds, and 2,000 bootstrap resamples at each of seven cutoffs.
- [x] Run the RF and LightGBM SHAP analyses for both years; verify all prediction and additivity checks.
- [x] Execute the spectral error diagnostics and review the generated figures.
- [x] Ensure that filenames, outputs, figures, manifests, and documentation do not reveal previous identifiers or group meanings.
- [ ] Reconcile the regenerated tables and figures with the revised manuscript.
- [ ] Add final article citation, acknowledgments, and an author-approved license.
