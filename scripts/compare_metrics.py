#!/usr/bin/env python3
"""Verify all primary-model summary statistics against archived reference values."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

KEY = ["year", "algorithm", "cutoff"]


def compare(reference: pd.DataFrame, actual: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Return mismatches, raising on incomplete or duplicate analysis keys."""
    if reference.duplicated(KEY).any() or actual.duplicated(KEY).any():
        raise ValueError("Duplicate year/algorithm/cutoff combinations")
    if not reference[KEY].sort_values(KEY).reset_index(drop=True).equals(
        actual[KEY].sort_values(KEY).reset_index(drop=True)
    ):
        missing = reference[KEY].merge(actual[KEY], on=KEY, how="left", indicator=True)
        extra = actual[KEY].merge(reference[KEY], on=KEY, how="left", indicator=True)
        raise ValueError(
            f"Missing reference rows: {len(missing[missing._merge.eq('left_only')])}; "
            f"unexpected rows: {len(extra[extra._merge.eq('left_only')])}"
        )
    columns = [c for c in reference.columns if c not in KEY + ["reference_source"]]
    absent = sorted(set(columns) - set(actual.columns))
    if absent:
        raise ValueError(f"Missing numeric metrics in observed results: {absent}")
    checked = reference.merge(actual, on=KEY, how="inner", suffixes=("_reference", "_actual"), validate="one_to_one")
    errors = []
    for column in columns:
        expected = checked[f"{column}_reference"].to_numpy(dtype=float)
        observed = checked[f"{column}_actual"].to_numpy(dtype=float)
        unequal = ~np.isclose(expected, observed, rtol=0, atol=tolerance, equal_nan=True)
        for i in np.flatnonzero(unequal):
            row = checked.iloc[i]
            errors.append({**{c: row[c] for c in KEY}, "metric": column,
                           "reference": expected[i], "observed": observed[i],
                           "difference": observed[i] - expected[i]})
    return pd.DataFrame(errors, columns=KEY + ["metric", "reference", "observed", "difference"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--reference", type=Path, default=Path("docs/reference_metrics.csv"))
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--differences", type=Path, default=Path("results/primary_metric_differences.csv"))
    args = parser.parse_args()
    reference = pd.read_csv(args.reference, dtype={"cutoff": str})
    observed = []
    for algorithm in ("RF", "LGBM"):
        for year in (2023, 2024):
            path = args.results / f"{algorithm}_{year}_revised_results" / "test_and_nested_cv_summary.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing full analysis: {path}")
            observed.append(pd.read_csv(path, dtype={"cutoff": str}))
    mismatches = compare(reference, pd.concat(observed, ignore_index=True), args.atol)
    args.differences.parent.mkdir(parents=True, exist_ok=True)
    mismatches.to_csv(args.differences, index=False)
    if not mismatches.empty:
        print(f"FAIL: {len(mismatches)} differences outside tolerance {args.atol:g}.")
        print(mismatches.head(30).to_string(index=False))
        print(f"All differences: {args.differences}")
        return 1
    print(f"PASS: {len(reference)} cutoff/model/year rows; all {len(reference.columns) - len(KEY) - 1} reference numeric fields match (atol={args.atol:g}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
