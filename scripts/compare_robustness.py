#!/usr/bin/env python3
"""Verify Phase 4 bootstrap, repeated-split, and county-held-out summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

KEY = ["year", "algorithm", "cutoff", "stage", "metric"]


def extract(output_root: Path) -> pd.DataFrame:
    values = []
    for algorithm in ("RF", "LGBM"):
        for year in (2023, 2024):
            folder = output_root / f"{algorithm}_{year}_Phase4_ONE_RUN"
            stages = (
                ("bootstrap", "heldout_bootstrap_95ci.csv", ("year", "algorithm", "cutoff", "metric")),
                ("repeated", "repeated_splits_summary.csv", ("year", "algorithm", "cutoff")),
                ("county", "county_holdout_pooled_summary.csv", ("year", "algorithm", "cutoff")),
            )
            for stage, filename, keys in stages:
                path = folder / filename
                if not path.is_file():
                    raise FileNotFoundError(f"Missing full Phase 4 summary: {path}")
                frame = pd.read_csv(path, dtype={"cutoff": str})
                for _, row in frame.iterrows():
                    for column in frame.columns:
                        if column in keys or column == "ci_method":
                            continue
                        if pd.api.types.is_numeric_dtype(frame[column]):
                            metric = f"{row['metric']}__{column}" if "metric" in keys else column
                            values.append({"year": year, "algorithm": algorithm,
                                           "cutoff": row["cutoff"], "stage": stage,
                                           "metric": metric, "value": float(row[column])})
    return pd.DataFrame(values)


def compare(reference: pd.DataFrame, actual: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    if reference.duplicated(KEY).any() or actual.duplicated(KEY).any():
        raise ValueError("Duplicate Phase 4 summary keys")
    both = reference.merge(actual, on=KEY, how="outer", suffixes=("_reference", "_actual"),
                           indicator=True, validate="one_to_one")
    if not both._merge.eq("both").all():
        missing = both.loc[both._merge.ne("both"), KEY + ["_merge"]]
        raise ValueError(f"Phase 4 results incomplete or extra: {missing.head(12).to_dict('records')}")
    expected = both.value_reference.to_numpy(dtype=float)
    observed = both.value_actual.to_numpy(dtype=float)
    mismatch = ~np.isclose(expected, observed, rtol=0, atol=tolerance, equal_nan=True)
    errors = both.loc[mismatch, KEY].copy()
    errors["reference"] = expected[mismatch]
    errors["observed"] = observed[mismatch]
    errors["difference"] = observed[mismatch] - expected[mismatch]
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--reference", type=Path, default=Path("docs/reference_phase4_metrics.csv"))
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--differences", type=Path, default=Path("results/phase4_metric_differences.csv"))
    args = parser.parse_args()
    reference = pd.read_csv(args.reference, dtype={"cutoff": str})
    actual = extract(args.results)
    errors = compare(reference, actual, args.atol)
    args.differences.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.differences, index=False)
    if not errors.empty:
        print(f"FAIL: {len(errors)} Phase 4 summary differences outside atol={args.atol:g}.")
        print(errors.head(30).to_string(index=False))
        print(f"All differences: {args.differences}")
        return 1
    print(f"PASS: {len(reference)} Phase 4 summary values match (atol={args.atol:g}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
