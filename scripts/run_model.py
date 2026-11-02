#!/usr/bin/env python3
"""Fit LightGBM or Random Forest for one study year."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from crop_type.modeling import RunConfig, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("RF", "LGBM"), required=True)
    parser.add_argument("--year", choices=(2023, 2024), type=int, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/Data.zip"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--partitions", type=Path, default=Path("data/reference_partitions.csv"),
                        help="Fixed revised-label field membership for the original test split")
    parser.add_argument("--new-split", action="store_true",
                        help="Generate a new split instead of using the archived test membership")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel search workers; use 1 for the reference analysis")
    parser.add_argument("--pilot", action="store_true",
                        help="One-cutoff smoke test with reduced searches and CV; NOT manuscript results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ref = None if args.new_split else args.partitions
    if ref is not None and not ref.exists():
        parser.error(f"Partition file not found: {ref}. Use --new-split for a newly sampled holdout.")
    result = run(RunConfig(args.algorithm, args.year, args.data, args.output,
                           partition_file=ref, pilot=args.pilot, workers=args.workers))
    print(result["summary"].to_string(index=False))
    print("Results:", result["output_dir"])


if __name__ == "__main__":
    main()
