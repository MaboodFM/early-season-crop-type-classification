#!/usr/bin/env python3
"""Run the LightGBM classification workflow for both years."""

from pathlib import Path
from crop_type.modeling import RunConfig, run

if __name__ == "__main__":
    for year in (2023, 2024):
        run(RunConfig("LGBM", year, Path("data/Data.zip"), Path("results"),
                      partition_file=Path("data/reference_partitions.csv")))
