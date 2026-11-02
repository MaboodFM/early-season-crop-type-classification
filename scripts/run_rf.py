#!/usr/bin/env python3
"""Run the Random Forest classification workflow for both years."""

from pathlib import Path
from crop_type.modeling import RunConfig, run

if __name__ == "__main__":
    for year in (2023, 2024):
        run(RunConfig("RF", year, Path("data/Data.zip"), Path("results"),
                      partition_file=Path("data/reference_partitions.csv")))
