#!/usr/bin/env python3
"""Run both classifiers against the same annual field partitions."""

from pathlib import Path
from crop_type.modeling import RunConfig, run

if __name__ == "__main__":
    for year in (2023, 2024):
        reports = {}
        for model in ("RF", "LGBM"):
            reports[model] = run(RunConfig(model, year, Path("data/Data.zip"), Path("results"),
                                           partition_file=Path("data/reference_partitions.csv")))
        assert reports["RF"]["test_signature"] == reports["LGBM"]["test_signature"]
        assert reports["RF"]["train_signature"] == reports["LGBM"]["train_signature"]
