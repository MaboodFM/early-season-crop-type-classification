#!/usr/bin/env python3
"""Run the complete primary, robustness, and SHAP workflow sequentially."""
import argparse
import logging
from pathlib import Path
from crop_type.modeling import RunConfig,run as run_model
from crop_type.robustness import RobustnessConfig,run as run_robustness
from crop_type.shap_analysis import run as run_shap
from crop_type.spectral_diagnostics import run as run_diagnostics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/Data.zip'))
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--partitions',type=Path,default=Path('data/reference_partitions.csv'))
    parser.add_argument('--repeated-memberships',type=Path,default=Path('data/reference_phase4_resplits.csv'))
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--pilot',action='store_true',help='Primary/Phase 4 smoke tests only (no SHAP)')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    for algorithm in ('RF','LGBM'):
        for year in (2023,2024):
            run_model(RunConfig(algorithm,year,args.data,args.output,
                                partition_file=args.partitions,pilot=args.pilot,workers=args.workers))
            run_robustness(RobustnessConfig(algorithm,year,args.data,args.output,
                                           args.partitions,args.repeated_memberships,
                                           args.pilot,args.workers))
        if not args.pilot:
            run_shap(algorithm,(2023,2024),args.data,args.output,args.partitions)
            run_diagnostics(args.data,args.output,algorithm,
                args.output/f"EndSeason_Error_Spectral_Diagnostics_{algorithm}_Results",
                args.partitions)


if __name__=='__main__':
    main()
