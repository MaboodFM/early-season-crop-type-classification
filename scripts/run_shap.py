#!/usr/bin/env python3
"""Explain the November 1 held-out RF or LightGBM predictions with Tree SHAP."""
import argparse
import logging
from pathlib import Path
from crop_type.shap_analysis import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm',choices=('RF','LGBM'),required=True)
    parser.add_argument('--years',nargs='+',type=int,choices=(2023,2024),default=[2023,2024])
    parser.add_argument('--data',type=Path,default=Path('data/Data.zip'))
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--partitions',type=Path,default=Path('data/reference_partitions.csv'))
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    print('Results:',run(args.algorithm,tuple(args.years),args.data,args.output,args.partitions))


if __name__=='__main__':
    main()
