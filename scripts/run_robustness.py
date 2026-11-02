#!/usr/bin/env python3
"""Bootstrap intervals, repeated field partitions, and county holdouts."""
import argparse
import logging
from pathlib import Path
from crop_type.robustness import RobustnessConfig, run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm',choices=('RF','LGBM'),required=True)
    parser.add_argument('--year',type=int,choices=(2023,2024),required=True)
    parser.add_argument('--data',type=Path,default=Path('data/Data.zip'))
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--partitions',type=Path,default=Path('data/reference_partitions.csv'))
    parser.add_argument('--repeated-memberships',type=Path,default=Path('data/reference_phase4_resplits.csv'))
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--pilot',action='store_true',help='Reduced nonpublication check')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    result=run(RobustnessConfig(args.algorithm,args.year,args.data,args.output,
                                args.partitions,args.repeated_memberships,args.pilot,args.workers))
    print('Results:',result)


if __name__=='__main__':
    main()
