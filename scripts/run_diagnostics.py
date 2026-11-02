#!/usr/bin/env python3
"""Generate 14-day spectral-profile diagnostics for held-out errors."""
import argparse
from pathlib import Path
from crop_type.spectral_diagnostics import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm',choices=('RF','LGBM'),required=True)
    parser.add_argument('--data',type=Path,default=Path('data/Data.zip'))
    parser.add_argument('--results',type=Path,default=Path('results'))
    parser.add_argument('--partitions',type=Path,default=Path('data/reference_partitions.csv'))
    parser.add_argument('--no-figures',action='store_true')
    args=parser.parse_args()
    output=args.results/f'EndSeason_Error_Spectral_Diagnostics_{args.algorithm}_Results'
    print('Results:',run(args.data,args.results,args.algorithm,output,args.partitions,not args.no_figures))


if __name__=='__main__':
    main()
