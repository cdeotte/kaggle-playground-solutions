#!/usr/bin/env python
"""Step 5: generator-temperature channels per outer fold (lib/temperature.py; target-free): exact-value counts of the
outer-training rows versus the original data, with a weighted power-law fit per field.
Writes cache/temperature/fold_<k>/{train,valid,test}.npy (rows x 56; the RealMLP uses the 39 screen-field channels).
Usage: python build_temperature.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import fold_list  # noqa: E402
from lib import temperature  # noqa: E402
from lib.paths import fold_indices, own_dir, read_original, read_test, read_train, save_arrays  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    train, test, original = read_train(), read_test(), read_original()
    for fold in fold_list(args.folds):
        fold_train, fold_valid = fold_indices(fold)
        save_arrays(own_dir('temperature', f'fold_{fold}'),
                    **temperature.build_fold(train, original, fold_train, fold_valid, test))
        print(f'fold {fold}: temperature channels done', flush=True)


if __name__ == '__main__':
    main()
