#!/usr/bin/env python
"""Steps 4-5: pair-lift banks per outer fold (lib/lift.py): pairlift (10 columns, step 4), cbank (9) and wbank (5)
(step 5). Outer-training rows: inner StratifiedKFold(5, shuffle=True, random_state=777) cross-fit; outer-validation and
test rows: state fit on all outer-training rows.
Writes cache/lift/fold_<k>/<bank>_{train,valid,test}.npy.
Usage: python build_lift_banks.py [--folds 0,1,2,3,4] [--banks pairlift,cbank,wbank]"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import fold_list  # noqa: E402
from lib.lift import BANKS, crossfit  # noqa: E402
from lib.paths import fold_indices, own_dir, read_test, read_train, save_arrays, target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--banks', default='pairlift,cbank,wbank')
    args = ap.parse_args()
    train, test, y = read_train(), read_test(), target()
    for name in args.banks.split(','):
        bank = BANKS[name]
        train_tokens, test_tokens = bank.tokenise(train), bank.tokenise(test)
        for fold in fold_list(args.folds):
            fold_train, fold_valid = fold_indices(fold)
            out = crossfit(bank, train_tokens, y, fold_train, fold_valid, test_tokens)
            save_arrays(own_dir('lift', f'fold_{fold}'), **{f'{name}_{role}': m for role, m in out.items()})
            print(f'fold {fold}: {name} done', flush=True)


if __name__ == '__main__':
    main()
