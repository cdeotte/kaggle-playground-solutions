#!/usr/bin/env python
"""Build the 10-column exact-pair lift bank of every outer fold (lib/pairlift.py) and cache it:
cache/pairlift/fold_<k>/{train,valid,test}_features.npy (outer-training rows inner-cross-fitted; outer-validation and
test rows from the full outer-training state). CPU only, about a minute per fold.
Usage: python build_pairlift.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import folds, read_test, read_train, target  # noqa: E402
from common.pairs import fold_list  # noqa: E402
from lib import pairlift  # noqa: E402
from lib.paths import fold_indices, pairlift_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    y = target()
    fold_ids = folds(y)
    train_tokens = pairlift.tokenise_frame(read_train())
    test_tokens = pairlift.tokenise_frame(read_test())
    for fold in fold_list(args.folds):
        train_idx, valid_idx = fold_indices(fold, fold_ids)
        matrices = pairlift.build_fold(train_tokens, test_tokens, y, train_idx, valid_idx)
        out = pairlift_dir(fold)
        for role, values in zip(('train', 'valid', 'test'), matrices):
            np.save(out / f'{role}_features.npy', values)
        print(f'fold {fold}: pair-lift bank {matrices[0].shape[1]} columns written', flush=True)


if __name__ == '__main__':
    main()
