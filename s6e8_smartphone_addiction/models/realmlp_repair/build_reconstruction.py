#!/usr/bin/env python
"""Step 3: cross-fitted XGBoost / autoencoder field reconstruction per outer fold (lib/reconstruction.py, target-free;
the original data enters only through its fixed class likelihood ratios). 3 hash groups of outer-training rows, each
reconstructed by 12 XGBoost experts and one autoencoder fit on the other two groups; outer-validation and test rows by
a fit on all outer-training rows. Seeds: XGBoost 12000 + 1000 fold + 100 group (+ field index), 32000 + 1000 fold for
the full fit; autoencoder 22000 + 1000 fold + 100 group, 42000 + fold.
Writes cache/reconstruction-production-v1/fold_<k>/{train,valid,test}_features.npy (189 columns: compact 71,
detailed 52, full 66; the RealMLP uses the compact view), train_indices.npy, valid_indices.npy and manifest.json
(feature_names, view_counts).
Usage: python build_reconstruction.py [--folds 0,1,2,3,4] [--threads 12]"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import TARGET, labels  # noqa: E402
from common.pairs import fold_list  # noqa: E402
from lib import reconstruction  # noqa: E402
from lib.paths import fold_indices, own_dir, read_original, read_test, read_train, save_fold_cache  # noqa: E402

CACHE = 'reconstruction-production-v1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--threads', type=int, default=12, help='XGBoost CPU threads')
    args = ap.parse_args()
    train, test, original = read_train(), read_test(), read_original()
    original_y = labels(original[TARGET])
    for fold in fold_list(args.folds):
        fold_train, fold_valid = fold_indices(fold)
        out, names, view_counts = reconstruction.build_fold(fold, train, fold_train, fold_valid, test, original,
                                                            original_y, args.threads)
        save_fold_cache(own_dir(CACHE, f'fold_{fold}'), out, names, fold, view_counts=view_counts)
        print(f'fold {fold}: reconstruction channels done {view_counts}', flush=True)


if __name__ == '__main__':
    main()
