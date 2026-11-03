#!/usr/bin/env python
"""Step 5: the 30 density-completion response channels per outer fold rebuilt from the global density mixture
(tempered 4-state mixture instead of the fold state; global lattice encoder). Rate tables: outer-training labels only,
hash-group complement tables for the training rows (lib/completion.py).
Writes cache/global_completion/fold_<k>/{train,valid,test}.npy (rows x 30; the RealMLP uses the 10 aggregates).
Usage: python build_global_completion.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import fold_list  # noqa: E402
from lib import completion  # noqa: E402
from lib.global_density import BATCH_SIZE, global_encoder, load_models, mixture_conditional  # noqa: E402
from lib.paths import fold_indices, own_dir, read_original, read_test, read_train, save_arrays, target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    import torch
    train, test, original, y = read_train(), read_test(), read_original(), target()
    encoder = global_encoder(train, test, original)
    models = load_models(torch, encoder, lambda c: own_dir('global_density', c) / 'model_state.pt')
    conditional = mixture_conditional(torch, models)
    for fold in fold_list(args.folds):
        fold_train, fold_valid = fold_indices(fold)
        out = completion.build_fold(torch, conditional, encoder, train, y, fold_train, fold_valid, test, BATCH_SIZE)
        save_arrays(own_dir('global_completion', f'fold_{fold}'), **out)
        print(f'fold {fold}: global completion channels done', flush=True)


if __name__ == '__main__':
    main()
