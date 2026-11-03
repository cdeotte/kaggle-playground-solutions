#!/usr/bin/env python
"""Step 2b: 30 density-completion response channels per outer fold (lib/completion.py) from that fold's density
state (build_fold_density.py). The lattice encoder is refit on the fold's outer-training predictors, exactly as when the
state was trained. Rate tables: outer-training labels only, hash-group complement tables for the training rows.
Writes cache/density_completion_response_production_features/fold_<k>/{train,valid,test}_features.npy (30 columns;
the RealMLP uses the 10 aggregates), train_indices.npy, valid_indices.npy and manifest.json (feature_names).
Usage: python build_completion.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import fold_list  # noqa: E402
from lib import completion  # noqa: E402
from lib.lattice import FOLD_CONFIG, LatticeEncoder, load_model  # noqa: E402
from lib.paths import fold_indices, own_dir, read_test, read_train, save_fold_cache, target  # noqa: E402

CACHE = 'density_completion_response_production_features'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    import torch
    train, test, y = read_train(), read_test(), target()
    for fold in fold_list(args.folds):
        fold_train, fold_valid = fold_indices(fold)
        encoder = LatticeEncoder(train.iloc[fold_train].reset_index(drop=True))
        model = load_model(torch, own_dir('fold_density', f'fold_{fold}') / 'model_state.pt', encoder.cardinalities,
                           FOLD_CONFIG)
        out = completion.build_fold(torch, completion.single_model_conditional(torch, model), encoder, train, y,
                                    fold_train, fold_valid, test, int(FOLD_CONFIG['batch_size']))
        save_fold_cache(own_dir(CACHE, f'fold_{fold}'), out, completion.feature_names(), fold)
        print(f'fold {fold}: completion channels done', flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
