#!/usr/bin/env python
"""One RealMLP (lib/realmlp.py) per outer fold on the base matrix plus the blocks of a chain step.

Pairs (chain order; columns = base 135 + source block + completion aggregates):
  realmlp_forensic_base                 base only                                                          135
  realmlp_forensic_density_completion   + 10 completion aggregates of the fold density state                145
  realmlp_forensic_reconstruction       + 71 reconstruction channels                                        216
  realmlp_forensic_pairlift             + 10 pair-lift channels                                            226
  realmlp_forensic_global_density       source = pairlift 10, cbank 9, temperature 39, global density 51,
                                        wbank 5 (reconstruction removed); completion from the global mixture 259
  realmlp_forensic_repair20             + 20 repair-posterior channels after wbank                         279
Each fold writes parts/<pair>_f<k>_{valid,test}.npy; when all 5 folds exist the pair is assembled into
preds/<pair>_{oof,test}.npy (test = mean of the 5 fold models).
Usage: python train.py <pair> [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import assemble, fold_list, save_part  # noqa: E402
from lib import completion, temperature  # noqa: E402
from lib.global_density import GDEN_NAMES, MFILL_NAMES  # noqa: E402
from lib.lift import BANKS  # noqa: E402
from lib.paths import (MODEL_DIR, fold_indices, folds, load_array, load_fold_cache, own_dir, read_test,  # noqa: E402
                       read_train, target)
from lib.repair import REPAIR_NAMES  # noqa: E402

GLOBAL_SOURCE = ['pairlift', 'cbank', 'temperature39', 'global_channels', 'wbank']
RECIPES = {
    'realmlp_forensic_base': [],
    'realmlp_forensic_density_completion': ['fold_completion'],
    'realmlp_forensic_reconstruction': ['reconstruction', 'fold_completion'],
    'realmlp_forensic_pairlift': ['reconstruction', 'pairlift', 'fold_completion'],
    'realmlp_forensic_global_density': GLOBAL_SOURCE + ['global_completion'],
    'realmlp_forensic_repair20': GLOBAL_SOURCE + ['repair20', 'global_completion'],
}
ROLES = ('train', 'valid', 'test')


def fold_block(directory, columns=None, prefix=''):
    out = {}
    for role in ROLES:
        m = load_array(directory, f'{prefix}{role}')
        out[role] = np.asarray(m if columns is None else m[:, columns], dtype=np.float32)
    return out


def global_block(directory, fold):
    """Blocks computed once for all train rows / test rows: slice the fold's rows."""
    fold_train, fold_valid = fold_indices(fold)
    train_matrix = load_array(directory, 'train')
    return {'train': np.asarray(train_matrix[fold_train], dtype=np.float32),
            'valid': np.asarray(train_matrix[fold_valid], dtype=np.float32),
            'test': np.asarray(load_array(directory, 'test'), dtype=np.float32)}


def load_block(name, fold):
    aggregates = completion.AGGREGATE_INDICES
    if name == 'fold_completion':
        return load_fold_cache(own_dir('density_completion_response_production_features', f'fold_{fold}'), aggregates)
    if name == 'global_completion':
        names = [completion.feature_names()[i] for i in aggregates]
        return names, fold_block(own_dir('global_completion', f'fold_{fold}'), aggregates)
    if name == 'reconstruction':  # compact view = first 71 columns
        return load_fold_cache(own_dir('reconstruction-production-v1', f'fold_{fold}'), range(71))
    if name in BANKS:
        return BANKS[name].names, fold_block(own_dir('lift', f'fold_{fold}'), prefix=f'{name}_')
    if name == 'temperature39':
        names = [temperature.feature_names()[i] for i in temperature.STRONG39]
        return names, fold_block(own_dir('temperature', f'fold_{fold}'), temperature.STRONG39)
    if name == 'global_channels':
        return GDEN_NAMES + MFILL_NAMES, global_block(own_dir('global_channels'), fold)
    if name == 'repair20':
        return list(REPAIR_NAMES), global_block(own_dir('repair'), fold)
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=list(RECIPES))
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    from lib.realmlp import fit_fold
    train, test, y = read_train(), read_test(), target()
    for fold in fold_list(args.folds):
        blocks = [load_block(name, fold) for name in RECIPES[args.pair]]
        valid_prediction, test_prediction = fit_fold(fold, train, test, y, blocks)
        save_part(MODEL_DIR, args.pair, fold, valid_prediction, test_prediction)
        from sklearn.metrics import roc_auc_score
        print(f'{args.pair} fold {fold}: AUC {roc_auc_score(y[fold_indices(fold)[1]], valid_prediction):.6f}',
              flush=True)
    assemble(MODEL_DIR, args.pair, folds(y), y)


if __name__ == '__main__':
    main()
