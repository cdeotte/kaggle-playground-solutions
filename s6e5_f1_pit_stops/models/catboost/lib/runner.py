"""Fold bookkeeping shared by the training scripts of this folder.

Each finished outer fold is kept in models/catboost/parts/ (common.pairs.save_part), so folds can run one at a time or in
separate processes; preds/<pair>_{oof,test}.npy is written once all 5 folds exist (test = mean of the 5 fold models)."""
import argparse
from pathlib import Path

import numpy as np

from common.data import folds, target
from common.pairs import assemble, fold_list, have_part

FOLDER = Path(__file__).resolve().parents[1]


def fold_parser(pairs=None):
    ap = argparse.ArgumentParser()
    if pairs is not None:
        ap.add_argument('pair', choices=list(pairs))
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--force', action='store_true', help='rerun folds that are already done')
    return ap


def todo_folds(pair, fold_arg, force=False):
    todo = []
    for k in fold_list(fold_arg):
        if have_part(FOLDER, pair, k) and not force:
            print(f'{pair} fold {k}: done', flush=True)
        else:
            todo.append(k)
    return todo


def finish(pair, clip=None):
    """Writes the pair once all 5 folds are done. clip: clip the averaged OOF / test probabilities to [clip, 1 - clip]."""
    y = target()
    oof = assemble(FOLDER, pair, folds(y), y)
    if oof is not None and clip:
        for split in ('oof', 'test'):
            path = FOLDER / 'preds' / f'{pair}_{split}.npy'
            np.save(path, np.clip(np.load(path), clip, 1 - clip).astype(np.float32))
