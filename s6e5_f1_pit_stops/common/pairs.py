"""Fold predictions and OOF / test pairs.

Every model writes its pair(s) to models/<model>/preds/:
  <pair>_oof.npy   one prediction per train row (train.csv order), made by the fold model that did not train on that row
  <pair>_test.npy  one prediction per test row (test.csv order), the mean of the 5 fold models
Values can be probabilities or logits; the stacker handles both.
While a model runs, each finished fold is kept in models/<model>/parts/, so folds can run one at a time or in parallel."""
from pathlib import Path

import numpy as np


def _part(model_dir, name, fold, split):
    return Path(model_dir) / 'parts' / f'{name}_f{fold}_{split}.npy'


def have_part(model_dir, name, fold):
    return _part(model_dir, name, fold, 'valid').exists() and _part(model_dir, name, fold, 'test').exists()


def save_part(model_dir, name, fold, valid_pred, test_pred):
    """valid_pred: predictions for the rows of fold `fold` (in train.csv order); test_pred: predictions for all test rows."""
    p = _part(model_dir, name, fold, 'valid')
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, np.asarray(valid_pred, np.float32))
    np.save(_part(model_dir, name, fold, 'test'), np.asarray(test_pred, np.float32))


def save_pair(model_dir, name, oof, test):
    d = Path(model_dir) / 'preds'
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f'{name}_oof.npy', np.asarray(oof, np.float32))
    np.save(d / f'{name}_test.npy', np.asarray(test, np.float32))


def assemble(model_dir, name, folds, y=None):
    """Once all 5 folds are done: OOF from the held-out predictions, test = mean of the 5 fold models. Writes the pair."""
    if not all(have_part(model_dir, name, k) for k in range(5)):
        print(f'{name}: not all folds are done yet', flush=True)
        return None
    oof, test = np.zeros(len(folds)), 0.0
    for k in range(5):
        oof[folds == k] = np.load(_part(model_dir, name, k, 'valid'))
        test = test + np.load(_part(model_dir, name, k, 'test')).astype(np.float64) / 5
    save_pair(model_dir, name, oof, test)
    if y is not None:
        from sklearn.metrics import roc_auc_score
        print(f'{name}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    return oof


def fold_list(arg):
    """'0,1,2,3,4' -> [0, 1, 2, 3, 4]"""
    return [int(k) for k in str(arg).split(',') if k != '']
