"""Fold predictions and OOF / test pairs (3-class probabilities, class order Low, Medium, High).

Every model writes its pair to models/<model>/preds/:
  <pair>_oof.npy   (n_train, 3): one prediction per train row (train.csv order), made by the fold model that did not train on
                   that row. Models with nested inner cross-validation save (5, n_train, 3) instead: slice k holds outer
                   fold k's inner-CV predictions for its training rows and its predictions for its held-out rows; the
                   stacker averages the slices (in logit space).
  <pair>_test.npy  (n_test, 3): one prediction per test row (test.csv order), the mean of the fold models.
While a model runs, each finished fold can be kept in models/<model>/parts/, so folds can run one at a time."""
from pathlib import Path

import numpy as np


def _part(model_dir, name, fold, split):
    return Path(model_dir) / 'parts' / f'{name}_f{fold}_{split}.npy'


def have_part(model_dir, name, fold):
    return _part(model_dir, name, fold, 'valid').exists() and _part(model_dir, name, fold, 'test').exists()


def save_part(model_dir, name, fold, valid_pred, test_pred):
    """valid_pred: (rows of fold `fold` in train.csv order, 3); test_pred: (n_test, 3)."""
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
    test = 0.0
    oof = None
    for k in range(5):
        v = np.load(_part(model_dir, name, k, 'valid'))
        if oof is None:
            oof = np.zeros((len(folds),) + v.shape[1:])
        oof[folds == k] = v
        test = test + np.load(_part(model_dir, name, k, 'test')).astype(np.float64) / 5
    save_pair(model_dir, name, oof, test)
    if y is not None:
        from sklearn.metrics import balanced_accuracy_score
        print(f'{name}: OOF balanced accuracy {balanced_accuracy_score(y, oof.argmax(1)):.6f}', flush=True)
    return oof


def fold_list(arg):
    """'0,1,2,3,4' -> [0, 1, 2, 3, 4]"""
    return [int(k) for k in str(arg).split(',') if k != '']
