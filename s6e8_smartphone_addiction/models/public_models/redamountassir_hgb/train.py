#!/usr/bin/env python
"""Public notebook "S6E8: HistGradientBoosting | LB 0.96945" (version 2) run on our 5 folds (the notebook uses the same
folds).
https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945

Features (lib/constrained_te.py): the 12 raw columns, missing count, 9 ratio features, constrained-imputation features,
frequency of every raw value over train + test, and per fold a TargetEncoder (cv=5, smooth 'auto') of the 12 raw
columns as strings fitted on the fold's training rows.
Model (the notebook's): scikit-learn HistGradientBoostingClassifier, lr 0.0463, 780 iterations (no early stopping),
96 leaf nodes, min_samples_leaf 267, l2 5.24, max_bins 255, max_features 0.473, native categoricals, seed 0.
Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from lib.constrained_te import fold_frames, load  # noqa: E402
from lib.runner import fold_args, run_folds  # noqa: E402

NAME = 'redamountassir_hgb'


def main():
    fold_ids = fold_args()
    x, xt, xs, xst, raw, y = load()

    def fit_fold(fold, tr, va):
        atr, ava, ate = fold_frames(x, xt, xs, xst, raw, y, tr, va)
        model = HistGradientBoostingClassifier(learning_rate=0.04625773730092041, max_iter=780, max_leaf_nodes=96,
                                               min_samples_leaf=267, l2_regularization=5.236015002068583,
                                               max_bins=255, max_features=0.47316360386000655, early_stopping=False,
                                               categorical_features='from_dtype', random_state=0)
        model.fit(atr, y[tr])
        return model.predict_proba(ava)[:, 1], model.predict_proba(ate)[:, 1]

    run_folds(NAME, fit_fold, fold_ids, y=y)


if __name__ == '__main__':
    main()
