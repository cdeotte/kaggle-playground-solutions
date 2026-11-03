#!/usr/bin/env python
"""Public notebook "S6E8: LGBM | LB 0.96965" (version 1) run on our 5 folds (the notebook uses the same folds).
https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965

Features (lib/constrained_te.py): the 12 raw columns, missing count, 9 ratio features, constrained-imputation features,
frequency of every raw value over train + test, and per fold a TargetEncoder (cv=5, smooth 'auto') of the 12 raw
columns as strings fitted on the fold's training rows.
Model (the notebook's): LGBMClassifier, lr 0.0463, 780 rounds (no early stopping), 96 leaves, min_child_samples 267,
lambda 5.24, max_bin 1023, colsample_bytree 0.473, native categoricals, seed 0. Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

import lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from lib.constrained_te import CAT, fold_frames, load  # noqa: E402
from lib.runner import fold_args, run_folds  # noqa: E402

NAME = 'redamountassir_lgbm'
THREADS = 24


def main():
    fold_ids = fold_args()
    x, xt, xs, xst, raw, y = load()

    def fit_fold(fold, tr, va):
        atr, ava, ate = fold_frames(x, xt, xs, xst, raw, y, tr, va)
        model = lgb.LGBMClassifier(objective='binary', learning_rate=0.04625773730092041, n_estimators=780,
                                   num_leaves=96, min_child_samples=267, reg_lambda=5.236015002068583, max_bin=1023,
                                   colsample_bytree=0.47316360386000655, n_jobs=THREADS, verbose=-1, random_state=0)
        model.fit(atr, y[tr], categorical_feature=CAT)
        return model.predict_proba(ava)[:, 1], model.predict_proba(ate)[:, 1]

    run_folds(NAME, fit_fold, fold_ids, y=y)


if __name__ == '__main__':
    main()
