#!/usr/bin/env python
"""YDF gradient boosted trees from the discussion post "YDF gives pretty good CV score off default parameters", run on
our 5 folds.
https://www.kaggle.com/competitions/playground-series-s6e3/discussion/679983

Features: the 19 raw competition columns; YDF handles the categorical (string) columns natively. No feature
engineering, no original rows.
Model: ydf GradientBoostedTreesLearner (CPU) with the post's parameters: max_depth 2, BEST_FIRST_GLOBAL growth, RANDOM
categorical splits, shrinkage 0.1, up to 10,000 trees with early stopping (300-tree look-ahead) on the validation fold.
Test = mean of the fold models.
Usage: python train.py"""
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import ydf

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, read_test, read_train  # noqa: E402
from lib.runner import run_folds  # noqa: E402

warnings.filterwarnings('ignore')
ydf.verbose(0)

FOLDER = 'siukeitin_ydf'
TEST_FOLDS = 10
PARAMS = dict(
    label=TARGET,
    task=ydf.Task.CLASSIFICATION,
    shrinkage=0.1,
    early_stopping_num_trees_look_ahead=300,
    max_depth=2,
    growing_strategy='BEST_FIRST_GLOBAL',
    categorical_algorithm='RANDOM',
    num_trees=10000,
)
TEST_PARAMS = {**PARAMS, 'shrinkage': 0.03, 'early_stopping_num_trees_look_ahead': 500, 'num_trees': 50_000}


def main():
    train = read_train().set_index(ID)
    test = read_test().set_index(ID)
    y = (train[TARGET] == 'Yes').astype(np.int32).values
    train_df = train.copy()
    train_df[TARGET] = y
    print(f'ydf {ydf.__version__} | training rows: {len(train_df):,} | test rows: {len(test):,}', flush=True)

    def fit_fold(fold, tr_idx, va_idx, n_folds):
        params = TEST_PARAMS if n_folds == TEST_FOLDS else PARAMS
        trn = train_df.iloc[tr_idx].copy()
        val = train_df.iloc[va_idx].copy()
        learner = ydf.GradientBoostedTreesLearner(**params)
        model = learner.train(trn, val)
        valid_pred = model.predict(val).reshape(-1)
        test_pred = model.predict(test).reshape(-1).astype(np.float32)
        del model, learner, trn, val
        gc.collect()
        return valid_pred, test_pred

    run_folds(FOLDER, FOLDER, fit_fold, test_folds=TEST_FOLDS, y=y)


if __name__ == '__main__':
    main()
