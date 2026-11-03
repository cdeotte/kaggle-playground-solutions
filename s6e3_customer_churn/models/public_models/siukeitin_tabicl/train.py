#!/usr/bin/env python
"""TabICL ensemble from a comment by @siukeitin on the discussion "TabICL baseline CV:0.91385,LB:0.91067" (by
@yunsuxiaozi), run on our 5 folds.
https://www.kaggle.com/competitions/playground-series-s6e3/discussion/681229

Features: the 19 raw competition columns (TabICL encodes the string columns itself). No feature engineering, no
original rows.
Model: a voting ensemble of 8 TabICLClassifier models (n_estimators 2 each, i.e. 16 ensemble members in all; kv_cache
on; random_state 0-7), each given all of the fold's training rows as context; the predicted probabilities are
averaged. No gradient training: TabICL is a pretrained in-context-learning foundation model (the library's default
TabICLv2 classifier checkpoint, downloaded on first use).
Test = mean of the fold models.
Usage: python train.py"""
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from tabicl import TabICLClassifier

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, read_test, read_train  # noqa: E402
from lib.runner import run_folds  # noqa: E402

FOLDER = 'siukeitin_tabicl'
TEST_FOLDS = 10
N_MODELS = 8
PARAMS = [
    {
        'device': 'cuda',
        'n_estimators': 2,
        'kv_cache': True,
        'random_state': i,
    }
    for i in range(N_MODELS)
]


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class VotingClassifier(ClassifierMixin, BaseEstimator):
    """Keeps the training rows; predict_proba fits one TabICLClassifier per parameter set on them (one at a time, to
    fit in GPU memory) and averages their probabilities."""

    def __init__(self, params):
        self.params = params

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.X_ = X.copy()
        self.y_ = np.asarray(y).copy()
        return self

    def predict_proba(self, X):
        probs = []
        for params in self.params:
            model = TabICLClassifier(**params)
            model.fit(self.X_, self.y_)
            probs.append(model.predict_proba(X))
            del model
            cleanup_cuda()
        return np.mean(probs, axis=0)


def main():
    train = read_train().set_index(ID)
    X_test = read_test().set_index(ID)
    y = (train.pop(TARGET) == 'Yes').astype(int).values
    X = train

    def fit_fold(fold, tr_idx, va_idx, n_folds):
        clf = VotingClassifier(PARAMS)
        clf.fit(X.iloc[tr_idx].copy(), y[tr_idx])
        va_pred = clf.predict_proba(X.iloc[va_idx].copy())[:, 1]
        test_pred = clf.predict_proba(X_test)[:, 1]
        del clf
        cleanup_cuda()
        return va_pred, test_pred

    run_folds(FOLDER, FOLDER, fit_fold, test_folds=TEST_FOLDS, y=y)


if __name__ == '__main__':
    main()
