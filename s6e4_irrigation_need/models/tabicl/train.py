#!/usr/bin/env python
"""Zero-shot TabICL, eight-member vote on the raw columns (pair tabicl_zero_shot_8seed_a).

Inputs: the 19 raw competition columns (the 8 categoricals stay strings; TabICL encodes them itself). No feature
engineering and no original rows. Per fold, eight pretrained TabICLClassifier models (n_estimators=2, kv_cache=True,
support_many_classes=True, random_state 0..7) get the fold's training rows as in-context examples (no weights are
trained) and predict the fold's validation rows and the test rows; the 8 probabilities are averaged. Inference is
deterministic. Test = mean of the 5 fold models.
Usage: python train.py"""
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from tabicl import TabICLClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CLASSES, ID, N_FOLDS, TARGET, TARGET_MAP, folds, model_dir, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

FOLDER = "tabicl"
PAIR = "tabicl_zero_shot_8seed_a"

N_MODELS = 8
PARAMS = [
    {
        "device": "cuda",
        "n_estimators": 2,
        "kv_cache": True,
        "random_state": i,
        "support_many_classes": True,
    }
    for i in range(N_MODELS)
]


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class VotingClassifier(ClassifierMixin, BaseEstimator):
    """Mean probability of one TabICLClassifier per parameter set; the members are fitted (context set) at predict time."""

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
    train = read_train()
    test = read_test()

    train[TARGET] = train[TARGET].map(TARGET_MAP)
    X = train.drop(columns=[TARGET]).set_index(ID)
    y = train[TARGET].astype(int).values
    X_test = test.set_index(ID)

    fold_id = folds(y)
    oof_pred = np.zeros((len(X), 3), dtype=np.float32)
    test_pred = np.zeros((len(X_test), 3), dtype=np.float32)
    fold_scores = []

    for fold in range(1, N_FOLDS + 1):
        tr_idx, va_idx = np.where(fold_id != fold - 1)[0], np.where(fold_id == fold - 1)[0]
        print(f"\n{'=' * 20} Fold {fold} {'=' * 20}")

        X_tr = X.iloc[tr_idx].copy()
        y_tr = y[tr_idx]
        X_va = X.iloc[va_idx].copy()
        y_va = y[va_idx]

        clf = VotingClassifier(PARAMS)
        clf.fit(X_tr, y_tr)

        va_pred = clf.predict_proba(X_va).astype(np.float32)
        oof_pred[va_idx] = va_pred

        fold_test_pred = clf.predict_proba(X_test).astype(np.float32)
        test_pred += fold_test_pred / N_FOLDS

        fold_ba = balanced_accuracy_score(y_va, va_pred.argmax(axis=1))
        fold_scores.append(fold_ba)
        print(f"Fold {fold} Balanced Accuracy: {fold_ba:.5f}")

        del clf, X_tr, y_tr, X_va, y_va, va_pred, fold_test_pred
        cleanup_cuda()

    cv_ba = balanced_accuracy_score(y, oof_pred.argmax(axis=1))
    print("\n" + "=" * 50)
    print(f"CV Balanced Accuracy (OOF): {cv_ba:.5f}")
    print(f"Mean Fold Balanced Accuracy: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print("Class order:", list(CLASSES))
    save_pair(model_dir(FOLDER), PAIR, oof_pred, test_pred)


if __name__ == "__main__":
    main()
