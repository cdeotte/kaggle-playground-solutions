#!/usr/bin/env python
"""Fold-local response logits and the nested logistic margin: cache/static/fold_<k>/.

For outer fold k (labels of the outer-train rows only):
  response_alpha1, response_alpha20   per-field target rate with smoothing alpha 1 / 20, as logits (13 columns each).
      Training rows: 5-fold cross-fit (KFold, seed 933000 + k); held-out and test rows: all outer-train labels.
  nested_response_margin              logit of a logistic regression (StandardScaler + LR, C 10) on the 26 response logits.
      Training rows: 5 margin splits (KFold, seed 880000 + k); in each split the LR is fit on the other 4/5 of outer-train,
      whose own response logits are again 5-fold cross-fitted (KFold, seed 990000 + 100 k + split), and applied to the
      held split encoded from those 4/5. Held-out and test rows: an LR fit on the cross-fitted logits of all outer-train rows,
      applied to logits encoded from all outer-train labels.
The XGBoost models start from this margin (base_margin).
Usage: python margin_cache.py [--folds 0,1,2,3,4] [--c 10]"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds, labels, read_test, read_train  # noqa: E402
from common.pairs import fold_list  # noqa: E402
from features import CACHE, FEATURES, exact_codes, save_arrays, update_manifest  # noqa: E402

ALPHAS = (1.0, 20.0)


def response_design(codes, y, fit, apply):
    """Smoothed per-field target rates fitted on `fit`, as logits: 13 columns for alpha 1, then 13 for alpha 20."""
    prior = float(y[fit].mean())
    columns = []
    for alpha in ALPHAS:
        for name in FEATURES:
            code = codes[name]
            size = int(code.max()) + 1
            count = np.bincount(code[fit], minlength=size).astype(np.float64)
            positive = np.bincount(code[fit], weights=y[fit], minlength=size).astype(np.float64)
            probability = (positive + alpha * prior) / (count + alpha)
            columns.append(logit(np.clip(probability[code[apply]], 1e-5, 1 - 1e-5)))
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def estimator(c_value):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c_value, solver='lbfgs', max_iter=400, tol=1e-7))


def one_margin_split(codes, y, margin_fit, margin_apply, seed, c_value):
    nested_train = np.empty((len(margin_fit), len(FEATURES) * len(ALPHAS)), dtype=np.float64)
    for nested_fit, nested_apply in KFold(5, shuffle=True, random_state=seed).split(np.zeros(len(margin_fit))):
        nested_train[nested_apply] = response_design(codes, y, margin_fit[nested_fit], margin_fit[nested_apply])
    model = estimator(c_value)
    model.fit(nested_train, y[margin_fit])
    return np.asarray(model.decision_function(response_design(codes, y, margin_fit, margin_apply)), dtype=np.float64)


def build_fold(k, y, f, codes, test_rows, c_value):
    outer_fit, outer_valid = np.flatnonzero(f != k), np.flatnonzero(f == k)
    margin_train = np.empty(len(outer_fit), dtype=np.float64)
    for split, (fit_rel, apply_rel) in enumerate(KFold(5, shuffle=True, random_state=880000 + k).split(np.zeros(len(outer_fit)))):
        margin_train[apply_rel] = one_margin_split(codes, y, outer_fit[fit_rel], outer_fit[apply_rel],
                                                   990000 + 100 * k + split, c_value)
    final_train = np.empty((len(outer_fit), len(FEATURES) * len(ALPHAS)), dtype=np.float64)
    for fit_rel, apply_rel in KFold(5, shuffle=True, random_state=933000 + k).split(np.zeros(len(outer_fit))):
        final_train[apply_rel] = response_design(codes, y, outer_fit[fit_rel], outer_fit[apply_rel])
    valid_design = response_design(codes, y, outer_fit, outer_valid)
    test_design = response_design(codes, y, outer_fit, test_rows)
    final_model = estimator(c_value)
    final_model.fit(final_train, y[outer_fit])
    margin_valid = np.asarray(final_model.decision_function(valid_design), dtype=np.float64)
    margin_test = np.asarray(final_model.decision_function(test_design), dtype=np.float64)

    n = len(FEATURES)
    folder = CACHE / 'static' / f'fold_{k}'
    save_arrays(folder, {
        'train_indices': outer_fit.astype(np.int64), 'valid_indices': outer_valid.astype(np.int64),
        'response_alpha1_train': final_train[:, :n].astype(np.float32), 'response_alpha1_valid': valid_design[:, :n],
        'response_alpha1_test': test_design[:, :n],
        'response_alpha20_train': final_train[:, n:].astype(np.float32), 'response_alpha20_valid': valid_design[:, n:],
        'response_alpha20_test': test_design[:, n:],
        'nested_response_margin_train': margin_train[:, None], 'nested_response_margin_valid': margin_valid[:, None],
        'nested_response_margin_test': margin_test[:, None],
    })
    names = [f'resp_logit__{name}' for name in FEATURES]
    update_manifest(folder, {'response_alpha1_names': names, 'response_alpha20_names': names,
                             'nested_response_margin_names': ['nested_response_margin_logit']})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--c', type=float, default=10.0)
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    train, test = read_train(), read_test()
    y = labels(train[TARGET])
    f = folds(y)
    codes = exact_codes(train, test)
    test_rows = len(train) + np.arange(len(test), dtype=np.int64)
    for k in fold_list(a.folds):
        build_fold(k, y, f, codes, test_rows, a.c)
        margin = np.load(CACHE / 'static' / f'fold_{k}' / 'nested_response_margin_train.npy')[:, 0]
        print(f'fold {k}: cross-fitted margin AUC on outer-train {roc_auc_score(y[f != k], margin):.6f}', flush=True)


if __name__ == '__main__':
    main()
