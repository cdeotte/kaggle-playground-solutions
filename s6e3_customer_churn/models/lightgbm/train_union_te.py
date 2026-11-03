#!/usr/bin/env python
"""LightGBM on high-cardinality keys, counts, original priors, bins and digits with per-fold target encoding; seven
presets on the feature recipe of lib/highcard_union_features.py.

The recipe adapts angelosmar1's public LightGBM notebook (2nd place, an earlier Playground competition):
https://www.kaggle.com/code/angelosmar1/s5e11-lgbm-cv-0-92813-2nd-place

Per fold (the shared 5 folds of common.data): every `_te` column is replaced by its scikit-learn TargetEncoder value
(binary, `inner_folds` shuffled inner folds, seed 42: the training rows are encoded out-of-fold, the validation and test
rows by the encoder fitted on all training rows); the other string columns stay pandas categories. LightGBM on 32
threads, seed 42.

Presets (pair: feature blocks; model):
  lgbm_highcard_bins_digits_te    base blocks (72 columns, 29 keys), TargetEncoder smooth 10; gbdt, 7,500 rounds,
                                  lr 0.01, max_depth 4, subsample 0.5, colsample_bytree 0.2, lambda_l1 10, lambda_l2 15
  lgbm_union_te                   base + union blocks (391 columns, 59 encoded), smooth "auto"; the same gbdt
  lgbm_union_dist_ngram_te        base + union + distribution / digit / n-gram blocks (466 columns, 81 encoded);
                                  the same gbdt with 10,000 rounds
  lgbm_union_dist_ngram_te_slow   the same features; 30,000 rounds, lr 0.0025
  lgbm_union_dist_ngram_te_20k    the same features; 20,000 rounds
  lgbm_union_dist_ngram_te_goss   the same features; GOSS (top_rate 0.2, other_rate 0.1), 63 leaves, no depth limit,
                                  up to 7,500 rounds with early stopping (300) on the validation fold
  lgbm_union_dist_ngram_te_dart   the same features; DART (drop_rate 0.05, skip_drop 0.6, max_drop 25), 31 leaves,
                                  max_depth 4, 7,500 rounds
Usage: python train_union_te.py <pair>"""
import argparse
import gc
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgbm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds  # noqa: E402
from common.pairs import save_pair  # noqa: E402
from models.lightgbm.lib.highcard_union_features import build_features  # noqa: E402
from models.lightgbm.lib.kfold import stratified_fold_ids  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5

GBDT = dict(
    objective="binary",
    metric="auc",
    n_estimators=7_500,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.5,
    colsample_bytree=0.2,
    lambda_l2=15.0,
    lambda_l1=10.0,
    random_state=SEED,
    verbosity=-1,
    num_threads=32,
)
GOSS = dict(
    objective="binary",
    metric="auc",
    n_estimators=7_500,
    learning_rate=0.01,
    max_depth=-1,
    num_leaves=63,
    colsample_bytree=0.2,
    lambda_l2=15.0,
    lambda_l1=10.0,
    boosting_type="goss",
    top_rate=0.2,
    other_rate=0.1,
    random_state=SEED,
    verbosity=-1,
    num_threads=32,
)
DART = dict(
    objective="binary",
    metric="auc",
    boosting_type="dart",
    n_estimators=7_500,
    learning_rate=0.01,
    max_depth=4,
    num_leaves=31,
    colsample_bytree=0.2,
    lambda_l2=15.0,
    lambda_l1=10.0,
    drop_rate=0.05,
    skip_drop=0.6,
    max_drop=25,
    uniform_drop=False,
    xgboost_dart_mode=False,
    random_state=SEED,
    verbosity=-1,
    num_threads=32,
)


@dataclass
class Recipe:
    union_blocks: bool
    extra_blocks: bool
    te_smooth: object
    params: dict
    early_stopping: int = 0
    inner_folds: int = 5
    test_folds: int = N_FOLDS
    test_inner_folds: int = 5
    test_params: dict = field(default_factory=dict)
    test_early_stopping: int = 0


RECIPES = {
    "lgbm_highcard_bins_digits_te": Recipe(
        union_blocks=False, extra_blocks=False, te_smooth=10.0, params=GBDT,
        test_folds=15, test_params=dict(n_estimators=8_750)),
    "lgbm_union_te": Recipe(
        union_blocks=True, extra_blocks=False, te_smooth="auto", params=GBDT,
        test_folds=15, test_inner_folds=10, test_params=dict(n_estimators=1_000_000, learning_rate=0.005),
        test_early_stopping=500),
    "lgbm_union_dist_ngram_te": Recipe(
        union_blocks=True, extra_blocks=True, te_smooth="auto", params=dict(GBDT, n_estimators=10_000),
        test_folds=15, test_inner_folds=10),
    "lgbm_union_dist_ngram_te_slow": Recipe(
        union_blocks=True, extra_blocks=True, te_smooth="auto",
        params=dict(GBDT, n_estimators=30_000, learning_rate=0.0025)),
    "lgbm_union_dist_ngram_te_20k": Recipe(
        union_blocks=True, extra_blocks=True, te_smooth="auto", params=dict(GBDT, n_estimators=20_000),
        test_folds=15, test_inner_folds=10),
    "lgbm_union_dist_ngram_te_goss": Recipe(
        union_blocks=True, extra_blocks=True, te_smooth="auto", params=GOSS, early_stopping=300),
    "lgbm_union_dist_ngram_te_dart": Recipe(
        union_blocks=True, extra_blocks=True, te_smooth="auto", params=DART),
}


def cross_validate(X_fe, T_fe, y, fold_ids, n_folds, inner_folds, te_smooth, params, early_stopping):
    oof = np.zeros(len(X_fe), dtype=np.float64)
    pred_test_folds = np.zeros((len(T_fe), n_folds), dtype=np.float64)
    to_target_encode = [c for c in X_fe.columns if c.endswith("_te")]

    for k in range(n_folds):
        tr_idx = np.where(fold_ids != k)[0]
        va_idx = np.where(fold_ids == k)[0]
        X_tr = X_fe.iloc[tr_idx].copy()
        y_tr = y[tr_idx].copy()
        X_va = X_fe.iloc[va_idx].copy()
        y_va = y[va_idx].copy()
        X_te = T_fe.copy()

        enc = TargetEncoder(cv=inner_folds, shuffle=True, random_state=SEED, smooth=te_smooth, target_type="binary")
        X_tr[to_target_encode] = enc.fit_transform(X_tr[to_target_encode], y_tr)
        X_va[to_target_encode] = enc.transform(X_va[to_target_encode])
        X_te[to_target_encode] = enc.transform(X_te[to_target_encode])

        callbacks = [lgbm.log_evaluation(1000)]
        if early_stopping:
            callbacks.append(lgbm.early_stopping(early_stopping, verbose=False))
        model = lgbm.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)
        best_iter = model.best_iteration_ if early_stopping else None

        p_va = model.predict_proba(X_va, num_iteration=best_iter)[:, 1]
        oof[va_idx] = p_va
        pred_test_folds[:, k] = model.predict_proba(X_te, num_iteration=best_iter)[:, 1]
        msg = f"fold {k + 1}/{n_folds}: AUC {roc_auc_score(y_va, p_va):.6f}"
        if early_stopping:
            msg += f", best_iteration {best_iter}"
        print(msg, flush=True)

        del X_tr, X_va, X_te, model
        gc.collect()

    return oof, pred_test_folds.mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(RECIPES))
    args = ap.parse_args()
    r = RECIPES[args.pair]

    X_fe, T_fe, y = build_features(union_blocks=r.union_blocks, extra_blocks=r.extra_blocks)
    oof, test_pred = cross_validate(X_fe, T_fe, y, folds(y), N_FOLDS, r.inner_folds, r.te_smooth, r.params,
                                    r.early_stopping)
    print(f"{args.pair}: OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)
    if r.test_folds != N_FOLDS:
        _, test_pred = cross_validate(X_fe, T_fe, y, stratified_fold_ids(y, r.test_folds), r.test_folds,
                                      r.test_inner_folds, r.te_smooth, dict(r.params, **r.test_params),
                                      r.test_early_stopping)
    save_pair(HERE, args.pair, oof, test_pred)


if __name__ == "__main__":
    main()
