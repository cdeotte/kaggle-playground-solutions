#!/usr/bin/env python
"""GPU CatBoost with native (ordered target statistics) categorical handling and Balanced class weights.

Recipes (pair name):
  catboost_native_cats_balanced              the 19 raw columns + a TotalCharges-missing flag (20 features; the 16
                                             non-numeric columns, SeniorCitizen included, are categoricals)
  catboost_native_cats_fold_stats_balanced   + statistics fitted on each fold's training rows for all 20 columns:
                                             value frequency, target mean and count per value, and 20 quantile bins
                                             (KBinsDiscretizer) of the 4 numeric columns as extra categoricals (84
                                             features)
Missing numerics are filled with the fold's training medians.
CatBoost (GPU): Logloss (AUC reported), depth 8, lr 0.03, l2 3, random_strength 0.5, Bayesian bagging_temperature 1,
border_count 254, auto_class_weights Balanced, up to 20,000 iterations with early stopping (500, on the validation
Logloss) (use_best_model=True), seed 42. Test = mean of the fold models.

Usage: python train_native_cats.py <pair>"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, labels, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402
from models.catboost.lib.cv import run_folds  # noqa: E402

FEATS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
         'MonthlyCharges', 'TotalCharges']
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATS = [c for c in FEATS if c not in NUMS]
FEATS2 = FEATS + ['TotalCharges_isna']
NUMS2 = NUMS + ['TotalCharges_isna']
COLS_ALL = NUMS2 + CATS
BIN_COLS = [f'bin_{c}' for c in NUMS2]
STAT_FEATURES = ([f'freq_{c}' for c in COLS_ALL] + [f'tmean_{c}' for c in COLS_ALL]
                 + [f'tcount_{c}' for c in COLS_ALL] + BIN_COLS)

RECIPES = {
    'catboost_native_cats_balanced': dict(fold_stats=False, test_folds=15, test_seed=42),
    'catboost_native_cats_fold_stats_balanced': dict(fold_stats=True, test_folds=15, test_seed=42),
}


def cb_params(seed):
    return dict(
        loss_function='Logloss',
        eval_metric='Logloss',
        custom_metric=['AUC'],
        iterations=20000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3.0,
        random_strength=0.5,
        bootstrap_type='Bayesian',
        bagging_temperature=1.0,
        border_count=254,
        auto_class_weights='Balanced',
        early_stopping_rounds=500,
        task_type='GPU',
        random_seed=seed,
        verbose=200,
        allow_writing_files=False,
    )


def fit_stats(df_tr, y_tr):
    """Frequency, target mean / count per value and quantile-bin edges of the fold's training rows."""
    fe = {'global_mean': float(np.mean(y_tr))}
    fe['freq'] = {c: df_tr[c].value_counts(normalize=True, dropna=False).to_dict() for c in COLS_ALL}
    stats_mean, stats_count = {}, {}
    y_ser = pd.Series(y_tr, name='y')
    for c in COLS_ALL:
        g = pd.DataFrame({'x': df_tr[c], 'y': y_ser}).groupby('x')['y']
        stats_mean[c] = g.mean().to_dict()
        stats_count[c] = g.count().to_dict()
    fe['tmean'] = stats_mean
    fe['tcount'] = stats_count
    kbd = KBinsDiscretizer(n_bins=20, strategy='quantile', encode='ordinal')
    kbd.fit(df_tr[NUMS2])
    fe['kbd'] = kbd
    return fe


def add_stats(df, fe):
    out = df.copy()
    for c in COLS_ALL:
        out[f'freq_{c}'] = out[c].map(fe['freq'][c]).fillna(0).astype(np.float32)
    gm = fe['global_mean']
    for c in COLS_ALL:
        out[f'tmean_{c}'] = out[c].map(fe['tmean'][c]).fillna(gm).astype(np.float32)
        out[f'tcount_{c}'] = out[c].map(fe['tcount'][c]).fillna(0).astype(np.float32)
    out[BIN_COLS] = fe['kbd'].transform(out[NUMS2]).astype(np.int32)
    for b in BIN_COLS:
        out[b] = out[b].astype(str)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=list(RECIPES))
    pair = ap.parse_args().pair
    cfg = RECIPES[pair]
    fold_stats = cfg['fold_stats']

    train, test = read_train(), read_test()
    y = labels(train[TARGET]).astype(np.int32)
    for df in (train, test):
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges_isna'] = df['TotalCharges'].isna().astype(np.int8)
        for c in NUMS:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        for c in CATS:
            df[c] = df[c].astype('object').fillna('__NA__').astype(str)
    X_all = train[FEATS2].copy()
    X_test_full = test[FEATS2].copy()
    features = FEATS2 + (STAT_FEATURES if fold_stats else [])
    cat_features = CATS + (BIN_COLS if fold_stats else [])
    print(f'features: {len(features)} ({len(cat_features)} categorical)', flush=True)

    def fit_fold(fold, tr_idx, va_idx, seed):
        X_tr = X_all.iloc[tr_idx].copy()
        X_va = X_all.iloc[va_idx].copy()
        y_tr, y_va = y[tr_idx], y[va_idx]
        med = X_tr[NUMS2].median(numeric_only=True)
        X_tr[NUMS2] = X_tr[NUMS2].fillna(med)
        X_va[NUMS2] = X_va[NUMS2].fillna(med)
        X_te = X_test_full.copy()
        X_te[NUMS2] = X_te[NUMS2].fillna(med)
        if fold_stats:
            fe = fit_stats(X_tr, y_tr)
            X_tr, X_va, X_te = (add_stats(df, fe)[features] for df in (X_tr, X_va, X_te))

        model = CatBoostClassifier(**cb_params(seed))
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_features, use_best_model=True)
        p_va = model.predict_proba(X_va)[:, 1].astype(np.float32)
        p_te = model.predict_proba(X_te)[:, 1].astype(np.float32)
        del X_tr, X_va, X_te, model
        gc.collect()
        return p_va, p_te

    oof, _ = run_folds(fit_fold, y, len(test), name=pair)
    _, test_pred = run_folds(fit_fold, y, len(test), n_folds=cfg['test_folds'], seed=cfg['test_seed'], name=pair)
    print(f'{pair}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(HERE, pair, oof, test_pred)


if __name__ == '__main__':
    main()
