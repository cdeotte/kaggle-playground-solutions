#!/usr/bin/env python
"""Selects the original-data KNN schemes (8 random + 8 smart) and their blend weights for
train_knn_nested_xgb.py xgb_nested_knn_k1_k80_top_schemes. Run it first.

Original-data KNN priors of the train rows (lib/knn_schemes.py; reference = the 7,043 original rows, numerics
standardized on them, k = 80, numerics in fp32): 30 random schemes (mean label of the 80 nearest original rows) and 30
smart schemes (inverse-distance weighted mean label, times train churn rate / original churn rate).
Per family: 5-fold XGBoost (the shared folds; GPU, lr 0.03, depth 4, min_child_weight 8, subsample 0.8,
colsample_bytree 0.35, alpha 1, lambda 8, native categoricals with max_cat_to_onehot 8, up to 3,000 rounds with early
stopping 150 (AUC) on the fold's validation rows, seed 42) on the 19 raw columns + the family's 30 prior columns. The
gain importance of every prior column is averaged over the 5 fold models; the 8 columns with the largest mean gain of
each family are kept, and their mean gains become the blend weights.
Writes cache/knn_scheme_selection.json (the kept columns) and cache/knn_scheme_importance_{random,smart}.csv
(per column: mean and std of the gain over the folds, number of folds with a non-zero gain).
Usage: python select_knn_schemes.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import TARGET, cache_dir, folds  # noqa: E402
from lib.knn_schemes import (CATEGORICAL_COLS, N_RANDOM_SCHEMES, N_SMART_SCHEMES, NUMERIC_COLS, SEED,  # noqa: E402
                             base_matrix, build_random_weight_schemes, build_smart_weight_schemes,
                             build_target_encoding_maps, cat_codes, fit_orig_importance_model, knn_random_full,
                             knn_smart_full, load_frames, log, make_target_rate_matrix, prefixed_colnames,
                             seed_everything, standardize_from_reference)

FOLDER = 'classic'
N_SPLITS = 5
N_NEIGHBORS = 80
GPU_BATCH_SIZE = 4096
TOPK_RANDOM = 8
TOPK_SMART = 8
IMPORTANCE_TYPE = 'gain'

XGB_SELECT_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=8,
    subsample=0.80,
    colsample_bytree=0.35,
    reg_alpha=1.0,
    reg_lambda=8.0,
    gamma=0.0,
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    device='cuda',
    enable_categorical=True,
    max_cat_to_onehot=8,
    random_state=SEED,
    early_stopping_rounds=150,
    verbosity=0,
)


def family_importance_cv(family_name, train_base, y, family_df, fold_id):
    """5-fold XGBoost on the base columns + one prior family; per-fold gain of the family's columns (rows = folds)."""
    family_cols = family_df.columns.tolist()
    oof = np.zeros(len(train_base), dtype=np.float32)
    importance_rows = []
    for k in range(N_SPLITS):
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        X_tr = train_base.iloc[tr_idx].copy()
        X_va = train_base.iloc[va_idx].copy()
        for c in family_cols:
            X_tr[c] = family_df.iloc[tr_idx][c].astype(np.float32).values
            X_va[c] = family_df.iloc[va_idx][c].astype(np.float32).values

        model = XGBClassifier(**XGB_SELECT_PARAMS)
        model.fit(X_tr, y[tr_idx], eval_set=[(X_va, y[va_idx])], verbose=200)
        oof[va_idx] = model.predict_proba(X_va)[:, 1].astype(np.float32)
        log(f'{family_name} fold {k}: AUC {roc_auc_score(y[va_idx], oof[va_idx]):.6f}')

        score = model.get_booster().get_score(importance_type=IMPORTANCE_TYPE)
        imp = pd.Series(0.0, index=X_tr.columns.tolist(), dtype=np.float64)
        for name, v in score.items():
            if name in imp.index:
                imp.loc[name] = float(v)
        imp_family = imp.reindex(family_cols).fillna(0.0)
        imp_family.name = f'fold_{k + 1}'
        importance_rows.append(imp_family)
    log(f'{family_name}: OOF AUC {roc_auc_score(y, oof):.6f}')
    return pd.DataFrame(importance_rows)


def summarize_importance(importance_df):
    out = pd.DataFrame({
        'feature': importance_df.columns,
        'mean_importance': importance_df.mean(axis=0).values,
        'std_importance': importance_df.std(axis=0).values,
        'nonzero_folds': (importance_df > 0).sum(axis=0).values,
    })
    return out.sort_values(['mean_importance', 'nonzero_folds', 'std_importance'],
                           ascending=[False, False, True]).reset_index(drop=True)


def main():
    seed_everything(SEED)
    out_dir = cache_dir(FOLDER)
    train, test, orig = load_frames()
    y = train[TARGET].values.astype(np.int8)

    orig_cat_codes, train_cat_codes = cat_codes(orig), cat_codes(train)
    orig_num = orig[NUMERIC_COLS].values.astype(np.float32)
    train_num = train[NUMERIC_COLS].values.astype(np.float32)
    orig_num_scaled, num_mu, num_sd = standardize_from_reference(orig_num, orig_num)
    train_num_scaled = ((train_num - num_mu) / num_sd).astype(np.float32)
    orig_y = orig[TARGET].values.astype(np.float32)

    log('original-data KNN priors, random schemes')
    random_schemes = build_random_weight_schemes(N_RANDOM_SCHEMES, n_cat=len(CATEGORICAL_COLS), n_num=len(NUMERIC_COLS),
                                                 seed=SEED)
    knn_random_train = knn_random_full(orig_cat_codes, orig_num_scaled, orig_y, train_cat_codes, train_num_scaled,
                                       random_schemes, prefixed_colnames('knnprior', range(N_RANDOM_SCHEMES)),
                                       N_NEIGHBORS, batch_size=GPU_BATCH_SIZE, fp16=False)

    log('original-data KNN priors, smart schemes')
    te_maps, te_global_mean = build_target_encoding_maps(orig)
    orig_cat_te = make_target_rate_matrix(orig, te_maps, te_global_mean)
    train_cat_te = make_target_rate_matrix(train, te_maps, te_global_mean)
    orig_importance = fit_orig_importance_model(orig)
    smart_schemes = build_smart_weight_schemes(orig_importance[CATEGORICAL_COLS].values.astype(np.float32),
                                               orig_importance[NUMERIC_COLS].values.astype(np.float32),
                                               N_SMART_SCHEMES, seed=SEED)
    knn_smart_train = knn_smart_full(orig_cat_te, orig_num_scaled, orig_y, train_cat_te, train_num_scaled,
                                     float(train[TARGET].mean()), float(orig[TARGET].mean()), smart_schemes,
                                     prefixed_colnames('knnsmart', range(N_SMART_SCHEMES)), N_NEIGHBORS,
                                     batch_size=GPU_BATCH_SIZE, fp16=False)
    del test

    train_base = base_matrix(train)
    fold_id = folds(y)
    selected = {}
    for family, family_df, topk in (('random', knn_random_train, TOPK_RANDOM), ('smart', knn_smart_train, TOPK_SMART)):
        summary = summarize_importance(family_importance_cv(f'{family} family', train_base, y, family_df, fold_id))
        summary.to_csv(out_dir / f'knn_scheme_importance_{family}.csv', index=False)
        selected[f'selected_{family}_cols'] = summary.head(topk)['feature'].tolist()
        log(f'{family} family: kept {selected[f"selected_{family}_cols"]}')

    with open(out_dir / 'knn_scheme_selection.json', 'w') as f:
        json.dump(dict(importance_type=IMPORTANCE_TYPE, **selected), f, indent=2)
    log(f'wrote {out_dir}')


if __name__ == '__main__':
    main()
