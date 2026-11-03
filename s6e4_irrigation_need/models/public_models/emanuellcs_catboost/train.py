#!/usr/bin/env python
"""Public notebook "Predicting Irrigation Need | XGBoost + CatBoost" (its CatBoost) in a nested 5 x 5 cross-validation
on our folds.
https://www.kaggle.com/code/emanuellcs/predicting-irrigation-need-xgboost-catboost

Features: the 19 raw columns; TE_ORIG_<col> for each of them (mean of the 0/1/2-coded target over the original rows
with that value, 0.5 for values the original data does not have); every pairwise string combination of the 19 columns
that has at most half as many distinct values as there are train + test + original rows. The 8 categoricals and the
pair combinations are CatBoost categorical features.
Nested CV: for outer fold k of our 5 folds, StratifiedKFold(5, shuffle=True, random_state=1001 + k) splits the outer
training rows; each inner model trains on its inner training rows + all 10,000 original rows and early-stops on its
inner validation rows. Slice k of the OOF holds the inner out-of-fold predictions for the outer training rows and the
mean of the 5 inner models for the outer held-out rows; test = mean of the 25 inner models.
Model: CatBoost (GPU) MultiClass, up to 15,000 iterations, lr 0.02, depth 6, balanced class weights, early stopping
500 on the MultiClass loss, seed 42 + inner fold (1-5).
Usage: python train.py [--folds 0,1,2,3,4]   (outer folds)"""
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, TARGET_MAP, read_original, read_test, read_train  # noqa: E402
from lib.runner import fold_args, run_nested_folds, work_dir, working_directory  # noqa: E402

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter('ignore')

FOLDER = 'emanuellcs_catboost'
N_INNER_SPLITS = 5
RANDOM_STATE = 42

CB_PARAMS = {
    'iterations': 15000,
    'learning_rate': 0.02,
    'depth': 6,
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'auto_class_weights': 'Balanced',
    'task_type': 'GPU',
    'random_seed': RANDOM_STATE,
    'early_stopping_rounds': 500,
    'verbose': False,
}


def build_features():
    train, test, orig = read_train(), read_test(), read_original()
    if ID in orig.columns:
        orig = orig.drop(columns=[ID])
    train[TARGET] = train[TARGET].map(TARGET_MAP)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP)

    features = [col for col in test.columns if col != ID]
    base_cat_features = test.select_dtypes(include=['object']).columns.tolist()

    # target means of the original data per value of every column
    for c in features:
        tmp = orig.groupby(c)[TARGET].mean().astype('float32')
        tmp.name = f'TE_ORIG_{c}'
        train = train.merge(tmp, on=c, how='left')
        test = test.merge(tmp, on=c, how='left')
        orig = orig.merge(tmp, on=c, how='left')
        train[tmp.name] = train[tmp.name].fillna(0.5)
        test[tmp.name] = test[tmp.name].fillna(0.5)
        orig[tmp.name] = orig[tmp.name].fillna(0.5)

    # 2-way string combinations, dropped when more than half of the train + test + original rows are distinct
    pair_columns = []
    for cols in combinations(features, 2):
        name = f'{cols[0]}-{cols[1]}'
        train[name] = train[cols[0]].astype(str) + '_' + train[cols[1]].astype(str)
        test[name] = test[cols[0]].astype(str) + '_' + test[cols[1]].astype(str)
        orig[name] = orig[cols[0]].astype(str) + '_' + orig[cols[1]].astype(str)
        combined, _ = pd.concat([train[name], test[name], orig[name]], ignore_index=True).factorize()
        if pd.Series(combined).nunique() > len(combined) // 2:
            train, test, orig = train.drop(name, axis=1), test.drop(name, axis=1), orig.drop(name, axis=1)
            continue
        pair_columns.append(name)
    print(f'pair combinations kept: {len(pair_columns)}', flush=True)

    final_features = [c for c in train.columns if c != TARGET and c != ID]
    x, y, x_test = train[final_features].copy(), train[TARGET].copy(), test[final_features].copy()
    x_orig, y_orig = orig[final_features].copy(), orig[TARGET].copy()
    all_cat_features = base_cat_features + pair_columns
    for df in [x, x_test, x_orig]:
        for col in all_cat_features:
            df[col] = df[col].astype(str)
    cat_idx = [x.columns.get_loc(c) for c in all_cat_features]
    return x, y, x_test, x_orig, y_orig, cat_idx


def main():
    args = fold_args()
    x, y, x_test, x_orig, y_orig, cat_idx = build_features()
    n_train, n_test = len(x), len(x_test)

    def fit_outer_fold(fold, outer_tr_idx, outer_va_idx):
        outer_fold = fold + 1
        x_outer_tr, y_outer_tr = x.iloc[outer_tr_idx].copy(), y.iloc[outer_tr_idx].copy()
        x_outer_va, y_outer_va = x.iloc[outer_va_idx].copy(), y.iloc[outer_va_idx].copy()
        outer_full_pred = np.zeros((n_train, 3), dtype=np.float32)
        outer_va_pred_sum = np.zeros((len(outer_va_idx), 3), dtype=np.float32)
        test_pred = np.zeros((n_test, 3), dtype=np.float32)
        outer_va_pool = Pool(x_outer_va, y_outer_va, cat_features=cat_idx)
        test_pool = Pool(x_test, cat_features=cat_idx)

        skf_inner = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=1000 + outer_fold)
        for inner_fold, (inner_tr_loc, inner_va_loc) in enumerate(skf_inner.split(x_outer_tr, y_outer_tr), start=1):
            x_tr, y_tr = x_outer_tr.iloc[inner_tr_loc].copy(), y_outer_tr.iloc[inner_tr_loc].copy()
            x_va, y_va = x_outer_tr.iloc[inner_va_loc].copy(), y_outer_tr.iloc[inner_va_loc].copy()
            # all original rows are appended to the inner training rows only
            x_tr = pd.concat([x_tr, x_orig], axis=0).reset_index(drop=True)
            y_tr = pd.concat([y_tr, y_orig], axis=0).reset_index(drop=True)
            train_pool = Pool(x_tr, y_tr, cat_features=cat_idx)
            val_pool = Pool(x_va, y_va, cat_features=cat_idx)

            with working_directory(work_dir(FOLDER)):
                model = CatBoostClassifier(**dict(CB_PARAMS, random_seed=RANDOM_STATE + inner_fold))
                model.fit(train_pool, eval_set=val_pool)

            p_inner_va = model.predict_proba(val_pool).astype(np.float32)
            outer_full_pred[outer_tr_idx[inner_va_loc]] = p_inner_va
            print(f'outer {outer_fold} inner {inner_fold}: balanced accuracy '
                  f'{balanced_accuracy_score(y_va, p_inner_va.argmax(1)):.5f}', flush=True)
            outer_va_pred_sum += model.predict_proba(outer_va_pool).astype(np.float32)
            test_pred += model.predict_proba(test_pool).astype(np.float32) / N_INNER_SPLITS
            del x_tr, x_va, y_tr, y_va, train_pool, val_pool, model, p_inner_va
            gc.collect()

        outer_full_pred[outer_va_idx] = outer_va_pred_sum / N_INNER_SPLITS
        return outer_full_pred, test_pred

    run_nested_folds(FOLDER, fit_outer_fold, args.folds, y=y.to_numpy())


if __name__ == '__main__':
    main()
