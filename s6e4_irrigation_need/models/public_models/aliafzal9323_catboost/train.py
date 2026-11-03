#!/usr/bin/env python
"""Public notebook "s6e4-0-978-xgb-cat-pairwise-te-magic" (its CatBoost) in a nested 5 x 5 cross-validation on our
folds.
https://www.kaggle.com/code/aliafzal9323/s6e4-0-978-xgb-cat-pairwise-te-magic

Features: the 8 categoricals factorized over train + test + original rows (CatBoost categorical features) and the 11
numerics; TE_ORIG_<col> for all 19 columns: the mean of the 0/1/2-coded target over the original rows with that value,
0.5 where there is none (the categoricals are integer codes when they are mapped, so their 8 TE_ORIG columns are the
constant 0.5); per inner fold, a multiclass TargetEncoder (cv=5, seed 42) of every pairwise combination of the 19
columns that has at most half as many distinct values as there are train + test + original rows (3 columns per pair),
fitted on the inner training rows. The original rows are only used for TE_ORIG.
Nested CV: for outer fold k of our 5 folds, StratifiedKFold(5, shuffle=True, random_state=1001 + k) splits the outer
training rows; each inner model early-stops on its inner validation rows. Slice k of the OOF holds the inner
out-of-fold predictions for the outer training rows and the mean of the 5 inner models for the outer held-out rows;
test = mean of the 25 inner models.
Model: CatBoost (GPU), up to 10,000 iterations, lr 0.03, depth 6, balanced class weights, early stopping 500 on
TotalF1:average=Macro, seed 42 + inner fold (1-5).
Usage: python train.py [--folds 0,1,2,3,4]   (outer folds)"""
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import TARGET, TARGET_MAP, read_original, read_test, read_train  # noqa: E402
from lib.runner import fold_args, run_nested_folds, work_dir, working_directory  # noqa: E402

warnings.filterwarnings('ignore')

FOLDER = 'aliafzal9323_catboost'
SEED = 42
N_INNER_SPLITS = 5

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
        'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']
BASE_FEATURES = CATS + NUMS + [f'TE_ORIG_{c}' for c in CATS + NUMS]

CB_PARAMS = {
    'task_type': 'GPU',
    'iterations': 10000,
    'learning_rate': 0.03,
    'depth': 6,
    'auto_class_weights': 'Balanced',
    'cat_features': CATS,
    'random_seed': SEED,
    'eval_metric': 'TotalF1:average=Macro',
    'early_stopping_rounds': 500,
    'verbose': 0,
}


def build_features():
    train = read_train().dropna(subset=[TARGET])
    test = read_test()
    orig = read_original()
    train[TARGET] = train[TARGET].map(TARGET_MAP)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP)
    y = train[TARGET].values

    combined = pd.concat([train, test, orig], axis=0, ignore_index=True)
    for c in CATS:
        combined[c], _ = pd.factorize(combined[c].astype(str))

    # target means of the original data (the categoricals are already integer codes here, so they map to the 0.5 fill)
    for c in CATS + NUMS:
        te_map = orig.groupby(c)[TARGET].mean()
        combined[f'TE_ORIG_{c}'] = combined[c].map(te_map).fillna(0.5)

    # pairwise combinations, dropped when more than half of the train + test + original rows are distinct
    interaction_cols = []
    for c1, c2 in combinations(NUMS + CATS, 2):
        col_name = f'{c1}_x_{c2}'
        combined[col_name] = combined[c1].astype(str) + '_' + combined[c2].astype(str)
        combined[col_name], _ = pd.factorize(combined[col_name])
        if combined[col_name].nunique() <= len(combined) // 2:
            interaction_cols.append(col_name)
        else:
            combined.drop(columns=[col_name], inplace=True)
    print(f'pair combinations kept: {len(interaction_cols)}', flush=True)

    train_processed = combined.iloc[:len(train)].copy()
    test_processed = combined.iloc[len(train):len(train) + len(test)].copy()
    return train_processed, y, test_processed, interaction_cols


def prepare_df(df_base, te_data, te_col_names):
    df_final = pd.concat([df_base[BASE_FEATURES].reset_index(drop=True),
                          pd.DataFrame(te_data, columns=te_col_names)], axis=1)
    for c in CATS:
        df_final[c] = df_final[c].astype('category')
    return df_final


def main():
    args = fold_args()
    train_processed, y, test_processed, interaction_cols = build_features()
    n_train, n_test = len(train_processed), len(test_processed)

    def fit_outer_fold(fold, outer_tr_idx, outer_va_idx):
        outer_fold = fold + 1
        x_outer_tr, y_outer_tr = train_processed.iloc[outer_tr_idx].copy(), y[outer_tr_idx].copy()
        x_outer_va, y_outer_va = train_processed.iloc[outer_va_idx].copy(), y[outer_va_idx].copy()
        outer_full_pred = np.zeros((n_train, 3), dtype=np.float32)
        outer_va_pred_sum = np.zeros((len(outer_va_idx), 3), dtype=np.float32)
        test_pred = np.zeros((n_test, 3), dtype=np.float32)

        skf_inner = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=1000 + outer_fold)
        for inner_fold, (inner_tr_idx, inner_va_idx) in enumerate(skf_inner.split(x_outer_tr, y_outer_tr), start=1):
            x_tr, y_tr = x_outer_tr.iloc[inner_tr_idx].copy(), y_outer_tr[inner_tr_idx].copy()
            x_va, y_va = x_outer_tr.iloc[inner_va_idx].copy(), y_outer_tr[inner_va_idx].copy()

            # target encoding of the pair combinations, fitted on the inner training rows
            te = TargetEncoder(target_type='multiclass', cv=5, random_state=SEED)
            te_tr = te.fit_transform(x_tr[interaction_cols], y_tr)
            te_va = te.transform(x_va[interaction_cols])
            te_outer_va = te.transform(x_outer_va[interaction_cols])
            te_te = te.transform(test_processed[interaction_cols])
            te_col_names = [f'TE_{i}' for i in range(te_tr.shape[1])]

            x_tr_final = prepare_df(x_tr, te_tr, te_col_names)
            x_va_final = prepare_df(x_va, te_va, te_col_names)
            x_outer_va_final = prepare_df(x_outer_va, te_outer_va, te_col_names)
            x_te_final = prepare_df(test_processed, te_te, te_col_names)

            with working_directory(work_dir(FOLDER)):
                model = cb.CatBoostClassifier(**dict(CB_PARAMS, random_seed=SEED + inner_fold))
                model.fit(x_tr_final, y_tr, eval_set=(x_va_final, y_va))

            p_inner_va = model.predict_proba(x_va_final).astype(np.float32)
            outer_full_pred[outer_tr_idx[inner_va_idx]] = p_inner_va
            print(f'outer {outer_fold} inner {inner_fold}: balanced accuracy '
                  f'{balanced_accuracy_score(y_va, p_inner_va.argmax(1)):.5f}', flush=True)
            outer_va_pred_sum += model.predict_proba(x_outer_va_final).astype(np.float32)
            test_pred += model.predict_proba(x_te_final).astype(np.float32) / N_INNER_SPLITS
            del x_tr, y_tr, x_va, y_va, x_tr_final, x_va_final, x_outer_va_final, x_te_final, model
            gc.collect()

        outer_full_pred[outer_va_idx] = outer_va_pred_sum / N_INNER_SPLITS
        return outer_full_pred, test_pred

    run_nested_folds(FOLDER, fit_outer_fold, args.folds, y=y)


if __name__ == '__main__':
    main()
