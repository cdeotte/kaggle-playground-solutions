#!/usr/bin/env python
"""GPU XGBoost with native categoricals and digit-position features (pair xgb_native_cat_digit_positions_b).

Features (71, label-free, built on train + test):
- the 8 categoricals (XGBoost native categoricals) and the 11 numerics (float32);
- digit positions of every numeric: the integer digits (ones, tens, hundreds, ... up to the largest value) and the
  first 2 decimal digits (44 columns);
- the rule formula: score = 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10]
  - 2 [Crop_Growth_Stage = Harvest] - 2 [Crop_Growth_Stage = Sowing] - [Mulching_Used = Yes], its 4 threshold flags and
  the one-hot formula class (score <= 0, 0 < score <= 3, score > 3).
Model: XGBoost multi:softprob, depth 3, lr 0.02, subsample / colsample_bytree 0.8, max_bin 16,000, up to 20,000 rounds,
early stopping 200 (mlogloss) on the validation fold, seed 42, no class weights. Competition rows only. Test = mean of
the 5 fold models.
Usage: python train_digit_positions_b.py [--folds 0,1,2,3,4] [--force]   (finished folds are kept in parts/)"""
import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, labels, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.simplefilter('ignore')

PAIR = 'xgb_native_cat_digit_positions_b'
SEED = 42
N_CLASSES = 3

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
        'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']

XGB_PARAMS = dict(
    n_estimators=20000,
    learning_rate=0.02,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    max_bin=16000,
    objective='multi:softprob',
    num_class=N_CLASSES,
    eval_metric='mlogloss',
    n_jobs=-1,
    random_state=SEED,
    early_stopping_rounds=200,
    device='cuda',
    enable_categorical=True,
)


def add_digit_features(df, base_cols, max_dec_digits=2):
    """Integer digit positions (ones, tens, hundreds, ...) + decimal digits of each numeric column."""
    digit_cols = []
    for col in base_cols:
        col_vals = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        max_val = float(col_vals.abs().max())
        max_int_digits = max(1, int(np.log10(max(max_val, 1))) + 1)
        for i in range(max_int_digits):
            n = f'{col}_digit_pos_{i + 1}'
            df[n] = ((col_vals // (10 ** i)) % 10).astype('int8')
            digit_cols.append(n)
        for i in range(1, max_dec_digits + 1):
            n = f'{col}_digit_dec_{i}'
            df[n] = ((col_vals * (10 ** i)).round().astype('int64') % 10).astype('int8')
            digit_cols.append(n)
    return df, digit_cols


def build_features():
    train = read_train().set_index(ID)
    test = read_test().set_index(ID)
    train[TARGET] = train[TARGET].map(TARGET_MAP).astype('int8')
    y = train[TARGET].values.astype('int32')
    for df in (train, test):
        for c in NUMS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

    test_copy = test.copy()
    test_copy[TARGET] = -1
    combined = pd.concat([train, test_copy], axis=0, ignore_index=True)
    for c in CATS:
        combined[c] = combined[c].astype(str).astype('category')

    combined, digit = add_digit_features(combined, NUMS, max_dec_digits=2)
    print(f'{len(digit)} digit features', flush=True)

    sm = combined['Soil_Moisture'].values.astype('float64')
    rf = combined['Rainfall_mm'].values.astype('float64')
    tc = combined['Temperature_C'].values.astype('float64')
    wnd = combined['Wind_Speed_kmh'].values.astype('float64')
    cgs = combined['Crop_Growth_Stage'].values
    mul = combined['Mulching_Used'].values
    score = ((sm < 25) * 2 + (rf < 300) * 2 + (tc > 30) + (wnd > 10)
             - (cgs == 'Harvest') * 2 - (cgs == 'Sowing') * 2 - (mul == 'Yes'))
    combined['formula_score'] = score.astype('float32')
    combined['sm_lt25'] = (sm < 25).astype('float32')
    combined['rf_lt300'] = (rf < 300).astype('float32')
    combined['tc_gt30'] = (tc > 30).astype('float32')
    combined['wnd_gt10'] = (wnd > 10).astype('float32')
    combined['formula_low'] = (score <= 0).astype('float32')
    combined['formula_medium'] = ((score > 0) & (score <= 3)).astype('float32')
    combined['formula_high'] = (score > 3).astype('float32')
    formula_cols = ['formula_score', 'sm_lt25', 'rf_lt300', 'tc_gt30', 'wnd_gt10',
                    'formula_low', 'formula_medium', 'formula_high']

    features = CATS + NUMS + digit + formula_cols
    print(f'Total features: {len(features)}', flush=True)
    train2 = combined.iloc[:len(train)].reset_index(drop=True)
    test2 = combined.iloc[len(train):].reset_index(drop=True)
    return train2[features], test2[features], y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--force', action='store_true', help='rerun folds that are already done')
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        X, X_test, y = build_features()
        fold_id = folds(y)
        for k in todo:
            tr_idx, va_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X.iloc[va_idx], y[va_idx]
            model = XGBClassifier(**XGB_PARAMS)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=500)
            p_va = model.predict_proba(X_va).astype('float32')
            p_te = model.predict_proba(X_test).astype('float32')
            save_part(HERE, PAIR, k, p_va, p_te)
            acc = balanced_accuracy_score(y_va, np.argmax(p_va, axis=1))
            print(f'{PAIR} fold {k}: balanced accuracy {acc:.5f}, best iteration '
                  f'{getattr(model, "best_iteration", None)}', flush=True)
            del model, X_tr, X_va, y_tr, y_va
            gc.collect()
    y = labels(read_train()[TARGET])
    assemble(HERE, PAIR, folds(y), y)


if __name__ == '__main__':
    main()
