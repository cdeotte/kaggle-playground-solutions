#!/usr/bin/env python
"""GPU XGBoost on target statistics of the numerics read as categories (pair xgb_nums_as_cats_te_stats_b).

Label-free features (built on train, test and the original rows):
- the 11 numerics (float32);
- the rule formula score = 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10]
  - 2 [Crop_Growth_Stage = Harvest] - 2 [Crop_Growth_Stage = Sowing] - [Mulching_Used = Yes], is_low, is_high, the 4
  threshold flags, Soil_Moisture / (Rainfall_mm + 1), Temperature_C x Wind_Speed_kmh, Soil_Moisture - 25,
  Rainfall_mm - 300;
- the frequency of every numeric value over train + test + original;
- per-class original-data priors of all 19 columns (57 columns).
Per fold, 19 keys (the 8 categoricals and the 11 numerics as strings rounded to 4 decimals) get
- std / min / max of the label (0, 1, 2) per key, out-of-fold over an inner StratifiedKFold(5, seed 42) of the training
  rows (fold-train statistics for the validation / test rows);
- 3 binary target encodings (one per class, sklearn TargetEncoder with its internal 5-fold cross fitting, seed 42).
The raw keys are dropped afterwards (204 columns).
Model: XGBoost multi:softprob, depth 6, lr 0.03, subsample / colsample_bytree 0.8, min_child_weight 5, alpha 0.1,
lambda 1, gamma 0.05, balanced class weights, up to 50,000 rounds, early stopping 500 (mlogloss) on the validation fold,
seed 42. Competition rows only. Test = mean of the 5 fold models. A fold takes about 2 minutes on one GPU.
Usage: python train_nums_as_cats_te_b.py [--folds 0,1,2,3,4] [--force]   (finished folds are kept in parts/)"""
import argparse
import gc
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'xgb_nums_as_cats_te_stats_b'
INNER_FOLDS = 5
N_CLASSES = 3
SEED = 42
STATS = ['std', 'min', 'max']

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Organic_Carbon', 'Electrical_Conductivity', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Temperature_C',
        'Soil_Moisture', 'Humidity', 'Previous_Irrigation_mm', 'Rainfall_mm']

XGB_PARAMS = {
    'n_estimators': 50000,
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'gamma': 0.05,
    'early_stopping_rounds': 500,
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'enable_categorical': True,
    'device': 'cuda',
    'tree_method': 'hist',
    'verbosity': 0,
    'random_state': SEED,
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def build_features(train, test, orig, y_orig):
    new_nums, num_as_cat = [], []
    for df in (train, test, orig):
        for c in CATS:
            df[c] = df[c].astype(str).fillna('missing').str.strip()
        for c in NUMS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

    # formula features
    for df in (train, test, orig):
        SM = df['Soil_Moisture']; RM = df['Rainfall_mm']
        TC = df['Temperature_C']; WS = df['Wind_Speed_kmh']
        CGS = df['Crop_Growth_Stage']; MU = df['Mulching_Used']
        score = ((SM < 25).astype(int) * 2 + (RM < 300).astype(int) * 2 +
                 (TC > 30).astype(int) + (WS > 10).astype(int) -
                 (CGS == 'Harvest').astype(int) * 2 - (CGS == 'Sowing').astype(int) * 2 -
                 (MU == 'Yes').astype(int))
        df['formula_score'] = score.astype('float32')
        df['is_low'] = (score <= 0).astype('float32')
        df['is_high'] = (score > 3).astype('float32')
        df['SM_thr'] = (SM < 25).astype('float32')
        df['RM_thr'] = (RM < 300).astype('float32')
        df['TC_thr'] = (TC > 30).astype('float32')
        df['WS_thr'] = (WS > 10).astype('float32')
        df['SM_div_RM'] = (SM / (RM + 1.0)).astype('float32')
        df['TC_x_WS'] = (TC * WS).astype('float32')
        df['SM_minus_25'] = (SM - 25.0).astype('float32')
        df['RM_minus_300'] = (RM - 300.0).astype('float32')
    new_nums += ['formula_score', 'is_low', 'is_high', 'SM_thr', 'RM_thr',
                 'TC_thr', 'WS_thr', 'SM_div_RM', 'TC_x_WS', 'SM_minus_25', 'RM_minus_300']

    # frequency encoding of the numerics
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for df in (train, test, orig):
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        new_nums.append(f'FREQ_{col}')

    # numerics as categorical strings
    for col in NUMS:
        nc = f'CAT_{col}'
        num_as_cat.append(nc)
        for df in (train, test, orig):
            df[nc] = df[col].round(4).astype(str)

    # per-class original priors of all 19 columns
    for col in CATS + NUMS:
        for k in range(N_CLASSES):
            pm = orig.groupby(col)['target'].apply(lambda g: (g == k).mean())
            gp = float((y_orig == k).mean())
            nc = f'ORIG_k{k}_{col}'
            for df in (train, test, orig):
                df[nc] = df[col].map(pm).fillna(gp).astype('float32')
            new_nums.append(nc)

    seen, features = set(), []
    for f in NUMS + CATS + new_nums + num_as_cat:
        if f not in seen:
            seen.add(f)
            features.append(f)
    te_columns = num_as_cat + CATS
    to_remove = num_as_cat + CATS
    log(f'Features before target encoding: {len(features)} | keys: {len(te_columns)}')
    return train, test, orig, features, te_columns, to_remove, num_as_cat


def add_nested_te_multiclass(X_tr, y_tr, X_val, X_te, te_columns, to_remove, num_as_cat):
    """Inner 5-fold label std / min / max per key + 3 per-class sklearn target encodings."""
    X_tr = X_tr.copy(); X_val = X_val.copy(); X_te = X_te.copy()
    skf_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    X_tr['_y'] = y_tr.astype('float32')

    for col in te_columns:
        for s in STATS:
            X_tr[f'TE1_{col}_{s}'] = np.float32(0)
    # training rows: statistics of the other inner folds
    for in_tr_idx, in_va_idx in skf_inner.split(X_tr, y_tr):
        for col in te_columns:
            tmp = X_tr.iloc[in_tr_idx].groupby(col, observed=False)['_y'].agg(STATS)
            tmp.columns = [f'TE1_{col}_{s}' for s in STATS]
            for s in STATS:
                nc = f'TE1_{col}_{s}'
                mapped = X_tr.iloc[in_va_idx][col].map(tmp[nc]).fillna(0).astype('float32').values
                X_tr.iloc[in_va_idx, X_tr.columns.get_loc(nc)] = mapped
    # validation / test rows: statistics of all fold-train rows
    for col in te_columns:
        tmp = X_tr.groupby(col, observed=False)['_y'].agg(STATS)
        tmp.columns = [f'TE1_{col}_{s}' for s in STATS]
        for s in STATS:
            nc = f'TE1_{col}_{s}'
            fv = float(X_tr['_y'].agg(s))
            X_val[nc] = X_val[col].map(tmp[nc]).fillna(fv).astype('float32')
            X_te[nc] = X_te[col].map(tmp[nc]).fillna(fv).astype('float32')
            X_tr[nc] = X_tr[nc].fillna(fv).astype('float32')
    X_tr.drop(columns=['_y'], inplace=True)

    for k in range(N_CLASSES):
        y_bin = (y_tr == k).astype(int)
        te = TargetEncoder(cv=INNER_FOLDS, shuffle=True, smooth='auto', target_type='binary', random_state=SEED)
        tc_k = [f'TE_k{k}_{col}' for col in te_columns]
        X_tr[tc_k] = te.fit_transform(X_tr[te_columns], y_bin).astype('float32')
        X_val[tc_k] = te.transform(X_val[te_columns]).astype('float32')
        X_te[tc_k] = te.transform(X_te[te_columns]).astype('float32')

    for df_ in (X_tr, X_val, X_te):
        for c in CATS + num_as_cat:
            if c in df_.columns:
                df_[c] = df_[c].astype(str).astype('category')
        df_.drop(columns=[c for c in to_remove if c in df_.columns], inplace=True)
    cols = list(X_tr.columns)
    return X_tr, X_val[cols].copy(), X_te[cols].copy(), cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--force', action='store_true', help='rerun folds that are already done')
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        train, test, orig = read_train(), read_test(), read_original()
        y_train = labels(train[TARGET])
        y_orig = labels(orig[TARGET])
        orig['target'] = y_orig
        train = train.drop(columns=[TARGET, ID], errors='ignore')
        test = test.drop(columns=[ID], errors='ignore')
        cls_counts = np.bincount(y_train)
        cls_weights_np = cls_counts.sum() / (N_CLASSES * cls_counts)
        train, test, orig, features, te_columns, to_remove, num_as_cat = build_features(train, test, orig, y_orig)
        fold_id = folds(y_train)
        for k in todo:
            t_fold = time.time()
            train_idx, val_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            X_tr = train.iloc[train_idx][features].reset_index(drop=True).copy()
            y_tr = y_train[train_idx].copy()
            X_val = train.iloc[val_idx][features].reset_index(drop=True).copy()
            y_val = y_train[val_idx].copy()
            X_te = test[features].reset_index(drop=True).copy()
            X_tr, X_val, X_te, cols = add_nested_te_multiclass(X_tr, y_tr, X_val, X_te, te_columns, to_remove,
                                                               num_as_cat)
            sw_tr = np.array([cls_weights_np[c] for c in y_tr], dtype=np.float32)
            model = xgb.XGBClassifier(**XGB_PARAMS)
            model.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_val, y_val)], verbose=500)
            val_probs = model.predict_proba(X_val)
            test_probs = model.predict_proba(X_te[cols])
            save_part(HERE, PAIR, k, val_probs, test_probs)
            bac = balanced_accuracy_score(y_val, val_probs.argmax(1))
            log(f'{PAIR} fold {k}: balanced accuracy {bac:.6f}, best iteration {model.best_iteration} '
                f'({time.time() - t_fold:.0f}s)')
            del X_tr, X_val, X_te, model
            gc.collect()
    y = labels(read_train()[TARGET])
    assemble(HERE, PAIR, folds(y), y)


if __name__ == '__main__':
    main()
