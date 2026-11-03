#!/usr/bin/env python
"""Shallow (depth 2) GPU XGBoost on nearest-original "snap" values and radix interactions (2 pairs).

Features (built once on train + test; the original rows only provide the snap grid and the priors):
- the 11 raw numerics; snap: each value replaced by the nearest value of the original data (<col>_snap) and the
  difference value - snap (<col>_snap_diff), for all 11 numerics;
- the last integer digit and the first 2 decimal digits of the 4 formula numerics (Soil_Moisture, Rainfall_mm,
  Temperature_C, Wind_Speed_kmh);
- the rule formula score = 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10]
  - 2 [Crop_Growth_Stage = Harvest] - 2 [Crop_Growth_Stage = Sowing] - [Mulching_Used = Yes], is_low (score <= 0),
  is_high (score > 3);
- the 8 categoricals as XGBoost native categoricals;
- radix integers round(100 x snap) + 100,000 x (category code) for every (radix numeric, categorical) pair, i.e. the
  snapped value within each category;
- per-class original-data priors (class shares of the original rows with the same value) of the 8 categoricals.
Model: XGBoost multi:softprob, depth 2, lr 0.1, min_child_weight 5, max_bin 16,000, subsample / colsample_bytree 0.85,
balanced class weights, early stopping 300 (mlogloss) on the validation fold, seed 42. Competition rows only.
Test = mean of the 5 fold models.

Recipes:
  xgb_snap_radix_5num_depth2_b    radix on 5 numerics (the 4 formula numerics + Humidity): 40 radix columns, 120 in total
  xgb_snap_radix_11num_depth2_b   radix on all 11 numerics: 88 radix columns, 168 in total
A fold takes 11-14 minutes on one GPU (about 10,000 rounds).
Usage: python train_snap_radix_b.py <pair> [--folds 0,1,2,3,4] [--force]   (finished folds are kept in parts/)"""
import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

SEED = 42
RADIX = 100_000

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Organic_Carbon', 'Electrical_Conductivity', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Temperature_C', 'Soil_Moisture',
        'Humidity', 'Previous_Irrigation_mm', 'Rainfall_mm']
FORMULA_NUMS = ['Soil_Moisture', 'Rainfall_mm', 'Temperature_C', 'Wind_Speed_kmh']

RECIPES = {
    'xgb_snap_radix_5num_depth2_b': dict(radix_nums=['Soil_Moisture', 'Rainfall_mm', 'Temperature_C', 'Wind_Speed_kmh',
                                                     'Humidity']),
    'xgb_snap_radix_11num_depth2_b': dict(radix_nums=NUMS),
}

XGB_PARAMS = dict(
    n_estimators=1_000_000,
    learning_rate=0.1,
    max_depth=2,
    min_child_weight=5,
    max_bin=16_000,
    subsample=0.85,
    colsample_bytree=0.85,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    tree_method='hist',
    enable_categorical=True,
    random_state=SEED,
    early_stopping_rounds=300,
    device='cuda',
    verbosity=0,
)


def build_features(radix_nums):
    t0 = time.time()
    train, test, orig = read_train(), read_test(), read_original()
    y = labels(train[TARGET]).astype(np.int32)
    orig['target'] = labels(orig[TARGET]).astype(np.int32)
    n_train = len(train)
    combo = pd.concat([train.drop(columns=[TARGET]), test.drop(columns=[ID])], ignore_index=True)

    # snap to the nearest original value (ties go to the smaller value)
    for col in NUMS:
        orig_vals = np.sort(orig[col].dropna().unique())
        vals = combo[col].values
        idx = np.clip(np.searchsorted(orig_vals, vals, side='left'), 1, len(orig_vals) - 1)
        left = orig_vals[idx - 1]
        right = orig_vals[idx]
        snap = np.where(np.abs(vals - left) <= np.abs(vals - right), left, right)
        combo[f'{col}_snap'] = snap.astype(np.float32)
        combo[f'{col}_snap_diff'] = (vals - snap).astype(np.float32)

    # digits of 100 x value
    for col in FORMULA_NUMS:
        scaled = (combo[col].values * 100).round().astype(np.int64)
        combo[f'{col}_d0'] = ((scaled // 100) % 10).astype(np.int16)
        combo[f'{col}_d1'] = ((scaled // 10) % 10).astype(np.int16)
        combo[f'{col}_d2'] = (scaled % 10).astype(np.int16)

    # formula
    SM = combo['Soil_Moisture']; RM = combo['Rainfall_mm']; TC = combo['Temperature_C']
    WS = combo['Wind_Speed_kmh']; CGS = combo['Crop_Growth_Stage']; MU = combo['Mulching_Used']
    score = ((SM < 25) * 2 + (RM < 300) * 2 + (TC > 30) + (WS > 10)
             - (CGS == 'Harvest') * 2 - (CGS == 'Sowing') * 2 - (MU == 'Yes'))
    combo['formula_score'] = score.astype(np.float32)
    combo['is_low'] = (score <= 0).astype(np.float32)
    combo['is_high'] = (score > 3).astype(np.float32)

    # radix: snapped value within each category
    cat_codes = {}
    for c in CATS:
        cat = pd.Categorical(combo[c].astype(str).fillna('NA'))
        cat_codes[c] = cat.codes.astype(np.int32)
        combo[c] = cat
    radix_cols = []
    for nc in radix_nums:
        snap_i = np.round(combo[f'{nc}_snap'].values * 100).astype(np.int32)
        for cc in CATS:
            name = f'RADIX_{nc}__{cc}'
            combo[name] = (snap_i + cat_codes[cc] * RADIX).astype(np.int32)
            radix_cols.append(name)

    # per-class original priors
    y_orig = orig['target'].values
    for c in CATS:
        for k in range(3):
            m = orig.groupby(c).apply(lambda g, kk=k: (g['target'] == kk).mean())
            combo[f'ORIG_k{k}_{c}'] = combo[c].astype(str).map(m).fillna((y_orig == k).mean()).astype(np.float32)

    all_feats = (NUMS + [f'{c}_snap' for c in NUMS] + [f'{c}_snap_diff' for c in NUMS]
                 + [f'{c}_{d}' for c in FORMULA_NUMS for d in ['d0', 'd1', 'd2']]
                 + ['formula_score', 'is_low', 'is_high'] + list(CATS) + radix_cols
                 + [f'ORIG_k{k}_{c}' for c in CATS for k in range(3)])
    all_feats = [c for c in all_feats if c in combo.columns]
    X_all = combo.iloc[:n_train][all_feats].copy().reset_index(drop=True)
    T_all = combo.iloc[n_train:][all_feats].copy().reset_index(drop=True)
    for c in CATS:
        X_all[c] = X_all[c].astype('object').astype('category')
        T_all[c] = T_all[c].astype('object').astype('category')
    print(f'Features: {len(all_feats)} (radix: {len(radix_cols)}), {time.time() - t0:.1f}s', flush=True)
    return X_all, T_all, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=list(RECIPES))
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--force', action='store_true', help='rerun folds that are already done')
    args = ap.parse_args()
    pair = args.pair
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, pair, k)]
    if todo:
        X_all, T_all, y = build_features(**RECIPES[pair])
        cls_counts = np.bincount(y)
        cls_weights = cls_counts.sum() / (3 * cls_counts)
        fold_id = folds(y)
        for k in todo:
            t0 = time.time()
            tr_idx, va_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            model = XGBClassifier(**XGB_PARAMS)
            model.fit(X_all.iloc[tr_idx], y[tr_idx], sample_weight=cls_weights[y[tr_idx]],
                      eval_set=[(X_all.iloc[va_idx], y[va_idx])], verbose=500)
            p_va = model.predict_proba(X_all.iloc[va_idx]).astype(np.float32)
            p_te = model.predict_proba(T_all).astype(np.float32)
            save_part(HERE, pair, k, p_va, p_te)
            bac = balanced_accuracy_score(y[va_idx], np.argmax(p_va, axis=1))
            print(f'{pair} fold {k}: balanced accuracy {bac:.5f}, best iteration {model.best_iteration} '
                  f'({time.time() - t0:.0f}s)', flush=True)
            del model
            gc.collect()
    y = labels(read_train()[TARGET])
    assemble(HERE, pair, folds(y), y)


if __name__ == '__main__':
    main()
