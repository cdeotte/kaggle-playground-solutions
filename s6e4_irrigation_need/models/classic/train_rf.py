#!/usr/bin/env python
"""cuML random forests on original-snap / decimal-digit features and per-class target encodings (two presets).

rf_water_snap_digit_te_a (stack A): the raw columns (including id) plus
  - Rainfall_mm and Previous_Irrigation_mm snapped to the nearest value of the original data, and value - snap;
  - their first and second decimal digits and a round-value flag (fractional part < 0.005);
  - water balance: rainfall + previous irrigation, soil moisture - that total, total / field area;
  - a managed-water count (mulching used + drip / sprinkler irrigation + water source other than rainwater);
  - the bigrams Crop_Type x Season and Irrigation_Type x Water_Source.
rf_snap_digit_formula_te_b (stack B): the raw columns (including id, numerics as float32) plus
  - a threshold score 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10]
    - 2 [stage Harvest] - 2 [stage Sowing] - [mulching used], with the flags score <= 0 and score > 3;
  - all 11 numerics snapped to the nearest original value, value - snap, and their first decimal digit;
  - the bigrams Crop_Type x Season, Crop_Growth_Stage x Mulching_Used, Irrigation_Type x Water_Source, Soil_Type x Region.
Per fold, every categorical (the 8 raw ones and the bigrams) is replaced by 3 one-vs-rest target encodings (cuML
TargetEncoder, 5 inner folds, smoothing 20, interleaved split) fitted on the fold's training rows.
cuML RandomForestClassifier: 1,500 trees, max_depth 16, max_features 0.3, min_samples_leaf 15, random_state 42,
n_streams 1. Test = mean of the 5 fold models. The original rows only provide the snap values.
Usage: python train_rf.py <preset> [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from cuml.ensemble import RandomForestClassifier
from cuml.preprocessing import TargetEncoder
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, NUMERICS, TARGET, folds, labels, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

FOLDER = 'classic'
SEED = 42
N_CLASSES = 3
CATS = list(CATEGORICALS)
NUMS = list(NUMERICS)

RF_PARAMS = dict(
    n_estimators=1500,
    max_depth=16,
    max_features=0.3,
    min_samples_leaf=15,
    random_state=SEED,
    n_streams=1,
)


def water_snap_features(train, test, orig):
    """Stack A features. Categoricals = the object columns (8 raw + 2 bigrams)."""
    orig_rain = np.sort(pd.to_numeric(orig['Rainfall_mm'], errors='coerce').dropna().unique())
    orig_prev = np.sort(pd.to_numeric(orig['Previous_Irrigation_mm'], errors='coerce').dropna().unique())

    def build(df):
        df = df.copy()
        rain = pd.to_numeric(df['Rainfall_mm'], errors='coerce')
        prev = pd.to_numeric(df['Previous_Irrigation_mm'], errors='coerce')
        df['Rain_snap'] = rain.apply(lambda x: orig_rain[np.abs(orig_rain - x).argmin()])
        df['PrevIrr_snap'] = prev.apply(lambda x: orig_prev[np.abs(orig_prev - x).argmin()])
        df['Rain_snap_diff'] = rain - df['Rain_snap']
        df['PrevIrr_snap_diff'] = prev - df['PrevIrr_snap']

        for col in ['Rainfall_mm', 'Previous_Irrigation_mm']:
            x = pd.to_numeric(df[col], errors='coerce')
            frac = x - np.floor(x)
            df[f'{col}_d1'] = np.floor(frac * 10).fillna(0).astype(int)
            df[f'{col}_d2'] = (np.floor(frac * 100) % 10).fillna(0).astype(int)
            df[f'{col}_is_round'] = (frac < 0.005).fillna(0).astype(int)

        total_water = rain.fillna(0) + prev.fillna(0)
        soil_moisture = pd.to_numeric(df['Soil_Moisture'], errors='coerce').fillna(0)
        field_area = pd.to_numeric(df['Field_Area_hectare'], errors='coerce').fillna(0)
        df['WaterInput_total'] = total_water
        df['Moisture_deviation'] = soil_moisture - total_water
        df['Water_to_area_ratio'] = total_water / (field_area + 1e-6)

        df['managed_water_count'] = (
            (df['Mulching_Used'] == 'Yes').astype(int)
            + df['Irrigation_Type'].isin(['Drip', 'Sprinkler']).astype(int)
            + (df['Water_Source'] != 'Rainwater').astype(int)
        )
        df['bi_Crop_Season'] = df['Crop_Type'].astype(str) + '__' + df['Season'].astype(str)
        df['bi_Irrigation_Water'] = df['Irrigation_Type'].astype(str) + '__' + df['Water_Source'].astype(str)
        return df

    train_fe, test_fe = build(train), build(test)
    cat_cols = [c for c in train_fe.columns if train_fe[c].dtype == 'object']
    return train_fe, test_fe, cat_cols


def snap_formula_features(train, test, orig):
    """Stack B features. Categoricals = the 8 raw ones + 4 bigrams."""
    train, test, orig = train.copy(), test.copy(), orig.copy()
    for col in NUMS:
        for df in (train, test, orig):
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

    def build(df):
        df = df.copy()
        sm = df['Soil_Moisture'].astype(float)
        rm = df['Rainfall_mm'].astype(float)
        tc = df['Temperature_C'].astype(float)
        ws = df['Wind_Speed_kmh'].astype(float)
        cgs = df['Crop_Growth_Stage'].astype(str)
        mu = df['Mulching_Used'].astype(str)
        score = ((sm < 25) * 2 + (rm < 300) * 2 + (tc > 30) + (ws > 10)
                 - (cgs == 'Harvest') * 2 - (cgs == 'Sowing') * 2 - (mu == 'Yes'))
        df['formula_score'] = score.astype(np.float32)
        df['is_low'] = (score <= 0).astype(np.float32)
        df['is_high'] = (score > 3).astype(np.float32)

        # nearest original value (ties -> the smaller value)
        for col in NUMS:
            orig_uniq = np.sort(orig[col].dropna().unique()).astype('float64')
            v = df[col].values.astype('float64')
            idx = np.clip(np.searchsorted(orig_uniq, v, side='left'), 0, len(orig_uniq) - 1)
            idx_l = np.clip(idx - 1, 0, len(orig_uniq) - 1)
            snap_r, snap_l = orig_uniq[idx], orig_uniq[idx_l]
            snap = np.where(np.abs(v - snap_l) <= np.abs(v - snap_r), snap_l, snap_r)
            df[f'{col}_snap'] = snap.astype(np.float32)
            df[f'{col}_snap_diff'] = (v - snap).astype(np.float32)

        for col in NUMS:
            frac = df[col].astype(float) - np.floor(df[col].astype(float))
            df[f'{col}_d1'] = (np.floor(frac * 10 + 1e-6) % 10).astype(np.float32)

        df['bi_CropType_Season'] = df['Crop_Type'].astype(str) + '__' + df['Season'].astype(str)
        df['bi_CropGrowth_Mulch'] = df['Crop_Growth_Stage'].astype(str) + '__' + df['Mulching_Used'].astype(str)
        df['bi_IrrigType_Source'] = df['Irrigation_Type'].astype(str) + '__' + df['Water_Source'].astype(str)
        df['bi_Soil_Region'] = df['Soil_Type'].astype(str) + '__' + df['Region'].astype(str)
        return df

    cat_cols = CATS + ['bi_CropType_Season', 'bi_CropGrowth_Mulch', 'bi_IrrigType_Source', 'bi_Soil_Region']
    return build(train), build(test), cat_cols


# preset -> (feature builder, dtype of the one-vs-rest target given to the encoder)
PRESETS = {
    'rf_water_snap_digit_te_a': (water_snap_features, np.int32),
    'rf_snap_digit_formula_te_b': (snap_formula_features, np.float32),
}


def to_numpy(a):
    if hasattr(a, 'to_numpy'):
        a = a.to_numpy()
    return np.asarray(a).ravel()


def add_class_target_encodings(X_tr, y_tr, X_va, X_te, cat_cols, y_dtype):
    """Each categorical -> 3 one-vs-rest target encodings (inner-fold values for the training rows)."""
    for col in cat_cols:
        for k in range(N_CLASSES):
            encoder = TargetEncoder(n_folds=5, smooth=20, split_method='interleaved')
            y_bin = (y_tr == k).astype(y_dtype)
            X_tr[f'{col}_TE_c{k}'] = to_numpy(encoder.fit_transform(X_tr[col].astype(str), y_bin)).astype(np.float32)
            X_va[f'{col}_TE_c{k}'] = to_numpy(encoder.transform(X_va[col].astype(str))).astype(np.float32)
            X_te[f'{col}_TE_c{k}'] = to_numpy(encoder.transform(X_te[col].astype(str))).astype(np.float32)
        for X in (X_tr, X_va, X_te):
            X.drop(columns=[col], inplace=True)
    return X_tr, X_va, X_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preset', choices=sorted(PRESETS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    pair = args.preset
    out = model_dir(FOLDER)

    train, test, orig = read_train(), read_test(), read_original()
    y = labels(train[TARGET]).astype(np.int32)
    build_features, te_y_dtype = PRESETS[pair]
    train_fe, test_fe, cat_cols = build_features(train.drop(columns=[TARGET]), test, orig)
    print(f'{pair}: {train_fe.shape[1]} columns before target encoding, categoricals {cat_cols}', flush=True)

    fold_id = folds(y)
    for k in fold_list(args.folds):
        if have_part(out, pair, k):
            print(f'{pair} fold {k}: done', flush=True)
            continue
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        X_tr = train_fe.iloc[tr_idx].copy()
        X_va = train_fe.iloc[va_idx].copy()
        X_te = test_fe.copy()
        X_tr, X_va, X_te = add_class_target_encodings(X_tr, y[tr_idx], X_va, X_te, cat_cols, te_y_dtype)

        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_tr.astype('float32').to_numpy(), y[tr_idx])
        p_va = np.asarray(model.predict_proba(X_va.astype('float32').to_numpy()), dtype=np.float32)
        p_te = np.asarray(model.predict_proba(X_te.astype('float32').to_numpy()), dtype=np.float32)
        save_part(out, pair, k, p_va, p_te)
        print(f'{pair} fold {k}: balanced accuracy {balanced_accuracy_score(y[va_idx], p_va.argmax(1)):.6f}', flush=True)
        del X_tr, X_va, X_te, model
        gc.collect()
    assemble(out, pair, fold_id, y)


if __name__ == '__main__':
    main()
