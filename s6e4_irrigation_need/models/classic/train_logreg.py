#!/usr/bin/env python
"""Multinomial logistic regression on one-hot categoricals, crosses, original-quantile bins and dense numerics
(pair logreg_ohe_crosses_bins_group_dense_a, stack A).

Features (built on train, test and original together; the original rows shape the bins and group statistics but are
not training rows):
  - dense (174, standardized per fold on the fold's training rows): the 11 numerics; 15 interactions (rainfall +
    previous irrigation, their difference and ratios, moisture / water, water / area, rain x moisture, previous
    irrigation x moisture, temperature x (100 - humidity), sunlight x temperature, wind x temperature, EC x pH,
    organic carbon x moisture, log water, log area); 16 category-gated numerics (e.g. rainfall if Summer, previous
    irrigation if Drip, moisture if mulched); value - group mean, value / (group mean + 1) and z-score of the 11
    numerics within Season, Irrigation_Type, Water_Source and Season x Irrigation_Type;
  - one-hot (4,518 columns, levels over train + test + original): the 8 categoricals, Season x Irrigation_Type, every
    numeric binned into 128-512 and into 16 bins of the sorted unique original values, the first decimal digit of 5
    numerics, 8 categorical crosses, 8 categorical x 16-bin crosses and 4 16-bin x 16-bin crosses.
Model per fold (504,000 x 4,692): PyTorch multinomial logistic regression on the GPU, full batch, Adam lr 0.05,
80 epochs, class-weighted cross-entropy (weights n / class count, normalized to mean 1), L2 weight decay
1 / (0.3 n) on the weights only. Test = mean of the 5 fold models.
--cuml first tries cuML LogisticRegression (L2, C 0.3, max_iter 3000, tol 1e-6) and falls back to the PyTorch model
if cuML fails; in our run cuML raised a CUDA error on these fold matrices in every fold, so the pair comes from the
PyTorch model (the default).
Usage: python train_logreg.py [--folds 0,1,2,3,4] [--cuml]   (finished folds are kept in parts/ and skipped)"""
import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, ID, NUMERICS, TARGET, TARGET_MAP, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'logreg_ohe_crosses_bins_group_dense_a'
FOLDER = 'classic'

NUMERIC_COLS = list(NUMERICS)
BASE_CAT_COLS = list(CATEGORICALS)

NUMERIC_BIN_CARDINALITY = {
    'Soil_pH': 128,
    'Soil_Moisture': 256,
    'Organic_Carbon': 128,
    'Electrical_Conductivity': 128,
    'Temperature_C': 192,
    'Humidity': 256,
    'Rainfall_mm': 512,
    'Sunlight_Hours': 128,
    'Wind_Speed_kmh': 128,
    'Field_Area_hectare': 128,
    'Previous_Irrigation_mm': 512,
}
LOW_CARD_BIN_CARDINALITY = {c: 16 for c in NUMERIC_COLS}
DECIMAL_DIGIT_COLS = ['Rainfall_mm', 'Previous_Irrigation_mm', 'Soil_Moisture', 'Temperature_C', 'Humidity']

LOGREG_PARAMS = {  # cuML (--cuml)
    'penalty': 'l2',
    'C': 0.30,
    'fit_intercept': True,
    'max_iter': 3000,
    'tol': 1e-6,
    'linesearch_max_iter': 100,
    'output_type': 'numpy',
}
PT_C = 0.30
PT_LR = 5e-2
PT_EPOCHS = 80

CAT_CAT_CROSSES = [
    ('Crop_Type', 'Season'),
    ('Crop_Type', 'Irrigation_Type'),
    ('Soil_Type', 'Crop_Type'),
    ('Water_Source', 'Irrigation_Type'),
    ('Region', 'Crop_Type'),
    ('Season', 'Crop_Growth_Stage'),
    ('Mulching_Used', 'Irrigation_Type'),
    ('Water_Source', 'Region'),
]
CAT_LOWNUM_CROSSES = [
    ('Crop_Type', 'Rainfall_mm_bin16'),
    ('Crop_Type', 'Previous_Irrigation_mm_bin16'),
    ('Irrigation_Type', 'Rainfall_mm_bin16'),
    ('Irrigation_Type', 'Soil_Moisture_bin16'),
    ('Water_Source', 'Humidity_bin16'),
    ('Season', 'Temperature_C_bin16'),
    ('Crop_Growth_Stage', 'Soil_Moisture_bin16'),
    ('Region', 'Field_Area_hectare_bin16'),
]
LOW_NUMGRID_CROSSES = [
    ('Rainfall_mm_bin16', 'Previous_Irrigation_mm_bin16'),
    ('Rainfall_mm_bin16', 'Soil_Moisture_bin16'),
    ('Temperature_C_bin16', 'Humidity_bin16'),
    ('Field_Area_hectare_bin16', 'Previous_Irrigation_mm_bin16'),
]
GATED_NUMERIC_SPECS = [
    ('Season', 'Summer', 'Rainfall_mm', 'rain_if_summer'),
    ('Season', 'Winter', 'Rainfall_mm', 'rain_if_winter'),
    ('Irrigation_Type', 'Drip', 'Previous_Irrigation_mm', 'prev_if_drip'),
    ('Irrigation_Type', 'Flood', 'Previous_Irrigation_mm', 'prev_if_flood'),
    ('Water_Source', 'Canal', 'Rainfall_mm', 'rain_if_canal'),
    ('Water_Source', 'Well', 'Rainfall_mm', 'rain_if_well'),
    ('Mulching_Used', 'Yes', 'Soil_Moisture', 'moist_if_mulch_yes'),
    ('Mulching_Used', 'No', 'Soil_Moisture', 'moist_if_mulch_no'),
    ('Crop_Growth_Stage', 'Mature', 'Rainfall_mm', 'rain_if_mature'),
    ('Crop_Growth_Stage', 'Seedling', 'Previous_Irrigation_mm', 'prev_if_seedling'),
    ('Crop_Type', 'Rice', 'Rainfall_mm', 'rain_if_rice'),
    ('Crop_Type', 'Wheat', 'Previous_Irrigation_mm', 'prev_if_wheat'),
    ('Region', 'North', 'Temperature_C', 'temp_if_north'),
    ('Region', 'South', 'Temperature_C', 'temp_if_south'),
    ('Season', 'Summer', 'water_input_total', 'water_if_summer'),
    ('Irrigation_Type', 'Sprinkler', 'moisture_to_water_ratio', 'mwr_if_sprinkler'),
]
GROUP_KEYS = [
    ['Season'],
    ['Irrigation_Type'],
    ['Water_Source'],
    ['Season', 'Irrigation_Type'],
]


def make_apply_bins_fn(value_to_bin, edges, out_name):
    interval_starts = np.array([a for a, _ in edges], dtype=np.float32)
    interval_ends = np.array([b for _, b in edges], dtype=np.float32)

    def apply_bins(s):
        arr = s.to_numpy(dtype=np.float32, copy=False)
        out = np.full(len(arr), -1, dtype=np.int32)
        for i, v in enumerate(arr):
            if pd.isna(v):
                out[i] = -1
            else:
                key = float(v)
                if key in value_to_bin:
                    out[i] = value_to_bin[key]
                else:
                    placed = False
                    for b in range(len(interval_starts)):
                        if interval_starts[b] <= v <= interval_ends[b]:
                            out[i] = b
                            placed = True
                            break
                    if not placed:
                        out[i] = 0 if v < interval_starts[0] else len(interval_starts) - 1
        return pd.Series(out, index=s.index, name=out_name, dtype='int32')

    return apply_bins


def first_decimal_digit(s):
    arr = s.to_numpy(dtype=np.float32, copy=False)
    out = np.full(len(arr), -1, dtype=np.int32)
    for i, v in enumerate(arr):
        if pd.isna(v):
            out[i] = -1
        else:
            out[i] = int(np.floor(np.abs(v) * 10.0 + 1e-6)) % 10
    return pd.Series(out, index=s.index, dtype='int32')


def add_crosses(df, pairs):
    new_cols = []
    for a, b in pairs:
        name = f'{a}__{b}'
        df[name] = df[a].astype(str) + '|' + df[b].astype(str)
        new_cols.append(name)
    return new_cols


def add_numeric_interaction_features(df):
    rain = df['Rainfall_mm'].to_numpy(dtype=np.float32, copy=False)
    prev = df['Previous_Irrigation_mm'].to_numpy(dtype=np.float32, copy=False)
    moist = df['Soil_Moisture'].to_numpy(dtype=np.float32, copy=False)
    temp = df['Temperature_C'].to_numpy(dtype=np.float32, copy=False)
    hum = df['Humidity'].to_numpy(dtype=np.float32, copy=False)
    area = df['Field_Area_hectare'].to_numpy(dtype=np.float32, copy=False)
    sun = df['Sunlight_Hours'].to_numpy(dtype=np.float32, copy=False)
    wind = df['Wind_Speed_kmh'].to_numpy(dtype=np.float32, copy=False)
    ec = df['Electrical_Conductivity'].to_numpy(dtype=np.float32, copy=False)
    ph = df['Soil_pH'].to_numpy(dtype=np.float32, copy=False)
    oc = df['Organic_Carbon'].to_numpy(dtype=np.float32, copy=False)

    out = pd.DataFrame(index=df.index)
    water = rain + prev
    denom = area + np.float32(1.0)
    out['water_input_total'] = water.astype(np.float32)
    out['water_input_diff'] = (prev - rain).astype(np.float32)
    out['moisture_to_water_ratio'] = (moist / (water + 1.0)).astype(np.float32)
    out['water_to_area_ratio'] = (water / denom).astype(np.float32)
    out['rain_x_moisture'] = (rain * moist).astype(np.float32)
    out['prev_x_moisture'] = (prev * moist).astype(np.float32)
    out['heat_stress_index'] = (temp * (100.0 - hum)).astype(np.float32)
    out['sun_heat_interaction'] = (sun * temp).astype(np.float32)
    out['wind_x_temp'] = (wind * temp).astype(np.float32)
    out['salinity_ph_interaction'] = (ec * ph).astype(np.float32)
    out['organic_x_moisture'] = (oc * moist).astype(np.float32)
    out['rain_over_prev1'] = (rain / (prev + 1.0)).astype(np.float32)
    out['prev_over_rain1'] = (prev / (rain + 1.0)).astype(np.float32)
    out['log_water'] = np.log1p(np.clip(water, 0, None)).astype(np.float32)
    out['log_area'] = np.log1p(np.clip(area, 0, None)).astype(np.float32)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_gated_numeric_features(df, specs):
    out = pd.DataFrame(index=df.index)
    for col, level, src_num, new_col in specs:
        mask = (df[col].astype(str).to_numpy(copy=False) == str(level)).astype(np.float32)
        out[new_col] = (mask * df[src_num].to_numpy(dtype=np.float32, copy=False)).astype(np.float32)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_group_relative_features(train_df, test_df, orig_df, group_cols, numeric_cols):
    combined = pd.concat([train_df[group_cols + numeric_cols], test_df[group_cols + numeric_cols],
                          orig_df[group_cols + numeric_cols]], axis=0, ignore_index=True).copy()
    grp = combined.groupby(group_cols, dropna=False)[numeric_cols].agg(['mean', 'std'])
    grp.columns = [f'{c}_{stat}' for c, stat in grp.columns]
    grp = grp.reset_index()

    def attach(df):
        merged = df[group_cols + numeric_cols].merge(grp, on=group_cols, how='left')
        out = pd.DataFrame(index=df.index)
        prefix = '__'.join(group_cols)
        for col in numeric_cols:
            base = pd.to_numeric(merged[col], errors='coerce').astype(np.float32)
            mean = pd.to_numeric(merged[f'{col}_mean'], errors='coerce').fillna(0).astype(np.float32)
            std = pd.to_numeric(merged[f'{col}_std'], errors='coerce').fillna(1).astype(np.float32).replace(0, 1)
            out[f'{prefix}__{col}_minus_mean'] = (base - mean).astype(np.float32)
            out[f'{prefix}__{col}_div_mean'] = (base / (mean + 1.0)).astype(np.float32)
            out[f'{prefix}__{col}_z'] = ((base - mean) / std).astype(np.float32)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return attach(train_df), attach(test_df), attach(orig_df)


def build_bins_from_orig(train_df, test_df, orig_df, col, desired_card, out_name):
    """Sorted unique original values split into equal-count chunks; values outside fall into the enclosing or edge bin."""
    uniq = np.sort(orig_df[col].dropna().unique())
    if len(uniq) == 0:
        raise ValueError(f'No valid unique values found in original for column {col}')
    chunks = np.array_split(uniq, min(desired_card, len(uniq)))
    value_to_bin, edges = {}, []
    for bin_id, arr in enumerate(chunks):
        for v in arr:
            value_to_bin[float(v)] = int(bin_id)
        edges.append((float(arr[0]), float(arr[-1])))
    apply_bins = make_apply_bins_fn(value_to_bin, edges, out_name)
    train_df[out_name] = apply_bins(train_df[col]).astype(str)
    test_df[out_name] = apply_bins(test_df[col]).astype(str)
    orig_df[out_name] = apply_bins(orig_df[col]).astype(str)


def preprocess():
    print('loading train / test / original', flush=True)
    train_df, test_df, orig_df = read_train(), read_test(), read_original()
    train_df = train_df.drop(columns=[ID])
    test_df = test_df.drop(columns=[ID])
    train_df[TARGET] = train_df[TARGET].map(TARGET_MAP).astype('int8')
    orig_df[TARGET] = orig_df[TARGET].map(TARGET_MAP).astype('int8')
    for df in [train_df, test_df, orig_df]:
        for c in BASE_CAT_COLS:
            df[c] = df[c].astype(str).fillna('__MISSING__')
        for c in NUMERIC_COLS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

    print('numeric interactions, bins, digits, crosses, gated and group features', flush=True)
    train_num_int = add_numeric_interaction_features(train_df)
    test_num_int = add_numeric_interaction_features(test_df)
    orig_num_int = add_numeric_interaction_features(orig_df)
    train_df = pd.concat([train_df, train_num_int], axis=1)
    test_df = pd.concat([test_df, test_num_int], axis=1)
    orig_df = pd.concat([orig_df, orig_num_int], axis=1)
    all_numeric_cols = NUMERIC_COLS + train_num_int.columns.tolist()

    for df in [train_df, test_df, orig_df]:
        df['Season__Irrigation_Type'] = df['Season'].astype(str) + '|' + df['Irrigation_Type'].astype(str)

    for col in NUMERIC_COLS:
        build_bins_from_orig(train_df, test_df, orig_df, col, NUMERIC_BIN_CARDINALITY[col], f'{col}_bin')
        build_bins_from_orig(train_df, test_df, orig_df, col, LOW_CARD_BIN_CARDINALITY[col], f'{col}_bin16')

    for col in DECIMAL_DIGIT_COLS:
        for df in [train_df, test_df, orig_df]:
            df[f'{col}_d1'] = first_decimal_digit(df[col]).astype(str)

    for pairs in [CAT_CAT_CROSSES, CAT_LOWNUM_CROSSES, LOW_NUMGRID_CROSSES]:
        for df in [train_df, test_df, orig_df]:
            add_crosses(df, pairs)

    train_gated = add_gated_numeric_features(train_df, GATED_NUMERIC_SPECS)
    test_gated = add_gated_numeric_features(test_df, GATED_NUMERIC_SPECS)
    orig_gated = add_gated_numeric_features(orig_df, GATED_NUMERIC_SPECS)
    gated_numeric_cols = train_gated.columns.tolist()
    train_df = pd.concat([train_df, train_gated], axis=1)
    test_df = pd.concat([test_df, test_gated], axis=1)
    orig_df = pd.concat([orig_df, orig_gated], axis=1)

    group_dense_cols = []
    for group_cols in GROUP_KEYS:
        tr_g, te_g, or_g = add_group_relative_features(train_df, test_df, orig_df, group_cols, NUMERIC_COLS)
        group_dense_cols.extend(tr_g.columns.tolist())
        train_df = pd.concat([train_df, tr_g], axis=1)
        test_df = pd.concat([test_df, te_g], axis=1)
        orig_df = pd.concat([orig_df, or_g], axis=1)

    final_numeric_cols = all_numeric_cols + gated_numeric_cols + group_dense_cols
    final_feature_cols = [c for c in train_df.columns if c != TARGET]
    final_cat_cols = [c for c in final_feature_cols if c not in final_numeric_cols]
    return train_df, test_df, orig_df, final_cat_cols, final_numeric_cols


def build_global_ohe(train_df, test_df, orig_df, cat_cols):
    combined = pd.concat([train_df[cat_cols], test_df[cat_cols], orig_df[cat_cols]], axis=0,
                         ignore_index=True).astype(str).fillna('__MISSING__')
    ohe_all = pd.get_dummies(combined, columns=cat_cols, prefix=cat_cols, prefix_sep='=', dtype=np.uint8)
    n_train, n_test = len(train_df), len(test_df)
    X_cat_train = np.ascontiguousarray(ohe_all.iloc[:n_train].to_numpy(dtype=np.float32, copy=False))
    X_cat_test = np.ascontiguousarray(ohe_all.iloc[n_train:n_train + n_test].to_numpy(dtype=np.float32, copy=False))
    del combined, ohe_all
    gc.collect()
    return X_cat_train, X_cat_test


def standardize_fold(train_num, valid_num, test_num):
    mean = np.nanmean(train_num, axis=0, dtype=np.float64).astype(np.float32)
    std = np.nanstd(train_num, axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    scaled = [np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
              for x in (train_num, valid_num, test_num)]
    return [np.ascontiguousarray(x) for x in scaled]


class MultiLogReg(nn.Module):
    def __init__(self, in_features, out_features=3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


def train_pytorch_logreg(X_tr, y_tr, X_va, X_te):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiLogReg(X_tr.shape[1], 3).to(device)

    class_counts = np.bincount(y_tr, minlength=3).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    weight_decay = 1.0 / (PT_C * max(1, len(y_tr)))
    optimizer = optim.Adam([
        {'params': model.linear.weight, 'weight_decay': weight_decay},
        {'params': model.linear.bias, 'weight_decay': 0.0},
    ], lr=PT_LR)

    model.train()
    for _ in range(PT_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_probs = torch.softmax(model(X_va_t), dim=1).cpu().numpy().astype(np.float32)
        test_probs = torch.softmax(model(X_te_t), dim=1).cpu().numpy().astype(np.float32)
    del model, X_tr_t, y_tr_t, X_va_t, X_te_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return val_probs, test_probs


def train_cuml_logreg(X_tr, y_tr, X_va, X_te, k):
    """cuML first, the PyTorch model if cuML fails."""
    import cupy as cp
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    try:
        model = cuLogisticRegression(**LOGREG_PARAMS)
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_va).astype(np.float32), model.predict_proba(X_te).astype(np.float32), 'cuml'
    except Exception as e:
        print(f'fold {k}: cuML failed ({type(e).__name__}: {e}), using the PyTorch model', flush=True)
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        return (*train_pytorch_logreg(X_tr, y_tr, X_va, X_te), 'pytorch')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--cuml', action='store_true', help='try cuML LogisticRegression first')
    args = ap.parse_args()
    out = model_dir(FOLDER)

    train_df, test_df, orig_df, cat_features, numeric_features = preprocess()
    X_cat_train, X_cat_test = build_global_ohe(train_df, test_df, orig_df, cat_features)
    X_num_train = np.ascontiguousarray(train_df[numeric_features].to_numpy(dtype=np.float32, copy=True))
    X_num_test = np.ascontiguousarray(test_df[numeric_features].to_numpy(dtype=np.float32, copy=True))
    y = train_df[TARGET].astype(np.int32).values
    print(f'one-hot columns {X_cat_train.shape[1]}, dense columns {X_num_train.shape[1]}', flush=True)

    fold_id = folds(y)
    for k in fold_list(args.folds):
        if have_part(out, PAIR, k):
            print(f'{PAIR} fold {k}: done', flush=True)
            continue
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        X_num_tr_sc, X_num_va_sc, X_num_te_sc = standardize_fold(X_num_train[tr_idx], X_num_train[va_idx], X_num_test)
        X_tr = np.ascontiguousarray(np.concatenate([X_num_tr_sc, X_cat_train[tr_idx]], axis=1).astype(np.float32, copy=False))
        X_va = np.ascontiguousarray(np.concatenate([X_num_va_sc, X_cat_train[va_idx]], axis=1).astype(np.float32, copy=False))
        X_te = np.ascontiguousarray(np.concatenate([X_num_te_sc, X_cat_test], axis=1).astype(np.float32, copy=False))
        y_tr = np.ascontiguousarray(y[tr_idx].astype(np.int32, copy=False))
        if args.cuml:
            p_va, p_te, backend = train_cuml_logreg(X_tr, y_tr, X_va, X_te, k)
        else:
            (p_va, p_te), backend = train_pytorch_logreg(X_tr, y_tr, X_va, X_te), 'pytorch'
        save_part(out, PAIR, k, p_va, p_te)
        print(f'{PAIR} fold {k}: balanced accuracy {balanced_accuracy_score(y[va_idx], p_va.argmax(axis=1)):.6f} '
              f'({backend}, {X_tr.shape[1]} columns)', flush=True)
        del X_num_tr_sc, X_num_va_sc, X_num_te_sc, X_tr, X_va, X_te, y_tr, p_va, p_te
        gc.collect()
    assemble(out, PAIR, fold_id, y)


if __name__ == '__main__':
    main()
