#!/usr/bin/env python
"""TabNet on numeric transforms, management / cross / bin features, train + test counts and original-data priors.

Features, built on train + test (the original rows are a reference only, not training rows):
- the 8 categoricals and the 11 numerics;
- water input total and gap, Soil_Moisture - mean water input, moisture / (water + 1), water / (area + 1),
  Temperature_C x (100 - Humidity), Sunlight_Hours x Temperature_C, Electrical_Conductivity x Soil_pH;
- log1p, sqrt and the value rounded to 0 and to 1 decimal of Rainfall_mm, Previous_Irrigation_mm, Soil_Moisture,
  Humidity and Temperature_C;
- mulch-yes and canal-source flags; Season x Region, Crop_Type x Irrigation_Type, Crop_Type x Crop_Growth_Stage;
- 6 categorical pairs; floor(x / 5) bins of Rainfall, Previous_Irrigation, Soil_Moisture and Humidity; rain bin x crop
  and previous-irrigation bin x irrigation type;
- original-data count and target mean (label codes Low 0, Medium 1, High 2; unseen -> global mean) of 7 categoricals;
- train + test count and a rare flag (count <= 25) of every string column, pair and bin.
Columns of string type or with '__' in the name (31, incl. the counts / rare flags of the pairs) are ordinal-encoded on
train + test (code + 1, unknown 0) and embedded by TabNet (dim ceil(log2(cardinality)) clipped to 3-16); the other 93
columns are numbers (missing -> train + test median). 124 inputs.
Model: pytorch-tabnet TabNetClassifier, n_d = n_a = 32, 5 steps, gamma 1.5, 2 independent + 2 shared GLU layers,
lambda_sparse 1e-4, momentum 0.02, gradient clip 2, entmax masks; Adam lr 2e-2, weight decay 1e-5, StepLR (20 epochs,
x0.7); batch 8192, virtual batch 1024, up to 300 epochs with early stopping on the held-out balanced accuracy (patience
30, best weights restored); seed 42. Test = mean of the 5 fold models.
Usage: python train_tabnet_crosses.py [--folds 0,1,2,3,4]"""
import argparse
import copy
import gc
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import OrdinalEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, NUMERICS, TARGET, TARGET_MAP, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'tabnet_crosses_counts_orig_priors_a'
MODEL_DIR = model_dir('neural_nets')
RANDOM_STATE = 42

N_D = 32
N_A = 32
N_STEPS = 5
GAMMA = 1.5
N_INDEPENDENT = 2
N_SHARED = 2
LAMBDA_SPARSE = 1e-4
MOMENTUM = 0.02
CLIP_VALUE = 2.0
CAT_EMB_DIM_DEFAULT = 3
CAT_EMB_DIM_MAX = 16
MAX_EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 8192
VIRTUAL_BATCH_SIZE = 1024
LR = 2e-2
WEIGHT_DECAY = 1e-5
RARE_THRESH = 25

BASE_NUM_COLS = list(NUMERICS)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_div(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a / np.where(np.abs(b) < 1e-12, np.nan, b)


def add_missing_flags(df):
    feats = {}
    for c in df.columns:
        if df[c].isna().any():
            feats[f'{c}_was_missing'] = df[c].isna().astype(np.int8)
    if feats:
        df = pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)
    return df


def add_numeric_features(df):
    df = df.copy()
    feats = {}
    rain = pd.to_numeric(df['Rainfall_mm'], errors='coerce').to_numpy(dtype=np.float64)
    prev = pd.to_numeric(df['Previous_Irrigation_mm'], errors='coerce').to_numpy(dtype=np.float64)
    moist = pd.to_numeric(df['Soil_Moisture'], errors='coerce').to_numpy(dtype=np.float64)
    temp = pd.to_numeric(df['Temperature_C'], errors='coerce').to_numpy(dtype=np.float64)
    hum = pd.to_numeric(df['Humidity'], errors='coerce').to_numpy(dtype=np.float64)
    area = pd.to_numeric(df['Field_Area_hectare'], errors='coerce').to_numpy(dtype=np.float64)
    ph = pd.to_numeric(df['Soil_pH'], errors='coerce').to_numpy(dtype=np.float64)
    ec = pd.to_numeric(df['Electrical_Conductivity'], errors='coerce').to_numpy(dtype=np.float64)
    sun = pd.to_numeric(df['Sunlight_Hours'], errors='coerce').to_numpy(dtype=np.float64)

    feats['water_input_total'] = np.nan_to_num(rain + prev, nan=0.0)
    feats['water_input_gap'] = np.nan_to_num(prev - rain, nan=0.0)
    feats['moisture_gap'] = np.nan_to_num(moist - (rain + prev) / 2.0, nan=0.0)
    feats['moisture_to_water_ratio'] = np.nan_to_num(safe_div(moist, rain + prev + 1.0), nan=0.0, posinf=0.0, neginf=0.0)
    feats['water_per_area'] = np.nan_to_num(safe_div(rain + prev, area + 1.0), nan=0.0, posinf=0.0, neginf=0.0)
    feats['heat_stress'] = np.nan_to_num(temp * (100.0 - hum), nan=0.0)
    feats['sun_heat'] = np.nan_to_num(sun * temp, nan=0.0)
    feats['salinity_ph'] = np.nan_to_num(ec * ph, nan=0.0)

    for col in ['Rainfall_mm', 'Previous_Irrigation_mm', 'Soil_Moisture', 'Humidity', 'Temperature_C']:
        x = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=np.float64)
        feats[f'{col}_log1p'] = np.log1p(np.clip(np.nan_to_num(x), 0, None))
        feats[f'{col}_sqrt'] = np.sqrt(np.clip(np.nan_to_num(x), 0, None))
        feats[f'{col}_round0'] = np.round(np.nan_to_num(x), 0).astype(np.float32)
        feats[f'{col}_round1'] = np.round(np.nan_to_num(x), 1).astype(np.float32)
    return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)


def add_management_features(df):
    df = df.copy()
    feats = {
        'mulch_yes': df['Mulching_Used'].astype(str).fillna('missing').eq('Yes').astype(np.int8).to_numpy(),
        'water_surface': df['Water_Source'].astype(str).fillna('missing').eq('Canal').astype(np.int8).to_numpy(),
        'season_region': df['Season'].astype(str).fillna('missing') + '__' + df['Region'].astype(str).fillna('missing'),
        'crop_irrigation': (df['Crop_Type'].astype(str).fillna('missing') + '__'
                            + df['Irrigation_Type'].astype(str).fillna('missing')),
        'crop_stage': (df['Crop_Type'].astype(str).fillna('missing') + '__'
                       + df['Crop_Growth_Stage'].astype(str).fillna('missing')),
    }
    return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)


def add_cross_features(df):
    df = df.copy()
    feats = {}
    pairs = [
        ('Crop_Type', 'Season'),
        ('Crop_Type', 'Irrigation_Type'),
        ('Water_Source', 'Irrigation_Type'),
        ('Soil_Type', 'Crop_Type'),
        ('Region', 'Crop_Type'),
        ('Crop_Growth_Stage', 'Season'),
    ]
    for a, b in pairs:
        feats[f'{a}__{b}'] = df[a].astype(str).fillna('missing') + '__' + df[b].astype(str).fillna('missing')

    for name, col in [('rain_bin5', 'Rainfall_mm'), ('prev_bin5', 'Previous_Irrigation_mm'),
                      ('moist_bin5', 'Soil_Moisture'), ('humidity_bin5', 'Humidity')]:
        feats[name] = np.floor(pd.to_numeric(df[col], errors='coerce').fillna(0).to_numpy() / 5.0).astype(np.int16)

    feats['rainbin_crop'] = feats['rain_bin5'].astype(str) + '__' + df['Crop_Type'].astype(str).fillna('missing')
    feats['prevbin_irrig'] = feats['prev_bin5'].astype(str) + '__' + df['Irrigation_Type'].astype(str).fillna('missing')
    return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)


def add_freq_features(train_df, test_df):
    combo = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    cat_like_cols = []
    for c in combo.columns:
        if combo[c].dtype == 'object' or str(combo[c].dtype).startswith('string') or '__' in c or c.endswith('bin5'):
            cat_like_cols.append(c)

    new_cols = {}
    for c in sorted(set(cat_like_cols)):
        s = combo[c].astype(str).fillna('missing')
        cnt = s.map(s.value_counts(dropna=False)).astype(np.int32)
        new_cols[f'{c}_cnt_all'] = cnt.to_numpy()
        new_cols[f'{c}_rare_all'] = (cnt.to_numpy() <= RARE_THRESH).astype(np.int8)
    if new_cols:
        combo = pd.concat([combo, pd.DataFrame(new_cols, index=combo.index)], axis=1)

    train_out = combo.iloc[:len(train_df)].copy()
    test_out = combo.iloc[len(train_df):].copy().reset_index(drop=True)
    return train_out, test_out


def add_orig_features(train_df, test_df, orig_df):
    y_orig = pd.to_numeric(orig_df[TARGET], errors='coerce')
    global_mean = float(y_orig.mean())
    keys = ['Crop_Type', 'Season', 'Irrigation_Type', 'Water_Source', 'Region', 'Mulching_Used', 'Crop_Growth_Stage']

    tr_new, te_new = {}, {}
    for k in keys:
        s = orig_df[k].astype(str).fillna('missing')
        vc = s.value_counts(dropna=False)
        mean_map = pd.DataFrame({k: s, TARGET: y_orig}).groupby(k)[TARGET].mean()
        tr_key = train_df[k].astype(str).fillna('missing')
        te_key = test_df[k].astype(str).fillna('missing')
        tr_new[f'orig_cnt_{k}'] = tr_key.map(vc).fillna(0).astype(np.float32).to_numpy()
        te_new[f'orig_cnt_{k}'] = te_key.map(vc).fillna(0).astype(np.float32).to_numpy()
        tr_new[f'orig_mean_{k}'] = tr_key.map(mean_map).fillna(global_mean).astype(np.float32).to_numpy()
        te_new[f'orig_mean_{k}'] = te_key.map(mean_map).fillna(global_mean).astype(np.float32).to_numpy()

    train_df = pd.concat([train_df, pd.DataFrame(tr_new, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(te_new, index=test_df.index)], axis=1)
    return train_df, test_df


def build_features(train_df, test_df, orig_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    orig_df = orig_df.copy()
    for df in (train_df, test_df, orig_df):
        for c in BASE_NUM_COLS:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    train_df = add_numeric_features(train_df)
    test_df = add_numeric_features(test_df)
    train_df = add_management_features(train_df)
    test_df = add_management_features(test_df)
    train_df = add_cross_features(train_df)
    test_df = add_cross_features(test_df)
    train_df, test_df = add_orig_features(train_df, test_df, orig_df)
    train_df, test_df = add_freq_features(train_df, test_df)
    train_df = add_missing_flags(train_df)
    test_df = add_missing_flags(test_df)
    return train_df, test_df


def prepare_tabnet_matrices(train_df, test_df):
    """Numbers first, then the ordinal-encoded categorical columns (sorted by name). The id column is not an input."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    cat_cols = []
    for c in train_df.columns:
        if c == ID:
            continue
        if train_df[c].dtype == 'object' or str(train_df[c].dtype).startswith('string') or '__' in c:
            if c in test_df.columns:
                cat_cols.append(c)
    cat_cols = sorted(set(cat_cols))
    num_cols = [c for c in train_df.columns if c not in cat_cols + [ID] and c in test_df.columns]

    for c in cat_cols:
        train_df[c] = train_df[c].astype('string').fillna('missing')
        test_df[c] = test_df[c].astype('string').fillna('missing')

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1,
                         dtype=np.float32)
    enc.fit(pd.concat([train_df[cat_cols], test_df[cat_cols]], axis=0, ignore_index=True))
    train_cat = enc.transform(train_df[cat_cols]).astype(np.float32) + 1.0
    test_cat = enc.transform(test_df[cat_cols]).astype(np.float32) + 1.0
    cat_dims = [int(max(train_cat[:, j].max(), test_cat[:, j].max())) + 1 for j in range(train_cat.shape[1])]
    cat_idxs = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    cat_emb_dim = [min(CAT_EMB_DIM_MAX, max(CAT_EMB_DIM_DEFAULT, int(np.ceil(np.log2(max(d, 2)))))) for d in cat_dims]

    X_num_train = train_df[num_cols].copy()
    X_num_test = test_df[num_cols].copy()
    for c in num_cols:
        X_num_train[c] = pd.to_numeric(X_num_train[c], errors='coerce')
        X_num_test[c] = pd.to_numeric(X_num_test[c], errors='coerce')
    med = pd.concat([X_num_train, X_num_test], axis=0, ignore_index=True).median(numeric_only=True)
    X_num_train = X_num_train.fillna(med).to_numpy(dtype=np.float32)
    X_num_test = X_num_test.fillna(med).to_numpy(dtype=np.float32)

    X_train = np.hstack([X_num_train, train_cat]).astype(np.float32)
    X_test = np.hstack([X_num_test, test_cat]).astype(np.float32)
    feature_names = list(num_cols) + [f'CAT__{c}' for c in cat_cols]
    return X_train, X_test, feature_names, cat_idxs, cat_dims, cat_emb_dim


def make_model(cat_idxs, cat_dims, cat_emb_dim):
    return TabNetClassifier(
        n_d=N_D, n_a=N_A, n_steps=N_STEPS, gamma=GAMMA, n_independent=N_INDEPENDENT, n_shared=N_SHARED,
        lambda_sparse=LAMBDA_SPARSE, momentum=MOMENTUM, clip_value=CLIP_VALUE,
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim,
        optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=LR, weight_decay=WEIGHT_DECAY),
        scheduler_fn=torch.optim.lr_scheduler.StepLR, scheduler_params=dict(step_size=20, gamma=0.7),
        mask_type='entmax', verbose=0, seed=RANDOM_STATE, device_name='auto',
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(RANDOM_STATE)
    train, test, orig = read_train(), read_test(), read_original()
    y = train[TARGET].map(TARGET_MAP).astype(np.int32).to_numpy()
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype(np.int32)
    print(f'train {train.shape}, test {test.shape}, original {orig.shape}', flush=True)

    train_fe, test_fe = build_features(train.drop(columns=[TARGET]).copy(), test.copy(), orig.copy())
    X_train, X_test, feature_names, cat_idxs, cat_dims, cat_emb_dim = prepare_tabnet_matrices(train_fe, test_fe)
    print(f'X_train {X_train.shape}, X_test {X_test.shape}, {len(cat_idxs)} categorical inputs', flush=True)

    fold_ids = folds(y)
    for k in fold_list(args.folds):
        t0 = time.time()
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        X_tr, y_tr = X_train[tr_idx], y[tr_idx]
        X_va, y_va = X_train[va_idx], y[va_idx]
        clf = make_model(cat_idxs=copy.deepcopy(cat_idxs), cat_dims=copy.deepcopy(cat_dims),
                         cat_emb_dim=copy.deepcopy(cat_emb_dim))
        clf.fit(X_train=X_tr, y_train=y_tr, eval_set=[(X_va, y_va)], eval_name=['valid'],
                eval_metric=['balanced_accuracy'], max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
                virtual_batch_size=VIRTUAL_BATCH_SIZE, num_workers=0, drop_last=False)
        va_pred = clf.predict_proba(X_va).astype(np.float32)
        te_pred = clf.predict_proba(X_test).astype(np.float32)
        print(f'fold {k + 1}/5 | BAC={balanced_accuracy_score(y_va, va_pred.argmax(axis=1)):.6f} | '
              f'best_epoch={getattr(clf, "best_epoch", None)} | time={time.time() - t0:.1f}s', flush=True)
        save_part(MODEL_DIR, PAIR, k, va_pred, te_pred)
        del X_tr, y_tr, X_va, y_va, clf, va_pred, te_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
