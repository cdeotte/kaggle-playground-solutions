#!/usr/bin/env python
"""Class-weighted CatBoost on numeric x category "radix" keys, quantile bins and nested target encodings
(pair catboost_radix_bins_nested_te_b).

Label-free features (computed on train + test; original rows only for the priors):
  - the 19 columns; a threshold score 2 (moisture < 25) + 2 (rainfall < 300) + (temperature > 30) + (wind > 10)
    - 2 (Harvest) - 2 (Sowing) - (mulching) and the flags score <= 0 / score > 3
  - each numeric: integer part mod 10 and mod 100, fractional part, frequency, log1p(|x|)
  - quantile-bin ids (strings) of soil moisture (40), rainfall (60), temperature (40), humidity (40), previous
    irrigation (60)
  - moisture / rainfall, temperature / humidity, rainfall - moisture
  - per-class original frequency of the 8 categoricals (24)
  - the 6 pairs of Crop_Type, Season, Region, Crop_Growth_Stage as strings
  - 88 radix keys, one per numeric x categorical: clip(round(10 x value), 0, 49999) + 50000 x (category code + 1)
Native CatBoost categories (107): the 8 categoricals, the 5 bins, the 6 pairs and the 88 radix keys.
Per fold: for the 8 categoricals and the 5 bins, the mean class index (0 Low, 1 Medium, 2 High) per value,
out-of-fold over an inner StratifiedKFold(5, seed 42 + fold number 1-5) of the training rows and from all training
rows for the validation / test rows.
Model: CatBoost GPU, MultiClass with eval_metric TotalF1, auto_class_weights Balanced, depth 7, lr 0.03, l2 8,
Bernoulli subsample 0.8, min_data_in_leaf 24, random_strength 0.8, Cosine score, 10 Newton leaf iterations,
254 borders, up to 20,000 iterations with early stopping (500) on the validation fold, seed 42.
Test = mean of the 5 fold models.
Usage: python train_radix_nested_te.py"""
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')
T0 = time.perf_counter()

PAIR = 'catboost_radix_bins_nested_te_b'
SEED = 42
FOLDS = 5

NUMS = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
        'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']
CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']

# radix key = clip(round(10 x value), 0, RADIX - 1) + (category code + 1) x RADIX
RADIX = 50_000

CAT_PARAMS = dict(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    auto_class_weights='Balanced',
    random_seed=SEED,
    iterations=20000,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=8.0,
    bootstrap_type='Bernoulli',
    subsample=0.8,
    grow_policy='SymmetricTree',
    min_data_in_leaf=24,
    random_strength=0.8,
    score_function='Cosine',
    leaf_estimation_iterations=10,
    leaf_estimation_method='Newton',
    border_count=254,
    od_type='Iter',
    od_wait=500,
    verbose=500,
    task_type='GPU',
)


def log(msg):
    print(f'[{time.perf_counter() - T0:7.1f}s] {msg}', flush=True)


def nested_target_encode(X_tr, y_tr, X_apply, col, n_splits=5, seed=42):
    """Mean class index of `col`: out-of-fold on X_tr (inner split), from all of X_tr for X_apply."""
    inner_kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    te_tr = np.zeros(len(X_tr), dtype='float32')
    global_mean = float(np.mean(y_tr))
    for it, iv in inner_kf.split(X_tr, y_tr):
        fit_map = pd.Series(y_tr[it], index=range(len(it))).groupby(
            X_tr.iloc[it][col].values).mean()
        te_tr[iv] = X_tr.iloc[iv][col].map(fit_map).fillna(global_mean).values
    full_map = pd.Series(y_tr, index=range(len(y_tr))).groupby(
        X_tr[col].values).mean()
    te_apply = X_apply[col].map(full_map).fillna(global_mean).values.astype('float32')
    return te_tr, te_apply


def build_features(train_df, test_df, orig_df, orig_y_arr):
    combined = pd.concat([train_df.copy(), test_df.copy()], axis=0, ignore_index=True)
    n_tr = len(train_df)

    for c in NUMS:
        combined[c] = combined[c].astype('float32')

    # threshold score and flags
    sm = combined['Soil_Moisture'].values
    rf = combined['Rainfall_mm'].values
    tc = combined['Temperature_C'].values
    ws = combined['Wind_Speed_kmh'].values
    combined['_formula_score'] = (
        (sm < 25).astype(int) * 2 + (rf < 300).astype(int) * 2 +
        (tc > 30).astype(int) + (ws > 10).astype(int) -
        (combined['Crop_Growth_Stage'].values == 'Harvest').astype(int) * 2 -
        (combined['Crop_Growth_Stage'].values == 'Sowing').astype(int) * 2 -
        (combined['Mulching_Used'].values == 'Yes').astype(int)
    ).astype('float32')
    combined['_is_low'] = (combined['_formula_score'] <= 0).astype('float32')
    combined['_is_high'] = (combined['_formula_score'] > 3).astype('float32')

    # digit / frequency / log features
    for c in NUMS:
        v = combined[c].values
        vf = np.floor(np.abs(v)).astype('int32')
        combined[f'{c}_mod10'] = (vf % 10).astype('float32')
        combined[f'{c}_mod100'] = (vf % 100).astype('float32')
        combined[f'{c}_frac'] = (v - np.floor(v)).astype('float32')
        freq = combined[c].value_counts(normalize=True)
        combined[f'{c}_freq'] = combined[c].map(freq).fillna(0).astype('float32')
        combined[f'{c}_log1p'] = np.log1p(np.abs(v)).astype('float32')

    # quantile-bin strings
    bin_cols = []
    for c, n_bins in [('Soil_Moisture', 40), ('Rainfall_mm', 60),
                      ('Temperature_C', 40), ('Humidity', 40),
                      ('Previous_Irrigation_mm', 60)]:
        bin_name = f'{c}_bin'
        combined[bin_name] = pd.qcut(
            combined[c], q=n_bins, labels=False, duplicates='drop'
        ).astype('Int32').astype(str)
        bin_cols.append(bin_name)

    # arithmetic interactions
    combined['_moist_div_rain'] = (sm / (rf + 1e-6)).astype('float32')
    combined['_temp_div_humid'] = (tc / (combined['Humidity'].values + 1e-6)).astype('float32')
    combined['_rain_minus_moist'] = (rf - sm).astype('float32')

    # per-class original priors of the categoricals
    for col in CATS:
        for k in range(3):
            tmp = orig_df.copy()
            tmp['_yk'] = (orig_y_arr == k).astype(float)
            prior = tmp.groupby(col)['_yk'].mean()
            overall = (orig_y_arr == k).mean()
            combined[f'ORIG_p{k}_{col}'] = combined[col].map(prior).fillna(overall).astype('float32')

    # category pairs
    top_cats = ['Crop_Type', 'Season', 'Region', 'Crop_Growth_Stage']
    for i, c1 in enumerate(top_cats):
        for c2 in top_cats[i + 1:]:
            combined[f'BI_{c1}_{c2}'] = (
                combined[c1].astype(str) + '_' + combined[c2].astype(str)
            )

    # radix keys: one integer per (numeric value, category) pair, used as a native category
    radix_cols = []
    for num_col in NUMS:
        base_i = np.round(combined[num_col].values * 10).astype('int64')
        base_i = np.clip(base_i, 0, RADIX - 1).astype('int32')
        for cat_col in CATS:
            cat_codes = pd.factorize(
                combined[cat_col].astype(str).fillna('<NA>'), sort=False
            )[0].astype('int32')
            rname = f'RX_{num_col}_{cat_col}'
            combined[rname] = (base_i + (cat_codes + 1) * RADIX).astype('int32')
            radix_cols.append(rname)

    train_out = combined.iloc[:n_tr].reset_index(drop=True)
    test_out = combined.iloc[n_tr:].reset_index(drop=True)

    bigram_cols = [c for c in combined.columns if c.startswith('BI_')]
    cat_cols = CATS + bin_cols + bigram_cols + radix_cols
    for c in radix_cols:
        train_out[c] = train_out[c].astype(str)
        test_out[c] = test_out[c].astype(str)
    feat_cols = [c for c in train_out.columns]
    log(f'Features: {len(feat_cols)} | cat={len(cat_cols)} | radix={len(radix_cols)} | bins={len(bin_cols)}')
    return train_out, test_out, cat_cols, feat_cols, bin_cols


def main():
    log('Loading data...')
    train = read_train().set_index(ID)
    test = read_test().set_index(ID)
    orig = read_original()

    y = train[TARGET].map(TARGET_MAP).astype(int).values
    y_orig = orig[TARGET].map(TARGET_MAP).astype(int).values
    train = train.drop(columns=[TARGET])
    orig = orig.drop(columns=[TARGET])
    for df in [train, test, orig]:
        for c in NUMS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
        for c in CATS:
            df[c] = df[c].fillna('Missing').astype(str)

    log('Building features...')
    train_feat, test_feat, cat_cols, feat_cols, bin_cols = build_features(train, test, orig, y_orig)
    nested_te_cols = CATS + bin_cols

    fold_ids = folds(y)
    oof_preds = np.zeros((len(train_feat), 3), dtype='float32')
    test_preds = np.zeros((len(test_feat), 3), dtype='float32')
    for k in range(FOLDS):
        fold = k + 1
        tr_idx = np.where(fold_ids != k)[0]
        val_idx = np.where(fold_ids == k)[0]
        log(f'--- Fold {k} ---')

        X_tr = train_feat.iloc[tr_idx][feat_cols].copy().reset_index(drop=True)
        X_val = train_feat.iloc[val_idx][feat_cols].copy().reset_index(drop=True)
        X_te = test_feat[feat_cols].copy().reset_index(drop=True)
        y_tr = y[tr_idx]
        y_val = y[val_idx]

        for c in nested_te_cols:
            if c in X_tr.columns:
                te_tr, te_va = nested_target_encode(X_tr, y_tr, X_val, c, n_splits=5, seed=SEED + fold)
                _, te_te = nested_target_encode(X_tr, y_tr, X_te, c, n_splits=5, seed=SEED + fold)
                nc = f'NTE_{c}'
                X_tr[nc] = te_tr
                X_val[nc] = te_va
                X_te[nc] = te_te

        fold_cat_cols = [c for c in cat_cols if c in X_tr.columns]
        tr_pool = Pool(X_tr, label=y_tr, cat_features=fold_cat_cols)
        val_pool = Pool(X_val, label=y_val, cat_features=fold_cat_cols)
        te_pool = Pool(X_te, cat_features=fold_cat_cols)

        model = CatBoostClassifier(**CAT_PARAMS)
        model.fit(tr_pool, eval_set=val_pool, use_best_model=True)

        val_prob = model.predict_proba(val_pool).astype('float32')
        test_prob = model.predict_proba(te_pool).astype('float32')
        oof_preds[val_idx] = val_prob
        test_preds += test_prob / FOLDS

        fold_acc = balanced_accuracy_score(y_val, np.argmax(val_prob, axis=1))
        log(f'Fold {k} balanced accuracy: {fold_acc:.5f} | best_iter: {model.best_iteration_}')

        del model, tr_pool, val_pool, te_pool
        gc.collect()

    log(f'{PAIR}: OOF balanced accuracy {balanced_accuracy_score(y, np.argmax(oof_preds, axis=1)):.6f}')
    save_pair(HERE, PAIR, oof_preds, test_preds)


if __name__ == '__main__':
    main()
