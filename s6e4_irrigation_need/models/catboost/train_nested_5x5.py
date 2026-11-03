#!/usr/bin/env python
"""Class-weighted CatBoost in nested 5x5 cross-validation on frequency / original-prior / n-gram features
(pair catboost_nested5x5_freq_priors_ngrams_b).

Label-free features:
  - the 19 columns
  - frequency of every numeric and categorical value over the train + original rows
  - soil moisture / (rainfall + 1), temperature / (humidity + 1), soil moisture / (temperature + 1)
  - flags: temperature >= 30, soil moisture < 20, rainfall == 0
  - per-class original frequency of the 8 categoricals and the 11 numerics (missing: 0.33)
  - soil moisture against the original rows of each class: percentile rank, z-score, and the distance to the class
    quartiles (q25 / q50 / q75)
  - every numeric value as a string; the 15 pairs of the first 6 categoricals and the 4 triples of the first 4
Native CatBoost categories: the 8 categoricals, the 11 numeric strings and the 19 n-grams.
Nested CV: for outer fold k (the shared 5 folds), an inner StratifiedKFold(5, seed 1000 + k + 1) of its training rows;
each inner model (seed 42 + inner fold number 1-5) predicts its inner validation rows, the outer validation rows and
the test rows. OOF slice k = the inner out-of-fold predictions on outer fold k's training rows and the mean of the 5
inner models on its validation rows; test = mean of all 25 models.
Model: CatBoost GPU, MultiClass, auto_class_weights Balanced, depth 6, lr 0.02, up to 15,000 iterations with early
stopping (500) on the inner validation rows.
Writes an OOF of shape (5, n_train, 3).
Usage: python train_nested_5x5.py"""
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
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "catboost_nested5x5_freq_priors_ngrams_b"
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5

NUMS = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity', 'Temperature_C', 'Humidity',
        'Rainfall_mm', 'Sunlight_Hours', 'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']
CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 'Irrigation_Type', 'Water_Source', 'Mulching_Used',
        'Region']

CB_PARAMS = {
    'iterations': 15000,
    'learning_rate': 0.02,
    'depth': 6,
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'auto_class_weights': 'Balanced',
    'task_type': 'GPU',
    'random_seed': 42,
    'early_stopping_rounds': 500,
    'verbose': False,
}


def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / (len(ref_sorted) + 1e-6)).astype('float32')


def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    return (np.zeros(len(values), dtype='float32') if sigma == 0 else ((values - mu) / sigma).astype('float32'))


def build_features():
    print("Loading data...", flush=True)
    train = read_train().set_index(ID)
    test = read_test().set_index(ID)
    orig = read_original()
    train[TARGET] = train[TARGET].map(TARGET_MAP)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP)

    new_nums = []
    num_as_cat = []

    # frequency encoding
    for col in NUMS + CATS:
        freq = pd.concat([train[col], orig[col]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        new_nums.append(f'FREQ_{col}')

    # arithmetic interactions
    for df in [train, test, orig]:
        df['moisture_rain_ratio'] = (df['Soil_Moisture'] / (df['Rainfall_mm'] + 1)).astype('float32')
        df['temp_humidity_ratio'] = (df['Temperature_C'] / (df['Humidity'] + 1)).astype('float32')
        df['moisture_temp_ratio'] = (df['Soil_Moisture'] / (df['Temperature_C'] + 1)).astype('float32')
    new_nums += ['moisture_rain_ratio', 'temp_humidity_ratio', 'moisture_temp_ratio']

    # flags
    for df in [train, test, orig]:
        df['is_high_temp'] = (df['Temperature_C'] >= 30).astype('float32')
        df['is_low_moisture'] = (df['Soil_Moisture'] < 20).astype('float32')
        df['is_no_rain'] = (df['Rainfall_mm'] == 0).astype('float32')
    new_nums += ['is_high_temp', 'is_low_moisture', 'is_no_rain']

    # per-class original priors
    for class_idx in [0, 1, 2]:
        orig[f'target_{class_idx}'] = (orig[TARGET] == class_idx).astype(int)
    for col in CATS + NUMS:
        for class_idx in [0, 1, 2]:
            tmp = orig.groupby(col)[f'target_{class_idx}'].mean().astype('float32')
            name = f"ORIG_proba_c{class_idx}_{col}"
            train = train.merge(tmp.rename(name), on=col, how="left")
            test = test.merge(tmp.rename(name), on=col, how="left")
            for df in [train, test]:
                df[name] = df[name].fillna(0.33).astype('float32')
            new_nums.append(name)

    # soil moisture against the original class distributions
    orig_sm_c0 = orig.loc[orig[TARGET] == 0, 'Soil_Moisture'].values
    orig_sm_c1 = orig.loc[orig[TARGET] == 1, 'Soil_Moisture'].values
    orig_sm_c2 = orig.loc[orig[TARGET] == 2, 'Soil_Moisture'].values
    for df in [train, test]:
        sm = df['Soil_Moisture'].values
        df['pctrank_c0_SM'] = pctrank_against(sm, orig_sm_c0)
        df['pctrank_c1_SM'] = pctrank_against(sm, orig_sm_c1)
        df['pctrank_c2_SM'] = pctrank_against(sm, orig_sm_c2)
        df['zscore_c0_SM'] = zscore_against(sm, orig_sm_c0)
        df['zscore_c1_SM'] = zscore_against(sm, orig_sm_c1)
        df['zscore_c2_SM'] = zscore_against(sm, orig_sm_c2)
    new_nums += ['pctrank_c0_SM', 'pctrank_c1_SM', 'pctrank_c2_SM', 'zscore_c0_SM', 'zscore_c1_SM', 'zscore_c2_SM']

    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        c0_q = np.quantile(orig_sm_c0, q_val) if len(orig_sm_c0) > 0 else 0
        c1_q = np.quantile(orig_sm_c1, q_val) if len(orig_sm_c1) > 0 else 0
        c2_q = np.quantile(orig_sm_c2, q_val) if len(orig_sm_c2) > 0 else 0
        for df in [train, test]:
            df[f'dist_To_c0_{q_label}'] = np.abs(df['Soil_Moisture'] - c0_q).astype('float32')
            df[f'dist_To_c1_{q_label}'] = np.abs(df['Soil_Moisture'] - c1_q).astype('float32')
            df[f'dist_To_c2_{q_label}'] = np.abs(df['Soil_Moisture'] - c2_q).astype('float32')
        new_nums += [f'dist_To_c0_{q_label}', f'dist_To_c1_{q_label}', f'dist_To_c2_{q_label}']

    # numerics as categories
    for col in NUMS:
        name = f'CAT_{col}'
        num_as_cat.append(name)
        for df in [train, test]:
            df[name] = df[col].astype(str)

    # category bigrams and trigrams
    top_cats = CATS[:6]
    bigram_cols = []
    trigram_cols = []
    for c1, c2 in combinations(top_cats, 2):
        name = f"BG_{c1}_{c2}"
        for df in [train, test]:
            df[name] = df[c1].astype(str) + "_" + df[c2].astype(str)
        bigram_cols.append(name)
    for c1, c2, c3 in combinations(top_cats[:4], 3):
        name = f"TG_{c1}_{c2}_{c3}"
        for df in [train, test]:
            df[name] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)
        trigram_cols.append(name)

    cb_features = NUMS + new_nums + CATS + num_as_cat + bigram_cols + trigram_cols
    cb_cat_features = CATS + num_as_cat + bigram_cols + trigram_cols
    for df in [train, test]:
        for col in cb_cat_features:
            df[col] = df[col].astype(str)
    cb_cat_idx = [train[cb_features].columns.get_loc(c) for c in cb_cat_features]
    return train, test, cb_features, cb_cat_idx


def main():
    train, test, cb_features, cb_cat_idx = build_features()
    y = train[TARGET].values
    n_train, n_test = len(train), len(test)

    fold_ids = folds(y)
    oof_matrix = np.zeros((N_OUTER_SPLITS, n_train, 3), dtype=np.float32)
    test_pred = np.zeros((n_test, 3), dtype=np.float32)
    for k in range(N_OUTER_SPLITS):
        outer_fold = k + 1
        outer_tr_idx = np.where(fold_ids != k)[0]
        outer_va_idx = np.where(fold_ids == k)[0]
        print(f"Outer fold {k}: {len(outer_tr_idx):,} training rows, {len(outer_va_idx):,} validation rows", flush=True)

        X_outer_tr = train.iloc[outer_tr_idx][cb_features].copy()
        y_outer_tr = train.iloc[outer_tr_idx][TARGET].copy()
        X_outer_va = train.iloc[outer_va_idx][cb_features].copy()
        y_outer_va = train.iloc[outer_va_idx][TARGET].copy()
        X_te = test[cb_features].copy()

        outer_full_pred = np.zeros((n_train, 3), dtype=np.float32)
        outer_va_pred_sum = np.zeros((len(outer_va_idx), 3), dtype=np.float32)
        skf_inner = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=1000 + outer_fold)
        outer_va_pool = Pool(X_outer_va, cat_features=cb_cat_idx)
        test_pool = Pool(X_te, cat_features=cb_cat_idx)

        for inner_fold, (inner_tr_loc, inner_va_loc) in enumerate(skf_inner.split(X_outer_tr, y_outer_tr), start=1):
            X_tr = X_outer_tr.iloc[inner_tr_loc]
            y_tr = y_outer_tr.iloc[inner_tr_loc]
            X_va = X_outer_tr.iloc[inner_va_loc]
            y_va = y_outer_tr.iloc[inner_va_loc]
            train_pool = Pool(X_tr, y_tr, cat_features=cb_cat_idx)
            val_pool = Pool(X_va, y_va, cat_features=cb_cat_idx)

            params = CB_PARAMS.copy()
            params['random_seed'] = 42 + inner_fold
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool)

            p_inner_va = model.predict_proba(val_pool).astype(np.float32)
            outer_full_pred[outer_tr_idx[inner_va_loc]] = p_inner_va
            inner_acc = balanced_accuracy_score(y_va, np.argmax(p_inner_va, axis=1))
            print(f"  outer {k} inner {inner_fold}: balanced accuracy {inner_acc:.5f}", flush=True)

            outer_va_pred_sum += model.predict_proba(outer_va_pool).astype(np.float32)
            test_pred += model.predict_proba(test_pool).astype(np.float32) / (N_OUTER_SPLITS * N_INNER_SPLITS)

            del X_tr, X_va, y_tr, y_va, train_pool, val_pool, model, p_inner_va
            gc.collect()

        outer_va_pred = outer_va_pred_sum / N_INNER_SPLITS
        outer_full_pred[outer_va_idx] = outer_va_pred
        oof_matrix[k] = outer_full_pred
        outer_va_acc = balanced_accuracy_score(y_outer_va, np.argmax(outer_va_pred, axis=1))
        print(f"Outer fold {k}: validation balanced accuracy {outer_va_acc:.5f}", flush=True)

        del X_outer_tr, y_outer_tr, X_outer_va, y_outer_va, X_te, outer_va_pool, test_pool
        del outer_full_pred, outer_va_pred_sum, outer_va_pred
        gc.collect()

    oof_mean = oof_matrix.mean(axis=0)
    print(f"{PAIR}: balanced accuracy of the slice mean {balanced_accuracy_score(y, np.argmax(oof_mean, axis=1)):.6f}",
          flush=True)
    save_pair(HERE, PAIR, oof_matrix, test_pred)


if __name__ == "__main__":
    main()
