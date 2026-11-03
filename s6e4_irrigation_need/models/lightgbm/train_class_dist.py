#!/usr/bin/env python
"""LightGBM on percentile ranks / z-scores against the original class distributions, with per-fold multiclass
target encodings (pair lgbm_class_pctrank_zscore_te_b).

Label-free features (train + test rows; the original rows are a reference; 242 numeric columns):
  - a threshold score 2 (moisture < 25) + 2 (rainfall < 300) + (temperature > 30) + (wind > 10) - 2 (Harvest)
    - 2 (Sowing) - (mulching), the flags score <= 0 / score > 3 and the 4 threshold flags
  - for every column with > 7 distinct values (the 11 numerics): count over train + test, original count, their ratio,
    mean original class index (0 Low, 1 Medium, 2 High)
  - log1p(train frequency / original frequency) of all 19 columns
  - train means of soil moisture, rainfall, temperature, wind speed per Crop_Growth_Stage, Mulching_Used, Crop_Type,
    Season
  - every numeric against the original rows: percentile rank overall and within each class, z-score within each
    class, High - Low percentile-rank gap, |z High| - |z Low|
  - soil moisture, rainfall, temperature, wind speed: distances to the class quartiles (q25 / q50 / q75) of the
    original rows and |x - median Low| - |x - median High|
  - percentile rank of soil moisture within the original rows of the same Crop_Growth_Stage and class, and of
    rainfall within the same Mulching_Used and class
  - the 11 numerics
Encoded keys (65): the 8 categoricals; per numeric round(x) // 2 x 2, floor(x), <= 1000 quantile bins of x and of
log1p(|x|); the 10 pairs of Crop_Growth_Stage, Mulching_Used, Crop_Type, Season, Region and 3 consecutive triples.
Per fold: scikit-learn TargetEncoder (multiclass, cv 5, shuffle, seed 42, smooth 'auto') fitted on the training rows
(out-of-fold for them), 195 columns.
Model: LightGBM multiclass, class_weight {High: 5}, lr 0.01, max_depth 4, subsample 0.5, colsample_bytree 0.2,
lambda_l2 15, lambda_l1 10, seed 42, 32 threads, up to 20,000 rounds with early stopping (500) on the validation fold.
Test = mean of the 5 fold models.
Usage: python train_class_dist.py"""
import gc
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')
T0 = time.perf_counter()

PAIR = 'lgbm_class_pctrank_zscore_te_b'
SEED = 42
N_FOLDS = 5
DRIFT_EPS = 1e-6
HIGH_CARD_THRESHOLD = 7
QCUT_BINS = 1000

NUMS = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
        'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']
CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']

LGB_PARAMS = dict(
    objective='multiclass',
    num_class=3,
    metric='multi_logloss',
    n_estimators=20_000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.5,
    colsample_bytree=0.2,
    lambda_l2=15.0,
    lambda_l1=10.0,
    random_state=SEED,
    verbosity=-1,
    num_threads=32,
    device='cpu',
    class_weight={2: 5},   # upweight the rare High class
)


def log(msg):
    print(f'[{time.perf_counter() - T0:7.1f}s] {msg}', flush=True)


def pctrank_against(values, reference):
    ref = np.sort(np.asarray(reference))
    if len(ref) == 0:
        return np.zeros(len(values), dtype='float32')
    return (np.searchsorted(ref, values) / len(ref)).astype('float32')


def zscore_against(values, reference):
    ref = np.asarray(reference)
    if len(ref) == 0:
        return np.zeros(len(values), dtype='float32')
    mu, sigma = np.mean(ref), np.std(ref)
    if sigma == 0:
        return np.zeros(len(values), dtype='float32')
    return ((values - mu) / sigma).astype('float32')


def reset(df):
    return df.reset_index(drop=True)


def build_features():
    log('Loading data...')
    train = read_train().set_index(ID)
    test = read_test().set_index(ID)
    orig = read_original()
    train[TARGET] = train[TARGET].map(TARGET_MAP).astype(int)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype(int)
    y = train[TARGET].values
    y_orig = orig[TARGET].values

    log('Building features on train + test...')
    combined = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
    N_tr = len(train)
    feats = {}

    # threshold score and flags
    SM = combined['Soil_Moisture'].values
    RM = combined['Rainfall_mm'].values
    TC = combined['Temperature_C'].values
    WS = combined['Wind_Speed_kmh'].values
    CGS = combined['Crop_Growth_Stage'].values
    MU = combined['Mulching_Used'].values
    score = ((SM < 25).astype(int) * 2 + (RM < 300).astype(int) * 2 +
             (TC > 30).astype(int) + (WS > 10).astype(int) -
             (CGS == 'Harvest').astype(int) * 2 - (CGS == 'Sowing').astype(int) * 2 -
             (MU == 'Yes').astype(int))
    feats['formula_score'] = score.astype('float32')
    feats['is_low'] = (score <= 0).astype('float32')
    feats['is_high'] = (score > 3).astype('float32')
    feats['soil_lt_25'] = (SM < 25).astype('float32')
    feats['temp_gt_30'] = (TC > 30).astype('float32')
    feats['rain_lt_300'] = (RM < 300).astype('float32')
    feats['wind_gt_10'] = (WS > 10).astype('float32')

    # counts, original counts, ratios, original mean class index
    high_card_cols = [c for c in NUMS + CATS if combined[c].nunique() > HIGH_CARD_THRESHOLD]
    for c in high_card_cols:
        vc_tr = combined[c].value_counts()
        feats[f'CE_{c}'] = combined[c].map(vc_tr).fillna(0).astype('float32').values
        vc_or = orig[c].value_counts()
        feats[f'CE_ORIG_{c}'] = combined[c].map(vc_or).fillna(0).astype('float32').values
        feats[f'CE_RATIO_{c}'] = (feats[f'CE_{c}'] / (feats[f'CE_ORIG_{c}'] + DRIFT_EPS)).astype('float32')
        te_orig_mean = orig.groupby(c, dropna=False)[TARGET].mean().to_dict()
        fallback = float(np.mean(y_orig))
        feats[f'TE_ORIG_{c}'] = combined[c].map(te_orig_mean).fillna(fallback).astype('float32').values

    # numeric bins as string keys (target-encoded per fold)
    num_bins = {}
    for col in NUMS:
        s = combined[col]
        num_bins[f'{col}_round_div2_te'] = (np.round(s).fillna(-1).astype(int) // 2 * 2).astype(str)
        num_bins[f'{col}_int_te'] = np.floor(s.fillna(-1)).astype(int).astype(str)
        try:
            q_eff = int(min(QCUT_BINS, max(2, s.nunique(dropna=True))))
            num_bins[f'{col}_quantiled_te'] = pd.qcut(s, q=q_eff, duplicates='drop').astype(str)
        except Exception:
            num_bins[f'{col}_quantiled_te'] = pd.Series(['NA'] * len(combined))
        log_s = np.log1p(np.abs(s.fillna(0)))
        try:
            q_eff2 = int(min(QCUT_BINS, max(2, log_s.nunique(dropna=True))))
            num_bins[f'{col}_log_binned_te'] = pd.qcut(log_s, q=q_eff2, duplicates='drop').astype(str)
        except Exception:
            num_bins[f'{col}_log_binned_te'] = pd.Series(['NA'] * len(combined))
    num_bins_df = reset(pd.DataFrame(num_bins))

    # category bigrams and consecutive trigrams (string keys)
    ngram_cols = ['Crop_Growth_Stage', 'Mulching_Used', 'Crop_Type', 'Season', 'Region']
    ngram_dict = {}
    for c1, c2 in combinations(ngram_cols, 2):
        ngram_dict[f'BI_{c1}_{c2}_te'] = combined[c1].astype(str) + '|' + combined[c2].astype(str)
    for i in range(len(ngram_cols) - 2):
        c1, c2, c3 = ngram_cols[i], ngram_cols[i + 1], ngram_cols[i + 2]
        ngram_dict[f'TRI_{c1}_{c2}_{c3}_te'] = (
            combined[c1].astype(str) + '|' + combined[c2].astype(str) + '|' + combined[c3].astype(str)
        )
    ngram_df = reset(pd.DataFrame(ngram_dict))

    # train vs original frequency drift
    for c in NUMS + CATS:
        p_tr = combined[:N_tr][c].value_counts(normalize=True)
        p_or = orig[c].value_counts(normalize=True)
        feats[f'DRIFT_{c}'] = np.log1p(
            combined[c].map(p_tr).fillna(0) / (combined[c].map(p_or).fillna(DRIFT_EPS) + DRIFT_EPS)
        ).astype('float32').values

    # group means from the train rows
    key_cats = ['Crop_Growth_Stage', 'Mulching_Used', 'Crop_Type', 'Season']
    key_nums = ['Soil_Moisture', 'Rainfall_mm', 'Temperature_C', 'Wind_Speed_kmh']
    tr_combined = combined.iloc[:N_tr]
    for cat in key_cats:
        for num in key_nums:
            name = f'MEAN_{num}_BY_{cat}'
            m = tr_combined.groupby(cat, dropna=False)[num].mean()
            feats[name] = combined[cat].map(m).fillna(float(train[num].mean())).astype('float32').values

    # percentile ranks / z-scores against the original class distributions
    for col in NUMS:
        vals = combined[col].values
        orig_all = orig[col].values
        refs = {k: orig.loc[y_orig == k, col].values for k in range(3)}
        feats[f'X_pctrank_all_{col}'] = pctrank_against(vals, orig_all)
        for k in range(3):
            feats[f'X_pctrank_c{k}_{col}'] = pctrank_against(vals, refs[k])
            feats[f'X_zscore_c{k}_{col}'] = zscore_against(vals, refs[k])
        feats[f'X_pctrank_gap_HL_{col}'] = (
            pctrank_against(vals, refs[2]) - pctrank_against(vals, refs[0])
        ).astype('float32')
        feats[f'X_zscore_gap_HL_{col}'] = (
            np.abs(zscore_against(vals, refs[2])) - np.abs(zscore_against(vals, refs[0]))
        ).astype('float32')

    # distances to the original class quartiles
    for col in ['Soil_Moisture', 'Rainfall_mm', 'Temperature_C', 'Wind_Speed_kmh']:
        vals = combined[col].values
        for k in range(3):
            ref = orig.loc[y_orig == k, col].values
            for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
                q = np.quantile(ref, q_val)
                feats[f'X_dist_c{k}_{q_label}_{col}'] = np.abs(vals - q).astype('float32')
        qH = np.quantile(orig.loc[y_orig == 2, col].values, 0.50)
        qL = np.quantile(orig.loc[y_orig == 0, col].values, 0.50)
        feats[f'X_qdist_gap_HL_{col}'] = (np.abs(vals - qL) - np.abs(vals - qH)).astype('float32')

    # percentile rank within the original rows of the same category and class
    for cat_col, num_col in [('Crop_Growth_Stage', 'Soil_Moisture'), ('Mulching_Used', 'Rainfall_mm')]:
        for k in range(3):
            cond = np.zeros(len(combined), dtype='float32')
            for cat_val in orig[cat_col].dropna().unique():
                mask = (combined[cat_col] == cat_val).values
                ref = orig.loc[(orig[cat_col] == cat_val) & (y_orig == k), num_col].values
                if len(ref) > 0 and mask.sum() > 0:
                    cond[mask] = pctrank_against(combined.loc[mask, num_col].values, ref)
            feats[f'X_cond_pctrank_{cat_col}_{num_col}_c{k}'] = cond

    # raw numerics; the categoricals are kept as strings for the encoder
    for col in NUMS:
        feats[col] = combined[col].values.astype('float32')
    for col in CATS:
        feats[col] = combined[col].values

    base_df = reset(pd.DataFrame(feats))
    base_df = pd.concat([base_df, num_bins_df, ngram_df], axis=1)
    te_cols = [c for c in base_df.columns if c.endswith('_te') or c in CATS]
    other_cols = [c for c in base_df.columns if c not in te_cols]
    log(f'Total features: {len(base_df.columns)}  (te_cols={len(te_cols)}, numeric={len(other_cols)})')

    X_base_tr = base_df.iloc[:N_tr].reset_index(drop=True)
    X_base_te = base_df.iloc[N_tr:].reset_index(drop=True)
    return X_base_tr, X_base_te, y, te_cols, other_cols


def main():
    X_base_tr, X_base_te, y, te_cols, other_cols = build_features()

    fold_ids = folds(y)
    oof_probs = np.zeros((len(X_base_tr), 3), dtype=np.float32)
    test_probs = np.zeros((len(X_base_te), 3), dtype=np.float32)
    for k in range(N_FOLDS):
        tr_idx = np.where(fold_ids != k)[0]
        val_idx = np.where(fold_ids == k)[0]
        log(f'---- Fold {k} ----')

        X_tr_fold = X_base_tr.iloc[tr_idx].reset_index(drop=True)
        y_tr_fold = y[tr_idx]
        X_val_fold = X_base_tr.iloc[val_idx].reset_index(drop=True)
        y_val_fold = y[val_idx]
        X_te_fold = X_base_te.copy()

        enc = TargetEncoder(target_type='multiclass', cv=5, random_state=SEED)
        te_enc_tr = enc.fit_transform(X_tr_fold[te_cols].astype(str), y_tr_fold)
        te_enc_val = enc.transform(X_val_fold[te_cols].astype(str))
        te_enc_te = enc.transform(X_te_fold[te_cols].astype(str))
        te_names = [f'TE_{i}' for i in range(te_enc_tr.shape[1])]

        X_tr_final = X_tr_fold[other_cols].copy()
        X_val_final = X_val_fold[other_cols].copy()
        X_te_final = X_te_fold[other_cols].copy()
        X_tr_final[te_names] = te_enc_tr.astype(np.float32)
        X_val_final[te_names] = te_enc_val.astype(np.float32)
        X_te_final[te_names] = te_enc_te.astype(np.float32)

        model = lgbm.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_tr_final, y_tr_fold,
            eval_set=[(X_val_final, y_val_fold)],
            callbacks=[lgbm.early_stopping(500, verbose=False), lgbm.log_evaluation(2000)],
        )

        oof_probs[val_idx] = model.predict_proba(X_val_final).astype(np.float32)
        test_probs += model.predict_proba(X_te_final).astype(np.float32) / N_FOLDS
        fold_bac = balanced_accuracy_score(y_val_fold, np.argmax(oof_probs[val_idx], axis=1))
        log(f'Fold {k} BAC={fold_bac:.5f}  best_iter={model.best_iteration_}')

        del X_tr_final, X_val_final, X_te_final, model
        gc.collect()

    log(f'{PAIR}: OOF balanced accuracy {balanced_accuracy_score(y, np.argmax(oof_probs, axis=1)):.6f}')
    save_pair(HERE, PAIR, oof_probs, test_probs)


if __name__ == '__main__':
    main()
