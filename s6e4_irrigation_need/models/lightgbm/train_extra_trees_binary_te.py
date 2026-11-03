#!/usr/bin/env python
"""Class-weighted extra-trees LightGBM on rounding / digit / bin artifacts with a two-stage nested per-class target
encoding (pair lgbm_extra_trees_two_stage_binary_te_b).

Label-free features (175 columns):
  - the 19 columns
  - soil moisture, rainfall, temperature, wind speed rounded to 1 and to 10
  - every numeric: the 3 lowest integer digits and the 2 first decimal digits
  - the 6 pairs of the units digits of soil moisture, rainfall, temperature, wind speed (strings) and the 4 units
    digits as strings
  - 32- and 64-quantile bins of the same 4 numerics (edges from the train rows)
  - 8 category pairs (strings)
  - a threshold score 2 (moisture < 25) + 2 (rainfall < 300) + (temperature > 30) + (wind > 10) - 2 (Harvest)
    - 2 (Sowing) - (mulching) and the flags score <= 0 / score > 3
  - per-class original frequency and original count of the 8 categoricals, 4 rounded values and 4 bins
Per fold, a binary scikit-learn TargetEncoder (smooth 'auto') of each class indicator, out-of-fold over an inner
StratifiedKFold(5, stratified on the indicator) of the training rows and fitted on all training rows for the
validation / test rows:
  - stage 1 (seed 42 + fold number 1-5): the 8 category pairs, the 6 digit pairs and the 4 digit strings, which are
    then dropped
  - stage 2 (seed 142 + fold number): the 11 numerics, the first 8 digit columns, the rounded values, the bins and the
    first 12 original-prior columns, which are kept
The remaining string columns (the 8 categoricals) are factorized over the fold's training + validation + test rows and
missing values set to -1.
Model: LightGBM multiclass, extra_trees, balanced class sample weights, 141 leaves, max_depth 7, lr 0.01,
bagging_fraction 0.7, feature_fraction 0.17, colsample_bynode 0.925, min_data_in_leaf 63, lambda_l1 3.3,
lambda_l2 5.7, max_bin 1008, 32 threads (default seed), up to 20,000 rounds with early stopping (200) on the
validation fold. Test = mean of the 5 fold models.
Usage: python train_extra_trees_binary_te.py"""
import gc
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'lgbm_extra_trees_two_stage_binary_te_b'
SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5
LABEL = 'target'

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Organic_Carbon', 'Electrical_Conductivity', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Temperature_C', 'Soil_Moisture',
        'Humidity', 'Previous_Irrigation_mm', 'Rainfall_mm']
FORMULA_NUMS = ['Soil_Moisture', 'Rainfall_mm', 'Temperature_C', 'Wind_Speed_kmh']
CAT_PAIRS = [
    ('Crop_Growth_Stage', 'Mulching_Used'), ('Crop_Growth_Stage', 'Season'),
    ('Crop_Type', 'Soil_Type'), ('Crop_Type', 'Season'),
    ('Crop_Type', 'Crop_Growth_Stage'), ('Region', 'Season'),
    ('Irrigation_Type', 'Water_Source'), ('Soil_Type', 'Region'),
]

LGB_PARAMS = dict(
    objective='multiclass',
    num_class=3,
    metric='multi_logloss',
    boosting_type='gbdt',
    n_estimators=20000,
    num_leaves=141,
    max_depth=7,
    learning_rate=0.01,
    bagging_fraction=0.7,
    colsample_bynode=0.925,
    feature_fraction=0.17,
    min_data_in_leaf=63,
    lambda_l1=3.3,
    lambda_l2=5.7,
    extra_trees=True,
    max_bin=1008,
    n_jobs=32,
    verbose=-1,
)


def build_features():
    train = read_train()
    test = read_test()
    orig = read_original()
    train[LABEL] = train[TARGET].map(TARGET_MAP).astype(np.int32)
    orig[LABEL] = orig[TARGET].map(TARGET_MAP).astype(np.int32)
    y = train[LABEL].values
    y_orig = orig[LABEL].values

    for c in [ID, TARGET]:
        for df in [train, test]:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
    orig.drop(columns=[ID], inplace=True, errors='ignore')
    print(f'Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}', flush=True)

    t0 = time.time()
    # rounded values
    round_cols = []
    for c in FORMULA_NUMS:
        for label, dec in [('1s', 0), ('10s', -1)]:
            name = f'{c}_round_{label}'
            for df in [train, test, orig]:
                df[name] = np.round(df[c].values, dec).astype(np.float32)
            round_cols.append(name)

    # integer and decimal digits
    digit_cols = []
    for c in NUMS:
        allv = pd.concat([train[c], test[c], orig[c]], ignore_index=True).fillna(0).astype(np.float64)
        n1, n2 = len(train), len(test)
        int_part = np.floor(np.abs(allv.values)).astype(np.int64)
        for i in range(3):
            name = f'{c}_digit{i}'
            vals = ((int_part // (10 ** i)) % 10).astype(np.int8)
            train[name] = vals[:n1]
            test[name] = vals[n1:n1 + n2]
            orig[name] = vals[n1 + n2:]
            digit_cols.append(name)
        frac = np.abs(allv.values) - np.floor(np.abs(allv.values))
        for i in range(2):
            name = f'{c}_dec_digit{i}'
            vals = (np.floor(frac * 10 ** (i + 1)) % 10).astype(np.int8)
            train[name] = vals[:n1]
            test[name] = vals[n1:n1 + n2]
            orig[name] = vals[n1 + n2:]
            digit_cols.append(name)

    # pairs of units digits (strings)
    digit_inter_cols = []
    key_digits = [f'{c}_digit0' for c in FORMULA_NUMS]
    for c1, c2 in combinations(key_digits, 2):
        name = f'DI_{c1}_{c2}'
        for df in [train, test, orig]:
            df[name] = df[c1].astype(str) + '_' + df[c2].astype(str)
        digit_inter_cols.append(name)

    # quantile bins (edges from the train rows)
    binning_cols = []
    for c in FORMULA_NUMS:
        for q in [32, 64]:
            name = f'{c}_qbin_{q}'
            try:
                _, bins = pd.qcut(train[c], q=q, labels=False, retbins=True, duplicates='drop')
                for df in [train, test, orig]:
                    df[name] = pd.cut(df[c], bins=bins, labels=False, include_lowest=True).fillna(-1).astype('int32')
            except Exception:
                for df in [train, test, orig]:
                    df[name] = 0
            binning_cols.append(name)

    # category pairs (strings)
    catpair_cols = []
    for a, b in CAT_PAIRS:
        name = f'CP_{a}_{b}'
        for df in [train, test, orig]:
            df[name] = df[a].astype(str) + '_' + df[b].astype(str)
        catpair_cols.append(name)

    # units digits as strings
    digit_cat_cols = []
    for c in [f'{n}_digit0' for n in FORMULA_NUMS]:
        name = f'DCAT_{c}'
        for df in [train, test, orig]:
            df[name] = df[c].astype(str)
        digit_cat_cols.append(name)

    # threshold score
    for df in [train, test, orig]:
        SM = df['Soil_Moisture']; RM = df['Rainfall_mm']; TC = df['Temperature_C']
        WS = df['Wind_Speed_kmh']; CGS = df['Crop_Growth_Stage']; MU = df['Mulching_Used']
        score = ((SM < 25) * 2 + (RM < 300) * 2 + (TC > 30) + (WS > 10)
                 - (CGS == 'Harvest') * 2 - (CGS == 'Sowing') * 2 - (MU == 'Yes'))
        df['formula_score'] = score.astype(np.float32)
        df['is_low'] = (score <= 0).astype(np.float32)
        df['is_high'] = (score > 3).astype(np.float32)
    formula_fe = ['formula_score', 'is_low', 'is_high']

    # per-class original frequencies and counts
    orig_cols = []
    safe_keys = CATS + round_cols[:4] + binning_cols[:4]
    for c in safe_keys:
        for k in range(3):
            m = orig.groupby(c).apply(lambda g, kk=k: (g[LABEL] == kk).mean())
            gm = float((y_orig == k).mean())
            name = f'ORIG_k{k}_{c}'
            for df in [train, test, orig]:
                df[name] = df[c].map(m).fillna(gm).astype(np.float32)
            orig_cols.append(name)
        cnt = orig.groupby(c).size()
        name = f'ORIG_cnt_{c}'
        for df in [train, test, orig]:
            df[name] = df[c].map(cnt).fillna(0).astype(np.float32)
        orig_cols.append(name)

    features = (NUMS + CATS + round_cols + digit_cols + digit_inter_cols + digit_cat_cols + binning_cols
                + catpair_cols + formula_fe + orig_cols)
    features = list(dict.fromkeys(features))
    te_feat = [c for c in catpair_cols + digit_inter_cols + digit_cat_cols if c in features]   # stage 1: dropped
    base_te_col = [c for c in NUMS + digit_cols[:8] + round_cols + binning_cols + orig_cols[:12] if c in features]  # stage 2: kept
    print(f'Features: {len(features)}, stage-1 TE: {len(te_feat)}, stage-2 TE: {len(base_te_col)} '
          f'({time.time() - t0:.1f}s)', flush=True)
    return train[features].copy(), test[features].copy(), y, te_feat, base_te_col


def _te_to_1d(x):
    if isinstance(x, np.ndarray):
        return x[:, 0].astype(np.float32) if x.ndim == 2 else x.astype(np.float32)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.iloc[:, 0].astype(np.float32).values if hasattr(x, 'iloc') and x.ndim == 2 else x.astype(np.float32).values
    return np.asarray(x).astype(np.float32)


def nested_multiclass_te(X_tr, y_tr, X_va, X_te, te_cols, inner_folds=5, seed=42, drop_original=False):
    X_tr, X_va, X_te = X_tr.copy(), X_va.copy(), X_te.copy()
    made = []
    skf_inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    for c in te_cols:
        if c not in X_tr.columns:
            continue
        tr_col = X_tr[c].astype('string').fillna('<NA>').to_frame(name=c)
        va_col = X_va[c].astype('string').fillna('<NA>').to_frame(name=c)
        te_col = X_te[c].astype('string').fillna('<NA>').to_frame(name=c)
        for k in range(3):
            y_k = (y_tr == k).astype(np.int32)
            oof_enc = np.zeros(len(X_tr), dtype=np.float32)
            for itr, iva in skf_inner.split(tr_col, y_k):
                enc = TargetEncoder(target_type='binary', smooth='auto')
                enc.fit(tr_col.iloc[itr], y_k.iloc[itr])
                oof_enc[iva] = _te_to_1d(enc.transform(tr_col.iloc[iva]))
            enc_full = TargetEncoder(target_type='binary', smooth='auto')
            enc_full.fit(tr_col, y_k)
            newc = f'TE_k{k}_{c}'
            X_tr[newc] = oof_enc
            X_va[newc] = _te_to_1d(enc_full.transform(va_col))
            X_te[newc] = _te_to_1d(enc_full.transform(te_col))
            made.append(newc)
        if drop_original:
            for df in [X_tr, X_va, X_te]:
                df.drop(columns=[c], inplace=True, errors='ignore')
    return X_tr, X_va, X_te, made


def factorize_tvt(tr, va, te, cols):
    tr, va, te = tr.copy(), va.copy(), te.copy()
    for c in cols:
        if c not in tr.columns:
            continue
        allv = pd.concat([tr[c].astype(str), va[c].astype(str), te[c].astype(str)], ignore_index=True)
        codes, _ = pd.factorize(allv, sort=False)
        n1, n2 = len(tr), len(va)
        tr[c] = codes[:n1].astype('int32')
        va[c] = codes[n1:n1 + n2].astype('int32')
        te[c] = codes[n1 + n2:].astype('int32')
    return tr, va, te


def main():
    X, X_test_all, y, te_feat, base_te_col = build_features()
    cls_counts = np.bincount(y)
    cls_weights = cls_counts.sum() / (3 * cls_counts)

    fold_ids = folds(y)
    oof_preds = np.zeros((len(X), 3), dtype=np.float32)
    test_preds = np.zeros((len(X_test_all), 3), dtype=np.float32)
    for k in range(N_FOLDS):
        fold = k + 1
        tr_idx = np.where(fold_ids != k)[0]
        va_idx = np.where(fold_ids == k)[0]
        print(f'Fold {k}', flush=True)
        t0 = time.time()

        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        X_va = X.iloc[va_idx].reset_index(drop=True)
        y_tr = pd.Series(y[tr_idx])
        y_va = y[va_idx]
        X_te = X_test_all.copy()

        # stage 1: category pairs and digit keys (originals dropped)
        X_tr, X_va, X_te, _ = nested_multiclass_te(
            X_tr, y_tr, X_va, X_te, te_feat,
            inner_folds=INNER_FOLDS, seed=SEED + fold, drop_original=True)
        # stage 2: numerics, digits, rounded values, bins, original priors (originals kept)
        X_tr, X_va, X_te, _ = nested_multiclass_te(
            X_tr, y_tr, X_va, X_te, base_te_col,
            inner_folds=INNER_FOLDS, seed=SEED + fold + 100, drop_original=False)

        remaining_cats = [c for c in X_tr.columns if X_tr[c].dtype.name in ['object', 'string']]
        X_tr, X_va, X_te = factorize_tvt(X_tr, X_va, X_te, remaining_cats)
        features = X_tr.columns.tolist()
        for df in [X_tr, X_va, X_te]:
            for c in features:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(-1).astype(np.float32)

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_tr[features], y_tr,
            sample_weight=cls_weights[y_tr.values],
            eval_set=[(X_va[features], y_va)],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)],
        )

        p_va = model.predict_proba(X_va[features], num_iteration=model.best_iteration_).astype(np.float32)
        p_te = model.predict_proba(X_te[features], num_iteration=model.best_iteration_).astype(np.float32)
        oof_preds[va_idx] = p_va
        test_preds += p_te / N_FOLDS
        bac = balanced_accuracy_score(y_va, np.argmax(p_va, axis=1))
        print(f'  Fold {k} BAC={bac:.5f} best_iter={model.best_iteration_} n_feat={len(features)} '
              f'({time.time() - t0:.0f}s)', flush=True)

        del X_tr, X_va, X_te, model
        gc.collect()

    print(f'{PAIR}: OOF balanced accuracy {balanced_accuracy_score(y, np.argmax(oof_preds, axis=1)):.6f}', flush=True)
    save_pair(HERE, PAIR, oof_preds, test_preds)


if __name__ == '__main__':
    main()
