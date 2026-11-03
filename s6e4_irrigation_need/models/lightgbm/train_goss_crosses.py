#!/usr/bin/env python
"""Class-weighted GOSS LightGBM on transforms, category crosses, bins, digits and original priors, with nested
per-class target encodings (pair lgbm_goss_crosses_binary_te_b).

Label-free features (242 columns):
  - the 19 columns
  - frequency and percentile rank (over train + test + original) and log1p, sqrt, 1 / (1 + x) of every numeric
  - a threshold score 2 (moisture < 25) + 2 (rainfall < 300) + (temperature > 30) + (wind > 10) - 2 (Harvest)
    - 2 (Sowing) - (mulching), the flags score <= 0 / score > 3, soil moisture - 25, rainfall - 300
  - 8 category pairs and 1 triple (strings)
  - soil moisture, rainfall, temperature, wind speed: 32 / 64 / 128-quantile bins (edges from the train rows), the
    units digit and the two decimal digits, the value rounded to an integer
  - count over train + test + original and rare flag (count <= 25) of the categoricals, pairs and triple
  - per-class original frequency, original count and log1p(train frequency / original frequency) of the 8
    categoricals and the first 4 pairs
  - original mean of soil moisture, rainfall, temperature, wind speed per value of each categorical
Per fold: for 25 columns (the 8 categoricals, the 9 crosses, 4 quantile-bin columns, the 4 rounded values) and each
class k, a binary scikit-learn TargetEncoder (smooth 'auto') of the class-k indicator, out-of-fold over an inner
StratifiedKFold(5, seed 42 + fold number 1-5, stratified on the indicator) of the training rows and fitted on all
training rows for the validation / test rows (75 columns). The 17 string columns are then factorized over the fold's
training + validation + test rows and missing values set to -1.
Model: LightGBM multiclass, GOSS (top_rate 0.2, other_rate 0.1), balanced class sample weights, 63 leaves, lr 0.01,
colsample_bytree 0.2, lambda_l2 15, lambda_l1 10, seed 42, 32 threads, up to 7500 rounds with early stopping (300)
on the validation fold. Test = mean of the 5 fold models.
Usage: python train_goss_crosses.py"""
import gc
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgbm
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

PAIR = 'lgbm_goss_crosses_binary_te_b'
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
CROSS_PAIRS = [
    ('Crop_Growth_Stage', 'Mulching_Used'), ('Crop_Growth_Stage', 'Season'),
    ('Crop_Type', 'Soil_Type'), ('Crop_Type', 'Season'),
    ('Crop_Type', 'Crop_Growth_Stage'), ('Region', 'Season'),
    ('Irrigation_Type', 'Water_Source'), ('Soil_Type', 'Region'),
]
TRIPLE_CATS = [('Crop_Type', 'Crop_Growth_Stage', 'Season')]

LGB_PARAMS = dict(
    objective='multiclass',
    num_class=3,
    metric='multi_logloss',
    n_estimators=7500,
    learning_rate=0.01,
    max_depth=-1,
    num_leaves=63,
    colsample_bytree=0.2,
    lambda_l2=15.0,
    lambda_l1=10.0,
    boosting_type='goss',
    top_rate=0.2,
    other_rate=0.1,
    random_state=SEED,
    verbosity=-1,
    num_threads=32,
)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


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
    log(f'Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}')

    log('Frequency / rank / numeric transforms')
    all_num = pd.concat([train[NUMS], test[NUMS], orig[NUMS]], ignore_index=True)
    for c in NUMS:
        freq = all_num[c].value_counts(normalize=True, dropna=False)
        for df in [train, test, orig]:
            df[f'FREQ_{c}'] = df[c].map(freq).fillna(0).astype('float32')
        r = all_num[c].rank(method='average', pct=True)
        n1, n2 = len(train), len(test)
        train[f'RANK_{c}'] = r.iloc[:n1].values.astype('float32')
        test[f'RANK_{c}'] = r.iloc[n1:n1 + n2].values.astype('float32')
        orig[f'RANK_{c}'] = r.iloc[n1 + n2:].values.astype('float32')
    for c in NUMS:
        for df in [train, test, orig]:
            v = df[c].astype('float32').clip(lower=0)
            df[f'LOG1P_{c}'] = np.log1p(v).astype('float32')
            df[f'SQRT_{c}'] = np.sqrt(v).astype('float32')
            df[f'INV1P_{c}'] = (1.0 / (1.0 + v)).astype('float32')

    log('Threshold score')
    for df in [train, test, orig]:
        SM = df['Soil_Moisture']; RM = df['Rainfall_mm']; TC = df['Temperature_C']
        WS = df['Wind_Speed_kmh']; CGS = df['Crop_Growth_Stage']; MU = df['Mulching_Used']
        score = ((SM < 25) * 2 + (RM < 300) * 2 + (TC > 30) + (WS > 10)
                 - (CGS == 'Harvest') * 2 - (CGS == 'Sowing') * 2 - (MU == 'Yes'))
        df['formula_score'] = score.astype('float32')
        df['is_low'] = (score <= 0).astype('float32')
        df['is_high'] = (score > 3).astype('float32')
        df['SM_thr'] = (SM - 25).astype('float32')
        df['RM_thr'] = (RM - 300).astype('float32')

    log('Category crosses')
    for a, b in CROSS_PAIRS:
        for df in [train, test, orig]:
            df[f'{a}__{b}'] = df[a].astype(str) + '|' + df[b].astype(str)
    for a, b, c in TRIPLE_CATS:
        for df in [train, test, orig]:
            df[f'{a}__{b}__{c}'] = df[a].astype(str) + '|' + df[b].astype(str) + '|' + df[c].astype(str)

    log('Quantile bins')
    qcut_cols = []
    for c in FORMULA_NUMS:
        for q in [32, 64, 128]:
            name = f'{c}_qcut_{q}'
            try:
                _, bins = pd.qcut(train[c], q=q, labels=False, retbins=True, duplicates='drop')
                for df in [train, test, orig]:
                    df[name] = pd.cut(df[c], bins=bins, labels=False, include_lowest=True).fillna(-1).astype('int32')
            except Exception:
                for df in [train, test, orig]:
                    df[name] = 0
            qcut_cols.append(name)

    log('Digits')
    for c in FORMULA_NUMS:
        allv = pd.concat([train[c], test[c], orig[c]], ignore_index=True).fillna(0)
        scaled = (allv * 100).round().astype(np.int64)
        n1, n2 = len(train), len(test)
        for k, (fn, mod) in {'int_last': (lambda s: s // 100, 10),
                             'd1': (lambda s: s // 10, 10),
                             'd2': (lambda s: s, 10)}.items():
            vals = (fn(scaled) % mod).astype(np.int16)
            name = f'{c}_digit_{k}'
            train[name] = vals.iloc[:n1].values
            test[name] = vals.iloc[n1:n1 + n2].values
            orig[name] = vals.iloc[n1 + n2:].values

    log('Rounded values')
    round_cols = []
    for c in FORMULA_NUMS:
        name = f'{c}_round0'
        for df in [train, test, orig]:
            df[name] = df[c].round(0).astype('float32')
        round_cols.append(name)

    log('Category counts / rare flags')
    all_cat_cols = CATS + [f'{a}__{b}' for a, b in CROSS_PAIRS] + [f'{a}__{b}__{c}' for a, b, c in TRIPLE_CATS]
    all_combined = pd.concat([train[all_cat_cols], test[all_cat_cols], orig[all_cat_cols]], ignore_index=True)
    for c in all_cat_cols:
        vc = all_combined[c].astype(str).value_counts(dropna=False)
        for df in [train, test, orig]:
            cnt = df[c].astype(str).map(vc).fillna(0).astype('float32')
            df[f'CAT_CNT_{c}'] = cnt.values
            df[f'CAT_RARE_{c}'] = (cnt <= 25).astype('float32').values

    log('Original priors')
    prior_cols = CATS + [f'{a}__{b}' for a, b in CROSS_PAIRS[:4]]
    for c in prior_cols:
        for k in range(3):
            m = orig.groupby(c).apply(lambda g, kk=k: (g[LABEL] == kk).mean())
            gm = float((y_orig == k).mean())
            for df in [train, test, orig]:
                df[f'ORIG_k{k}_{c}'] = df[c].map(m).fillna(gm).astype('float32')
        cnt_map = orig.groupby(c).size()
        for df in [train, test, orig]:
            df[f'ORIG_cnt_{c}'] = df[c].map(cnt_map).fillna(0).astype('float32')

    log('Drift ratio')
    for c in prior_cols:
        p_tr = train[c].value_counts(normalize=True, dropna=False)
        p_or = orig[c].value_counts(normalize=True, dropna=False)
        for df in [train, test]:
            df[f'DRIFT_{c}'] = np.log1p(df[c].map(p_tr).fillna(0) / (df[c].map(p_or).fillna(1e-6) + 1e-6)).astype('float32')
        orig[f'DRIFT_{c}'] = np.float32(0.0)

    log('Original group means')
    for cat in CATS:
        for num in FORMULA_NUMS:
            grp = orig.groupby(cat)[num].mean()
            for df in [train, test, orig]:
                df[f'GMEAN_{cat}_{num}'] = df[cat].map(grp).fillna(orig[num].mean()).astype('float32')
    log(f'After features: train={train.shape}, test={test.shape}')

    cat_cols_model = [c for c in train.columns if train[c].dtype.name in ['object', 'string']]
    te_cols = (CATS
               + [f'{a}__{b}' for a, b in CROSS_PAIRS]
               + [f'{a}__{b}__{c}' for a, b, c in TRIPLE_CATS]
               + qcut_cols[:4] + round_cols)
    te_cols = [c for c in te_cols if c in train.columns]
    log(f'String columns: {len(cat_cols_model)} | encoded columns: {len(te_cols)}')
    return train.drop(columns=[LABEL]), test.copy(), y, cat_cols_model, te_cols


def _te_to_1d(x):
    if isinstance(x, np.ndarray):
        return x[:, 0].astype(np.float32) if x.ndim == 2 else x.astype(np.float32)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.iloc[:, 0].astype(np.float32).values if hasattr(x, 'iloc') and x.ndim == 2 else x.astype(np.float32).values
    return np.asarray(x).astype(np.float32)


def nested_multiclass_te(X_tr, y_tr, X_va, X_te, te_cols, inner_folds=5, seed=42):
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
    return X_tr, X_va, X_te, made


def factorize_tvt(tr, va, te, cols):
    tr, va, te = tr.copy(), va.copy(), te.copy()
    for c in cols:
        allv = pd.concat([tr[c].astype(str), va[c].astype(str), te[c].astype(str)], ignore_index=True)
        codes, _ = pd.factorize(allv, sort=False)
        n1, n2 = len(tr), len(va)
        tr[c] = codes[:n1].astype('int32')
        va[c] = codes[n1:n1 + n2].astype('int32')
        te[c] = codes[n1 + n2:].astype('int32')
    return tr, va, te


def main():
    X, X_test_all, y, cat_cols_model, te_cols = build_features()
    cls_counts = np.bincount(y)
    cls_weights = cls_counts.sum() / (3 * cls_counts)

    fold_ids = folds(y)
    oof_preds = np.zeros((len(X), 3), dtype=np.float32)
    test_preds = np.zeros((len(X_test_all), 3), dtype=np.float32)
    for k in range(N_FOLDS):
        fold = k + 1
        tr_idx = np.where(fold_ids != k)[0]
        va_idx = np.where(fold_ids == k)[0]
        log(f'Fold {k}')
        t0 = time.time()

        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        X_va = X.iloc[va_idx].reset_index(drop=True)
        y_tr = pd.Series(y[tr_idx])
        y_va = y[va_idx]
        X_te = X_test_all.copy()

        X_tr, X_va, X_te, _ = nested_multiclass_te(
            X_tr, y_tr, X_va, X_te, te_cols=te_cols,
            inner_folds=INNER_FOLDS, seed=SEED + fold)

        cats_fold = [c for c in cat_cols_model if c in X_tr.columns]
        X_tr, X_va, X_te = factorize_tvt(X_tr, X_va, X_te, cats_fold)
        features = X_tr.columns.tolist()
        for df in [X_tr, X_va, X_te]:
            for c in features:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(-1).astype(np.float32)

        model = lgbm.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_tr[features], y_tr,
            sample_weight=cls_weights[y_tr.values],
            eval_set=[(X_va[features], y_va)],
            callbacks=[
                lgbm.log_evaluation(500),
                lgbm.early_stopping(300, verbose=False),
            ],
        )

        p_va = model.predict_proba(X_va[features], num_iteration=model.best_iteration_).astype(np.float32)
        p_te = model.predict_proba(X_te[features], num_iteration=model.best_iteration_).astype(np.float32)
        oof_preds[va_idx] = p_va
        test_preds += p_te / N_FOLDS
        bac = balanced_accuracy_score(y_va, np.argmax(p_va, axis=1))
        log(f'  Fold {k} BAC={bac:.5f} best_iter={model.best_iteration_} ({time.time() - t0:.0f}s)')

        del X_tr, X_va, X_te, model
        gc.collect()

    log(f'{PAIR}: OOF balanced accuracy {balanced_accuracy_score(y, np.argmax(oof_preds, axis=1)):.6f}')
    save_pair(HERE, PAIR, oof_preds, test_preds)


if __name__ == '__main__':
    main()
