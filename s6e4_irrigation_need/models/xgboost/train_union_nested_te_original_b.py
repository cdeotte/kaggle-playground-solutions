#!/usr/bin/env python
"""GPU XGBoost on a union of label-free blocks + nested per-class target encodings, with the original rows appended to
every training fold (pair xgb_union_nested_te_original_b).

Label-free features (built once on train, test and the original rows):
- the 19 raw columns; frequency and percentile rank of the 11 numerics over train + test + original;
  log1p / sqrt / 1 / (1 + x) of the numerics;
- the rule formula score = 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10]
  - 2 [Crop_Growth_Stage = Harvest] - 2 [Crop_Growth_Stage = Sowing] - [Mulching_Used = Yes], is_low, is_high and the 4
  threshold distances;
- 8 categorical pairs and 2 triples; 16 / 32 / 64-quantile bins (edges from the train rows) of the 4 formula numerics;
  the last integer digit, the first 2 decimal digits and the rounded value of the formula numerics;
- counts (over train + test + original) and rare flags (count <= 25) of the categoricals, pairs and triples;
- per-class original-data priors and original counts of the categoricals and the first 4 pairs, the drift
  log1p(train share / original share) of the same 12 keys, and flags "value occurs in the original data".
Per fold: 26 keys (categoricals, pairs, triples, 4 quantile bins, 4 rounded values) get 3 binary target encodings each
(one per class, sklearn TargetEncoder, smooth 'auto'), out-of-fold over an inner StratifiedKFold(5, seed 42 + fold,
fold = 1..5) of the competition training rows and fitted on all of them for the validation / test rows; the string
columns are integer-coded over training + validation + test rows. The 10,000 original rows are then appended to the
training rows (their target encodings set to 0, their codes taken from the training rows' codes).
Model: XGBoost multi:softprob, depth 5, lr 0.025, min_child_weight 5, subsample 0.8, colsample_bytree 0.45, alpha 0.1,
lambda 1, gamma 0.05, max_bin 16,000, balanced class weights (original rows included), up to 20,000 rounds, early
stopping 300 (mlogloss) on the validation fold, seed 42. Test = mean of the 5 fold models.
Usage: python train_union_nested_te_original_b.py [--folds 0,1,2,3,4] [--force]   (finished folds are kept in parts/)"""
import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'xgb_union_nested_te_original_b'
SEED = 42
INNER_FOLDS = 5
LABEL = 'target'

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Organic_Carbon', 'Electrical_Conductivity', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Temperature_C', 'Soil_Moisture',
        'Humidity', 'Previous_Irrigation_mm', 'Rainfall_mm']
FORMULA_NUMS = ['Soil_Moisture', 'Rainfall_mm', 'Temperature_C', 'Wind_Speed_kmh']

CROSS_PAIRS = [
    ('Crop_Growth_Stage', 'Mulching_Used'),
    ('Crop_Growth_Stage', 'Season'),
    ('Crop_Type', 'Soil_Type'),
    ('Crop_Type', 'Season'),
    ('Crop_Type', 'Crop_Growth_Stage'),
    ('Region', 'Season'),
    ('Irrigation_Type', 'Water_Source'),
    ('Soil_Type', 'Region'),
]
TRIPLE_CATS = [
    ('Crop_Type', 'Crop_Growth_Stage', 'Season'),
    ('Crop_Type', 'Mulching_Used', 'Region'),
]
QCUT_BINS = [16, 32, 64]
RARE_MIN = 25

XGB_PARAMS = dict(
    n_estimators=20000,
    learning_rate=0.025,
    max_depth=5,
    min_child_weight=5,
    subsample=0.80,
    colsample_bytree=0.45,
    reg_alpha=0.10,
    reg_lambda=1.00,
    gamma=0.05,
    max_bin=16000,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    tree_method='hist',
    device='cuda',
    early_stopping_rounds=300,
    random_state=SEED,
    enable_categorical=False,
    verbosity=0,
)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def safe_str(s):
    return s.astype('string').fillna('<NA>')


def build_features():
    train, test, orig = read_train(), read_test(), read_original()
    train[LABEL] = labels(train[TARGET]).astype(np.int32)
    orig[LABEL] = labels(orig[TARGET]).astype(np.int32)
    y = train[LABEL].values
    for c in [ID, TARGET]:
        for df in [train, test]:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
    if ID in orig.columns:
        orig.drop(columns=[ID], inplace=True, errors='ignore')

    log('frequency / rank')
    all_num = pd.concat([train[NUMS], test[NUMS], orig[NUMS]], axis=0, ignore_index=True)
    n1, n2 = len(train), len(test)
    for c in NUMS:
        freq = all_num[c].value_counts(normalize=True, dropna=False)
        for df in [train, test, orig]:
            df[f'FREQ_{c}'] = df[c].map(freq).fillna(0).astype('float32')
        r = all_num[c].rank(method='average', pct=True)
        train[f'RANK_{c}'] = r.iloc[:n1].values.astype('float32')
        test[f'RANK_{c}'] = r.iloc[n1:n1 + n2].values.astype('float32')
        orig[f'RANK_{c}'] = r.iloc[n1 + n2:].values.astype('float32')

    log('log1p / sqrt / inverse transforms')
    for c in NUMS:
        for df in [train, test, orig]:
            v = df[c].astype('float32').clip(lower=0)
            df[f'LOG1P_{c}'] = np.log1p(v).astype('float32')
            df[f'SQRT_{c}'] = np.sqrt(v).astype('float32')
            df[f'INV1P_{c}'] = (1.0 / (1.0 + v)).astype('float32')

    log('formula')
    for df in [train, test, orig]:
        SM = df['Soil_Moisture']; RM = df['Rainfall_mm']; TC = df['Temperature_C']
        WS = df['Wind_Speed_kmh']; CGS = df['Crop_Growth_Stage']; MU = df['Mulching_Used']
        score = ((SM < 25) * 2 + (RM < 300) * 2 + (TC > 30) * 1 + (WS > 10) * 1
                 - (CGS == 'Harvest') * 2 - (CGS == 'Sowing') * 2 - (MU == 'Yes') * 1)
        df['formula_score'] = score.astype('float32')
        df['is_low'] = (score <= 0).astype('float32')
        df['is_high'] = (score > 3).astype('float32')
        df['SM_thr'] = (SM - 25).astype('float32')
        df['RM_thr'] = (RM - 300).astype('float32')
        df['TC_thr'] = (TC - 30).astype('float32')
        df['WS_thr'] = (WS - 10).astype('float32')

    log('categorical pairs / triples')
    for a, b in CROSS_PAIRS:
        for df in [train, test, orig]:
            df[f'{a}__{b}'] = safe_str(df[a]) + '|' + safe_str(df[b])
    for a, b, c in TRIPLE_CATS:
        for df in [train, test, orig]:
            df[f'{a}__{b}__{c}'] = safe_str(df[a]) + '|' + safe_str(df[b]) + '|' + safe_str(df[c])

    log('quantile bins')
    qcut_cols_made = []
    for c in FORMULA_NUMS:
        for q in QCUT_BINS:
            name = f'{c}_qcut_{q}'
            try:
                _, bins = pd.qcut(train[c], q=q, labels=False, retbins=True, duplicates='drop')
                for df in [train, test, orig]:
                    df[name] = pd.cut(df[c], bins=bins, labels=False, include_lowest=True)
                    df[name] = df[name].fillna(-1).astype('int32')
            except Exception:
                for df in [train, test, orig]:
                    df[name] = 0
            qcut_cols_made.append(name)

    log('digits')
    for c in FORMULA_NUMS:
        allv = pd.concat([train[c], test[c], orig[c]], ignore_index=True).fillna(0)
        scaled = (allv * 100).round().astype(np.int64)
        for k, (a_fn, b) in {'int_last': (lambda s: s // 100, 10),
                             'dec0': (lambda s: s // 10, 10),
                             'dec1': (lambda s: s, 10)}.items():
            vals = (a_fn(scaled) % b).astype(np.int16)
            name = f'{c}_digit_{k}'
            train[name] = vals.iloc[:n1].values
            test[name] = vals.iloc[n1:n1 + n2].values
            orig[name] = vals.iloc[n1 + n2:].values

    log('rounded values')
    round_cols_made = []
    for c in FORMULA_NUMS:
        name = f'{c}_round0'
        for df in [train, test, orig]:
            df[name] = df[c].round(0).astype('float32')
        round_cols_made.append(name)

    log('category counts / rare flags')
    all_cat_cols = CATS + [f'{a}__{b}' for a, b in CROSS_PAIRS] + [f'{a}__{b}__{c}' for a, b, c in TRIPLE_CATS]
    all_cat_combined = pd.concat([train[all_cat_cols], test[all_cat_cols], orig[all_cat_cols]], ignore_index=True)
    for c in all_cat_cols:
        vc = all_cat_combined[c].astype('string').value_counts(dropna=False)
        for df in [train, test, orig]:
            cnt = df[c].astype('string').map(vc).fillna(0).astype('float32')
            df[f'CAT_CNT_{c}'] = cnt.values
            df[f'CAT_RARE_{c}'] = (cnt <= RARE_MIN).astype('float32').values

    log('per-class original priors')
    y_orig = orig[LABEL].values
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

    log('drift ratios')
    drift_cols = CATS + [f'{a}__{b}' for a, b in CROSS_PAIRS[:4]]
    for c in drift_cols:
        p_tr = train[c].value_counts(normalize=True, dropna=False)
        p_or = orig[c].value_counts(normalize=True, dropna=False)
        for df in [train, test]:
            df[f'DRIFT_{c}'] = np.log1p(df[c].map(p_tr).fillna(0) / (df[c].map(p_or).fillna(1e-6) + 1e-6)).astype('float32')
        orig[f'DRIFT_{c}'] = np.float32(0.0)

    for c in CATS:
        orig_vals = set(orig[c].unique())
        for df in [train, test]:
            df[f'ORIG_SEEN_{c}'] = df[c].isin(orig_vals).astype('float32')
        orig[f'ORIG_SEEN_{c}'] = np.float32(1.0)
    log(f'after the feature build: train {train.shape}, test {test.shape}')

    cat_cols_model = [c for c in train.columns if train[c].dtype.name in ['object', 'string']]
    cat_cols_model = list(dict.fromkeys(cat_cols_model))
    te_cols = (CATS
               + [f'{a}__{b}' for a, b in CROSS_PAIRS]
               + [f'{a}__{b}__{c}' for a, b, c in TRIPLE_CATS]
               + qcut_cols_made[:4]
               + round_cols_made)
    te_cols = [c for c in te_cols if c in train.columns]
    log(f'string columns: {len(cat_cols_model)}, target-encoded keys: {len(te_cols)}, columns: {train.shape[1]}')
    X = train.drop(columns=[LABEL]).copy()
    return X, test.copy(), orig, y, te_cols, cat_cols_model


def _te_to_1d(x):
    if isinstance(x, np.ndarray):
        return x[:, 0].astype(np.float32) if x.ndim == 2 else x.astype(np.float32)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.iloc[:, 0].astype(np.float32).values if x.ndim == 2 else x.astype(np.float32).values
    return np.asarray(x).astype(np.float32)


def nested_multiclass_te(X_tr, y_tr, X_va, X_te, te_cols, inner_folds=5, seed=42):
    """3 binary target encodings per column (one per class), nested inside the outer fold."""
    X_tr, X_va, X_te = X_tr.copy(), X_va.copy(), X_te.copy()
    made = []
    skf_inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    for c in te_cols:
        if c not in X_tr.columns:
            continue
        tr_col = safe_str(X_tr[c]).to_frame(name=c)
        va_col = safe_str(X_va[c]).to_frame(name=c)
        te_col = safe_str(X_te[c]).to_frame(name=c)
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
        allv = pd.concat([safe_str(tr[c]), safe_str(va[c]), safe_str(te[c])], ignore_index=True)
        codes, _ = pd.factorize(allv, sort=False)
        n1, n2 = len(tr), len(va)
        tr[c] = codes[:n1].astype('int32')
        va[c] = codes[n1:n1 + n2].astype('int32')
        te[c] = codes[n1 + n2:].astype('int32')
    return tr, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--force', action='store_true', help='rerun folds that are already done')
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        X, X_test_all, orig, y, te_cols, cat_cols_model = build_features()
        cls_counts = np.bincount(y)
        cls_weights = cls_counts.sum() / (3 * cls_counts)
        fold_id = folds(y)
        for k in todo:
            fold = k + 1
            log(f'fold {k}')
            t0 = time.time()
            tr_idx, va_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            X_tr = X.iloc[tr_idx].reset_index(drop=True)
            X_va = X.iloc[va_idx].reset_index(drop=True)
            y_tr = pd.Series(y[tr_idx])
            y_va = y[va_idx]
            X_te = X_test_all.copy()

            X_tr, X_va, X_te, te_made = nested_multiclass_te(X_tr, y_tr, X_va, X_te, te_cols=te_cols,
                                                             inner_folds=INNER_FOLDS, seed=SEED + fold)
            cats_fold = [c for c in cat_cols_model if c in X_tr.columns]
            X_tr, X_va, X_te = factorize_tvt(X_tr, X_va, X_te, cats_fold)
            features = X_tr.columns.tolist()
            for df in [X_tr, X_va, X_te]:
                for c in features:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(-1).astype(np.float32)
            sw = cls_weights[y_tr.values]

            # original rows appended to the training rows: target encodings 0, codes from the training rows
            X_or = orig.drop(columns=[LABEL, TARGET], errors='ignore').copy()
            for c in te_made:
                if c not in X_or.columns:
                    X_or[c] = np.float32(0.0)
            for c in cats_fold:
                if c in X_or.columns:
                    X_or[c] = pd.to_numeric(X_or[c].astype('string').map(
                        dict(zip(safe_str(X.iloc[tr_idx][c]).unique(), range(999)))),
                        errors='coerce').fillna(-1).astype(np.float32)
                else:
                    X_or[c] = np.float32(-1)
            for c in features:
                if c not in X_or.columns:
                    X_or[c] = np.float32(0.0)
                X_or[c] = pd.to_numeric(X_or[c], errors='coerce').fillna(-1).astype(np.float32)
            X_tr_aug = pd.concat([X_tr[features], X_or[features]], ignore_index=True)
            y_tr_aug = np.concatenate([y_tr.values, orig[LABEL].values])
            sw_aug = np.concatenate([sw, cls_weights[orig[LABEL].values]])

            model = XGBClassifier(**XGB_PARAMS)
            model.fit(X_tr_aug, y_tr_aug, sample_weight=sw_aug, eval_set=[(X_va[features], y_va)], verbose=500)
            p_va = model.predict_proba(X_va[features]).astype(np.float32)
            p_te = model.predict_proba(X_te[features]).astype(np.float32)
            save_part(HERE, PAIR, k, p_va, p_te)
            bac = balanced_accuracy_score(y_va, np.argmax(p_va, axis=1))
            log(f'{PAIR} fold {k}: balanced accuracy {bac:.5f}, best iteration {model.best_iteration} '
                f'({time.time() - t0:.0f}s)')
            del X_tr, X_va, X_te, X_or, X_tr_aug, model
            gc.collect()
    y = labels(read_train()[TARGET])
    assemble(HERE, PAIR, folds(y), y)


if __name__ == '__main__':
    main()
