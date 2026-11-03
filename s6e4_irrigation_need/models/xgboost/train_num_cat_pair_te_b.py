#!/usr/bin/env python
"""GPU XGBoost on per-class target encodings of numeric x categorical pairs + digit features (pair
xgb_num_cat_pair_te_digits_b).

Base features (101, label-free, built on train + test; the original rows only provide the priors):
- the 11 numerics and the 8 categoricals (XGBoost native categoricals);
- digit features of every numeric: up to 4 integer digit positions (ones, tens, ...) and the first 2 decimal digits;
- the rule formula score = 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10]
  - 2 [Crop_Growth_Stage = Harvest] - 2 [Crop_Growth_Stage = Sowing] - [Mulching_Used = Yes], is_low, is_high;
- per-class original-data priors of the 8 categoricals; the count of every numeric value over train + test.
Target encodings (per fold, 297 columns): keys = the 88 (numeric value, categorical) pairs and the 11 numeric values
(exact value strings); for each key and class k a binary target encoding of [label = k] with cuML TargetEncoder
(smooth 10), out-of-fold over an inner StratifiedKFold(5, seed 42 + fold, fold = 1..5) on the fold's training rows and
fitted on all training rows for the validation / test rows.
Model: XGBoost multi:softprob, depth 3, lr 0.02, subsample / colsample_bytree 0.8, balanced class weights, up to 20,000
rounds, early stopping 200 (mlogloss) on the validation fold, seed 42. Competition rows only. Test = mean of the 5 fold
models. A fold takes about 6 minutes on one GPU.
Usage: python train_num_cat_pair_te_b.py [--folds 0,1,2,3,4] [--force]   (finished folds are kept in parts/)"""
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
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'xgb_num_cat_pair_te_digits_b'
SEED = 42
INNER_TE_SPLITS = 5
TE_SMOOTH = 10.0
LABEL = 'target'

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Organic_Carbon', 'Electrical_Conductivity', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Temperature_C', 'Soil_Moisture',
        'Humidity', 'Previous_Irrigation_mm', 'Rainfall_mm']

XGB_PARAMS = dict(
    n_estimators=20000,
    learning_rate=0.02,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=SEED,
    early_stopping_rounds=200,
    device='cuda',
    enable_categorical=True,
    verbosity=0,
)


def numeric_to_key(s):
    x = pd.to_numeric(s, errors='coerce').astype(np.float64).fillna(-999999.123456)
    return x.map(lambda v: f'{v:.6f}').str.rstrip('0').str.rstrip('.').replace('', '0').astype(str)


def build_features():
    t0 = time.time()
    train, test, orig = read_train(), read_test(), read_original()
    y = labels(train[TARGET]).astype(np.int32)
    orig[LABEL] = labels(orig[TARGET]).astype(np.int32)
    train = train.drop(columns=[ID, TARGET])
    test = test.drop(columns=[ID])
    n_train = len(train)
    combined = pd.concat([train, test], ignore_index=True)

    # 1. digit features
    digit_cols = []
    for col in NUMS:
        vals = pd.to_numeric(combined[col], errors='coerce').fillna(0)
        max_val = float(vals.abs().max())
        max_int_digits = max(1, int(np.log10(max_val + 1)) + 1)
        for i in range(min(max_int_digits, 4)):
            name = f'{col}_digit_pos_{i + 1}'
            combined[name] = ((vals // (10 ** i)) % 10).astype('int16')
            digit_cols.append(name)
        if (vals % 1 != 0).any():
            for i in range(1, 3):
                name = f'{col}_digit_dec_{i}'
                combined[name] = ((vals * (10 ** i)).round().astype(np.int64) % 10).astype('int16')
                digit_cols.append(name)

    # 2. formula features
    SM = combined['Soil_Moisture']; RM = combined['Rainfall_mm']; TC = combined['Temperature_C']
    WS = combined['Wind_Speed_kmh']; CGS = combined['Crop_Growth_Stage']; MU = combined['Mulching_Used']
    score = ((SM < 25) * 2 + (RM < 300) * 2 + (TC > 30) + (WS > 10)
             - (CGS == 'Harvest') * 2 - (CGS == 'Sowing') * 2 - (MU == 'Yes'))
    combined['formula_score'] = score.astype(np.float32)
    combined['is_low'] = (score <= 0).astype(np.float32)
    combined['is_high'] = (score > 3).astype(np.float32)
    formula_cols = ['formula_score', 'is_low', 'is_high']

    # 3. per-class original priors
    y_orig = orig[LABEL].values
    prior_cols = []
    for c in CATS:
        for k in range(3):
            m = orig.groupby(c).apply(lambda g, kk=k: (g[LABEL] == kk).mean())
            gm = float((y_orig == k).mean())
            name = f'ORIG_k{k}_{c}'
            combined[name] = combined[c].map(m).fillna(gm).astype(np.float32)
            prior_cols.append(name)

    # 4. target-encoding keys: numeric x categorical pairs + numeric values, as integer codes
    cat_str = {c: combined[c].astype(str).fillna('__MISSING__') for c in CATS}
    num_key = {n: numeric_to_key(combined[n]) for n in NUMS}
    te_source = pd.DataFrame()
    pair_te_names = []
    for n in NUMS:
        for c in CATS:
            col = f'TEPAIR__{n}__X__{c}'
            te_source[col] = (num_key[n] + '__' + cat_str[c]).astype(str)
            pair_te_names.append(col)
    num_te_names = []
    for n in NUMS:
        col = f'TENUM__{n}'
        te_source[col] = num_key[n].astype(str)
        num_te_names.append(col)
    for c in pair_te_names + num_te_names:
        codes, _ = pd.factorize(te_source[c])
        te_source[c] = codes.astype(np.int32)
    train_te_raw = te_source.iloc[:n_train].reset_index(drop=True)
    test_te_raw = te_source.iloc[n_train:].reset_index(drop=True)

    # 5. count encoding
    cnt_cols = []
    for n in NUMS:
        name = f'CNT_{n}'
        vc = combined[n].value_counts()
        combined[name] = combined[n].map(vc).fillna(0).astype(np.float32)
        cnt_cols.append(name)

    base_features = NUMS + CATS + digit_cols + formula_cols + prior_cols + cnt_cols
    base_features = [c for c in base_features if c in combined.columns]
    train2 = combined.iloc[:n_train][base_features].reset_index(drop=True)
    test2 = combined.iloc[n_train:][base_features].reset_index(drop=True)
    for c in CATS:
        train2[c] = train2[c].astype('category')
        test2[c] = test2[c].astype('category')
    print(f'Base features: {len(base_features)}, TE pair keys: {len(pair_te_names)}, TE numeric keys: '
          f'{len(num_te_names)} ({time.time() - t0:.1f}s)', flush=True)
    return train2, test2, train_te_raw, test_te_raw, y


def make_leak_free_te_features(X_tr_raw, y_tr, X_va_raw, X_te_raw, outer_seed, inner_splits, smooth):
    """3 binary target encodings per key column (one per class), nested inner CV for the training rows."""
    import cudf
    import cupy as cp
    from cuml.preprocessing import TargetEncoder as CuTE

    te_cols = X_tr_raw.columns.tolist()
    n_out = len(te_cols) * 3
    tr_enc = np.zeros((len(X_tr_raw), n_out), dtype=np.float32)
    va_enc = np.zeros((len(X_va_raw), n_out), dtype=np.float32)
    te_enc = np.zeros((len(X_te_raw), n_out), dtype=np.float32)
    skf_inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=outer_seed)

    for k in range(3):
        y_k = (y_tr == k).astype(np.int32)
        y_k_cu = cudf.Series(y_k)
        for t, col in enumerate(te_cols):
            col_idx = t * 3 + k
            tr_col_cu = cudf.DataFrame({col: cudf.Series(X_tr_raw[col].values).astype('int32')})
            va_col_cu = cudf.DataFrame({col: cudf.Series(X_va_raw[col].values).astype('int32')})
            te_col_cu = cudf.DataFrame({col: cudf.Series(X_te_raw[col].values).astype('int32')})
            # inner out-of-fold encodings for the training rows
            oof_inner = np.zeros(len(X_tr_raw), dtype=np.float32)
            for itr, iva in skf_inner.split(np.zeros(len(X_tr_raw)), y_k):
                enc = CuTE(n_folds=0, smooth=smooth)
                itr_cu = cudf.DataFrame({col: cudf.Series(X_tr_raw[col].values[itr]).astype('int32')})
                iva_cu = cudf.DataFrame({col: cudf.Series(X_tr_raw[col].values[iva]).astype('int32')})
                enc.fit(itr_cu, cudf.Series(y_k[itr]))
                oof_inner[iva] = cp.asnumpy(enc.transform(iva_cu)).ravel()
            tr_enc[:, col_idx] = oof_inner
            # fitted on all training rows for the validation / test rows
            enc_full = CuTE(n_folds=0, smooth=smooth)
            enc_full.fit(tr_col_cu, y_k_cu)
            va_enc[:, col_idx] = cp.asnumpy(enc_full.transform(va_col_cu)).ravel()
            te_enc[:, col_idx] = cp.asnumpy(enc_full.transform(te_col_cu)).ravel()
        print(f'    class {k} target encodings done ({len(te_cols)} columns)', flush=True)

    out_names = [f'TE_k{k}_{col}' for col in te_cols for k in range(3)]
    return (pd.DataFrame(tr_enc, columns=out_names),
            pd.DataFrame(va_enc, columns=out_names),
            pd.DataFrame(te_enc, columns=out_names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--force', action='store_true', help='rerun folds that are already done')
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        train2, test2, train_te_raw, test_te_raw, y = build_features()
        cls_counts = np.bincount(y)
        cls_weights = cls_counts.sum() / (3 * cls_counts)
        fold_id = folds(y)
        for k in todo:
            fold = k + 1
            t0 = time.time()
            tr_idx, va_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            X_tr_base = train2.iloc[tr_idx].reset_index(drop=True)
            X_va_base = train2.iloc[va_idx].reset_index(drop=True)
            X_te_base = test2.reset_index(drop=True)
            y_tr, y_va = y[tr_idx], y[va_idx]

            tr_te, va_te, te_te = make_leak_free_te_features(
                train_te_raw.iloc[tr_idx].reset_index(drop=True), y_tr,
                train_te_raw.iloc[va_idx].reset_index(drop=True),
                test_te_raw.reset_index(drop=True),
                outer_seed=SEED + fold, inner_splits=INNER_TE_SPLITS, smooth=TE_SMOOTH,
            )
            X_tr = pd.concat([X_tr_base, tr_te], axis=1)
            X_va = pd.concat([X_va_base, va_te], axis=1)
            X_te = pd.concat([X_te_base, te_te], axis=1)
            for c in tr_te.columns:
                X_tr[c] = X_tr[c].astype(np.float32)
                X_va[c] = X_va[c].astype(np.float32)
                X_te[c] = X_te[c].astype(np.float32)
            print(f'  train {X_tr.shape}, valid {X_va.shape}, test {X_te.shape}', flush=True)

            model = XGBClassifier(**XGB_PARAMS)
            model.fit(X_tr, y_tr, sample_weight=cls_weights[y_tr], eval_set=[(X_va, y_va)], verbose=500)
            p_va = model.predict_proba(X_va).astype(np.float32)
            p_te = model.predict_proba(X_te).astype(np.float32)
            save_part(HERE, PAIR, k, p_va, p_te)
            bac = balanced_accuracy_score(y_va, np.argmax(p_va, axis=1))
            print(f'{PAIR} fold {k}: balanced accuracy {bac:.5f}, best iteration {model.best_iteration} '
                  f'({time.time() - t0:.0f}s)', flush=True)
            del X_tr, X_va, X_te, tr_te, va_te, te_te, model
            gc.collect()
    y = labels(read_train()[TARGET])
    assemble(HERE, PAIR, folds(y), y)


if __name__ == '__main__':
    main()
