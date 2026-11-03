#!/usr/bin/env python
"""XGBoost pairwise rankers + multinomial logistic calibration (pair xgb_rank_ordinal_ovr_platt_b).

Features (91; built once, the original rows only provide statistics):
- the 19 raw columns (categoricals ordinal-encoded);
- the rule formula score = 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10]
  - 2 [Crop_Growth_Stage = Harvest] - 2 [Crop_Growth_Stage = Sowing] - [Mulching_Used = Yes], is_low (score <= 0),
  is_high (score > 3), Soil_Moisture - 25 and Rainfall_mm - 300;
- the frequency of every numeric value over train + test + original;
- the last integer digit and the first 2 decimal digits of the 4 formula numerics;
- 8 categorical pairs (ordinal-encoded);
- per-class original-data priors (class shares of the original rows with the same value) of the 8 categoricals and of
  the first 4 pairs.
Stage 1: per fold, 4 XGBoost rankers (rank:pairwise) on random query groups of 256 rows: one ordinal ranker on the label
(Low < Medium < High) and one one-vs-rest ranker per class. Depth 6, lr 0.03, subsample / colsample_bytree 0.85, up to
4000 rounds, early stopping 200 on the validation fold (NDCG over random validation groups). Their out-of-fold scores
(test = mean of the 5 fold models) are 4 raw scores.
Stage 2: per fold, a class-balanced multinomial logistic regression (C 1, standardized inputs) on the 4 raw scores + the
91 features, fitted on the rows of the 4 other folds (whose raw scores are out-of-fold). Test = mean of the 5 calibrators.
Usage: python train_rank_platt_b.py   (one process; the rankers take a few minutes on one GPU)"""
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBRanker

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'xgb_rank_ordinal_ovr_platt_b'
SEED = 42
N_FOLDS = 5
GROUP_SIZE = 256
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

XGB_PARAMS = dict(
    objective='rank:pairwise',
    tree_method='hist',
    device='cuda',
    max_depth=6,
    min_child_weight=1.0,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    n_estimators=4000,
    learning_rate=0.03,
    random_state=SEED,
    n_jobs=-1,
    early_stopping_rounds=200,
)


def build_features():
    train, test, orig = read_train(), read_test(), read_original()
    train[LABEL] = labels(train[TARGET]).astype(np.int32)
    orig[LABEL] = labels(orig[TARGET]).astype(np.int32)
    y = train[LABEL].values
    y_orig = orig[LABEL].values
    for c in [ID, TARGET]:
        for df in [train, test]:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
    print(f'Train: {train.shape}, Test: {test.shape}', flush=True)
    t0 = time.time()

    # 1. formula features
    for df in [train, test, orig]:
        SM = df['Soil_Moisture']; RM = df['Rainfall_mm']; TC = df['Temperature_C']
        WS = df['Wind_Speed_kmh']; CGS = df['Crop_Growth_Stage']; MU = df['Mulching_Used']
        score = ((SM < 25) * 2 + (RM < 300) * 2 + (TC > 30) + (WS > 10)
                 - (CGS == 'Harvest') * 2 - (CGS == 'Sowing') * 2 - (MU == 'Yes'))
        df['formula_score'] = score.astype(np.float32)
        df['is_low'] = (score <= 0).astype(np.float32)
        df['is_high'] = (score > 3).astype(np.float32)
        df['SM_thr'] = (SM - 25).astype(np.float32)
        df['RM_thr'] = (RM - 300).astype(np.float32)

    # 2. frequency encoding over train + test + original
    all_num = pd.concat([train[NUMS], test[NUMS], orig[NUMS]], ignore_index=True)
    for c in NUMS:
        freq = all_num[c].value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{c}'] = df[c].map(freq).fillna(0).astype(np.float32)

    # 3. last integer digit and first 2 decimal digits (digits of 100 x value)
    n1, n2 = len(train), len(test)
    digit_fns = {'d0': (lambda s: s // 100, 10), 'd1': (lambda s: s // 10, 10), 'd2': (lambda s: s, 10)}
    for c in FORMULA_NUMS:
        allv = pd.concat([train[c], test[c], orig[c]], ignore_index=True).fillna(0)
        scaled = (allv * 100).round().astype(np.int64)
        for k, (fn, mod) in digit_fns.items():
            vals = (fn(scaled) % mod).astype(np.int16)
            name = f'{c}_digit_{k}'
            train[name] = vals.iloc[:n1].values
            test[name] = vals.iloc[n1:n1 + n2].values
            orig[name] = vals.iloc[n1 + n2:].values

    # 4. categorical pairs
    for a, b in CAT_PAIRS:
        for df in [train, test, orig]:
            df[f'{a}__{b}'] = df[a].astype(str) + '_' + df[b].astype(str)

    # 5. per-class original priors
    prior_cols = CATS + [f'{a}__{b}' for a, b in CAT_PAIRS[:4]]
    for c in prior_cols:
        for k in range(3):
            m = orig.groupby(c).apply(lambda g, kk=k: (g[LABEL] == kk).mean())
            gm = float((y_orig == k).mean())
            for df in [train, test, orig]:
                df[f'ORIG_k{k}_{c}'] = df[c].map(m).fillna(gm).astype(np.float32)

    # 6. ordinal-encode the string columns (fitted on train)
    cat_cols = [c for c in train.columns if train[c].dtype == 'object']
    for c in cat_cols:
        train[c] = train[c].fillna('Missing')
        test[c] = test[c].fillna('Missing')
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train[cat_cols] = enc.fit_transform(train[cat_cols]).astype(np.float32)
    test[cat_cols] = enc.transform(test[cat_cols]).astype(np.float32)

    features = [c for c in train.columns if c != LABEL]
    X_all = train[features].values.astype(np.float32)
    X_test = test[features].values.astype(np.float32)
    print(f'Features: {len(features)} ({time.time() - t0:.1f}s)', flush=True)
    return X_all, X_test, y


def make_groups(n, group_size, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    grp_ids = np.zeros(n, dtype=np.int32)
    n_full = n // group_size
    for g in range(n_full):
        grp_ids[idx[g * group_size:(g + 1) * group_size]] = g
    if n_full * group_size < n:
        grp_ids[idx[n_full * group_size:]] = n_full
    _, counts = np.unique(grp_ids, return_counts=True)
    return grp_ids, counts


def train_one_ranker(X_all, X_test, y_target, splits, oof_raw, test_raw, col_idx, label, seed_offset):
    for fold, tr_idx, va_idx in splits:
        t0 = time.time()
        X_tr, y_tr = X_all[tr_idx], y_target[tr_idx]
        X_va = X_all[va_idx]
        tr_grp_ids, tr_grp_counts = make_groups(len(X_tr), GROUP_SIZE, SEED + fold + seed_offset)
        tr_sort = np.argsort(tr_grp_ids)
        va_grp_ids, va_grp_counts = make_groups(len(X_va), GROUP_SIZE, SEED + fold + seed_offset + 50)
        va_sort = np.argsort(va_grp_ids)

        model = XGBRanker(**XGB_PARAMS)
        model.fit(
            X_tr[tr_sort], y_tr[tr_sort], group=tr_grp_counts,
            eval_set=[(X_va[va_sort], y_target[va_idx][va_sort])], eval_group=[va_grp_counts],
            verbose=500,
        )
        va_scores_s = model.predict(X_va[va_sort])
        va_unsort = np.argsort(va_sort)
        oof_raw[va_idx, col_idx] = va_scores_s[va_unsort].astype(np.float32)
        test_raw[:, col_idx] += model.predict(X_test).astype(np.float32) / N_FOLDS
        print(f'  {label} fold {fold} done ({time.time() - t0:.0f}s)', flush=True)
        del model
        gc.collect()


def main():
    X_all, X_test, y = build_features()
    fold_id = folds(y)
    splits = [(k + 1, np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)) for k in range(N_FOLDS)]

    # stage 1: 1 ordinal + 3 one-vs-rest rankers -> 4 raw scores
    oof_raw = np.zeros((len(X_all), 4), dtype=np.float32)
    test_raw = np.zeros((len(X_test), 4), dtype=np.float32)
    print('Ordinal ranker (target 0 < 1 < 2)...', flush=True)
    train_one_ranker(X_all, X_test, y, splits, oof_raw, test_raw, col_idx=0, label='Ordinal', seed_offset=0)
    for k in range(3):
        lbl = ['Low', 'Medium', 'High'][k]
        print(f'One-vs-rest ranker for class {k} ({lbl})...', flush=True)
        train_one_ranker(X_all, X_test, (y == k).astype(np.int32), splits, oof_raw, test_raw, col_idx=k + 1,
                         label=f'OVR-{lbl}', seed_offset=(k + 1) * 100)

    # stage 2: calibration on the 4 raw scores + the features
    calib_oof = np.hstack([oof_raw, X_all]).astype(np.float32)
    calib_test = np.hstack([test_raw, X_test]).astype(np.float32)
    print(f'Calibration input: {calib_oof.shape[1]} columns (4 raw scores + {X_all.shape[1]} features)', flush=True)
    oof_preds = np.zeros((len(X_all), 3), dtype=np.float32)
    test_preds = np.zeros((len(X_test), 3), dtype=np.float32)
    for fold, tr_idx, va_idx in splits:
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(calib_oof[tr_idx])
        X_va_sc = scaler.transform(calib_oof[va_idx])
        X_te_sc = scaler.transform(calib_test)
        # lbfgs on 3 classes = multinomial logistic regression
        calib = LogisticRegression(solver='lbfgs', C=1.0, max_iter=2000, class_weight='balanced', random_state=SEED)
        calib.fit(X_tr_sc, y[tr_idx])
        oof_preds[va_idx] = calib.predict_proba(X_va_sc).astype(np.float32)
        test_preds += calib.predict_proba(X_te_sc).astype(np.float32) / N_FOLDS
        bac = balanced_accuracy_score(y[va_idx], np.argmax(oof_preds[va_idx], axis=1))
        print(f'{PAIR} fold {fold - 1}: balanced accuracy {bac:.5f}', flush=True)

    save_pair(HERE, PAIR, oof_preds, test_preds)
    print(f'{PAIR}: OOF balanced accuracy {balanced_accuracy_score(y, oof_preds.argmax(1)):.6f}', flush=True)


if __name__ == '__main__':
    main()
