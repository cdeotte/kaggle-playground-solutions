#!/usr/bin/env python
"""GPU XGBoost with target statistics around two snapped anchors, Rainfall_mm and Soil_Moisture (pair
xgb_rain_moisture_snap_anchor_te_b).

Base features (120, label-free, built on train + test; the original rows only provide the snap grid):
- the 8 categoricals (integer codes over train + test, XGBoost categoricals) and the 11 numerics;
- snap: each numeric value replaced by the nearest original value (<col>_snap) and value - snap (<col>_snap_diff);
- the rule formula score = 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10]
  - 2 [Crop_Growth_Stage = Harvest] - 2 [Crop_Growth_Stage = Sowing] - [Mulching_Used = Yes], is_low, is_high;
- anchors Rainfall_mm_snap and Soil_Moisture_snap: category x anchor keys code x 1e6 + round(10 x anchor) (16), the first
  6 decimal digits of each anchor (12, XGBoost categoricals) and the 10 pairs of its first 5 digits (20);
- the 28 pairs of the 8 categoricals (dropped after encoding).
Per fold, out-of-fold over an inner KFold(5, shuffle, seed 42) of the training rows (all training rows for the
validation / test rows): per anchor value the label (0, 1, 2) mean / std / count / nunique / median / min / max / skew,
its 5, 10, 40, 45, 55, 60, 90, 95 % quantiles and the 3 class shares; the 3 class shares of every category x anchor key,
digit pair and categorical pair; count / nunique and std / count ratios per anchor.
Model: XGBoost multi:softprob, depth 6, colsample_bynode 0.3, subsample 0.8, min_child_weight 10, lr 0.01, balanced
class weights, up to 50,000 rounds, early stopping 500 (mlogloss) on the validation fold, seed 42. Competition rows only.
Test = mean of the 5 fold models.
Usage: python train_snap_anchor_te_b.py [--folds 0,1,2,3,4] [--force]   (finished folds are kept in parts/)"""
import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'xgb_rain_moisture_snap_anchor_te_b'
INNER_FOLDS = 5
RANDOM_STATE = 42
N_CLASSES = 3

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
        'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']

# the two snapped numerics with the most distinct original values
ANCHOR1 = 'Rainfall_mm_snap'
ANCHOR2 = 'Soil_Moisture_snap'
ANCHOR_COLS = [ANCHOR1, ANCHOR2]

STATS_TE1 = ['mean', 'std', 'count', 'nunique', 'median', 'min', 'max', 'skew']
QUANTILES = [5, 10, 40, 45, 55, 60, 90, 95]

XGB_PARAMS = dict(
    device='cuda',
    tree_method='hist',
    enable_categorical=True,
    objective='multi:softprob',
    num_class=N_CLASSES,
    eval_metric='mlogloss',
    max_depth=6,
    colsample_bynode=0.3,
    subsample=0.8,
    min_child_weight=10,
    learning_rate=0.01,
    n_estimators=50_000,
    early_stopping_rounds=500,
    random_state=RANDOM_STATE,
    verbosity=0,
)


def safe_map(keys, mapping):
    return keys.map(mapping).astype('float32').to_numpy()


def factorize_int_codes(tr_s, te_s):
    comb = pd.concat([tr_s, te_s], axis=0, ignore_index=True)
    codes, _ = pd.factorize(comb, sort=False)
    codes = codes.astype('int32')
    return (pd.Series(codes[:len(tr_s)], index=tr_s.index),
            pd.Series(codes[len(tr_s):len(tr_s) + len(te_s)], index=te_s.index))


def compute_snap(values_f64, orig_uniq_sorted_f64):
    idx = np.searchsorted(orig_uniq_sorted_f64, values_f64, side='left')
    idx = np.clip(idx, 0, len(orig_uniq_sorted_f64) - 1)
    idx_l = np.clip(idx - 1, 0, len(orig_uniq_sorted_f64) - 1)
    snap_r = orig_uniq_sorted_f64[idx]
    snap_l = orig_uniq_sorted_f64[idx_l]
    return np.where(np.abs(values_f64 - snap_l) <= np.abs(values_f64 - snap_r), snap_l, snap_r).astype(np.float32)


def build_features():
    train, test, orig = read_train(), read_test(), read_original()
    print(f'Train: {train.shape}, Test: {test.shape}, Original: {orig.shape}', flush=True)
    y = labels(train[TARGET]).astype('int32')
    train = train.drop(columns=[TARGET])
    for col in NUMS:
        for df in (train, test, orig):
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

    # snap all 11 numerics
    for col in NUMS:
        orig_uniq = np.sort(orig[col].dropna().unique()).astype('float64')
        for df in (train, test):
            v = df[col].values.astype('float64')
            snap = compute_snap(v, orig_uniq)
            df[f'{col}_snap'] = snap
            df[f'{col}_snap_diff'] = (v - snap.astype('float64')).astype(np.float32)
    snap_cols = [f'{c}_snap' for c in NUMS]
    diff_cols = [f'{c}_snap_diff' for c in NUMS]

    for df in (train, test):
        SM = df['Soil_Moisture'].astype(float)
        RM = df['Rainfall_mm'].astype(float)
        TC = df['Temperature_C'].astype(float)
        WS = df['Wind_Speed_kmh'].astype(float)
        CGS = df['Crop_Growth_Stage'].astype(str)
        MU = df['Mulching_Used'].astype(str)
        score = ((SM < 25) * 2 + (RM < 300) * 2 + (TC > 30) + (WS > 10)
                 - (CGS == 'Harvest') * 2 - (CGS == 'Sowing') * 2 - (MU == 'Yes'))
        df['formula_score'] = score.astype(np.float32)
        df['is_low'] = (score <= 0).astype(np.float32)
        df['is_high'] = (score > 3).astype(np.float32)
    formula_cols = ['formula_score', 'is_low', 'is_high']

    for c in CATS:
        tr_c, te_c = factorize_int_codes(train[c], test[c])
        train[c] = tr_c
        test[c] = te_c

    new_train, new_test = {}, {}
    combo, digits_all = [], []
    # 1) category x anchor keys: code * 1M + round(anchor * 10)
    for anc_col in ANCHOR_COLS:
        anc_int_tr = (train[anc_col].astype('float32') * 10).round().astype('int64')
        anc_int_te = (test[anc_col].astype('float32') * 10).round().astype('int64')
        anc_tag = anc_col.replace('_snap', '').replace('_', '')
        for c in CATS:
            n = f'{c}_x_{anc_tag}'
            new_train[n] = (train[c].astype('int64') * 1_000_000 + anc_int_tr).astype('int64')
            new_test[n] = (test[c].astype('int64') * 1_000_000 + anc_int_te).astype('int64')
            combo.append(n)
    # 2) 6 decimal digits of each anchor
    for anc_col in ANCHOR_COLS:
        anc_tag = anc_col.replace('_snap', '').replace('_', '')
        anc_fill_tr = train[anc_col].fillna(0.0).astype('float64')
        anc_fill_te = test[anc_col].fillna(0.0).astype('float64')
        for k in range(1, 7):
            dn = f'digit_{anc_tag}_{k}'
            new_train[dn] = ((anc_fill_tr * (10 ** k)) % 10).astype('int8')
            new_test[dn] = ((anc_fill_te * (10 ** k)) % 10).astype('int8')
            digits_all.append(dn)
    # 3) pairs of the first 5 digits
    for anc_col in ANCHOR_COLS:
        anc_tag = anc_col.replace('_snap', '').replace('_', '')
        for i in range(4):
            for j in range(i + 1, 5):
                dn_i = f'digit_{anc_tag}_{i + 1}'
                dn_j = f'digit_{anc_tag}_{j + 1}'
                n = f'dp_{anc_tag}_{i + 1}_{j + 1}'
                new_train[n] = ((new_train[dn_i].astype('int16') + 1) * 11
                                + (new_train[dn_j].astype('int16') + 1)).astype('int16')
                new_test[n] = ((new_test[dn_i].astype('int16') + 1) * 11
                               + (new_test[dn_j].astype('int16') + 1)).astype('int16')
                combo.append(n)
    # 4) categorical pair ids
    pairs = []
    for i, c1 in enumerate(CATS[:-1]):
        c1_tr = train[c1].astype('int64')
        c1_te = test[c1].astype('int64')
        for c2 in CATS[i + 1:]:
            n = f'{c1}__{c2}'
            c2_tr = train[c2].astype('int64')
            c2_te = test[c2].astype('int64')
            m2 = int(c2_tr.max() + 2)
            new_train[n] = ((c1_tr + 1) * m2 + (c2_tr + 1)).astype('int32')
            new_test[n] = ((c1_te + 1) * m2 + (c2_te + 1)).astype('int32')
            combo.append(n)
            pairs.append(n)
    train = pd.concat([train, pd.DataFrame(new_train)], axis=1).copy()
    test = pd.concat([test, pd.DataFrame(new_test)], axis=1).copy()

    base_features = CATS + NUMS + snap_cols + diff_cols + formula_cols + digits_all + combo
    cat_for_xgb = set(CATS + digits_all)
    print(f'Base features: {len(base_features)}; combos: {len(combo)}; digits: {len(digits_all)}', flush=True)
    return train, test, y, base_features, combo, pairs, cat_for_xgb


def run_fold(tr_idx, va_idx, train, test, y, base_features, combo, pairs, cat_for_xgb):
    Xtr_base = train.iloc[tr_idx][base_features].reset_index(drop=True)
    ytr = y[tr_idx]
    Xva_base = train.iloc[va_idx][base_features].reset_index(drop=True)
    yva = y[va_idx]
    Xte_base = test[base_features].reset_index(drop=True)
    ntr = len(Xtr_base)

    te1_names = []
    for anc in ANCHOR_COLS:
        te1_names += [f'TE1_{anc}_{s}' for s in STATS_TE1]
        te1_names += [f'TE1_{anc}_q{q}' for q in QUANTILES]
        te1_names += [f'TE1_{anc}_frac_k{k}' for k in range(N_CLASSES)]
    te2_names = [f'TE2_{col}_frac_k{k}' for col in combo for k in range(N_CLASSES)]

    # training rows: statistics of the other inner folds
    inner = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    te1_store = {n: np.full(ntr, np.nan, 'float32') for n in te1_names}
    te2_store = {n: np.full(ntr, np.nan, 'float32') for n in te2_names}
    for inner_fold, (tr2_idx, va2_idx) in enumerate(inner.split(Xtr_base, ytr), start=1):
        print(f'  inner fold {inner_fold}/{INNER_FOLDS}', flush=True)
        Xtr2 = Xtr_base.iloc[tr2_idx]
        ytr2 = ytr[tr2_idx]
        for anc in ANCHOR_COLS:
            g = pd.DataFrame({'_anc_': Xtr2[anc].values, '_y_': ytr2})
            agg = g.groupby('_anc_')['_y_'].agg(STATS_TE1)
            keys_va = Xtr_base.iloc[va2_idx][anc]
            for s in STATS_TE1:
                te1_store[f'TE1_{anc}_{s}'][va2_idx] = safe_map(keys_va, agg[s])
            for q in QUANTILES:
                qmap = g.groupby('_anc_')['_y_'].quantile(q / 100.0)
                te1_store[f'TE1_{anc}_q{q}'][va2_idx] = safe_map(keys_va, qmap)
            for k in range(N_CLASSES):
                frac_map = g.groupby('_anc_')['_y_'].apply(lambda arr, _k=k: (arr == _k).mean())
                te1_store[f'TE1_{anc}_frac_k{k}'][va2_idx] = safe_map(keys_va, frac_map)
        for col in combo:
            g2 = pd.DataFrame({'_c_': Xtr2[col].values, '_y_': ytr2})
            for k in range(N_CLASSES):
                frac_map = g2.groupby('_c_')['_y_'].apply(lambda arr, _k=k: (arr == _k).mean())
                te2_store[f'TE2_{col}_frac_k{k}'][va2_idx] = safe_map(Xtr_base.iloc[va2_idx][col], frac_map)
        gc.collect()
    Xtr = pd.concat([Xtr_base, pd.DataFrame(te1_store), pd.DataFrame(te2_store)], axis=1)

    # validation / test rows: statistics of all training rows
    va_te, te_te = {}, {}
    for anc in ANCHOR_COLS:
        g_full = pd.DataFrame({'_anc_': Xtr_base[anc].values, '_y_': ytr})
        agg_full = g_full.groupby('_anc_')['_y_'].agg(STATS_TE1)
        va_anc = Xva_base[anc]
        te_anc = Xte_base[anc]
        for s in STATS_TE1:
            nm = f'TE1_{anc}_{s}'
            va_te[nm] = safe_map(va_anc, agg_full[s])
            te_te[nm] = safe_map(te_anc, agg_full[s])
        for q in QUANTILES:
            qmap = g_full.groupby('_anc_')['_y_'].quantile(q / 100.0)
            nm = f'TE1_{anc}_q{q}'
            va_te[nm] = safe_map(va_anc, qmap)
            te_te[nm] = safe_map(te_anc, qmap)
        for k in range(N_CLASSES):
            frac_map = g_full.groupby('_anc_')['_y_'].apply(lambda arr, _k=k: (arr == _k).mean())
            nm = f'TE1_{anc}_frac_k{k}'
            va_te[nm] = safe_map(va_anc, frac_map)
            te_te[nm] = safe_map(te_anc, frac_map)
    for col in combo:
        g2 = pd.DataFrame({'_c_': Xtr_base[col].values, '_y_': ytr})
        for k in range(N_CLASSES):
            frac_map = g2.groupby('_c_')['_y_'].apply(lambda arr, _k=k: (arr == _k).mean())
            nm = f'TE2_{col}_frac_k{k}'
            va_te[nm] = safe_map(Xva_base[col], frac_map)
            te_te[nm] = safe_map(Xte_base[col], frac_map)
    Xva = pd.concat([Xva_base, pd.DataFrame(va_te)], axis=1)
    Xte = pd.concat([Xte_base, pd.DataFrame(te_te)], axis=1)

    # ratio features
    for anc in ANCHOR_COLS:
        te_count = f'TE1_{anc}_count'
        te_nuniq = f'TE1_{anc}_nunique'
        te_std = f'TE1_{anc}_std'
        for Xdf in (Xtr, Xva, Xte):
            extra = {}
            if te_count in Xdf.columns and te_nuniq in Xdf.columns:
                extra[f'TE1_{anc}_count_per_nunique'] = (
                    Xdf[te_count] / Xdf[te_nuniq].replace(0, np.nan)).astype('float32').to_numpy()
            if te_std in Xdf.columns and te_count in Xdf.columns:
                extra[f'TE1_{anc}_std_per_count'] = (
                    Xdf[te_std] / Xdf[te_count].replace(0, np.nan)).astype('float32').to_numpy()
            for k2, v2 in extra.items():
                Xdf[k2] = v2

    # the raw categorical pair ids are dropped (their encodings stay)
    if pairs:
        drop_p = [c for c in pairs if c in Xtr.columns]
        Xtr = Xtr.drop(columns=drop_p)
        Xva = Xva.drop(columns=drop_p)
        Xte = Xte.drop(columns=drop_p)

    cols = list(Xtr.columns)
    feature_types = ['c' if c in cat_for_xgb else 'q' for c in cols]
    sw = compute_sample_weight('balanced', ytr).astype('float32')
    model = XGBClassifier(**XGB_PARAMS, feature_types=feature_types)
    model.fit(Xtr[cols], ytr, sample_weight=sw, eval_set=[(Xva[cols], yva)], verbose=500)
    p_va = model.predict_proba(Xva[cols]).astype('float32')
    p_te = model.predict_proba(Xte[cols]).astype('float32')
    best_iter = model.best_iteration
    del Xtr_base, Xva_base, Xte_base, Xtr, Xva, Xte, model
    gc.collect()
    return p_va, p_te, yva, best_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--force', action='store_true', help='rerun folds that are already done')
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        train, test, y, base_features, combo, pairs, cat_for_xgb = build_features()
        fold_id = folds(y)
        for k in todo:
            print(f'fold {k}', flush=True)
            p_va, p_te, yva, best_iter = run_fold(np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k), train,
                                                  test, y, base_features, combo, pairs, cat_for_xgb)
            save_part(HERE, PAIR, k, p_va, p_te)
            print(f'{PAIR} fold {k}: balanced accuracy {balanced_accuracy_score(yva, p_va.argmax(1)):.6f}, '
                  f'best iteration {best_iter}', flush=True)
    y = labels(read_train()[TARGET])
    assemble(HERE, PAIR, folds(y), y)


if __name__ == '__main__':
    main()
