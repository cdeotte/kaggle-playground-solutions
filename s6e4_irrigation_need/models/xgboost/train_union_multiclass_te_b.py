#!/usr/bin/env python
"""GPU XGBoost on three label-free pipelines + fold-local multiclass target encodings, with the original rows appended to
every training fold (2 pairs).

Label-free features (built once on train, test and the original rows; 441 numeric columns):
- pipeline A: the 11 numerics; the rule-formula flags (Soil_Moisture < 25, Rainfall_mm < 300, Temperature_C > 30,
  Wind_Speed_kmh > 10, stage Harvest / Sowing, Mulching Yes), the score 2 + 2 + 1 + 1 - 2 - 2 - 1 over them, is_low,
  is_high; frequency and percentile rank of the numerics over train + test + original; log1p / sqrt; 8 arithmetic
  interactions (evapotranspiration proxy, water deficit, soil quality, total water, moisture / rain, moisture /
  temperature, vapour-pressure deficit, dryness); per-class original priors of the categoricals; all 28 categorical
  pairs and 3 triples (string keys); counts and rare flags (count <= 50) of the categoricals, pairs and triples;
- pipeline B: digits (x 10^k) % 10 for k = -2..2 of every numeric;
- LGBM-style block: round(x) // 2, int(x) and up-to-5000-quantile bins of every numeric (string keys); value counts over
  train + test of the 19 columns; original-data mean label and count per value of the 19 columns; drift
  log1p(train share / original share) per value of the 19 columns;
- the 171 pairs of the 19 columns as integer codes, kept if they have at most n / 2 levels (135 kept).
Per fold, on the competition training rows + all 10,000 original rows:
- label std per key (categoricals, pairs, triples, numerics as strings; 50 keys), out-of-fold over an inner
  StratifiedKFold(5, seed 42) (all training rows for the validation / test rows);
- sklearn TargetEncoder(target_type multiclass, cv 5, seed 42) on 3 groups of keys (the 50 keys above, the 135 pairs,
  the 33 numeric bins), plus the row mean / std / min / max of each group's encodings (1157 columns).
Model: XGBoost multi:softprob, depth 6, lr 0.03, subsample 0.8, colsample_bytree 0.4, min_child_weight 5, alpha 0.1,
lambda 1, gamma 0.05, balanced class weights, early stopping 500 (mlogloss) on the validation fold, seed 42.
Test = mean of the 5 fold models.

Recipes:
  xgb_union_multiclass_te_original_b             as above (max_bin 256; the drift columns are 0 on the original rows)
  xgb_union_multiclass_te_original_prebinned_b   drift also computed on the original rows; every column except
                                                 Soil_Moisture, Rainfall_mm, Temperature_C, Wind_Speed_kmh is cut into
                                                 256 quantile bins of the fold's training rows; max_bin 16,000
Usage: python train_union_multiclass_te_b.py <pair> [--folds 0,1,2,3,4] [--force]   (finished folds are kept in parts/)"""
import argparse
import gc
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, TARGET_MAP, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

SEED = 42
INNER_FOLDS = 5

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
        'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']

BIN_SMALL = 256
BIN_EXCLUDE_COLS = {'Soil_Moisture', 'Rainfall_mm', 'Temperature_C', 'Wind_Speed_kmh'}
BIN_VERBOSE_EVERY = 150

XGB_PARAMS = {
    'n_estimators': 1_000_000,
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.4,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'gamma': 0.05,
    'early_stopping_rounds': 500,
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'device': 'cuda',
    'tree_method': 'hist',
    'verbosity': 0,
    'random_state': SEED,
}

RECIPES = {
    'xgb_union_multiclass_te_original_b': dict(drift_on_original=False, prebin=False, xgb={}),
    'xgb_union_multiclass_te_original_prebinned_b': dict(drift_on_original=True, prebin=True, xgb={'max_bin': 16000}),
}


# ------------------------------------------------------------------ pre-binning
def _make_bin_edges_from_train(x, n_bins):
    x = x.astype(np.float64, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    if x.size > 2_000_000:
        rs = np.random.RandomState(123)
        x = x[rs.choice(x.size, size=2_000_000, replace=False)]
    if np.unique(x).size <= n_bins:
        return None
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    try:
        edges = np.nanquantile(x, qs)
    except Exception:
        return None
    edges = np.unique(edges)
    if edges.size < 3:
        return None
    return edges


def _apply_edges(s, edges):
    x = pd.to_numeric(s, errors='coerce').to_numpy(dtype=np.float64, copy=False)
    out = np.full(x.shape, -1, dtype=np.int32)
    m = np.isfinite(x)
    out[m] = np.searchsorted(edges[1:-1], x[m], side='right').astype(np.int32)
    return out


def bin_matrices_like_xgb(X_tr, X_va, X_te, n_bins=256, exclude_cols=None, log_prefix=''):
    """Quantile bins (edges from the training rows) for every column with more than n_bins distinct values."""
    if exclude_cols is None:
        exclude_cols = set()
    cols_to_bin = [c for c in X_tr.columns if c not in exclude_cols]
    t0 = time.time()
    print(f'{log_prefix}binning {len(cols_to_bin):,}/{len(X_tr.columns):,} columns -> {n_bins} bins | '
          f'exclude={sorted(list(exclude_cols & set(X_tr.columns)))}', flush=True)
    for j, c in enumerate(cols_to_bin, 1):
        edges = _make_bin_edges_from_train(X_tr[c].to_numpy(dtype=np.float64, copy=False), n_bins)
        if edges is None:
            for df in (X_tr, X_va, X_te):
                if df[c].dtype == 'float64':
                    df[c] = df[c].astype('float32')
            continue
        X_tr[c] = _apply_edges(X_tr[c], edges).astype('int16')
        X_va[c] = _apply_edges(X_va[c], edges).astype('int16')
        X_te[c] = _apply_edges(X_te[c], edges).astype('int16')
        if j % BIN_VERBOSE_EVERY == 0:
            print(f'{log_prefix}  ...binned {j:,}/{len(cols_to_bin):,}', flush=True)
    print(f'{log_prefix}binning done in {time.time() - t0:.1f}s', flush=True)
    return X_tr, X_va, X_te


# ------------------------------------------------------------------ features
def build_features(train_raw, test_raw, orig_raw, drift_on_original):
    train = train_raw.copy()
    test = test_raw.copy()
    orig = orig_raw.copy()
    for c in CATS:
        for df in (train, test, orig):
            df[c] = df[c].astype(str).fillna('missing').str.strip()
    for c in NUMS:
        for df in (train, test, orig):
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

    A_NEW_CATS = []

    # --- formula features
    for df in (train, test, orig):
        SM, RM = df['Soil_Moisture'], df['Rainfall_mm']
        TC, WS = df['Temperature_C'], df['Wind_Speed_kmh']
        CGS, MU = df['Crop_Growth_Stage'], df['Mulching_Used']
        df['A_soil_lt_25'] = (SM < 25).astype('float32')
        df['A_rain_lt_300'] = (RM < 300).astype('float32')
        df['A_temp_gt_30'] = (TC > 30).astype('float32')
        df['A_wind_gt_10'] = (WS > 10).astype('float32')
        df['A_is_harvest'] = (CGS == 'Harvest').astype('float32')
        df['A_is_sowing'] = (CGS == 'Sowing').astype('float32')
        df['A_mulch_yes'] = (MU == 'Yes').astype('float32')
        df['A_magic_score'] = (
            2 * df['A_soil_lt_25'] + 2 * df['A_rain_lt_300'] + df['A_temp_gt_30'] + df['A_wind_gt_10']
            - 2 * df['A_is_harvest'] - 2 * df['A_is_sowing'] - df['A_mulch_yes']
        ).astype('float32')
        df['A_is_low'] = (df['A_magic_score'] <= 0).astype('float32')
        df['A_is_high'] = (df['A_magic_score'] > 3).astype('float32')

    # --- frequency
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]], axis=0).value_counts(normalize=True)
        for df in (train, test, orig):
            df[f'A_FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')

    # --- percentile rank over all rows
    _all_num = pd.concat([train[NUMS], test[NUMS], orig[NUMS]], axis=0, ignore_index=True)
    n_tr, n_te = len(train), len(test)
    for col in NUMS:
        r = _all_num[col].rank(method='average', pct=True)
        train[f'A_RANK_{col}'] = r.iloc[:n_tr].values.astype('float32')
        test[f'A_RANK_{col}'] = r.iloc[n_tr:n_tr + n_te].values.astype('float32')
        orig[f'A_RANK_{col}'] = r.iloc[n_tr + n_te:].values.astype('float32')

    # --- log1p / sqrt
    for df in (train, test, orig):
        for col in NUMS:
            v = df[col].astype('float32').clip(lower=0)
            df[f'A_LOG1P_{col}'] = np.log1p(v).astype('float32')
            df[f'A_SQRT_{col}'] = np.sqrt(v).astype('float32')

    # --- arithmetic interactions
    for df in (train, test, orig):
        df['A_ET_proxy'] = (df['Temperature_C'] * df['Wind_Speed_kmh'] * df['Sunlight_Hours']) / (df['Humidity'] + 1)
        df['A_water_deficit'] = df['Soil_Moisture'] - (df['Rainfall_mm'] + df['Previous_Irrigation_mm']) * 0.1
        df['A_soil_quality'] = df['Organic_Carbon'] / (df['Electrical_Conductivity'] + 0.1)
        df['A_total_water'] = (df['Rainfall_mm'] + df['Previous_Irrigation_mm']).astype('float32')
        df['A_moist_rain'] = df['Soil_Moisture'] / (df['Rainfall_mm'] + 1)
        df['A_moist_temp'] = df['Soil_Moisture'] / (df['Temperature_C'] + 1)
        df['A_VPD'] = df['Temperature_C'] * (1 - df['Humidity'] / 100)
        df['A_dryness'] = df['Temperature_C'] * df['Sunlight_Hours'] / (df['Rainfall_mm'] + 1)

    # --- per-class original priors of the categoricals
    y_orig = orig[TARGET].map(TARGET_MAP).astype(int)
    for col in CATS:
        for k in range(3):
            name = f'A_ORIG_{col}_k{k}'
            m = (y_orig == k).groupby(orig[col]).mean()
            m.name = name
            for df in (train, test, orig):
                df[name] = df[col].map(m).fillna((y_orig == k).mean()).astype('float32')

    # --- categorical pairs and triples
    cross_pairs = list(combinations(CATS, 2))
    cross_trips = [
        ('Crop_Growth_Stage', 'Mulching_Used', 'Crop_Type'),
        ('Season', 'Region', 'Soil_Type'),
        ('Irrigation_Type', 'Water_Source', 'Crop_Type'),
    ]
    for a, b in cross_pairs:
        name = f'A_{a}__{b}'
        for df in (train, test, orig):
            df[name] = df[a].astype(str) + '|' + df[b].astype(str)
        A_NEW_CATS.append(name)
    for a, b, c_ in cross_trips:
        name = f'A_{a}__{b}__{c_}'
        for df in (train, test, orig):
            df[name] = df[a].astype(str) + '|' + df[b].astype(str) + '|' + df[c_].astype(str)
        A_NEW_CATS.append(name)

    all_cats_count = CATS + A_NEW_CATS
    _all_cat = pd.concat([train[all_cats_count], test[all_cats_count], orig[all_cats_count]], axis=0,
                         ignore_index=True)
    for c in all_cats_count:
        vc = _all_cat[c].value_counts(dropna=False)
        for df in (train, test, orig):
            df[f'A_CNT_{c}'] = df[c].map(vc).fillna(0).astype('float32')
            df[f'A_RARE_{c}'] = (df[f'A_CNT_{c}'] <= 50).astype('float32')

    # --- numerics as strings (target-encoding keys)
    A_NUM_AS_CAT = []
    for col in NUMS:
        n = f'A_CAT_{col}'
        for df in (train, test, orig):
            df[n] = df[col].astype(str)
        A_NUM_AS_CAT.append(n)

    # --- pipeline B: digits (x 10^k) % 10, k = -2..2
    for col in NUMS:
        for k in range(-2, 3):
            n = f'B_digit_{col}_d{k}'
            for df in (train, test, orig):
                df[n] = ((df[col].astype('float64') * (10 ** k)) % 10).fillna(-1).astype('int8')

    # --- LGBM-style numeric bins (target-encoding keys)
    L_TE_COLS = []
    comb = pd.concat([train[NUMS], test[NUMS]], axis=0, ignore_index=True)
    for col in NUMS:
        s = comb[col].astype('float32')
        s_tr = s.iloc[:len(train)]
        s_te = s.iloc[len(train):]
        n = f'L_{col}_round_div2_te'
        train[n] = (s_tr.round().fillna(-1).astype('int64') // 2).astype(str).values
        test[n] = (s_te.round().fillna(-1).astype('int64') // 2).astype(str).values
        orig[n] = (orig[col].round().fillna(-1).astype('int64') // 2).astype(str)
        L_TE_COLS.append(n)
        n = f'L_{col}_int_te'
        train[n] = s_tr.fillna(-1).astype('int64').astype(str).values
        test[n] = s_te.fillna(-1).astype('int64').astype(str).values
        orig[n] = orig[col].fillna(-1).astype('int64').astype(str)
        L_TE_COLS.append(n)
        try:
            q = int(min(5000, max(2, s.nunique(dropna=True))))
            bins_q = pd.qcut(s, q=q, duplicates='drop', retbins=True)[1]
            train[f'L_{col}_quantiled_te'] = pd.cut(s_tr, bins=bins_q, include_lowest=True).astype(str).values
            test[f'L_{col}_quantiled_te'] = pd.cut(s_te, bins=bins_q, include_lowest=True).astype(str).values
            orig[f'L_{col}_quantiled_te'] = pd.cut(orig[col], bins=bins_q, include_lowest=True).astype(str)
        except Exception:
            train[f'L_{col}_quantiled_te'] = 'NA'
            test[f'L_{col}_quantiled_te'] = 'NA'
            orig[f'L_{col}_quantiled_te'] = 'NA'
        L_TE_COLS.append(f'L_{col}_quantiled_te')

    # --- value counts over train + test
    for col in NUMS + CATS:
        vc_all = pd.concat([train[col], test[col]], axis=0).value_counts()
        n = f'L_CE_{col}'
        train[n] = train[col].map(vc_all).fillna(0).astype('float32')
        test[n] = test[col].map(vc_all).fillna(0).astype('float32')
        orig[n] = orig[col].map(vc_all).fillna(0).astype('float32')

    # --- original mean label and count per value
    y_orig_num = orig[TARGET].map(TARGET_MAP).astype(float)
    for col in NUMS + CATS:
        if col in orig.columns:
            mean_m = orig.groupby(col, observed=False)[TARGET].apply(lambda g: g.map(TARGET_MAP).mean())
            size_m = orig.groupby(col, observed=False)[TARGET].size()
            for df in (train, test, orig):
                df[f'L_ORIGMEAN_{col}'] = df[col].map(mean_m).fillna(y_orig_num.mean()).astype('float32')
                df[f'L_ORIGSIZE_{col}'] = df[col].map(size_m).fillna(0).astype('float32')

    # --- drift log1p(train share / original share)
    eps = 1e-6
    drift_frames = (train, test, orig) if drift_on_original else (train, test)
    for col in NUMS + CATS:
        p_tr = train[col].value_counts(normalize=True)
        p_or = orig[col].value_counts(normalize=True)
        n = f'B_DRIFT_{col}'
        for df in drift_frames:
            df[n] = np.log1p(df[col].map(p_tr).fillna(0) / (df[col].map(p_or).fillna(eps) + eps)).astype('float32')

    # --- pairs of the 19 columns as integer codes (kept if at most n / 2 levels)
    B_PAIRS = []
    print('Building the pairs of the 19 columns...', flush=True)
    for col_a, col_b in combinations(NUMS + CATS, 2):
        name = f'B_pair_{col_a}__{col_b}'
        tr_s = train[col_a].astype(str) + '_' + train[col_b].astype(str)
        te_s = test[col_a].astype(str) + '_' + test[col_b].astype(str)
        og_s = orig[col_a].astype(str) + '_' + orig[col_b].astype(str)
        combined = pd.concat([tr_s, te_s, og_s], ignore_index=True)
        codes, _ = combined.factorize(sort=False)
        if pd.Series(codes).nunique() > len(combined) // 2:
            continue
        train[name] = codes[:len(train)]
        test[name] = codes[len(train):len(train) + len(test)]
        orig[name] = codes[len(train) + len(test):]
        B_PAIRS.append(name)
    print(f'Pairs kept: {len(B_PAIRS)}', flush=True)

    A_TE_COLUMNS = CATS + A_NEW_CATS + A_NUM_AS_CAT
    print(f'A keys: {len(A_TE_COLUMNS)} | pair keys: {len(B_PAIRS)} | bin keys: {len(L_TE_COLS)} | '
          f'columns: {len(train.columns)}', flush=True)
    return train, test, orig, A_TE_COLUMNS, B_PAIRS, L_TE_COLS


def add_multiclass_te(X_tr, X_va, X_te, y_tr, X_tr_b, X_va_b, X_te_b, cols, prefix):
    """sklearn multiclass TargetEncoder (cross-fitted on the training rows) + row mean / std / min / max."""
    te = TargetEncoder(target_type='multiclass', cv=INNER_FOLDS, random_state=SEED)
    enc_tr = te.fit_transform(X_tr[cols].values, y_tr)
    enc_va = te.transform(X_va[cols].values)
    enc_te = te.transform(X_te[cols].values)
    names = [f'{prefix}_TE_{i}' for i in range(enc_tr.shape[1])]
    X_tr_b[names] = enc_tr.astype('float32')
    X_va_b[names] = enc_va.astype('float32')
    X_te_b[names] = enc_te.astype('float32')
    for agg in ['mean', 'std', 'min', 'max']:
        X_tr_b[f'{prefix}_TEagg_{agg}'] = getattr(pd.DataFrame(enc_tr), agg)(axis=1).astype('float32').values
        X_va_b[f'{prefix}_TEagg_{agg}'] = getattr(pd.DataFrame(enc_va), agg)(axis=1).astype('float32').values
        X_te_b[f'{prefix}_TEagg_{agg}'] = getattr(pd.DataFrame(enc_te), agg)(axis=1).astype('float32').values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=list(RECIPES))
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--force', action='store_true', help='rerun folds that are already done')
    args = ap.parse_args()
    pair, recipe = args.pair, RECIPES[args.pair]
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, pair, k)]
    if todo:
        train_raw = read_train().dropna(subset=[TARGET])
        test_raw = read_test()
        orig_raw = read_original()
        print(f'Train: {train_raw.shape}, Test: {test_raw.shape}, Original: {orig_raw.shape}', flush=True)
        train, test, orig, A_TE_COLUMNS, B_TE_COLUMNS, LGBM_TE_COLS = build_features(
            train_raw, test_raw, orig_raw, recipe['drift_on_original'])
        y_train = train[TARGET].map(TARGET_MAP).values.astype(int)
        y_orig = orig[TARGET].map(TARGET_MAP).values.astype(int)
        base_features = [c for c in train.columns if c not in [TARGET, 'id'] and train[c].dtype != object]
        print(f'Base numeric features: {len(base_features)}', flush=True)
        xgb_params = {**XGB_PARAMS, **recipe['xgb']}
        fold_id = folds(y_train)

        for k in todo:
            tr_idx, val_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            X_tr_comp = train.iloc[tr_idx]
            X_va = train.iloc[val_idx]
            y_va = y_train[val_idx]
            # original rows appended to the training rows
            X_tr = pd.concat([X_tr_comp, orig], ignore_index=True)
            y_tr = np.concatenate([y_train[tr_idx], y_orig])
            X_tr_b = X_tr[base_features].copy().fillna(0).astype('float32')
            X_va_b = X_va[base_features].copy().fillna(0).astype('float32')
            X_te_b = test[base_features].copy().fillna(0).astype('float32')

            # label std per key: inner folds for the training rows, all training rows for validation / test
            X_trA = X_tr[A_TE_COLUMNS].copy().reset_index(drop=True)
            X_trA['_y_'] = y_tr
            skf_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
            te1_std_arr = np.full((len(X_tr), len(A_TE_COLUMNS)), np.nan, dtype='float32')
            for in_tr, in_va in skf_inner.split(X_trA, y_tr):
                X_in_tr = X_trA.iloc[in_tr]
                X_in_va = X_trA.iloc[in_va]
                for ci, col in enumerate(A_TE_COLUMNS):
                    stats = X_in_tr.groupby(col, observed=False)['_y_'].agg(['mean', 'std'])
                    std_m = stats['std'].fillna(0).rename(f'te1_std_{col}')
                    mapped_std = X_in_va[[col]].join(std_m, on=col)[f'te1_std_{col}'].values
                    te1_std_arr[in_va, ci] = mapped_std.astype('float32')
            te1_std_arr = np.nan_to_num(te1_std_arr, nan=0.0)
            X_tr_b[[f'te1_std_{c}' for c in A_TE_COLUMNS]] = te1_std_arr
            for col in A_TE_COLUMNS:
                stats_full = X_trA.groupby(col, observed=False)['_y_'].std().fillna(0).rename(f'te1_std_{col}')
                X_va_b[f'te1_std_{col}'] = X_va[col].map(stats_full).fillna(0).values.astype('float32')
                X_te_b[f'te1_std_{col}'] = test[col].map(stats_full).fillna(0).values.astype('float32')

            # multiclass target encodings of the 3 key groups
            add_multiclass_te(X_tr, X_va, test, y_tr, X_tr_b, X_va_b, X_te_b, A_TE_COLUMNS, 'A')
            if len(B_TE_COLUMNS) > 0:
                add_multiclass_te(X_tr, X_va, test, y_tr, X_tr_b, X_va_b, X_te_b, B_TE_COLUMNS, 'B')
            if len(LGBM_TE_COLS) > 0:
                add_multiclass_te(X_tr, X_va, test, y_tr, X_tr_b, X_va_b, X_te_b, LGBM_TE_COLS, 'L')

            if recipe['prebin']:
                keep_unbinned = {c for c in BIN_EXCLUDE_COLS if c in X_tr_b.columns}
                print(f'before binning: X_tr={X_tr_b.shape} | keep_unbinned={sorted(list(keep_unbinned))}', flush=True)
                X_tr_b, X_va_b, X_te_b = bin_matrices_like_xgb(X_tr_b, X_va_b, X_te_b, n_bins=BIN_SMALL,
                                                               exclude_cols=keep_unbinned, log_prefix=f'[fold {k}] ')

            sw = compute_sample_weight('balanced', y_tr).astype('float32')
            model = XGBClassifier(**xgb_params)
            model.fit(X_tr_b, y_tr, sample_weight=sw, eval_set=[(X_va_b, y_va)], verbose=500)
            p_va = model.predict_proba(X_va_b).astype('float32')
            p_te = model.predict_proba(X_te_b).astype('float32')
            save_part(HERE, pair, k, p_va, p_te)
            print(f'{pair} fold {k}: balanced accuracy {balanced_accuracy_score(y_va, p_va.argmax(1)):.5f}, '
                  f'best iteration {model.best_iteration}', flush=True)
            del X_tr_b, X_va_b, X_te_b, model
            gc.collect()
    y = labels(read_train()[TARGET])
    assemble(HERE, pair, folds(y), y)


if __name__ == '__main__':
    main()
