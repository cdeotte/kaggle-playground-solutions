#!/usr/bin/env python
"""Shallow GPU XGBoost on nearest-original "snap" values, decimal digits and radix interactions.

Features (built once on train + test; the original rows only provide the snap targets and the priors):
- the 10 low-cardinality columns (strings + numerics with < 100 values) as native categoricals and the 4 continuous
  columns (LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress) as float32;
- digit blocks of the 4 continuous columns written with 6 decimals: the 6-digit decimal string, decimal digits 1-3,
  the last two integer digits and the decimal length (categoricals);
- snap: every train / test value that does not occur in the original data is replaced by the nearest original value
  (<col>_snap) and the difference snap - value (<col>_snap_diff);
- radix interactions: round(1000 x snap) as an integer, and that integer + RADIX x (code of a low-cardinality column)
  for each of the 10 low-cardinality columns (RADIX = 50,000,000), i.e. the snapped value within each category;
- smoothed original target means (smoothing 25) and log original counts of the 10 low-cardinality columns.
Model: XGBoost hist on GPU with native categoricals, min_child_weight 5, max_bin 16,000, subsample 0.85,
colsample_bytree 0.85, early stopping 300 on the fold's validation rows (AUC), seed 42 + fold (fold = 1..5).
Probabilities clipped to [1e-7, 1 - 1e-7]. Test = mean of the 5 fold models. The original rows are never training rows.

Recipes:
  xgb_snap_radix_depth2             depth 2, lr 0.05, up to 50,000 rounds; 114 columns
  xgb_snap_radix_lapnumber_depth1   depth 1 (stumps), lr 0.1, up to 1,000,000 rounds, plus an extra radix block on
                                    LapNumber (round(1000 x LapNumber) + RADIX x code); 125 columns
The feature build (mostly the snap replacement) takes 1.5-2 hours on CPU; the folds then take 1-2 minutes each.
Usage: python train_snap_radix_xgb.py <pair> [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

FOLDER = 'classic'
RANDOM_STATE = 42
N_THREADS = 32
RADIX = 50_000_000
SNAP_COLS = ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress']

XGB_BASE = dict(
    min_child_weight=5,
    max_bin=16000,
    subsample=0.85,
    colsample_bytree=0.85,
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    enable_categorical=True,
    random_state=RANDOM_STATE,
    n_jobs=N_THREADS,
    early_stopping_rounds=300,
    device='cuda',
)
RECIPES = {
    'xgb_snap_radix_depth2': dict(extra_radix_col=None,
                                  xgb=dict(n_estimators=50000, learning_rate=0.05, max_depth=2)),
    'xgb_snap_radix_lapnumber_depth1': dict(extra_radix_col='LapNumber',
                                            xgb=dict(n_estimators=1000000, learning_rate=0.1, max_depth=1)),
}


def safe_name(c):
    return ''.join(ch if ch.isalnum() else '_' for ch in str(c)).strip('_')


def numeric_to_string(s, decimals=6):
    x = pd.to_numeric(s, errors='coerce').astype(np.float64)
    return x.map(lambda v: f'{v:.{decimals}f}' if np.isfinite(v) else 'nan').astype('string')


def infer_low_card(train, test):
    cols = []
    feature_cols = [c for c in train.columns if c not in [ID, TARGET]]
    for c in feature_cols:
        if pd.api.types.is_object_dtype(train[c]) or str(train[c].dtype).startswith('string'):
            cols.append(c)
        else:
            nun = pd.concat([train[c], test[c]], ignore_index=True).nunique(dropna=False)
            if nun < 100:
                cols.append(c)
    first = [c for c in ['Driver', 'Compound', 'Race', 'Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position',
                         'Position_Change'] if c in feature_cols]
    out = []
    for c in first + cols:
        if c not in out:
            out.append(c)
    return out


def build_snap_map(combo_s, orig_s, colname):
    """{value string not in the original data: nearest original value string}"""
    combo_str = combo_s.astype('string')
    orig_str = orig_s.astype('string')
    orig_unique = pd.Series(orig_str.dropna().unique(), dtype='string')
    orig_num = pd.to_numeric(orig_unique, errors='coerce').astype('float64')
    keep = orig_num.notna()
    orig_unique = orig_unique[keep].reset_index(drop=True)
    orig_num = orig_num[keep].to_numpy()
    combo_unique = pd.Series(combo_str.dropna().unique(), dtype='string')
    rare = combo_unique[~combo_unique.isin(orig_unique)].reset_index(drop=True)
    rare_num = pd.to_numeric(rare, errors='coerce').astype('float64')
    ok = rare_num.notna()
    rare = rare[ok].reset_index(drop=True)
    rare_num = rare_num[ok].to_numpy()
    snap_map = {}
    if len(rare) == 0 or len(orig_num) == 0:
        return snap_map
    order = np.argsort(orig_num)
    orig_num = orig_num[order]
    orig_unique = orig_unique.iloc[order].reset_index(drop=True)
    pos = np.searchsorted(orig_num, rare_num)
    left = np.clip(pos - 1, 0, len(orig_num) - 1)
    right = np.clip(pos, 0, len(orig_num) - 1)
    choose_right = np.abs(rare_num - orig_num[right]) < np.abs(rare_num - orig_num[left])
    nearest = np.where(choose_right, right, left)
    for s_val, idx in zip(rare.tolist(), nearest):
        snap_map[s_val] = orig_unique.iloc[int(idx)]
    print(f'{colname}: snap {len(snap_map):,} values to the nearest original value', flush=True)
    return snap_map


def apply_snap(combo_df, orig_df, col):
    snap_col = f'{safe_name(col)}_snap'
    diff_col = f'{safe_name(col)}_snap_diff'
    combo_df[snap_col] = combo_df[col].astype('string')
    mp = build_snap_map(combo_df[col], orig_df[col], col)
    if mp:
        combo_df[snap_col] = combo_df[snap_col].replace(mp)
    orig_num = pd.to_numeric(combo_df[col], errors='coerce')
    snap_num = pd.to_numeric(combo_df[snap_col], errors='coerce')
    combo_df[diff_col] = (snap_num - orig_num).astype('float32')
    return snap_col, diff_col


def add_digit_block(combo, col, prefix):
    s = numeric_to_string(combo[col])
    parts = s.str.split('.', n=1, expand=True)
    left = parts[0].str.replace('-', '', regex=False).str.zfill(5)
    dec = parts[1].fillna('').str.pad(6, side='right', fillchar='0')
    combo[f'{prefix}_dec'] = dec.astype('string')
    combo[f'{prefix}0'] = dec.str[0].fillna('-1').astype('string')
    combo[f'{prefix}1'] = dec.str[1].fillna('-1').astype('string')
    combo[f'{prefix}2'] = dec.str[2].fillna('-1').astype('string')
    combo[f'{prefix}_int0'] = left.str[-1].fillna('-1').astype('string')
    combo[f'{prefix}_int1'] = left.str[-2].fillna('-1').astype('string')
    combo[f'{prefix}_len'] = dec.str.len().fillna(0).astype('int16')
    return [f'{prefix}_dec', f'{prefix}0', f'{prefix}1', f'{prefix}2', f'{prefix}_int0', f'{prefix}_int1', f'{prefix}_len']


def build_features(extra_radix_col=None):
    train, test, orig = read_train(), read_test(), read_original()
    y = pd.to_numeric(train[TARGET], errors='coerce').fillna(0).astype(np.int8).to_numpy()
    y_orig = pd.to_numeric(orig[TARGET], errors='coerce').fillna(0).astype(np.int8).to_numpy()
    test[TARGET] = 0
    combo = pd.concat([train, test], axis=0, ignore_index=True)
    n_train = len(train)
    low_card_cols = infer_low_card(train, test)
    snap_cols = [c for c in SNAP_COLS if c in combo.columns and c in orig.columns]
    print(f'low-cardinality columns: {low_card_cols}\nsnap columns: {snap_cols}', flush=True)

    for col in snap_cols:
        combo[col] = numeric_to_string(combo[col])
        orig[col] = numeric_to_string(orig[col])

    digit_cols = []
    for col in snap_cols:
        digit_cols += add_digit_block(combo, col, safe_name(col))

    snap_created = []
    for col in snap_cols:
        snap_created += list(apply_snap(combo, orig, col))

    for col in snap_cols + [c for c in snap_created if c.endswith('_snap')]:
        combo[col] = pd.to_numeric(combo[col], errors='coerce').astype('float32')
        med = float(np.nanmedian(combo.loc[:n_train - 1, col].to_numpy()))
        combo[col] = combo[col].fillna(med).astype('float32')
    for col in [c for c in snap_created if c.endswith('_diff')]:
        combo[col] = pd.to_numeric(combo[col], errors='coerce').fillna(0).astype('float32')

    lowcard_codes = {}
    for c in low_card_cols:
        s = combo[c].astype('string').fillna('NA')
        cat = pd.Categorical(s)
        lowcard_codes[c] = cat.codes.astype('int32')
        combo[c] = cat

    radix_cols = []
    for snap_col in [c for c in snap_created if c.endswith('_snap')]:
        p = safe_name(snap_col)
        snap_i = f'{p}_i'
        combo[snap_i] = np.round(combo[snap_col].astype('float32') * 1000.0).astype('int32')
        radix_cols.append(snap_i)
        for c in low_card_cols:
            newc = f'{p}_radix_{safe_name(c)}'
            combo[newc] = (combo[snap_i].astype('int64') + lowcard_codes[c].astype('int64') * RADIX).astype('int64')
            radix_cols.append(newc)
    if extra_radix_col is not None and extra_radix_col in combo.columns:
        combo['extra_radix_i'] = np.round(pd.to_numeric(combo[extra_radix_col], errors='coerce').fillna(0).astype('float32') * 1000.0).astype('int64')
        combo['extra_radix_i'] = np.clip(combo['extra_radix_i'], 0, RADIX - 1).astype('int64')
        radix_cols.append('extra_radix_i')
        for c in low_card_cols:
            newc = f'extra_radix_{safe_name(c)}'
            combo[newc] = (combo['extra_radix_i'] + lowcard_codes[c].astype('int64') * RADIX).astype('int64')
            radix_cols.append(newc)
    print(f'radix features: {len(radix_cols)}', flush=True)

    # original target priors / counts (only the low-cardinality columns exist in the original data)
    prior_cols = low_card_cols + [c for c in digit_cols if c in combo.columns]
    orig_base = orig.copy()
    for c in low_card_cols:
        if c in orig_base.columns:
            orig_base[c] = orig_base[c].astype(str).fillna('NA')
    for c in prior_cols:
        if c not in combo.columns or c not in orig_base.columns:
            continue
        stat = pd.DataFrame({'k': orig_base[c].astype(str), 'y': y_orig.astype('float32')}).groupby('k')['y'].agg(['mean', 'count'])
        smooth = (stat['mean'] * stat['count'] + float(y_orig.mean()) * 25.0) / (stat['count'] + 25.0)
        combo[f'orig_mean_{safe_name(c)}'] = combo[c].astype(str).map(smooth).fillna(float(y_orig.mean())).astype('float32')
        combo[f'orig_logcnt_{safe_name(c)}'] = np.log1p(combo[c].astype(str).map(stat['count']).fillna(0)).astype('float32')

    base_feats = low_card_cols + snap_cols
    engineered = digit_cols + snap_created + radix_cols + [c for c in combo.columns if c.startswith('orig_mean_') or c.startswith('orig_logcnt_')]
    all_feats = []
    for c in base_feats + engineered:
        if c in combo.columns and c not in all_feats and c not in [ID, TARGET]:
            all_feats.append(c)

    for c in digit_cols:
        if c in combo.columns:
            combo[c] = combo[c].astype('category')
    X_all = combo.iloc[:n_train][all_feats].copy()
    T_all = combo.iloc[n_train:][all_feats].copy()
    for c in X_all.select_dtypes(include=['category']).columns.tolist():
        X_all[c] = X_all[c].astype('object').astype('category')
        T_all[c] = T_all[c].astype('object').astype('category')
    print(f'{len(all_feats)} features, train {X_all.shape}, test {T_all.shape}', flush=True)
    return X_all, T_all, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=list(RECIPES))
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    pair, r = args.pair, RECIPES[args.pair]
    out = model_dir(FOLDER)
    todo = [k for k in fold_list(args.folds) if not have_part(out, pair, k)]
    if not todo:
        y = read_train()[TARGET].astype(np.int8).to_numpy()
        assemble(out, pair, folds(y), y)
        return
    X_all, T_all, y = build_features(r['extra_radix_col'])
    fold_id = folds(y)
    for k in todo:
        fold = k + 1
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        X_tr, y_tr, X_va, y_va = X_all.iloc[tr_idx], y[tr_idx], X_all.iloc[va_idx], y[va_idx]
        model = xgb.XGBClassifier(**{**XGB_BASE, **r['xgb'], 'random_state': RANDOM_STATE + fold})
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=200)
        va_pred = np.clip(model.predict_proba(X_va)[:, 1].astype(np.float32), 1e-7, 1 - 1e-7)
        te_pred = np.clip(model.predict_proba(T_all)[:, 1].astype(np.float32), 1e-7, 1 - 1e-7)
        save_part(out, pair, k, va_pred, te_pred)
        print(f'{pair} fold {k}: AUC {roc_auc_score(y_va, va_pred):.6f}, best iteration '
              f'{getattr(model, "best_iteration", -1)}', flush=True)
        del model
        gc.collect()
    assemble(out, pair, fold_id, y)


if __name__ == '__main__':
    main()
