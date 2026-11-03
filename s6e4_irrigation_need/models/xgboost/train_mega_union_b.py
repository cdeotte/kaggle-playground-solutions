#!/usr/bin/env python
"""GPU XGBoost on a large union of label-free feature blocks + nested per-class target encodings (pair
xgb_mega_union_snap_radix_nested_te_b).

Label-free blocks (built once on train + test + original rows, 882 columns; the original rows are not training rows):
- the raw columns (including the row id) and, for all 11 numerics, the nearest original value of every value that does
  not occur in the original data (<col>_snap) and snap - value (<col>_snap_diff);
- frequency and percentile rank of the 11 numerics and their snaps; log1p / sqrt / 1 / (1 + x) transforms;
- the rule formula (score, is_low / is_medium / is_high), threshold distances (Soil_Moisture - 25, Rainfall_mm - 300,
  Temperature_C - 30, Wind_Speed_kmh - 10) and 4 interactions of the formula numerics;
- 12-quantile bins of the numerics and their snaps, 6 categorical pairs and 1 triple; their counts and rare flags;
- drift log1p(train share / original share) of the categoricals and snaps; per-class original-data priors of the first
  20 of (categoricals, bins, snaps); rounded numerics; digit positions of the numerics and snaps;
- an LGBM-style block: every column with more than 12 distinct values as a string key (`*_te`), its count, its
  per-class original priors / original count and its train / original count ratio; rounded, integer, 10-quantile and
  log-10-quantile bins of the numerics (`*_te`); the 4 integer + 2 decimal string digits of the numerics and snaps and the
  15 pairs of the first 6 decimal-digit columns (`*_te`);
- radix integers round(100 x snap) + 100,000 x (category code) of 5 numerics x 8 categoricals and the category codes;
- 6 categorical pair keys (capped at 20,000 levels).
Per fold: every `*_te` key gets 3 binary target encodings (one per class, sklearn TargetEncoder, smooth 'auto'),
out-of-fold over an inner StratifiedKFold(5, seed 42) of the training rows and fitted on all training rows for the
validation / test rows; the remaining string columns are integer-coded (1066 columns). The training / validation codes
come from the training + validation rows, the test codes from the test rows (two calls of final_factorize).
Model: XGBoost multi:softprob, depth 6, lr 0.03, subsample 0.8, colsample_bytree 0.45, min_child_weight 6, gamma 0.1,
alpha 0.5, lambda 4, max_bin 4096, balanced class weights, early stopping 200 (mlogloss) on the validation fold, seed 42.
Competition rows only. Test = mean of the 5 fold models.
The feature build takes about 10 minutes, a fold 10-16 minutes (mostly the target encodings) on one GPU.
Usage: python train_mega_union_b.py [--folds 0,1,2,3,4] [--force]   (finished folds are kept in parts/)"""
import argparse
import gc
import itertools
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
from common.data import TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'xgb_mega_union_snap_radix_nested_te_b'
SEED = 42
INNER_FOLDS = 5
N_THREADS = 32
N_CLASSES = 3

HIGH_CARD_THRESHOLD = 12
PAIR_MAX_UNIQUE = 20000
RADIX = 100_000
CAT_RARE_THRESH = 50

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Organic_Carbon', 'Electrical_Conductivity', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Temperature_C', 'Soil_Moisture',
        'Humidity', 'Previous_Irrigation_mm', 'Rainfall_mm']

CROSS_PAIRS = [
    ('Crop_Type', 'Season'),
    ('Soil_Type', 'Region'),
    ('Irrigation_Type', 'Crop_Type'),
    ('Season', 'Crop_Growth_Stage'),
    ('Water_Source', 'Irrigation_Type'),
    ('Region', 'Crop_Growth_Stage'),
]
TRIPLE = [('Crop_Type', 'Season', 'Region')]
PAIR_FEATURES = [
    ('Crop_Type', 'Season'),
    ('Soil_Type', 'Irrigation_Type'),
    ('Region', 'Crop_Growth_Stage'),
    ('Crop_Type', 'Region'),
    ('Season', 'Water_Source'),
    ('Mulching_Used', 'Crop_Growth_Stage'),
]
ROUND_SPECS = {
    'Temperature_C': [0, 1],
    'Soil_Moisture': [0, 1],
    'Rainfall_mm': [0, 1, -1],
    'Wind_Speed_kmh': [0, 1],
    'Humidity': [0, 1],
    'Previous_Irrigation_mm': [0, 1, -1],
    'Soil_pH': [1],
    'Organic_Carbon': [1],
}
DIGIT_BASE_COLS = NUMS[:]
RADIX_NUMS = ['Soil_Moisture', 'Temperature_C', 'Rainfall_mm', 'Humidity', 'Previous_Irrigation_mm']

XGB_PARAMS = dict(
    n_estimators=1_000_000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.80,
    colsample_bytree=0.45,
    min_child_weight=6,
    gamma=0.10,
    reg_alpha=0.50,
    reg_lambda=4.00,
    max_bin=4096,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    tree_method='hist',
    enable_categorical=False,
    device='cuda',
    random_state=SEED,
    n_jobs=N_THREADS,
    early_stopping_rounds=200,
    verbosity=0,
)

T0 = time.perf_counter()


def log(msg):
    print(f'[{time.perf_counter() - T0:7.0f}s] {msg}', flush=True)


def dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ------------------------------------------------------------------ snap
def build_snap_map(combo_vals, orig_vals, colname):
    """Map every value that does not occur in the original data to the nearest original value."""
    combo_u = np.unique(combo_vals[~np.isnan(combo_vals)])
    orig_u = np.unique(orig_vals[~np.isnan(orig_vals)])
    not_in = np.setdiff1d(combo_u, orig_u)
    if len(not_in) == 0:
        log(f'{colname}: no snap needed')
        return {}
    orig_sorted = np.sort(orig_u)
    pos = np.searchsorted(orig_sorted, not_in)
    snap = {}
    for v, p in zip(not_in, pos):
        if p <= 0:
            snap[v] = orig_sorted[0]
        elif p >= len(orig_sorted):
            snap[v] = orig_sorted[-1]
        else:
            lo, hi = orig_sorted[p - 1], orig_sorted[p]
            snap[v] = lo if abs(v - lo) <= abs(v - hi) else hi
    log(f'{colname}: snapping {len(snap):,} non-original values')
    return snap


def apply_snap(combined, orig_df, col):
    v = combined[col].astype('float32').values
    vo = orig_df[col].astype('float32').values
    smap = build_snap_map(v, vo, col)
    snapped = v.copy()
    for src, dst in smap.items():
        snapped[v == src] = dst
    combined[f'{col}_snap'] = snapped.astype('float32')
    combined[f'{col}_snap_diff'] = (snapped - v).astype('float32')


# ------------------------------------------------------------------ feature blocks
def add_freq_rank_features(combined, num_cols):
    out = {}
    for c in num_cols:
        if c not in combined.columns:
            continue
        s = combined[c].astype('float32')
        vc = s.value_counts(normalize=True, dropna=False)
        out[f'FREQ_{c}'] = s.map(vc).fillna(0).astype('float32')
        out[f'RANK_{c}'] = s.rank(method='average', pct=True).astype('float32')
    return pd.DataFrame(out, index=combined.index)


def add_num_transforms(combined, num_cols):
    out = {}
    for c in num_cols:
        if c not in combined.columns:
            continue
        v = combined[c].astype('float32')
        out[f'LOG1P_{c}'] = np.log1p(v.clip(lower=0)).astype('float32')
        out[f'SQRT_{c}'] = np.sqrt(v.clip(lower=0)).astype('float32')
        out[f'INV1P_{c}'] = (1.0 / (1.0 + v.clip(lower=0))).astype('float32')
    return pd.DataFrame(out, index=combined.index)


def add_formula_features(combined):
    SM = combined['Soil_Moisture'].astype('float32')
    RM = combined['Rainfall_mm'].astype('float32')
    TC = combined['Temperature_C'].astype('float32')
    WS = combined['Wind_Speed_kmh'].astype('float32')
    CGS = combined['Crop_Growth_Stage'].astype('string')
    MU = combined['Mulching_Used'].astype('string')
    score = ((SM < 25).astype(int) * 2
             + (RM < 300).astype(int) * 2
             + (TC > 30).astype(int)
             + (WS > 10).astype(int)
             - (CGS == 'Harvest').astype(int) * 2
             - (CGS == 'Sowing').astype(int) * 2
             - (MU == 'Yes').astype(int))
    out = {}
    out['formula_score'] = score.astype('float32')
    out['is_low'] = (score <= 0).astype('float32')
    out['is_medium'] = ((score > 0) & (score <= 3)).astype('float32')
    out['is_high'] = (score > 3).astype('float32')
    out['SM_thr'] = (SM - 25).astype('float32')
    out['RM_thr'] = (RM - 300).astype('float32')
    out['TC_thr'] = (TC - 30).astype('float32')
    out['WS_thr'] = (WS - 10).astype('float32')
    out['SM_x_RM'] = (SM * RM).astype('float32')
    out['TC_x_WS'] = (TC * WS).astype('float32')
    out['RM_div_SM_p1'] = (RM / (SM + 1.0)).astype('float32')
    out['SM_div_RM_p1'] = (SM / (RM + 1.0)).astype('float32')
    return pd.DataFrame(out, index=combined.index)


def add_bins_and_cross_cats(combined):
    out = pd.DataFrame(index=combined.index)
    for col in NUMS:
        if col not in combined.columns:
            continue
        s = combined[col].astype('float32')
        q = min(12, max(2, int(s.nunique(dropna=True))))
        try:
            out[f'{col}_bin'] = pd.qcut(s, q=q, duplicates='drop').astype('string')
        except Exception:
            out[f'{col}_bin'] = s.round(1).astype('string')
        snap_c = f'{col}_snap'
        if snap_c in combined.columns:
            ss = combined[snap_c].astype('float32')
            qs = min(12, max(2, int(ss.nunique(dropna=True))))
            try:
                out[f'{snap_c}_bin'] = pd.qcut(ss, q=qs, duplicates='drop').astype('string')
            except Exception:
                out[f'{snap_c}_bin'] = ss.round(1).astype('string')
    for a, b in CROSS_PAIRS:
        if a in combined.columns and b in combined.columns:
            out[f'{a}__{b}'] = combined[a].astype('string') + '|' + combined[b].astype('string')
    for a, b, c in TRIPLE:
        if all(x in combined.columns for x in [a, b, c]):
            out[f'{a}__{b}__{c}'] = (combined[a].astype('string') + '|'
                                     + combined[b].astype('string') + '|'
                                     + combined[c].astype('string'))
    return out


def add_cat_counts_and_rare(combined, cat_cols, rare_thresh=50):
    out = {}
    for c in cat_cols:
        if c not in combined.columns:
            continue
        vc = combined[c].astype('string').value_counts(dropna=False)
        cnt = combined[c].astype('string').map(vc).fillna(0).astype('float32')
        out[f'CAT_CNT_{c}'] = cnt
        out[f'CAT_RARE_{c}'] = (cnt <= rare_thresh).astype('float32')
    return pd.DataFrame(out, index=combined.index)


def add_drift_ratio_from_orig(combined, n_train, n_test, cols, eps=1e-6):
    """log1p(train share / original share) of every value."""
    orig_df = combined.iloc[n_train + n_test:].copy()
    out = {}
    for c in cols:
        if c not in combined.columns:
            continue
        p_tr = combined.iloc[:n_train][c].astype('string').value_counts(normalize=True, dropna=False)
        p_or = orig_df[c].astype('string').value_counts(normalize=True, dropna=False)
        num = combined[c].astype('string').map(p_tr).fillna(0)
        den = combined[c].astype('string').map(p_or).fillna(eps) + eps
        out[f'DRIFT_{c}'] = np.log1p(num / den).astype('float32')
    return pd.DataFrame(out, index=combined.index)


def add_orig_proba_features(combined, n_train, n_test, y_orig, cols):
    """Per-class original-data priors of each column."""
    orig_df = combined.iloc[n_train + n_test:].copy()
    orig_df['_y_'] = y_orig.astype('int32')
    overall_means = [(y_orig == k).mean() for k in range(N_CLASSES)]
    out = {}
    for c in cols:
        if c not in orig_df.columns:
            continue
        for k in range(N_CLASSES):
            m = orig_df.groupby(c, dropna=False)['_y_'].apply(lambda y: (y == k).mean())
            out[f'ORIG_proba_{c}_k{k}'] = combined[c].map(m).fillna(overall_means[k]).astype('float32')
    return pd.DataFrame(out, index=combined.index)


def add_round_features(combined, round_specs):
    out = {}
    for c, rr in round_specs.items():
        if c not in combined.columns:
            continue
        s = combined[c].astype('float32')
        for r in rr:
            out[f'B_round_{c}_r{r}'] = s.round(r).astype('float32')
    return pd.DataFrame(out, index=combined.index)


def add_digit_position_features(combined, base_cols, max_dec_digits=2):
    out = {}
    for col in base_cols:
        if col not in combined.columns:
            continue
        vals = combined[col].astype('float32').fillna(0)
        abs_vals = vals.abs().round(max_dec_digits)
        max_val = int(abs_vals.max()) if len(abs_vals) else 0
        max_int_digits = 1 if max_val == 0 else int(np.log10(max(max_val, 1))) + 1
        for i in range(max_int_digits):
            out[f'{col}_digit_pos_{i + 1}'] = ((abs_vals // (10 ** i)).astype(np.int64) % 10).astype('int16')
        for i in range(1, max_dec_digits + 1):
            out[f'{col}_digit_dec_{i}'] = ((abs_vals * (10 ** i)).round().astype(np.int64) % 10).astype('int16')
    return pd.DataFrame(out, index=combined.index)


def add_string_digit_features(combined, cols):
    out = {}
    for col in cols:
        if col not in combined.columns:
            continue
        s = combined[col].astype('string').fillna('NA')
        parts = s.str.split('.', n=1, expand=True)
        if parts.shape[1] == 1:
            left = parts[0].fillna('')
            right = pd.Series([''] * len(s), index=s.index)
        else:
            left, right = parts[0].fillna(''), parts[1].fillna('')
        left = left.str.replace('-', '', regex=False)
        right = right.str.pad(2, side='right', fillchar='0')
        for i in range(4):
            out[f'{col}_digit{i}'] = left.str.zfill(4).str[-4 + i].fillna('0')
        for i in range(2):
            out[f'{col}_decimal_digit{i}'] = right.str[i].fillna('0')
    return out


def create_count_features(X, cols_list):
    out = {}
    for cols in cols_list:
        name = f'count({"_".join(cols)})'
        out[name] = X.groupby(list(cols), dropna=False).transform('size').astype('float32')
    return pd.DataFrame(out, index=X.index)


def create_groupby_orig_features(X_keys, col, orig_df, y_orig):
    """Per-class original-data mean and original count of a key column."""
    out = {}
    xk = X_keys[col].astype('string')
    od = orig_df[col].astype('string')
    od_df = pd.DataFrame({'k': od, 'y': y_orig})
    for k in range(N_CLASSES):
        m = od_df.groupby('k')['y'].apply(lambda yy: (yy == k).mean())
        out[f'L_gb_{col}_k{k}'] = xk.map(m).fillna((y_orig == k).mean()).astype('float32')
    size_m = od_df.groupby('k')['y'].size()
    out[f'L_gb_{col}_size'] = xk.map(size_m).fillna(0).astype('float32')
    return pd.DataFrame(out, index=X_keys.index)


def add_lgbm_port_features(combined, n_train, n_test, y_orig):
    out_blocks = []
    orig_df = combined.iloc[n_train + n_test:].copy()
    base_cols = [c for c in combined.columns if c != TARGET]
    high_card = [c for c in base_cols if combined[c].astype('string').nunique(dropna=False) > HIGH_CARD_THRESHOLD]
    if high_card:
        # string keys, target-encoded inside the folds
        te1 = combined[high_card].astype('string').rename(columns={c: f'{c}_te' for c in high_card})
        out_blocks.append(te1)
        ce1 = create_count_features(combined[high_card].astype('string'), [(c,) for c in high_card]).add_prefix('L_')
        out_blocks.append(ce1)
        for c in high_card:
            if c in orig_df.columns:
                out_blocks.append(create_groupby_orig_features(combined[[c]], c, orig_df[[c]], y_orig))
        for c in high_card:
            vc_comb = combined.iloc[:n_train][c].astype('string').value_counts(dropna=False)
            vc_orig = orig_df[c].astype('string').value_counts(dropna=False)
            ratio = (combined[c].astype('string').map(vc_comb).fillna(0)
                     / (combined[c].astype('string').map(vc_orig).fillna(0) + 1))
            out_blocks.append(pd.DataFrame({f'L_count_ratio_{c}': ratio.astype('float32')}, index=combined.index))

    num_bins = {}
    for col in NUMS:
        if col not in combined.columns:
            continue
        s = combined[col].astype('float32')
        num_bins[f'L_{col}_round_te'] = s.round(1).astype('string')
        num_bins[f'L_{col}_int_te'] = s.fillna(-1).astype('int64').astype('string')
        try:
            q = int(min(10, max(2, s.nunique(dropna=True))))
            num_bins[f'L_{col}_quantiled_te'] = pd.qcut(s, q=q, duplicates='drop').astype('string')
        except Exception:
            num_bins[f'L_{col}_quantiled_te'] = pd.Series(['NA'] * len(combined), index=combined.index, dtype='string')
        try:
            slog = np.log1p(s.clip(lower=0))
            bins = np.unique(np.nanquantile(slog.dropna(), np.linspace(0, 1, 11)))
            num_bins[f'L_{col}_log_binned_te'] = (
                pd.cut(slog, bins=bins, duplicates='drop').astype('string') if len(bins) >= 2
                else pd.Series(['NA'] * len(combined), index=combined.index, dtype='string'))
        except Exception:
            num_bins[f'L_{col}_log_binned_te'] = pd.Series(['NA'] * len(combined), index=combined.index, dtype='string')
    if num_bins:
        out_blocks.append(pd.DataFrame(num_bins, index=combined.index))

    digit_src = [c for c in NUMS if c in combined.columns]
    digit_src += [f'{c}_snap' for c in NUMS if f'{c}_snap' in combined.columns]
    if digit_src:
        digit_df = pd.DataFrame(add_string_digit_features(combined, digit_src), index=combined.index)
        out_blocks.append(digit_df)
        dec_cols = [c for c in digit_df.columns if 'decimal_digit' in c][:8]
        if len(dec_cols) >= 2:
            combs = {}
            for a, b in itertools.combinations(dec_cols[:6], 2):
                combs[f'L_digits_{a}__{b}_te'] = digit_df[a].astype('string') + '|' + digit_df[b].astype('string')
            out_blocks.append(pd.DataFrame(combs, index=combined.index))

    if not out_blocks:
        return pd.DataFrame(index=combined.index)
    return pd.concat(out_blocks, axis=1)


def add_radix_features(combined):
    """Category codes and radix integers round(100 x snap) + RADIX x (category code)."""
    code_df = {}
    for c in CATS:
        if c in combined.columns:
            codes, _ = pd.factorize(combined[c].astype('string'), sort=False)
            code_df[f'code_{c}'] = pd.Series(codes, index=combined.index).astype('int32')
    code_df = pd.DataFrame(code_df, index=combined.index)
    out2 = {}
    for num_c in RADIX_NUMS:
        snap_c = f'{num_c}_snap'
        use_c = snap_c if snap_c in combined.columns else num_c
        if use_c not in combined.columns:
            continue
        v_i = np.round(combined[use_c].fillna(0).astype('float32') * 100.0).astype('int32')
        for c in CATS:
            k = f'code_{c}'
            if k in code_df.columns:
                out2[f'radix_{num_c}_{c}'] = (v_i + code_df[k] * RADIX).astype('int32')
    return pd.concat([code_df, pd.DataFrame(out2, index=combined.index)], axis=1)


def make_pair_feature_from_combined(combined, cols, pair_max_unique=20000):
    name = 'B_pair_' + '__'.join(cols)
    tmp = combined[list(cols)].astype('string').fillna('NA')
    joined = tmp.iloc[:, 0]
    for c in cols[1:]:
        joined = joined + '|' + tmp[c]
    vc = joined.value_counts(dropna=False)
    if len(vc) > pair_max_unique:
        keep = set(vc.nlargest(pair_max_unique).index)
        joined = joined.where(joined.isin(keep), other='__OTHER__')
    return pd.Series(joined, index=combined.index, name=name, dtype='string')


# ------------------------------------------------------------------ fold encodings
def nested_target_encode_mc(train_raw, apply_raws, y, te_cols):
    """3 binary target encodings (one per class) per *_te column: out-of-fold over the inner folds for the training rows,
    an encoder fitted on all training rows for each frame of apply_raws (validation, test)."""
    y = np.asarray(y)
    new_train = pd.DataFrame(index=train_raw.index)
    new_apply = [pd.DataFrame(index=a.index) for a in apply_raws]
    if len(te_cols) == 0:
        return new_train, new_apply
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    for c in te_cols:
        s_train = train_raw[c].astype('string').fillna('NA')
        s_apply = [a[c].astype('string').fillna('NA') for a in apply_raws]
        for k in range(N_CLASSES):
            y_bin = (y == k).astype(np.float32)
            oof = np.zeros(len(train_raw), dtype=np.float32)
            for itr, iva in inner_cv.split(s_train, y):
                enc = TargetEncoder(cv=2, smooth='auto', target_type='binary')
                enc.fit(s_train.iloc[itr].to_frame(c), y_bin[itr])
                oof[iva] = np.asarray(enc.transform(s_train.iloc[iva].to_frame(c))).ravel().astype(np.float32)
            enc_full = TargetEncoder(cv=2, smooth='auto', target_type='binary')
            enc_full.fit(s_train.to_frame(c), y_bin)
            new_name = c.replace('_te', f'_TE_k{k}') if c.endswith('_te') else f'{c}_TE_k{k}'
            new_train[new_name] = oof
            for s, out in zip(s_apply, new_apply):
                out[new_name] = np.asarray(enc_full.transform(s.to_frame(c))).ravel().astype(np.float32)
    return new_train, new_apply


def final_factorize(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    cat_cols = [c for c in train_df.columns if str(train_df[c].dtype).startswith(('object', 'string', 'category'))]
    for c in cat_cols:
        comb = pd.concat([train_df[c].astype('string'), test_df[c].astype('string')], axis=0, ignore_index=True)
        codes, _ = pd.factorize(comb, sort=False)
        codes = codes.astype('int32') + 1
        codes[codes < 1] = 0
        train_df[c] = codes[:len(train_df)]
        test_df[c] = codes[len(train_df):]
    for df in (train_df, test_df):
        for c in df.columns:
            if str(df[c].dtype).startswith(('object', 'string', 'category')):
                df[c] = pd.factorize(df[c].astype('string'), sort=False)[0].astype('int32') + 1
            elif str(df[c].dtype) == 'bool':
                df[c] = df[c].astype('int8')
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    return train_df, test_df


# ------------------------------------------------------------------ build
def build_features():
    log('loading train / test / original')
    train, test, orig = read_train(), read_test(), read_original()
    for df in [train, test, orig]:
        for c in NUMS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    y = labels(train[TARGET]).astype('int32')
    y_orig = labels(orig[TARGET]).astype('int32')
    train = train.drop(columns=[TARGET])
    orig = orig.drop(columns=[TARGET])
    n_train, n_test = len(train), len(test)

    # train + test + original (the id column is kept, NaN on the original rows)
    combined = pd.concat([train, test, orig], axis=0, ignore_index=True)
    log('snap features for all 11 numerics')
    for col in NUMS:
        if col in combined.columns:
            apply_snap(combined, orig, col)
    digit_base_cols_ext = DIGIT_BASE_COLS + [f'{c}_snap' for c in NUMS if f'{c}_snap' in combined.columns]

    blocks = []
    log('frequency / rank')
    blocks.append(add_freq_rank_features(combined, NUMS + [f'{c}_snap' for c in NUMS]))
    log('numeric transforms')
    blocks.append(add_num_transforms(combined, NUMS + [f'{c}_snap' for c in NUMS]))
    log('formula')
    blocks.append(add_formula_features(combined))
    log('bins + categorical crosses')
    block_bins = add_bins_and_cross_cats(combined)
    blocks.append(block_bins)
    log('category counts + rare flags')
    combined_tmp = pd.concat([combined, block_bins], axis=1)
    cat_for_counts = dedup_keep_order(CATS + [f'{a}__{b}' for a, b in CROSS_PAIRS]
                                      + [f'{a}__{b}__{c}' for a, b, c in TRIPLE])
    cat_for_counts = [c for c in cat_for_counts if c in combined_tmp.columns]
    blocks.append(add_cat_counts_and_rare(combined_tmp, cat_for_counts, rare_thresh=CAT_RARE_THRESH))
    log('drift ratios')
    drift_cols = CATS + [f'{c}_snap' for c in NUMS if f'{c}_snap' in combined.columns]
    blocks.append(add_drift_ratio_from_orig(combined, n_train, n_test, drift_cols))
    log('original priors')
    orig_proba_cols = dedup_keep_order(CATS + [f'{c}_bin' for c in NUMS if f'{c}_bin' in block_bins.columns]
                                       + [f'{c}_snap' for c in NUMS])
    orig_proba_cols = [c for c in orig_proba_cols if c in combined_tmp.columns]
    blocks.append(add_orig_proba_features(combined_tmp, n_train, n_test, y_orig, orig_proba_cols[:20]))
    log('rounded values')
    blocks.append(add_round_features(combined, ROUND_SPECS))
    log('digit positions')
    blocks.append(add_digit_position_features(combined, digit_base_cols_ext, max_dec_digits=2))
    log('LGBM-style block')
    combined_for_lgbm = pd.concat([combined, block_bins], axis=1)
    blocks.append(add_lgbm_port_features(combined_for_lgbm, n_train, n_test, y_orig))
    log('radix')
    blocks.append(add_radix_features(combined))
    log('capped categorical pair keys')
    pair_block = {}
    combined_pairs = pd.concat([combined, block_bins], axis=1)
    for cols in PAIR_FEATURES:
        if all(c in combined_pairs.columns for c in cols):
            s = make_pair_feature_from_combined(combined_pairs, cols, pair_max_unique=PAIR_MAX_UNIQUE)
            pair_block[s.name] = s
    if pair_block:
        blocks.append(pd.DataFrame(pair_block, index=combined.index))

    X_all = pd.concat([combined] + blocks, axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()].copy()
    log(f'all rows after the feature build: {X_all.shape}')
    X_train_raw = X_all.iloc[:n_train].reset_index(drop=True)
    X_test_raw = X_all.iloc[n_train:n_train + n_test].reset_index(drop=True)
    return X_train_raw, X_test_raw, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--force', action='store_true', help='rerun folds that are already done')
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        X_train_raw, X_test_raw, y = build_features()
        cls_counts = np.bincount(y, minlength=N_CLASSES)
        cls_weights = cls_counts.sum() / (N_CLASSES * cls_counts)
        fold_id = folds(y)
        for k in todo:
            t_fold = time.perf_counter()
            tr_idx, va_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            X_tr = X_train_raw.iloc[tr_idx].reset_index(drop=True).copy()
            X_va = X_train_raw.iloc[va_idx].reset_index(drop=True).copy()
            X_te = X_test_raw.reset_index(drop=True).copy()
            y_tr, y_va = y[tr_idx], y[va_idx]

            te_cols = [c for c in X_tr.columns if c.endswith('_te')]
            if te_cols:
                tr_te, (va_te, te_te) = nested_target_encode_mc(X_tr[te_cols], [X_va[te_cols], X_te[te_cols]], y_tr,
                                                                te_cols)
                X_tr = pd.concat([X_tr.drop(columns=te_cols).reset_index(drop=True), tr_te.reset_index(drop=True)], axis=1)
                X_va = pd.concat([X_va.drop(columns=te_cols).reset_index(drop=True), va_te.reset_index(drop=True)], axis=1)
                X_te = pd.concat([X_te.drop(columns=te_cols).reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)

            X_tr, X_va = final_factorize(X_tr, X_va)
            _, X_te = final_factorize(X_tr, X_te)
            X_tr = X_tr.loc[:, ~X_tr.columns.duplicated()].copy()
            X_va = X_va[X_tr.columns].copy()
            X_te = X_te[X_tr.columns].copy()

            model = XGBClassifier(**XGB_PARAMS)
            model.fit(X_tr, y_tr, sample_weight=cls_weights[y_tr], eval_set=[(X_va, y_va)], verbose=500)
            p_va = model.predict_proba(X_va).astype(np.float32)
            p_te = model.predict_proba(X_te).astype(np.float32)
            save_part(HERE, PAIR, k, p_va, p_te)
            bac = balanced_accuracy_score(y_va, p_va.argmax(axis=1))
            log(f'{PAIR} fold {k}: balanced accuracy {bac:.5f}, best iteration {getattr(model, "best_iteration", None)}, '
                f'{X_tr.shape[1]} features, {time.perf_counter() - t_fold:.0f}s')
            del X_tr, X_va, X_te, model, p_va, p_te
            gc.collect()
    y = labels(read_train()[TARGET])
    assemble(HERE, PAIR, folds(y), y)


if __name__ == '__main__':
    main()
