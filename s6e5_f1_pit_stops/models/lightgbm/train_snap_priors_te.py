#!/usr/bin/env python
"""LightGBM on a wide union of label-free features, original-data priors and per-fold nested target encodings
(pair lgbm_snap_priors_pairs_nested_te).

Columns are downcast (float32 where lossless to ~5e-4) on load. Features of the train / test rows, built with the original rows
as a reference (the original rows are not training rows):
  - snap: LapTime_Delta, Cumulative_Degradation, LapTime (s) snapped to the nearest value present in the original data, the
    difference to it, the first 3 decimals (strings) and the number of decimals of the raw and of the snapped value
  - A_FREQ / A_RANK: frequency and percentile rank of every numeric over train + test + original; signed log1p, signed sqrt,
    1 / (1 + |x|) and missing flags; tyre / lap-time / degradation relations; race-context flags and their counts
  - bins, quantile codes (40-80 bins), rounded and integer values, the decimal strings and 9 categorical crosses
    (all string keys)
  - category counts and rare flags (<= 50 rows), original target mean and row count of every category and numeric value and
    of 5 pairs, train / original frequency drift of the categories
  - B_pair: every pair of 21 key columns with at most 20,000 distinct values (string keys)
  - L_: 5000-quantile codes, integer values and values rounded to 2 of the 5 main numerics and the snapped values (string
    keys); counts and original target means of Compound, Race, Driver, Compound x Stint, Race x Compound
Per fold, every string column that is a pair / L_ / *_te key or has more than 7 levels is replaced by a nested target
encoding (target mean; the training rows get 5-fold inner out-of-fold values, seed 42 + fold number; validation and test rows
the means of all training rows); the other string columns are factorized.
Model: LightGBM (CPU, 32 threads), lr 0.01, 63 leaves, depth 6, min_child_samples 200, subsample 0.72 every round,
colsample_bytree 0.22, alpha 8, lambda 14, max_bin 511, seed 42; up to 12,000 rounds with early stopping (400) on the
validation fold. Test = mean of the 5 fold models.
Usage: python train_snap_priors_te.py"""
import gc
import itertools
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'lgbm_snap_priors_pairs_nested_te'
SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5
N_THREADS = 32

LGB_PARAMS = dict(
    objective='binary',
    metric='auc',
    boosting_type='gbdt',
    n_estimators=12000,
    learning_rate=0.01,
    num_leaves=63,
    max_depth=6,
    min_child_samples=200,
    subsample=0.72,
    subsample_freq=1,
    colsample_bytree=0.22,
    reg_alpha=8.0,
    reg_lambda=14.0,
    min_split_gain=0.0,
    max_bin=511,
    random_state=SEED,
    verbosity=-1,
    num_threads=N_THREADS,
    n_jobs=N_THREADS,
    device_type='cpu',
)

PAIR_MAX_UNIQUE = 20_000
CAT_RARE_THRESH = 50
HIGH_CARD_THRESHOLD = 7
QCUT_BINS = 5000

TYRE_COL = 'TyreLife'
DELTA_COL = 'LapTime_Delta'
DEGR_COL = 'Cumulative_Degradation'
LAPTIME_COL = 'LapTime (s)'
EXTRA_NUMERIC_COLS = ['LapTime (s)', 'RaceProgress', 'LapNumber', 'Position', 'Position_Change', 'PitStop']
SNAP_SOURCE_COLS = ['LapTime_Delta', 'Cumulative_Degradation', 'LapTime (s)']


def to_binary_target(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype('int8')
    return s.astype(str).str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0}).astype('int8')


def safe_name(col):
    out = str(col)
    for a, b in [(' ', '_'), ('(', ''), (')', ''), ('/', '_'), ('-', '_'), ('.', '_'), ('%', 'pct'), ('|', '_'), (',', '_'), ('<', 'lt'), ('>', 'gt')]:
        out = out.replace(a, b)
    return out


def dedup_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def as_key(s):
    return s.astype('string').fillna('missing').astype(str)


def signed_log1p(s):
    s = pd.to_numeric(s, errors='coerce').astype('float64')
    return np.sign(s) * np.log1p(np.abs(s))


def signed_sqrt(s):
    s = pd.to_numeric(s, errors='coerce').astype('float64')
    return np.sign(s) * np.sqrt(np.abs(s))


def reduce_mem(df):
    for c in df.columns:
        if c in [ID, TARGET]:
            continue
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast='float')
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast='integer')
    return df


def sanitize_columns(df):
    counts = {}
    new_cols = []
    for c in df.columns:
        base = safe_name(c)
        if base not in counts:
            counts[base] = 0
            new_cols.append(base)
        else:
            counts[base] += 1
            new_cols.append(f'{base}__dup{counts[base]}')
    out = df.copy()
    out.columns = new_cols
    return out


def consistent_factorize(train_df, valid_df, test_df, cols):
    out_tr, out_va, out_te = train_df.copy(), valid_df.copy(), test_df.copy()
    for c in cols:
        allv = pd.concat([as_key(out_tr[c]), as_key(out_va[c]), as_key(out_te[c])], axis=0, ignore_index=True)
        codes, _ = pd.factorize(allv, sort=True)
        n_tr, n_va = len(out_tr), len(out_va)
        out_tr[c] = codes[:n_tr].astype('int32')
        out_va[c] = codes[n_tr:n_tr + n_va].astype('int32')
        out_te[c] = codes[n_tr + n_va:].astype('int32')
    return out_tr, out_va, out_te


def safe_qcut_codes(series_list, q):
    all_s = pd.concat([pd.to_numeric(s, errors='coerce') for s in series_list], axis=0, ignore_index=True)
    q = int(min(q, max(2, all_s.nunique(dropna=True))))
    try:
        codes = pd.qcut(all_s.rank(method='first'), q=q, labels=False, duplicates='drop').astype('float32')
    except Exception:
        codes = pd.Series(np.zeros(len(all_s), dtype='float32'))
    out, start = [], 0
    for s in series_list:
        out.append(codes.iloc[start:start + len(s)].reset_index(drop=True))
        start += len(s)
    return out


def nearest_original_values(values, orig_values):
    vals = pd.to_numeric(values, errors='coerce').to_numpy(dtype='float64')
    base = pd.to_numeric(orig_values, errors='coerce').dropna().unique().astype('float64')
    if len(base) == 0:
        return np.full(len(vals), np.nan, dtype='float64')
    base.sort()
    idx = np.searchsorted(base, vals)
    idx0 = np.clip(idx - 1, 0, len(base) - 1)
    idx1 = np.clip(idx, 0, len(base) - 1)
    left = base[idx0]
    right = base[idx1]
    choose_right = np.abs(right - vals) < np.abs(vals - left)
    snap = np.where(choose_right, right, left)
    snap[np.isnan(vals)] = np.nan
    return snap


def add_snap_features(train, test, orig, cols):
    """Adds the snapped values, their differences, decimal digits and lengths to the three frames (in place)."""
    snap_cols, digit_cols = [], []
    for c in cols:
        if c not in train.columns or c not in orig.columns:
            continue
        s = safe_name(c)
        snap_col = f'{s}_snap'
        diff_col = f'{s}_snap_diff'
        train[snap_col] = nearest_original_values(train[c], orig[c]).astype('float32')
        test[snap_col] = nearest_original_values(test[c], orig[c]).astype('float32')
        orig[snap_col] = pd.to_numeric(orig[c], errors='coerce').astype('float32')
        for df in [train, test, orig]:
            df[diff_col] = (pd.to_numeric(df[c], errors='coerce') - pd.to_numeric(df[snap_col], errors='coerce')).astype('float32')
        snap_cols.extend([snap_col, diff_col])

        for prefix_source, source_col in [(f'{s}_raw', c), (f'{s}_snap', snap_col)]:
            vals_all = pd.concat([train[source_col], test[source_col], orig[source_col]], axis=0, ignore_index=True)
            text = pd.to_numeric(vals_all, errors='coerce').abs().round(8).astype('string')
            frac = text.str.split('.').str[1].fillna('')
            n_tr, n_te = len(train), len(test)
            for pos in range(3):
                name = f'{prefix_source}_d{pos}'
                dig = frac.str[pos].fillna('-1')
                train[name] = dig.iloc[:n_tr].to_numpy()
                test[name] = dig.iloc[n_tr:n_tr + n_te].to_numpy()
                orig[name] = dig.iloc[n_tr + n_te:].to_numpy()
                digit_cols.append(name)
            name = f'{prefix_source}_len'
            ln = frac.str.len().fillna(-1).astype('float32')
            train[name] = ln.iloc[:n_tr].to_numpy(dtype='float32')
            test[name] = ln.iloc[n_tr:n_tr + n_te].to_numpy(dtype='float32')
            orig[name] = ln.iloc[n_tr + n_te:].to_numpy(dtype='float32')
            snap_cols.append(name)
    return snap_cols, digit_cols


def load_data():
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = to_binary_target(train[TARGET])
    orig[TARGET] = to_binary_target(orig[TARGET])
    return reduce_mem(train), reduce_mem(test), reduce_mem(orig)


def add_global_rank_freq_num(train_df, test_df, orig_df, num_cols):
    out_tr, out_te, out_or = {}, {}, {}
    all_num = pd.concat([train_df[num_cols], test_df[num_cols], orig_df[num_cols]], axis=0, ignore_index=True)
    n_tr, n_te = len(train_df), len(test_df)
    for c in num_cols:
        s = pd.to_numeric(all_num[c], errors='coerce')
        freq = s.value_counts(normalize=True, dropna=False)
        rank = s.rank(method='average', pct=True)
        sn = safe_name(c)
        out_tr[f'A_FREQ_{sn}'] = s.iloc[:n_tr].map(freq).astype('float32').to_numpy()
        out_te[f'A_FREQ_{sn}'] = s.iloc[n_tr:n_tr + n_te].map(freq).astype('float32').to_numpy()
        out_or[f'A_FREQ_{sn}'] = s.iloc[n_tr + n_te:].map(freq).astype('float32').to_numpy()
        out_tr[f'A_RANK_{sn}'] = rank.iloc[:n_tr].astype('float32').to_numpy()
        out_te[f'A_RANK_{sn}'] = rank.iloc[n_tr:n_tr + n_te].astype('float32').to_numpy()
        out_or[f'A_RANK_{sn}'] = rank.iloc[n_tr + n_te:].astype('float32').to_numpy()
    return pd.DataFrame(out_tr, index=train_df.index), pd.DataFrame(out_te, index=test_df.index), pd.DataFrame(out_or, index=orig_df.index)


def add_num_transforms(df, num_cols, prefix='A'):
    out = {}
    for c in num_cols:
        if c not in df.columns:
            continue
        sn = safe_name(c)
        v = pd.to_numeric(df[c], errors='coerce').astype('float32')
        out[f'{prefix}_LOG1P_{sn}'] = signed_log1p(v).astype('float32').to_numpy()
        out[f'{prefix}_SQRT_{sn}'] = signed_sqrt(v).astype('float32').to_numpy()
        out[f'{prefix}_INV1P_ABS_{sn}'] = (1.0 / (1.0 + np.abs(v))).astype('float32').to_numpy()
        out[f'{prefix}_MISS_{sn}'] = v.isna().astype('float32').to_numpy()
    return pd.DataFrame(out, index=df.index)


def add_f1_relationships(df, prefix='A'):
    out = {}
    tyre = pd.to_numeric(df.get(TYRE_COL, 0), errors='coerce').fillna(0).astype('float32')
    delta = pd.to_numeric(df.get(DELTA_COL, 0), errors='coerce').fillna(0).astype('float32')
    degr = pd.to_numeric(df.get(DEGR_COL, 0), errors='coerce').fillna(0).astype('float32')
    lap_time = pd.to_numeric(df.get(LAPTIME_COL, 0), errors='coerce').fillna(0).astype('float32')
    progress = pd.to_numeric(df.get('RaceProgress', 0), errors='coerce').fillna(0).astype('float32')
    lap = pd.to_numeric(df.get('LapNumber', 0), errors='coerce').fillna(0).astype('float32')
    pos = pd.to_numeric(df.get('Position', 0), errors='coerce').fillna(0).astype('float32')
    pos_chg = pd.to_numeric(df.get('Position_Change', 0), errors='coerce').fillna(0).astype('float32')
    pitstop = pd.to_numeric(df.get('PitStop', 0), errors='coerce').fillna(0).astype('float32')

    t1 = tyre.clip(lower=1)
    out[f'{prefix}_TYRE_X_DELTA'] = (tyre * delta).astype('float32').to_numpy()
    out[f'{prefix}_DEGR_DIV_TYRE'] = (degr / t1).astype('float32').to_numpy()
    out[f'{prefix}_DEGR_MINUS_TYRE_DELTA'] = (degr - tyre * delta).astype('float32').to_numpy()
    out[f'{prefix}_DELTA_DIV_DEGR1P'] = (delta / (1.0 + np.abs(degr))).astype('float32').to_numpy()
    out[f'{prefix}_TYRE_DIV_DELTA1P'] = (tyre / (1.0 + np.abs(delta))).astype('float32').to_numpy()
    out[f'{prefix}_LAPTIME_PLUS_DELTA'] = (lap_time + delta).astype('float32').to_numpy()
    out[f'{prefix}_LAP_X_PROGRESS'] = (lap * progress).astype('float32').to_numpy()
    out[f'{prefix}_POSITION_X_CHANGE'] = (pos * pos_chg).astype('float32').to_numpy()
    out[f'{prefix}_PITSTOP_X_TYRE'] = (pitstop * tyre).astype('float32').to_numpy()

    for c in SNAP_SOURCE_COLS:
        s = safe_name(c)
        snap_col = f'{s}_snap'
        diff_col = f'{s}_snap_diff'
        if snap_col in df.columns:
            snap = pd.to_numeric(df[snap_col], errors='coerce').fillna(0).astype('float32')
            out[f'{prefix}_{s}_SNAP_ABS'] = np.abs(snap).astype('float32').to_numpy()
        if diff_col in df.columns:
            diff = pd.to_numeric(df[diff_col], errors='coerce').fillna(0).astype('float32')
            out[f'{prefix}_{s}_SNAP_DIFF_ABS'] = np.abs(diff).astype('float32').to_numpy()
    return pd.DataFrame(out, index=df.index)


def add_state_counts(df, prefix='A'):
    """Race-context flags and three counts of them."""
    out = {}
    comp = as_key(df.get('Compound', pd.Series([''] * len(df), index=df.index))).str.lower()
    position = pd.to_numeric(df.get('Position', 0), errors='coerce').fillna(0)
    pos_change = pd.to_numeric(df.get('Position_Change', 0), errors='coerce').fillna(0)
    pitstop = pd.to_numeric(df.get('PitStop', 0), errors='coerce').fillna(0)
    tyre = pd.to_numeric(df.get('TyreLife', 0), errors='coerce').fillna(0)
    progress = pd.to_numeric(df.get('RaceProgress', 0), errors='coerce').fillna(0)
    delta = pd.to_numeric(df.get('LapTime_Delta', 0), errors='coerce').fillna(0)

    flags = pd.DataFrame(index=df.index)
    flags['soft'] = comp.str.contains('soft').astype('float32')
    flags['medium'] = comp.str.contains('medium').astype('float32')
    flags['hard'] = comp.str.contains('hard').astype('float32')
    flags['wet_inter'] = (comp.str.contains('wet') | comp.str.contains('inter')).astype('float32')
    flags['front_position'] = (position <= 3).astype('float32')
    flags['points_position'] = (position <= 10).astype('float32')
    flags['position_loss'] = (pos_change < 0).astype('float32')
    flags['position_gain'] = (pos_change > 0).astype('float32')
    flags['has_pitstop'] = (pitstop > 0).astype('float32')
    flags['old_tyre'] = (tyre >= 20).astype('float32')
    flags['late_race'] = (progress >= 0.70).astype('float32')
    flags['slow_lap'] = (delta > 0).astype('float32')

    out[f'{prefix}_SERVICE_YES_COUNT'] = flags[['soft', 'front_position', 'points_position', 'position_gain']].sum(axis=1).astype('float32').to_numpy()
    out[f'{prefix}_SERVICE_NO_COUNT'] = flags[['hard', 'position_loss', 'has_pitstop', 'old_tyre', 'slow_lap']].sum(axis=1).astype('float32').to_numpy()
    out[f'{prefix}_SERVICE_OTHER_COUNT'] = flags[['medium', 'wet_inter', 'late_race']].sum(axis=1).astype('float32').to_numpy()
    out[f'{prefix}_HAS_FAMILY'] = flags[['front_position', 'points_position']].any(axis=1).astype('float32').to_numpy()
    for c in flags.columns:
        out[f'{prefix}_FLAG_{c}'] = flags[c].astype('float32').to_numpy()
    return pd.DataFrame(out, index=df.index)


def add_bins_cross_and_digits(train_df, test_df, orig_df, snap_cols, digit_cols):
    use_nums = [c for c in [TYRE_COL, DELTA_COL, DEGR_COL, LAPTIME_COL, 'RaceProgress'] + [c for c in snap_cols if c.endswith('_snap')] if c in train_df.columns and c in test_df.columns and c in orig_df.columns]
    all_num = pd.concat([train_df[use_nums], test_df[use_nums], orig_df[use_nums]], axis=0, ignore_index=True)
    n_tr, n_te = len(train_df), len(test_df)
    out_all = {}

    if TYRE_COL in use_nums:
        bins = [-1, 0, 1, 3, 6, 10, 15, 20, 30, 50, 999999]
        out_all['A_TyreLife_bin'] = pd.cut(all_num[TYRE_COL], bins=bins, labels=False, include_lowest=True).astype('Int64').astype('string')
    if 'LapNumber' in train_df.columns:
        all_lap = pd.concat([train_df['LapNumber'], test_df['LapNumber'], orig_df['LapNumber']], axis=0, ignore_index=True)
        bins = [-1, 1, 5, 10, 15, 20, 30, 40, 50, 70, 999999]
        out_all['A_LapNumber_bin'] = pd.cut(pd.to_numeric(all_lap, errors='coerce'), bins=bins, labels=False, include_lowest=True).astype('Int64').astype('string')

    qspec = {DELTA_COL: 60, DEGR_COL: 80, LAPTIME_COL: 60, 'RaceProgress': 40}
    for c in [x for x in snap_cols if x.endswith('_snap')]:
        qspec[c] = 60
    for c, q in qspec.items():
        if c in all_num.columns:
            qv = safe_qcut_codes([train_df[c], test_df[c], orig_df[c]], q=q)
            vals = pd.concat(qv, axis=0, ignore_index=True).fillna(-1).astype('int32').astype('string')
            out_all[f'A_qcut_{safe_name(c)}'] = vals

    round_map = {TYRE_COL: 1.0, DELTA_COL: 0.05, DEGR_COL: 0.05, LAPTIME_COL: 0.01, 'RaceProgress': 0.01}
    for c in [x for x in snap_cols if x.endswith('_snap')]:
        round_map[c] = 0.05
    for c, step in round_map.items():
        if c in all_num.columns:
            s = pd.to_numeric(all_num[c], errors='coerce')
            sn = safe_name(c)
            out_all[f'A_round_{sn}'] = (np.round(s / step) * step).astype('float32').astype('string')
            out_all[f'A_int_{sn}'] = np.floor(s.fillna(-9999)).astype('int32').astype('string')

    for c in digit_cols:
        if c in train_df.columns and c in test_df.columns and c in orig_df.columns:
            out_all[f'A_digit_{safe_name(c)}'] = pd.concat([train_df[c], test_df[c], orig_df[c]], axis=0, ignore_index=True).astype('string')

    all_cat_source = pd.concat([train_df, test_df, orig_df], axis=0, ignore_index=True)
    cross_pairs = [
        ('Driver', 'Race'), ('Driver', 'Compound'), ('Race', 'Compound'),
        ('Compound', 'Stint'), ('Race', 'Stint'), ('PitStop', 'Compound'),
        ('Position', 'Compound'), ('TyreLife', 'Compound')
    ]
    for a, b in cross_pairs:
        if a in all_cat_source.columns and b in all_cat_source.columns:
            out_all[f'A_{safe_name(a)}__{safe_name(b)}_te'] = as_key(all_cat_source[a]) + '|' + as_key(all_cat_source[b])
    if all(c in all_cat_source.columns for c in ['Race', 'Compound', 'Stint']):
        out_all['A_Race__Compound__Stint_te'] = as_key(all_cat_source['Race']) + '|' + as_key(all_cat_source['Compound']) + '|' + as_key(all_cat_source['Stint'])

    out_all = pd.DataFrame(out_all)
    out_tr = out_all.iloc[:n_tr].set_index(train_df.index)
    out_te = out_all.iloc[n_tr:n_tr + n_te].set_index(test_df.index)
    out_or = out_all.iloc[n_tr + n_te:].set_index(orig_df.index)
    return out_tr, out_te, out_or


def add_cat_count_and_rare(train_df, test_df, orig_df, cols, rare_thresh=50):
    out_tr, out_te, out_or = {}, {}, {}
    all_cat = pd.concat([train_df[cols].astype('string'), test_df[cols].astype('string'), orig_df[cols].astype('string')], axis=0, ignore_index=True)
    n_tr, n_te = len(train_df), len(test_df)
    for c in cols:
        vc = all_cat[c].value_counts(dropna=False)
        cnt = all_cat[c].map(vc).astype('float32')
        sn = safe_name(c)
        out_tr[f'A_CAT_CNT_{sn}'] = cnt.iloc[:n_tr].to_numpy()
        out_te[f'A_CAT_CNT_{sn}'] = cnt.iloc[n_tr:n_tr + n_te].to_numpy()
        out_or[f'A_CAT_CNT_{sn}'] = cnt.iloc[n_tr + n_te:].to_numpy()
        out_tr[f'A_CAT_RARE_{sn}'] = (cnt.iloc[:n_tr] <= rare_thresh).astype('float32').to_numpy()
        out_te[f'A_CAT_RARE_{sn}'] = (cnt.iloc[n_tr:n_tr + n_te] <= rare_thresh).astype('float32').to_numpy()
        out_or[f'A_CAT_RARE_{sn}'] = (cnt.iloc[n_tr + n_te:] <= rare_thresh).astype('float32').to_numpy()
    return pd.DataFrame(out_tr, index=train_df.index), pd.DataFrame(out_te, index=test_df.index), pd.DataFrame(out_or, index=orig_df.index)


def add_orig_single_priors(train_df, test_df, orig_df, cols):
    out_tr, out_te, out_or = {}, {}, {}
    base_mean = float(orig_df[TARGET].mean())
    for c in cols:
        if c not in orig_df.columns:
            continue
        grp = orig_df.groupby(c, dropna=False)[TARGET]
        mean_map = grp.mean()
        size_map = grp.size()
        sn = safe_name(c)
        out_tr[f'A_ORIG_MEAN_{sn}'] = train_df[c].map(mean_map).fillna(base_mean).astype('float32').to_numpy()
        out_te[f'A_ORIG_MEAN_{sn}'] = test_df[c].map(mean_map).fillna(base_mean).astype('float32').to_numpy()
        out_or[f'A_ORIG_MEAN_{sn}'] = orig_df[c].map(mean_map).fillna(base_mean).astype('float32').to_numpy()
        out_tr[f'A_ORIG_CNT_{sn}'] = train_df[c].map(size_map).fillna(0).astype('float32').to_numpy()
        out_te[f'A_ORIG_CNT_{sn}'] = test_df[c].map(size_map).fillna(0).astype('float32').to_numpy()
        out_or[f'A_ORIG_CNT_{sn}'] = orig_df[c].map(size_map).fillna(0).astype('float32').to_numpy()
    return pd.DataFrame(out_tr, index=train_df.index), pd.DataFrame(out_te, index=test_df.index), pd.DataFrame(out_or, index=orig_df.index)


def add_orig_pair_priors(train_df, test_df, orig_df, pairs):
    out_tr, out_te, out_or = {}, {}, {}
    base_mean = float(orig_df[TARGET].mean())
    for c1, c2 in pairs:
        if c1 not in orig_df.columns or c2 not in orig_df.columns:
            continue
        name = f'A_PAIR_{safe_name(c1)}__{safe_name(c2)}'
        key_o = as_key(orig_df[c1]) + '|' + as_key(orig_df[c2])
        mean_map = pd.DataFrame({'k': key_o, 'y': orig_df[TARGET].astype('float32')}).groupby('k')['y'].mean()
        size_map = pd.DataFrame({'k': key_o}).groupby('k').size()
        for df, out in [(train_df, out_tr), (test_df, out_te), (orig_df, out_or)]:
            key = as_key(df[c1]) + '|' + as_key(df[c2])
            out[f'{name}_orig_mean'] = key.map(mean_map).fillna(base_mean).astype('float32').to_numpy()
            out[f'{name}_orig_cnt'] = key.map(size_map).fillna(0).astype('float32').to_numpy()
    return pd.DataFrame(out_tr, index=train_df.index), pd.DataFrame(out_te, index=test_df.index), pd.DataFrame(out_or, index=orig_df.index)


def add_drift_ratio(train_df, test_df, orig_df, cols, eps=1e-6):
    out_tr, out_te, out_or = {}, {}, {}
    for c in cols:
        if c not in orig_df.columns:
            continue
        p_tr = train_df[c].value_counts(normalize=True, dropna=False)
        p_or = orig_df[c].value_counts(normalize=True, dropna=False)
        sn = safe_name(c)
        for df, out in [(train_df, out_tr), (test_df, out_te), (orig_df, out_or)]:
            out[f'A_DRIFT_{sn}'] = np.log1p(df[c].map(p_tr).fillna(0) / (df[c].map(p_or).fillna(eps) + eps)).astype('float32').to_numpy()
    return pd.DataFrame(out_tr, index=train_df.index), pd.DataFrame(out_te, index=test_df.index), pd.DataFrame(out_or, index=orig_df.index)


def create_pair_features(train_df, test_df, orig_df, base_cols, pair_max_unique=20000):
    out_tr, out_te, out_or = {}, {}, {}
    for a, b in itertools.combinations(base_cols, 2):
        if a not in train_df.columns or b not in train_df.columns:
            continue
        sa = pd.concat([train_df[a], test_df[a], orig_df[a]], axis=0, ignore_index=True).astype('string')
        sb = pd.concat([train_df[b], test_df[b], orig_df[b]], axis=0, ignore_index=True).astype('string')
        key = (sa + '|' + sb).astype('string')
        if key.nunique(dropna=False) <= pair_max_unique:
            name = f'B_pair_{safe_name(a)}__{safe_name(b)}'
            n_tr, n_te = len(train_df), len(test_df)
            out_tr[name] = key.iloc[:n_tr].to_numpy()
            out_te[name] = key.iloc[n_tr:n_tr + n_te].to_numpy()
            out_or[name] = key.iloc[n_tr + n_te:].to_numpy()
    return pd.DataFrame(out_tr, index=train_df.index), pd.DataFrame(out_te, index=test_df.index), pd.DataFrame(out_or, index=orig_df.index)


def add_quantile_count_features(train_df, test_df, orig_df, snap_cols):
    """L_ block: quantile / integer / rounded keys of the main numerics, group counts and original group target means."""
    num_candidates = [c for c in [TYRE_COL, DELTA_COL, DEGR_COL, LAPTIME_COL, 'RaceProgress'] + [c for c in snap_cols if c.endswith('_snap')] if c in train_df.columns]
    out_tr, out_te, out_or = {}, {}, {}

    for col in num_candidates:
        s_all = pd.concat([train_df[col], test_df[col], orig_df[col]], axis=0, ignore_index=True)
        s_num = pd.to_numeric(s_all, errors='coerce')
        n_tr, n_te = len(train_df), len(test_df)
        sn = safe_name(col)

        q = int(min(QCUT_BINS, max(2, s_num.nunique(dropna=True))))
        try:
            qcat = pd.qcut(s_num.rank(method='first'), q=q, labels=False, duplicates='drop').astype('Int64').astype('string')
        except Exception:
            qcat = pd.Series(['0'] * len(s_all), dtype='string')
        out_tr[f'L_{sn}_quantiled_te'] = qcat.iloc[:n_tr].to_numpy()
        out_te[f'L_{sn}_quantiled_te'] = qcat.iloc[n_tr:n_tr + n_te].to_numpy()
        out_or[f'L_{sn}_quantiled_te'] = qcat.iloc[n_tr + n_te:].to_numpy()

        intval = np.floor(s_num.fillna(-9999)).astype('int32').astype('string')
        out_tr[f'L_{sn}_int_te'] = intval.iloc[:n_tr].to_numpy()
        out_te[f'L_{sn}_int_te'] = intval.iloc[n_tr:n_tr + n_te].to_numpy()
        out_or[f'L_{sn}_int_te'] = intval.iloc[n_tr + n_te:].to_numpy()

        round_div2 = (np.round(s_num / 2.0) * 2.0).astype('float32').astype('string')
        out_tr[f'L_{sn}_round_div2_te'] = round_div2.iloc[:n_tr].to_numpy()
        out_te[f'L_{sn}_round_div2_te'] = round_div2.iloc[n_tr:n_tr + n_te].to_numpy()
        out_or[f'L_{sn}_round_div2_te'] = round_div2.iloc[n_tr + n_te:].to_numpy()

    all_df = pd.concat([train_df, test_df, orig_df], axis=0, ignore_index=True)
    n_tr, n_te = len(train_df), len(test_df)
    count_groups = [('Compound',), ('Race',), ('Driver',), ('Compound', 'Stint'), ('Race', 'Compound')]
    for cols in count_groups:
        if not all(c in all_df.columns for c in cols):
            continue
        name = f'L_count_{"__".join(safe_name(c) for c in cols)}'
        key = as_key(all_df[cols[0]])
        for c in cols[1:]:
            key = key + '|' + as_key(all_df[c])
        counts = key.map(key.value_counts(dropna=False)).astype('float32')
        out_tr[name] = counts.iloc[:n_tr].to_numpy()
        out_te[name] = counts.iloc[n_tr:n_tr + n_te].to_numpy()
        out_or[name] = counts.iloc[n_tr + n_te:].to_numpy()

    for cols in count_groups:
        if not all(c in orig_df.columns for c in cols):
            continue
        name = '__'.join(safe_name(c) for c in cols)
        key_o = as_key(orig_df[cols[0]])
        for c in cols[1:]:
            key_o = key_o + '|' + as_key(orig_df[c])
        mean_map = pd.DataFrame({'k': key_o, 'y': orig_df[TARGET].astype('float32')}).groupby('k')['y'].mean()
        size_map = pd.DataFrame({'k': key_o}).groupby('k').size()
        base_mean = float(orig_df[TARGET].mean())
        for df, out in [(train_df, out_tr), (test_df, out_te), (orig_df, out_or)]:
            key = as_key(df[cols[0]])
            for c in cols[1:]:
                key = key + '|' + as_key(df[c])
            out[f'L_groupby_{name}_mean_orig'] = key.map(mean_map).fillna(base_mean).astype('float32').to_numpy()
            out[f'L_groupby_{name}_size_orig'] = key.map(size_map).fillna(0).astype('float32').to_numpy()

    return pd.DataFrame(out_tr, index=train_df.index), pd.DataFrame(out_te, index=test_df.index), pd.DataFrame(out_or, index=orig_df.index)


def nested_target_encode(train_col, y_train, valid_col, test_col, n_splits=5, seed=42):
    train_col = as_key(train_col)
    valid_col = as_key(valid_col)
    test_col = as_key(test_col)
    global_mean = float(np.mean(y_train))
    oof_enc = np.full(len(train_col), global_mean, dtype=np.float32)
    skf_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_s = pd.Series(y_train, index=train_col.index)
    for sub_tr_idx, sub_va_idx in skf_inner.split(train_col, y_train):
        sub_tr = train_col.iloc[sub_tr_idx]
        sub_y = y_s.iloc[sub_tr_idx]
        sub_map = sub_y.groupby(sub_tr).mean()
        oof_enc[sub_va_idx] = train_col.iloc[sub_va_idx].map(sub_map).fillna(global_mean).astype('float32').to_numpy()
    full_map = y_s.groupby(train_col).mean()
    va = valid_col.map(full_map).fillna(global_mean).astype('float32').to_numpy()
    te = test_col.map(full_map).fillna(global_mean).astype('float32').to_numpy()
    return oof_enc, va, te


def clean_for_lgb(df):
    out = df.copy()
    for c in out.columns:
        if c == ID:
            continue
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype('int8')
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].astype('int32')
        else:
            out[c] = pd.to_numeric(out[c], errors='coerce').astype('float32')
    if ID in out.columns:
        out = out.drop(columns=[ID])
    out = out.replace([np.inf, -np.inf], np.nan).fillna(-999.0)
    return reduce_mem(out)


def build_features():
    """Returns X_train, X_test (id kept, dropped before training), the TE / factorized string columns and y."""
    train, test, orig = load_data()
    base_cats = [c for c in ['Driver', 'Compound', 'Race', 'Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position', 'Position_Change'] if c in train.columns]
    base_nums = [c for c in [TYRE_COL, DELTA_COL, DEGR_COL, LAPTIME_COL, 'RaceProgress'] if c in train.columns]
    snap_cols, digit_cols = add_snap_features(train, test, orig, [c for c in SNAP_SOURCE_COLS if c in train.columns])
    for df in [train, test, orig]:
        for c in base_cats:
            df[c] = as_key(df[c])
        for c in base_nums + snap_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

    work_tr = train.drop(columns=[TARGET]).copy()
    work_te = test.copy()
    work_or = orig.copy()

    all_nums = [c for c in base_nums + EXTRA_NUMERIC_COLS + snap_cols if c in work_tr.columns and c in work_te.columns and c in work_or.columns]
    all_nums = dedup_keep_order(all_nums)
    all_cats = [c for c in base_cats if c in work_tr.columns and c in work_te.columns and c in work_or.columns]

    A_freq_tr, A_freq_te, A_freq_or = add_global_rank_freq_num(work_tr, work_te, work_or, [c for c in all_nums if c in work_tr.columns])
    A_num_tr = add_num_transforms(work_tr, all_nums)
    A_num_te = add_num_transforms(work_te, all_nums)
    A_rel_tr = add_f1_relationships(work_tr)
    A_rel_te = add_f1_relationships(work_te)
    A_state_tr = add_state_counts(work_tr)
    A_state_te = add_state_counts(work_te)
    A_bin_tr, A_bin_te, A_bin_or = add_bins_cross_and_digits(work_tr, work_te, work_or, snap_cols, digit_cols)

    count_cols = dedup_keep_order(all_cats + [c for c in ['A_TyreLife_bin', 'A_LapNumber_bin'] if c in A_bin_tr.columns])
    count_src_tr = pd.concat([work_tr[[c for c in all_cats if c in work_tr.columns]], A_bin_tr[[c for c in count_cols if c in A_bin_tr.columns]]], axis=1)
    count_src_te = pd.concat([work_te[[c for c in all_cats if c in work_te.columns]], A_bin_te[[c for c in count_cols if c in A_bin_te.columns]]], axis=1)
    count_src_or = pd.concat([work_or[[c for c in all_cats if c in work_or.columns]], A_bin_or[[c for c in count_cols if c in A_bin_or.columns]]], axis=1)
    count_cols_present = [c for c in count_cols if c in count_src_tr.columns]
    A_cnt_tr, A_cnt_te, _ = add_cat_count_and_rare(count_src_tr, count_src_te, count_src_or, count_cols_present, rare_thresh=CAT_RARE_THRESH)
    prior_src_tr = count_src_tr.copy()
    prior_src_te = count_src_te.copy()
    prior_src_or = count_src_or.copy()
    for c in all_nums:
        if c in work_tr.columns and c not in prior_src_tr.columns:
            prior_src_tr[c] = work_tr[c]
        if c in work_te.columns and c not in prior_src_te.columns:
            prior_src_te[c] = work_te[c]
        if c in work_or.columns and c not in prior_src_or.columns:
            prior_src_or[c] = work_or[c]
    prior_src_or[TARGET] = work_or[TARGET]
    A_op_tr, A_op_te, _ = add_orig_single_priors(prior_src_tr, prior_src_te, prior_src_or, dedup_keep_order(count_cols_present + all_nums))
    A_dr_tr, A_dr_te, _ = add_drift_ratio(count_src_tr, count_src_te, count_src_or, [c for c in all_cats if c in count_src_tr.columns])

    snap_pair_priors = [
        (f'{safe_name(DELTA_COL)}_snap', 'Compound'),
        (f'{safe_name(DEGR_COL)}_snap', 'Compound'),
        (f'{safe_name(LAPTIME_COL)}_snap', 'Race'),
        ('Compound', 'Stint'),
        ('Race', 'Compound'),
    ]
    A_pp_tr, A_pp_te, _ = add_orig_pair_priors(work_tr, work_te, work_or, [p for p in snap_pair_priors if p[0] in work_tr.columns and p[1] in work_tr.columns])

    pair_base = dedup_keep_order([c for c in ['Driver', 'Race', 'Compound', 'Stint', 'PitStop', 'Position', 'TyreLife', 'A_TyreLife_bin', 'A_LapNumber_bin'] + [c for c in A_bin_tr.columns if c.startswith('A_qcut_')][:6] + digit_cols[:8] if c in pd.concat([work_tr, A_bin_tr], axis=1).columns])
    pair_src_tr = pd.concat([work_tr, A_bin_tr], axis=1)
    pair_src_te = pd.concat([work_te, A_bin_te], axis=1)
    pair_src_or = pd.concat([work_or, A_bin_or], axis=1)
    B_pair_tr, B_pair_te, _ = create_pair_features(pair_src_tr, pair_src_te, pair_src_or, pair_base, pair_max_unique=PAIR_MAX_UNIQUE)

    L_tr, L_te, _ = add_quantile_count_features(work_tr, work_te, work_or, snap_cols)

    X_train = pd.concat([
        work_tr.reset_index(drop=True), A_freq_tr.reset_index(drop=True), A_num_tr.reset_index(drop=True), A_rel_tr.reset_index(drop=True),
        A_state_tr.reset_index(drop=True), A_bin_tr.reset_index(drop=True), A_cnt_tr.reset_index(drop=True), A_op_tr.reset_index(drop=True),
        A_dr_tr.reset_index(drop=True), A_pp_tr.reset_index(drop=True), B_pair_tr.reset_index(drop=True), L_tr.reset_index(drop=True)
    ], axis=1)
    X_test = pd.concat([
        work_te.reset_index(drop=True), A_freq_te.reset_index(drop=True), A_num_te.reset_index(drop=True), A_rel_te.reset_index(drop=True),
        A_state_te.reset_index(drop=True), A_bin_te.reset_index(drop=True), A_cnt_te.reset_index(drop=True), A_op_te.reset_index(drop=True),
        A_dr_te.reset_index(drop=True), A_pp_te.reset_index(drop=True), B_pair_te.reset_index(drop=True), L_te.reset_index(drop=True)
    ], axis=1)

    X_train = X_train.loc[:, ~X_train.columns.duplicated()].copy()
    X_test = X_test.loc[:, X_train.columns].copy()
    X_train = sanitize_columns(X_train)
    X_test = sanitize_columns(X_test)

    string_cols = [c for c in X_train.columns if str(X_train[c].dtype) in ('object', 'string') or pd.api.types.is_string_dtype(X_train[c])]
    te_cols = []
    for c in string_cols:
        nunq = pd.concat([X_train[c], X_test[c]], axis=0).astype('string').nunique(dropna=False)
        if c.endswith('_te') or c.startswith('B_pair_') or c.startswith('L_') or nunq > HIGH_CARD_THRESHOLD:
            te_cols.append(c)
    te_cols = dedup_keep_order(te_cols)
    cat_factor_cols = [c for c in string_cols if c not in te_cols and c != ID]
    y = train[TARGET].astype('int8').to_numpy()
    print(f'X_train {X_train.shape}, X_test {X_test.shape}, target-encoded string columns {len(te_cols)}, '
          f'factorized string columns {len(cat_factor_cols)}', flush=True)
    return X_train, X_test, te_cols, cat_factor_cols, y


def main():
    X_train, X_test, te_cols, cat_factor_cols, y = build_features()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_train), dtype=np.float32)
    test_pred = np.zeros(len(X_test), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr = X_train.iloc[tr_idx].copy()
        y_tr = y[tr_idx]
        X_va = X_train.iloc[va_idx].copy()
        y_va = y[va_idx]
        X_te = X_test.copy()

        te_blocks_tr, te_blocks_va, te_blocks_te = {}, {}, {}
        for c in te_cols:
            tr_enc, va_enc, te_enc = nested_target_encode(X_tr[c], y_tr, X_va[c], X_te[c], n_splits=INNER_FOLDS, seed=SEED + fold)
            te_blocks_tr[f'TE_{c}'] = tr_enc
            te_blocks_va[f'TE_{c}'] = va_enc
            te_blocks_te[f'TE_{c}'] = te_enc
        if te_blocks_tr:
            X_tr = pd.concat([X_tr.drop(columns=te_cols), pd.DataFrame(te_blocks_tr, index=X_tr.index)], axis=1)
            X_va = pd.concat([X_va.drop(columns=te_cols), pd.DataFrame(te_blocks_va, index=X_va.index)], axis=1)
            X_te = pd.concat([X_te.drop(columns=te_cols), pd.DataFrame(te_blocks_te, index=X_te.index)], axis=1)

        fac_cols = [c for c in X_tr.columns if c in cat_factor_cols]
        if fac_cols:
            X_tr, X_va, X_te = consistent_factorize(X_tr, X_va, X_te, fac_cols)
        X_tr = clean_for_lgb(X_tr)
        X_va = clean_for_lgb(X_va)
        X_te = clean_for_lgb(X_te)

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc',
                  callbacks=[lgb.early_stopping(400, verbose=False), lgb.log_evaluation(0)])
        oof[va_idx] = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1].astype('float32')
        test_pred += model.predict_proba(X_te, num_iteration=model.best_iteration_)[:, 1].astype('float32') / N_FOLDS
        print(f'fold {fold - 1}: AUC {roc_auc_score(y_va, oof[va_idx]):.6f}, best_iteration {model.best_iteration_}', flush=True)
        del X_tr, X_va, X_te, model
        gc.collect()

    print(f'{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
