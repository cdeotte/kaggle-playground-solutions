#!/usr/bin/env python
"""GPU XGBoost on a union of three feature pipelines with nested target statistics and pre-binning (pair
xgb_union_te_prebinned). Only competition rows are trained on; the original rows feed counts, ranks, drift ratios and
original target priors.

Label-free features (train + test + original):
  - Pipeline A: count / rank / signed log / signed sqrt / 1 / (1 + |x|) of LapTime (s), LapTime_Delta and
    Cumulative_Degradation; 11 race-state interactions and 12 state flags; tyre-life and lap bins, 40 / 60 quantile bins of
    the three timing columns, 8 category crosses and Race x Compound x Stint; counts and rare flags of every category;
    original-target priors of every category / bin and of the exact timing values;
  - Pipeline B: 5 quantile bins, rounding to tens / units, digits 10^-2..10^2 and 2-decimal fractional parts of the timing
    columns (factorized over all rows); train / original frequency ratios, original target mean and count of every key,
    category means of the timing columns over train, pairs of keys with <= 20,000 levels;
  - LGBM-port keys: the raw columns with > 7 levels and, for 6 numerics, round(x / 2), floor, up to 5000 quantile bins,
    200-8000 equal-width bins and log bins; extra digit columns and 28 digit pairs; counts and original target priors.
Per fold (nested, inner StratifiedKFold(5, shuffle=True, random_state=42) for the training rows): smoothed (20) target
means of the A / B / LGBM-port keys, target std / min / max of the A keys, mean / std / min / max of the B and LGBM-port
keys, counts of the B pairs; the raw keys are then dropped. Every column with > 256 distinct values except the three raw
timing columns is pre-binned to 256 quantile bins of the fold's training rows (the timing columns keep their values;
XGBoost max_bin 16000).
Model: depth 6, lr 0.03, subsample 0.8, colsample_bytree 0.4, min_child_weight 5, alpha 0.1, lambda 1, gamma 0.05,
max_bin 16000, early stopping 500 on the validation fold, random_state 42. Test = mean of the 5 fold models.
Usage: python train_union_prebinned.py [--folds 0,1,2,3,4] [--force]"""
from __future__ import annotations

import argparse
import gc
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

PAIR = "xgb_union_te_prebinned"
ID_COL = ID
N_FOLDS = 5
INNER_FOLDS = 5
SEED = 42

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
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'enable_categorical': False,
    'device': 'cuda',
    'tree_method': 'hist',
    'max_bin': 16000,
    'verbosity': 0,
    'random_state': SEED,
}

# Feature knobs.
Q_LIST = [5]
MAKE_PAIR_FEATURES = True
PAIR_MAX_UNIQUE = 20_000
DRIFT_EPS = 1e-6
HIGH_CARD_THRESHOLD = 7
QCUT_BINS = 5000
CUT_BINS = {
    'LapTime (s)': 5000,
    'LapTime_Delta': 5000,
    'Cumulative_Degradation': 8000,
    'RaceProgress': 3000,
    'TyreLife': 200,
    'LapNumber': 300,
}
ROUND_SPECS = {
    'LapTime (s)': [-1, 0],
    'LapTime_Delta': [-1, 0],
    'Cumulative_Degradation': [-1, 0],
}
DIGIT_SPECS = {
    'LapTime (s)': list(range(-2, 2)),
    'LapTime_Delta': list(range(-2, 3)),
    'Cumulative_Degradation': list(range(-2, 3)),
}

# Pre-binning: most features to <= 256 bins, the key continuous timing values kept fine-grained.
BIN_SMALL = 256
BIN_EXCLUDE_COLS = {'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation'}
BIN_VERBOSE_EVERY = 150


def log(msg: str) -> None:
    print(msg, flush=True)


def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast='float')
        elif pd.api.types.is_integer_dtype(df[c]) and c not in [ID_COL, TARGET]:
            df[c] = pd.to_numeric(df[c], downcast='integer')
    return df


def safe_name(col: str) -> str:
    return (
        str(col)
        .replace(' ', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('/', '_')
        .replace('-', '_')
        .replace('.', '_')
        .replace('%', 'pct')
    )


def signed_log1p(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce').astype('float64')
    return np.sign(s) * np.log1p(np.abs(s))


def signed_sqrt(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce').astype('float64')
    return np.sign(s) * np.sqrt(np.abs(s))


def to_binary_target(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype('int8')
    return s.astype(str).str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0}).astype('int8')


def as_str_key(s: pd.Series) -> pd.Series:
    return s.astype('string').fillna('__NA__').astype(str)


def add_count_from_all(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame | None, col: str, name: str) -> None:
    parts = [train[col], test[col]]
    if orig is not None and col in orig.columns:
        parts.append(orig[col])
    vc = pd.concat(parts, axis=0).astype('string').fillna('__NA__').value_counts(dropna=False)
    for df in [train, test] + ([] if orig is None else [orig]):
        if col in df.columns:
            df[name] = df[col].astype('string').fillna('__NA__').map(vc).fillna(0).astype('float32')


def add_rank_from_all(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame | None, col: str, name: str) -> None:
    parts = [pd.to_numeric(train[col], errors='coerce'), pd.to_numeric(test[col], errors='coerce')]
    if orig is not None and col in orig.columns:
        parts.append(pd.to_numeric(orig[col], errors='coerce'))
    all_values = pd.concat(parts, axis=0)
    ranks = all_values.rank(method='average', pct=True).astype('float32')
    n_train, n_test = len(train), len(test)
    train[name] = ranks.iloc[:n_train].to_numpy(dtype='float32')
    test[name] = ranks.iloc[n_train:n_train + n_test].to_numpy(dtype='float32')
    if orig is not None and col in orig.columns:
        orig[name] = ranks.iloc[n_train + n_test:].to_numpy(dtype='float32')


def safe_qcut_codes(series_list, q: int) -> list[pd.Series]:
    all_s = pd.concat([pd.to_numeric(s, errors='coerce') for s in series_list], axis=0)
    try:
        codes = pd.qcut(all_s.rank(method='first'), q=q, labels=False, duplicates='drop').astype('float32')
    except Exception:
        codes = pd.Series(np.zeros(len(all_s), dtype='float32'), index=all_s.index)
    out = []
    start = 0
    for s in series_list:
        out.append(codes.iloc[start:start + len(s)].reset_index(drop=True).astype('float32'))
        start += len(s)
    return out


def safe_cut_codes(series_list, bins: int) -> list[pd.Series]:
    all_s = pd.concat([pd.to_numeric(s, errors='coerce') for s in series_list], axis=0)
    try:
        codes = pd.cut(all_s, bins=bins, labels=False, duplicates='drop').astype('float32')
    except Exception:
        codes = pd.Series(np.zeros(len(all_s), dtype='float32'), index=all_s.index)
    out = []
    start = 0
    for s in series_list:
        out.append(codes.iloc[start:start + len(s)].reset_index(drop=True).astype('float32'))
        start += len(s)
    return out


def factorize_across(frames: list[pd.DataFrame], cols: list[str]) -> None:
    for col in cols:
        present = [df for df in frames if col in df.columns]
        if not present:
            continue
        vals = pd.concat([as_str_key(df[col]) for df in present], axis=0)
        codes, _ = pd.factorize(vals, sort=True)
        start = 0
        for df in present:
            df[col] = codes[start:start + len(df)].astype('int32')
            start += len(df)


def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.is_unique:
        return df
    counts = {}
    new_cols = []
    for c in df.columns:
        if c not in counts:
            counts[c] = 0
            new_cols.append(c)
        else:
            counts[c] += 1
            new_cols.append(f'{c}__dup{counts[c]}')
    out = df.copy()
    out.columns = new_cols
    return out


def prepare_for_xgb(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype('int8')
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.replace([np.inf, -np.inf], np.nan).fillna(-999.0)
    return reduce_mem(out)


def _make_bin_edges_from_train(x, n_bins):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    xs = x
    if x.size > 2_000_000:
        rs = np.random.RandomState(123)
        xs = x[rs.choice(x.size, size=2_000_000, replace=False)]
    if np.unique(xs).size <= n_bins:
        return None
    try:
        edges = np.nanquantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    except Exception:
        return None
    edges = np.unique(edges)
    if edges.size < 3:
        return None
    return edges


def _apply_edges_to_series(s, edges):
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    out = np.full(x.shape, -1, dtype=np.int32)
    m = np.isfinite(x)
    out[m] = np.searchsorted(edges[1:-1], x[m], side="right").astype(np.int32)
    return out


def bin_matrices_like_xgb(X_tr, X_va, X_te, n_bins=256, exclude_cols=None, log_prefix=""):
    """Replace every column (except `exclude_cols`) with its bin index among <= n_bins quantile bins of the training rows."""
    if exclude_cols is None:
        exclude_cols = set()
    cols = list(X_tr.columns)
    cols_to_bin = [c for c in cols if c not in exclude_cols]
    log(f"{log_prefix}binning {len(cols_to_bin)}/{len(cols)} columns to at most {n_bins} bins")
    for j, c in enumerate(cols_to_bin, start=1):
        edges = _make_bin_edges_from_train(X_tr[c].to_numpy(dtype=np.float64, copy=False), n_bins=n_bins)
        if edges is None:
            for df in (X_tr, X_va, X_te):
                if df[c].dtype == "float64":
                    df[c] = df[c].astype("float32")
            continue
        X_tr[c] = _apply_edges_to_series(X_tr[c], edges)
        X_va[c] = _apply_edges_to_series(X_va[c], edges)
        X_te[c] = _apply_edges_to_series(X_te[c], edges)
        if n_bins <= 32000:
            X_tr[c] = X_tr[c].astype("int16")
            X_va[c] = X_va[c].astype("int16")
            X_te[c] = X_te[c].astype("int16")
        if j % BIN_VERBOSE_EVERY == 0:
            log(f"{log_prefix}binned {j}/{len(cols_to_bin)} columns")
    return X_tr, X_va, X_te


def load_data():
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = to_binary_target(train[TARGET])
    if TARGET in orig.columns:
        orig[TARGET] = to_binary_target(orig[TARGET])
    # Align original to competition feature names.
    keep_orig = [c for c in train.columns if c != ID_COL and c in orig.columns]
    orig = orig[keep_orig].copy()
    train = reduce_mem(train)
    test = reduce_mem(test)
    orig = reduce_mem(orig)
    log(f'train={train.shape}, test={test.shape}, original={orig.shape}')
    return train, test, orig


def infer_feature_groups(train: pd.DataFrame, test: pd.DataFrame):
    feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL]]
    cats, nums = [], []
    both = pd.concat([train[feature_cols], test[feature_cols]], axis=0, ignore_index=True)
    for c in feature_cols:
        if pd.api.types.is_object_dtype(both[c]) or pd.api.types.is_string_dtype(both[c]) or pd.api.types.is_categorical_dtype(both[c]):
            cats.append(c)
        else:
            nunique = both[c].nunique(dropna=False)
            if nunique < 100:
                cats.append(c)
            else:
                nums.append(c)
    log(f'CATS ({len(cats)}): {cats}')
    log(f'NUMS ({len(nums)}): {nums}')
    return cats, nums


def add_pipeline_a_features(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, base_cats: list[str], base_nums: list[str]):
    log('Building Pipeline A features')
    a_cats = [c for c in ['Driver', 'Compound', 'Race', 'Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position', 'Position_Change'] if c in train.columns]
    a_nums = [c for c in ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation'] if c in train.columns]
    if not a_nums:
        a_nums = base_nums[:3]

    for df in [train, test, orig]:
        for c in a_cats:
            if c in df.columns:
                df[c] = as_str_key(df[c])
        for c in a_nums + ['TyreLife', 'LapNumber', 'RaceProgress', 'Position', 'Position_Change', 'PitStop']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

    # Frequency, rank, and monotone transforms for high-card numeric artifact columns.
    for c in a_nums:
        s = safe_name(c)
        add_count_from_all(train, test, orig, c, f'A_FREQ_{s}')
        add_rank_from_all(train, test, orig, c, f'A_RANK_{s}')
        for df in [train, test, orig]:
            df[f'A_LOG1P_{s}'] = signed_log1p(df[c]).astype('float32')
            df[f'A_SQRT_{s}'] = signed_sqrt(df[c]).astype('float32')
            df[f'A_INV1P_ABS_{s}'] = (1.0 / (1.0 + np.abs(pd.to_numeric(df[c], errors='coerce')))).astype('float32')
            df[f'A_MISSING_{s}'] = df[c].isna().astype('int8')

    # Race-state interactions.
    def add_f1_interactions(df: pd.DataFrame) -> None:
        tyre = pd.to_numeric(df.get('TyreLife', 0), errors='coerce').fillna(0).astype('float64')
        lap = pd.to_numeric(df.get('LapNumber', 0), errors='coerce').fillna(0).astype('float64')
        progress = pd.to_numeric(df.get('RaceProgress', 0), errors='coerce').fillna(0).astype('float64')
        position = pd.to_numeric(df.get('Position', 0), errors='coerce').fillna(0).astype('float64')
        pos_change = pd.to_numeric(df.get('Position_Change', 0), errors='coerce').fillna(0).astype('float64')
        pitstop = pd.to_numeric(df.get('PitStop', 0), errors='coerce').fillna(0).astype('float64')
        lap_time = pd.to_numeric(df.get('LapTime (s)', 0), errors='coerce').fillna(0).astype('float64')
        lap_delta = pd.to_numeric(df.get('LapTime_Delta', 0), errors='coerce').fillna(0).astype('float64')
        degr = pd.to_numeric(df.get('Cumulative_Degradation', 0), errors='coerce').fillna(0).astype('float64')

        df['A_degradation_deviation'] = (degr - tyre * lap_delta).astype('float32')
        df['A_abs_degradation_dev'] = np.abs(df['A_degradation_deviation']).astype('float32')
        df['A_delta_to_degradation_ratio'] = (lap_delta / (np.abs(degr) + 1.0)).astype('float32')
        df['A_degradation_to_delta_ratio'] = (degr / (np.abs(lap_delta) + 1.0)).astype('float32')
        df['A_avg_degradation_per_tyre'] = (degr / (tyre + 1.0)).astype('float32')
        df['A_avg_laptime_per_lap'] = (lap_time / (lap + 1.0)).astype('float32')
        df['A_tyre_x_delta'] = (tyre * lap_delta).astype('float32')
        df['A_tyre_x_degradation'] = (tyre * degr).astype('float32')
        df['A_lap_x_progress'] = (lap * progress).astype('float32')
        df['A_position_x_change'] = (position * pos_change).astype('float32')
        df['A_pitstop_x_tyre'] = (pitstop * tyre).astype('float32')

        comp = as_str_key(df.get('Compound', pd.Series([''] * len(df), index=df.index))).str.lower()
        indicators = pd.DataFrame(index=df.index)
        indicators['soft'] = comp.str.contains('soft').astype('int8')
        indicators['medium'] = comp.str.contains('medium').astype('int8')
        indicators['hard'] = comp.str.contains('hard').astype('int8')
        indicators['wet_or_inter'] = (comp.str.contains('wet') | comp.str.contains('inter')).astype('int8')
        indicators['front_position'] = (position <= 3).astype('int8')
        indicators['points_position'] = (position <= 10).astype('int8')
        indicators['position_loss'] = (pos_change < 0).astype('int8')
        indicators['position_gain'] = (pos_change > 0).astype('int8')
        indicators['has_pitstop'] = (pitstop > 0).astype('int8')
        indicators['old_tyre'] = (tyre >= 20).astype('int8')
        indicators['late_race'] = (progress >= 0.70).astype('int8')
        indicators['slow_lap'] = (lap_delta > 0).astype('int8')
        for c in indicators.columns:
            df[f'A_STATE_{c}'] = indicators[c].astype('int8')
        df['A_state_risk_count'] = indicators[['old_tyre', 'late_race', 'slow_lap', 'position_loss', 'has_pitstop']].sum(axis=1).astype('int8')
        df['A_state_context_count'] = indicators.sum(axis=1).astype('int8')

    for df in [train, test, orig]:
        add_f1_interactions(df)

    # Bins and cross features.
    a_new_cats = []
    if 'TyreLife' in train.columns:
        bins = [-1, 0, 1, 3, 5, 8, 12, 16, 20, 30, 50, 10_000]
        for df in [train, test, orig]:
            df['A_TyreLife_bin'] = pd.cut(pd.to_numeric(df['TyreLife'], errors='coerce'), bins=bins, labels=False, include_lowest=True).astype('float32').fillna(-1).astype('int16').astype(str)
        a_new_cats.append('A_TyreLife_bin')
    if 'LapNumber' in train.columns:
        bins = [-1, 1, 5, 10, 15, 20, 30, 40, 50, 70, 1000]
        for df in [train, test, orig]:
            df['A_LapNumber_bin'] = pd.cut(pd.to_numeric(df['LapNumber'], errors='coerce'), bins=bins, labels=False, include_lowest=True).astype('float32').fillna(-1).astype('int16').astype(str)
        a_new_cats.append('A_LapNumber_bin')

    for c in a_nums:
        q = 60 if c == 'Cumulative_Degradation' else 40
        name = f'A_{safe_name(c)}_qbin'
        qs = safe_qcut_codes([train[c], test[c], orig[c]], q=q)
        for df, vals in zip([train, test, orig], qs):
            df[name] = vals.fillna(-1).astype('int16').astype(str)
        a_new_cats.append(name)

    cross_pairs = [
        ('Driver', 'Race'), ('Driver', 'Compound'), ('Race', 'Compound'),
        ('Compound', 'Stint'), ('Race', 'Stint'), ('PitStop', 'Compound'),
        ('Position', 'Compound'), ('TyreLife', 'Compound')
    ]
    for c1, c2 in cross_pairs:
        if c1 in train.columns and c2 in train.columns:
            name = f'A_{safe_name(c1)}__{safe_name(c2)}'
            for df in [train, test, orig]:
                df[name] = as_str_key(df[c1]) + '_' + as_str_key(df[c2])
            a_new_cats.append(name)
    triple = ('Race', 'Compound', 'Stint')
    if all(c in train.columns for c in triple):
        name = 'A_Race__Compound__Stint'
        for df in [train, test, orig]:
            df[name] = as_str_key(df[triple[0]]) + '_' + as_str_key(df[triple[1]]) + '_' + as_str_key(df[triple[2]])
        a_new_cats.append(name)

    # Counts and rare flags for raw/new categorical columns.
    for c in a_cats + a_new_cats:
        add_count_from_all(train, test, orig, c, f'A_COUNT_{safe_name(c)}')
        for df in [train, test, orig]:
            df[f'A_RARE_{safe_name(c)}'] = (df[f'A_COUNT_{safe_name(c)}'] <= 3).astype('int8')

    # Original target priors by source-like keys. These use only original labels.
    prior_cols = a_cats + a_nums + a_new_cats
    for c in prior_cols:
        if c not in orig.columns:
            continue
        name = f'A_ORIG_PROBA_{safe_name(c)}'
        mapping = orig.groupby(c, dropna=False)[TARGET].mean()
        for df in [train, test, orig]:
            df[name] = df[c].map(mapping).astype('float32').fillna(0.5)

    # Numeric-as-category exact artifact copies.
    a_num_as_cat = []
    for c in a_nums:
        name = f'A_NUMCAT_{safe_name(c)}'
        for df in [train, test, orig]:
            df[name] = as_str_key(pd.to_numeric(df[c], errors='coerce').round(6))
        a_num_as_cat.append(name)

    log(f'Pipeline A: {len(a_cats)} cats, {len(a_nums)} nums, {len(a_new_cats)} new cats, {len(a_num_as_cat)} num-as-cat')
    return a_cats, a_nums, a_new_cats, a_num_as_cat


def add_qcut_bins(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, num_cols: list[str], q_list: list[int]) -> list[str]:
    new_cols = []
    for c in num_cols:
        for q in q_list:
            name = f'B_{safe_name(c)}_q{q}'
            vals = safe_qcut_codes([train[c], test[c], orig[c]], q=q)
            for df, v in zip([train, test, orig], vals):
                df[name] = v.fillna(-1).astype('int16').astype(str)
            new_cols.append(name)
    return new_cols


def add_round_features(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, round_specs: dict[str, list[int]]) -> list[str]:
    new_cols = []
    for c, ndigits_list in round_specs.items():
        if c not in train.columns:
            continue
        for ndigits in ndigits_list:
            suffix = str(ndigits).replace('-', 'm')
            name = f'B_{safe_name(c)}_round_{suffix}'
            for df in [train, test, orig]:
                df[name] = pd.to_numeric(df[c], errors='coerce').round(ndigits).astype('float32').astype(str)
            new_cols.append(name)
    return new_cols


def add_digit_features(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, digit_specs: dict[str, list[int]]) -> list[str]:
    new_cols = []
    for c, positions in digit_specs.items():
        if c not in train.columns:
            continue
        for pos in positions:
            name = f'B_DIG_{safe_name(c)}_{pos}'
            scale = 10 ** (-pos)
            for df in [train, test, orig]:
                vals = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('float64')
                df[name] = (np.floor(np.abs(vals) * scale).astype('int64') % 10).astype('int8').astype(str)
            new_cols.append(name)
    return new_cols


def add_orig_te_ce(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, cols: list[str]) -> list[str]:
    new_cols = []
    y_orig = orig[TARGET].astype('float32')
    global_mean = float(y_orig.mean())
    for c in cols:
        if c not in train.columns or c not in orig.columns:
            continue
        key_orig = orig[c]
        mean_map = orig.groupby(key_orig, dropna=False)[TARGET].mean()
        size_map = orig.groupby(key_orig, dropna=False)[TARGET].size()
        te_name = f'B_ORIG_TE_{safe_name(c)}'
        ce_name = f'B_ORIG_CE_{safe_name(c)}'
        for df in [train, test, orig]:
            df[te_name] = df[c].map(mean_map).astype('float32').fillna(global_mean)
            df[ce_name] = df[c].map(size_map).astype('float32').fillna(0)
        new_cols.extend([te_name, ce_name])
    return new_cols


def add_group_means(train: pd.DataFrame, test: pd.DataFrame, group_cols: list[str], num_cols: list[str]) -> list[str]:
    new_cols = []
    for g in group_cols:
        if g not in train.columns:
            continue
        for n in num_cols:
            if n not in train.columns:
                continue
            name = f'B_MEAN_{safe_name(n)}_BY_{safe_name(g)}'
            mapping = train.groupby(g, dropna=False)[n].mean()
            fallback = float(pd.to_numeric(train[n], errors='coerce').mean())
            train[name] = train[g].map(mapping).astype('float32').fillna(fallback)
            test[name] = test[g].map(mapping).astype('float32').fillna(fallback)
            new_cols.append(name)
    return new_cols


def add_drift_ratio(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, cols: list[str]) -> list[str]:
    new_cols = []
    for c in cols:
        if c not in train.columns or c not in orig.columns:
            continue
        tr_vc = as_str_key(train[c]).value_counts(normalize=True)
        orig_vc = as_str_key(orig[c]).value_counts(normalize=True)
        name = f'B_DRIFT_RATIO_{safe_name(c)}'
        for df in [train, test, orig]:
            keys = as_str_key(df[c])
            tr_p = keys.map(tr_vc).fillna(0).astype('float32')
            or_p = keys.map(orig_vc).fillna(0).astype('float32')
            df[name] = ((tr_p + DRIFT_EPS) / (or_p + DRIFT_EPS)).astype('float32')
        new_cols.append(name)
    return new_cols


def make_pair_features(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, cols: list[str], pair_max_unique: int) -> list[str]:
    new_cols = []
    for c1, c2 in itertools.combinations(cols, 2):
        if c1 not in train.columns or c2 not in train.columns:
            continue
        sample = as_str_key(train[c1]).str.cat(as_str_key(train[c2]), sep='_')
        if sample.nunique(dropna=False) > pair_max_unique:
            continue
        name = f'B_PAIR_{safe_name(c1)}__{safe_name(c2)}'
        for df in [train, test, orig]:
            if c1 in df.columns and c2 in df.columns:
                df[name] = as_str_key(df[c1]).str.cat(as_str_key(df[c2]), sep='_')
        new_cols.append(name)
    return new_cols


def add_pipeline_b_features(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, base_nums: list[str]):
    log('Building Pipeline B features')
    b_nums = [c for c in ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation'] if c in train.columns]
    if not b_nums:
        b_nums = base_nums[:3]

    exclude = set([TARGET, ID_COL]) | set(b_nums)
    b_cats = [c for c in train.columns if c not in exclude and not c.startswith('A_') and not c.startswith('B_')]

    b_bins = add_qcut_bins(train, test, orig, b_nums, Q_LIST)
    b_round = add_round_features(train, test, orig, ROUND_SPECS)
    b_digits = add_digit_features(train, test, orig, DIGIT_SPECS)

    b_num_cat = []
    for c in b_nums:
        name = f'B_NUMCAT_{safe_name(c)}'
        for df in [train, test, orig]:
            df[name] = as_str_key(pd.to_numeric(df[c], errors='coerce').round(6))
        b_num_cat.append(name)

    # Fractional components of the timing columns.
    frac_cols = []
    for c in b_nums:
        name = f'B_FRAC2_{safe_name(c)}'
        for df in [train, test, orig]:
            vals = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('float64')
            df[name] = np.round(np.abs(vals) - np.floor(np.abs(vals)), 2).astype('float32').astype(str)
        frac_cols.append(name)

    factorize_cols = b_cats + b_num_cat + b_bins + b_round + b_digits + frac_cols
    factorize_across([train, test, orig], factorize_cols)

    add_drift_ratio(train, test, orig, b_cats + b_num_cat + b_bins + b_round + frac_cols)
    add_orig_te_ce(train, test, orig, b_cats + b_num_cat + b_bins + b_round + b_digits + frac_cols)
    add_group_means(train, test, b_cats, b_nums)

    b_pairs = []
    if MAKE_PAIR_FEATURES:
        pair_base = b_cats + b_num_cat + b_bins + b_round[:3]
        b_pairs = make_pair_features(train, test, orig, pair_base, PAIR_MAX_UNIQUE)
        factorize_across([train, test, orig], b_pairs)

    log(f'Pipeline B: {len(b_cats)} cats, {len(b_nums)} nums, {len(b_pairs)} pairs, {len(b_digits)} digit cols')
    return b_nums, b_cats, b_num_cat, b_bins, b_round, b_digits, frac_cols, b_pairs


def add_lgbm_port_features(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame):
    log('Building LGBM-port features')
    raw_cols = [c for c in train.columns if c not in [TARGET, ID_COL] and not c.startswith('A_') and not c.startswith('B_') and not c.startswith('L_')]
    combined = pd.concat([train[raw_cols], test[raw_cols]], axis=0, ignore_index=True)
    high_card_cols = [c for c in raw_cols if combined[c].nunique(dropna=False) > HIGH_CARD_THRESHOLD]

    l_te_cols = []
    for c in high_card_cols:
        te_col = f'L_{safe_name(c)}_te'
        cnt_col = f'L_count_{safe_name(c)}'
        for df in [train, test, orig]:
            if c in df.columns:
                df[te_col] = as_str_key(df[c])
        add_count_from_all(train, test, orig, c, cnt_col)
        l_te_cols.append(te_col)

        if c in orig.columns:
            mean_map = orig.groupby(c, dropna=False)[TARGET].mean()
            size_map = orig.groupby(c, dropna=False)[TARGET].size()
            mean_name = f'L_orig_mean_{safe_name(c)}'
            size_name = f'L_orig_size_{safe_name(c)}'
            ratio_name = f'L_count_ratio_{safe_name(c)}'
            global_mean = float(orig[TARGET].mean())
            for df in [train, test, orig]:
                df[mean_name] = df[c].map(mean_map).astype('float32').fillna(global_mean)
                df[size_name] = df[c].map(size_map).astype('float32').fillna(0)
                df[ratio_name] = (df[cnt_col].astype('float32') / (df[size_name].astype('float32') + 1.0)).astype('float32')

    numeric_bin_cols = [c for c in ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'TyreLife', 'LapNumber'] if c in train.columns]
    for c in numeric_bin_cols:
        s = safe_name(c)
        vals = [pd.to_numeric(train[c], errors='coerce'), pd.to_numeric(test[c], errors='coerce'), pd.to_numeric(orig[c], errors='coerce')]
        # Round/divide-by-two and integer artifact categories.
        for df in [train, test, orig]:
            v = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('float64')
            df[f'L_{s}_round_div2_te'] = np.round(v / 2.0, 0).astype('int64').astype(str)
            df[f'L_{s}_int_te'] = np.floor(v).astype('int64').astype(str)
        l_te_cols.extend([f'L_{s}_round_div2_te', f'L_{s}_int_te'])

        qvals = safe_qcut_codes(vals, q=min(QCUT_BINS, max(10, pd.concat(vals).nunique(dropna=True))))
        for df, qv in zip([train, test, orig], qvals):
            df[f'L_{s}_quantiled_te'] = qv.fillna(-1).astype('int32').astype(str)
        l_te_cols.append(f'L_{s}_quantiled_te')

        bins = CUT_BINS.get(c, 3000)
        cvals = safe_cut_codes(vals, bins=bins)
        for df, cv in zip([train, test, orig], cvals):
            df[f'L_{s}_binned_te'] = cv.fillna(-1).astype('int32').astype(str)
        l_te_cols.append(f'L_{s}_binned_te')

        log_vals = [signed_log1p(v) for v in vals]
        lvals = safe_cut_codes(log_vals, bins=min(bins, 3000))
        for df, lv in zip([train, test, orig], lvals):
            df[f'L_{s}_log_binned_te'] = lv.fillna(-1).astype('int32').astype(str)
        l_te_cols.append(f'L_{s}_log_binned_te')

    # Extra digit columns of the high-card numeric columns.
    digit_cols = []
    for c, positions in DIGIT_SPECS.items():
        if c not in train.columns:
            continue
        s = safe_name(c)
        for pos in positions:
            name = f'L_digit_{s}_{pos}_te'
            scale = 10 ** (-pos)
            for df in [train, test, orig]:
                v = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('float64')
                df[name] = (np.floor(np.abs(v) * scale).astype('int64') % 10).astype('int8').astype(str)
            digit_cols.append(name)
    for c1, c2 in itertools.combinations(digit_cols[:8], 2):
        name = f'L_digit_pair_{safe_name(c1)}__{safe_name(c2)}'
        for df in [train, test, orig]:
            df[name] = as_str_key(df[c1]) + '_' + as_str_key(df[c2])
        l_te_cols.append(name)
    l_te_cols.extend(digit_cols)

    log(f'LGBM-port: {len(high_card_cols)} high-card raw cols, {len(l_te_cols)} TE cols')
    return list(dict.fromkeys(l_te_cols)), high_card_cols


def count_encode_fold(X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame, cols: list[str]) -> list[str]:
    new_cols = []
    for c in cols:
        if c not in X_tr.columns:
            continue
        name = f'CE_{safe_name(c)}'
        vc = as_str_key(X_tr[c]).value_counts(dropna=False)
        X_tr[name] = as_str_key(X_tr[c]).map(vc).fillna(0).astype('float32')
        X_va[name] = as_str_key(X_va[c]).map(vc).fillna(0).astype('float32')
        X_te[name] = as_str_key(X_te[c]).map(vc).fillna(0).astype('float32')
        new_cols.append(name)
    return new_cols


def _smooth_mean_map(keys: pd.Series, y: pd.Series, smooth: float = 20.0):
    tmp = pd.DataFrame({'key': as_str_key(keys), 'y': y.astype('float32')})
    stats = tmp.groupby('key')['y'].agg(['mean', 'count'])
    global_mean = float(y.mean())
    values = (stats['mean'] * stats['count'] + global_mean * smooth) / (stats['count'] + smooth)
    return values, global_mean


def target_encode_mean_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    cols: list[str],
    prefix: str,
    inner_folds: int = 5,
    smooth: float = 20.0,
) -> list[str]:
    new_cols = []
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=SEED)
    y_tr = pd.Series(y_tr).reset_index(drop=True)
    for c in cols:
        if c not in X_tr.columns:
            continue
        name = f'{prefix}_TE_{safe_name(c)}'
        tr_encoded = np.zeros(len(X_tr), dtype='float32')
        xcol = X_tr[c].reset_index(drop=True)
        for in_tr_idx, in_va_idx in inner.split(X_tr, y_tr):
            mapping, global_mean = _smooth_mean_map(xcol.iloc[in_tr_idx], y_tr.iloc[in_tr_idx], smooth=smooth)
            tr_encoded[in_va_idx] = as_str_key(xcol.iloc[in_va_idx]).map(mapping).fillna(global_mean).to_numpy(dtype='float32')
        full_map, global_mean = _smooth_mean_map(xcol, y_tr, smooth=smooth)
        X_tr[name] = tr_encoded
        X_va[name] = as_str_key(X_va[c]).map(full_map).fillna(global_mean).astype('float32')
        X_te[name] = as_str_key(X_te[c]).map(full_map).fillna(global_mean).astype('float32')
        new_cols.append(name)
    return new_cols


def target_encode_aggs_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    cols: list[str],
    aggs: list[str],
    prefix: str,
    inner_folds: int = 5,
) -> list[str]:
    new_cols = []
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=SEED)
    y_tr = pd.Series(y_tr).reset_index(drop=True).astype('float32')
    for c in cols:
        if c not in X_tr.columns:
            continue
        xcol = X_tr[c].reset_index(drop=True)
        for agg in aggs:
            name = f'{prefix}_{agg}_{safe_name(c)}'
            tr_encoded = np.zeros(len(X_tr), dtype='float32')
            for in_tr_idx, in_va_idx in inner.split(X_tr, y_tr):
                tmp = pd.DataFrame({'key': as_str_key(xcol.iloc[in_tr_idx]), 'y': y_tr.iloc[in_tr_idx].to_numpy(dtype='float32')})
                mp = tmp.groupby('key')['y'].agg(agg)
                tr_encoded[in_va_idx] = as_str_key(xcol.iloc[in_va_idx]).map(mp).fillna(0).to_numpy(dtype='float32')
            tmp_full = pd.DataFrame({'key': as_str_key(xcol), 'y': y_tr.to_numpy(dtype='float32')})
            mp_full = tmp_full.groupby('key')['y'].agg(agg)
            X_tr[name] = tr_encoded
            X_va[name] = as_str_key(X_va[c]).map(mp_full).fillna(0).astype('float32')
            X_te[name] = as_str_key(X_te[c]).map(mp_full).fillna(0).astype('float32')
            new_cols.append(name)
    return new_cols


def factorize_remaining_objects(X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame) -> None:
    obj_cols = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]
    for c in obj_cols:
        vals = pd.concat([as_str_key(X_tr[c]), as_str_key(X_va[c]), as_str_key(X_te[c])], axis=0)
        codes, _ = pd.factorize(vals, sort=True)
        n1, n2 = len(X_tr), len(X_va)
        X_tr[c] = codes[:n1].astype('int32')
        X_va[c] = codes[n1:n1 + n2].astype('int32')
        X_te[c] = codes[n1 + n2:].astype('int32')


def build_features():
    train, test, orig = load_data()
    base_cats, base_nums = infer_feature_groups(train, test)
    a_cats, _, a_new_cats, a_num_as_cat = add_pipeline_a_features(train, test, orig, base_cats, base_nums)
    _, b_cats, b_num_cat, _, _, _, _, b_pairs = add_pipeline_b_features(train, test, orig, base_nums)
    lgbm_te_cols, _ = add_lgbm_port_features(train, test, orig)

    # Remove accidental duplicate columns before selecting model features.
    train = deduplicate_columns(train)
    test = deduplicate_columns(test)

    base_union_features = [c for c in train.columns if c not in [TARGET, ID_COL]]
    keys = dict(
        a_te=[c for c in a_num_as_cat + a_cats + a_new_cats if c in train.columns],
        b_te=[c for c in b_num_cat + b_cats + b_pairs if c in train.columns],
        b_ce=[c for c in b_pairs if c in train.columns],
        lgbm_te=[c for c in lgbm_te_cols if c in train.columns],
    )
    keys['drop'] = list(dict.fromkeys(keys['a_te'] + keys['b_te'] + keys['lgbm_te']))
    log(f'Base union feature count before fold TE/drop: {len(base_union_features)}')
    log(f"A_TE_COLUMNS={len(keys['a_te'])}, B_TE_COLUMNS={len(keys['b_te'])}, B_CE_COLUMNS={len(keys['b_ce'])}, "
        f"LGBM_TE_COLUMNS={len(keys['lgbm_te'])}")

    y = train[TARGET].astype('int8').copy()
    X_all = deduplicate_columns(train[base_union_features].copy())
    X_test_all = test[base_union_features].copy()[X_all.columns].copy()
    log(f'Final pre-fold feature matrix: train={X_all.shape}, test={X_test_all.shape}')
    return X_all, X_test_all, y, keys


def run_fold(fold, tr_idx, va_idx, X_all, X_test_all, y, keys):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    X_tr = X_all.iloc[tr_idx].reset_index(drop=True).copy()
    y_tr = y.iloc[tr_idx].reset_index(drop=True)
    X_va = X_all.iloc[va_idx].reset_index(drop=True).copy()
    y_va = y.iloc[va_idx].reset_index(drop=True)
    X_te = X_test_all.copy().reset_index(drop=True)

    log(f'Fold {fold}: adding nested TE/stat encodings')
    target_encode_aggs_fold(X_tr, y_tr, X_va, X_te, keys['a_te'], ['std', 'min', 'max'], prefix='A_TE1', inner_folds=INNER_FOLDS)
    target_encode_mean_fold(X_tr, y_tr, X_va, X_te, keys['a_te'], prefix='A', inner_folds=INNER_FOLDS, smooth=20.0)
    count_encode_fold(X_tr, X_va, X_te, keys['b_ce'])
    target_encode_mean_fold(X_tr, y_tr, X_va, X_te, keys['b_te'], prefix='B', inner_folds=INNER_FOLDS, smooth=20.0)
    target_encode_aggs_fold(X_tr, y_tr, X_va, X_te, keys['b_te'], ['mean', 'std', 'min', 'max'], prefix='B_TEagg', inner_folds=INNER_FOLDS)
    target_encode_mean_fold(X_tr, y_tr, X_va, X_te, keys['lgbm_te'], prefix='L', inner_folds=INNER_FOLDS, smooth=20.0)
    target_encode_aggs_fold(X_tr, y_tr, X_va, X_te, keys['lgbm_te'], ['mean', 'std', 'min', 'max'], prefix='L_TEagg', inner_folds=INNER_FOLDS)

    drop_cols = [c for c in keys['drop'] if c in X_tr.columns]
    if drop_cols:
        X_tr = X_tr.drop(columns=drop_cols)
        X_va = X_va.drop(columns=drop_cols)
        X_te = X_te.drop(columns=drop_cols)

    factorize_remaining_objects(X_tr, X_va, X_te)
    X_tr = prepare_for_xgb(X_tr)
    X_va = prepare_for_xgb(X_va)
    X_te = prepare_for_xgb(X_te)

    keep_unbinned = {c for c in BIN_EXCLUDE_COLS if c in X_tr.columns}
    X_tr, X_va, X_te = bin_matrices_like_xgb(X_tr, X_va, X_te, BIN_SMALL, keep_unbinned, log_prefix=f'Fold {fold}: ')

    log(f'Fold {fold}: train shape={X_tr.shape}, valid shape={X_va.shape}')
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=200)
    va_pred = model.predict_proba(X_va)[:, 1].astype('float32')
    te_pred = model.predict_proba(X_te)[:, 1].astype('float32')
    log(f'{PAIR} fold {fold}: AUC {roc_auc_score(y_va, va_pred):.6f}, best_iteration {getattr(model, "best_iteration", None)}')
    del X_tr, X_va, X_te, model
    gc.collect()
    return va_pred, te_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        X_all, X_test_all, y, keys = build_features()
        f = folds(y.values)
        for k in todo:
            va_pred, te_pred = run_fold(k, np.flatnonzero(f != k), np.flatnonzero(f == k), X_all, X_test_all, y, keys)
            save_part(HERE, PAIR, k, va_pred, te_pred)
    y = read_train()[TARGET].values.astype('int8')
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
