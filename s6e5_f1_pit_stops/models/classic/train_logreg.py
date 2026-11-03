#!/usr/bin/env python
"""cuML L2 logistic regression on a dense one-hot + dense numeric matrix (pair logreg_ohe_crosses_splines_group_dense).

Features (built once on train + test + original; the original rows only shape the features, they are not training rows):
- base: string columns and numeric columns with < 100 distinct values over train + test are categoricals; the 5
  continuous columns (TyreLife, LapTime_Delta, Cumulative_Degradation, LapTime (s), RaceProgress) are numerics
  (median-filled). TyreLife is both.
- numeric interactions (products, ratios to TyreLife + 1, squares, signed logs), spline hinges max(x - knot, 0) at
  TyreLife 1/3/6/12/24/48 and at the 20/40/60/80% quantiles of the original data for LapTime_Delta,
  Cumulative_Degradation and LapTime (s);
- bins built from the sorted unique original values (equal-count chunks of unique values: TyreLife 100, LapTime_Delta 512,
  Cumulative_Degradation 2048, LapTime (s) 512; and 16 / 16 / 32 / 16 for the low-cardinality versions), first decimal
  digits of LapTime_Delta, Cumulative_Degradation, LapTime (s);
- categorical crosses (10 cat x cat, 8 cat x low bin, 4 low bin x low bin), gated numerics (a numeric times a category
  indicator, e.g. TyreLife if SOFT) and gated hinges;
- group-relative features: value - group mean, value / |group mean| and z-score of 7 numerics within Race, Compound,
  PitStop, Race x Compound, Compound x Stint (group statistics over train + test + original).
Categoricals are one-hot encoded over train + test + original, levels with fewer than 30 rows (or beyond the 2,500 most
frequent) pooled into one rare level: 8,143 one-hot + 167 numeric columns.
Per outer fold: numerics standardized with the fold's training rows; cuML LogisticRegression (L2, C 0.3, max_iter 3000,
tol 1e-6, linesearch_max_iter 100) on the training rows; probabilities clipped to [1e-7, 1 - 1e-7].
Test = mean of the 5 fold models. Needs a large-memory machine (the dense one-hot matrices take about 20 GB).
Usage: python train_logreg.py [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import gc
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

try:
    cp.cuda.set_pinned_memory_allocator(None)
except Exception:
    pass

PAIR = 'logreg_ohe_crosses_splines_group_dense'
FOLDER = 'classic'

# column roles of the feature recipe (first written for tenure / monthly charges / total charges columns; the feature
# names below keep those labels)
TENURE_COL = 'TyreLife'
MONTHLY_COL = 'LapTime_Delta'
TOTAL_COL = 'Cumulative_Degradation'
LAPTIME_COL = 'LapTime (s)'
PROGRESS_COL = 'RaceProgress'

ALL_BASE_NUMERIC = [TENURE_COL, MONTHLY_COL, TOTAL_COL, LAPTIME_COL, PROGRESS_COL]
DECIMAL_DIGIT_COLS = [MONTHLY_COL, TOTAL_COL, LAPTIME_COL]

NUMERIC_BIN_CARDINALITY = {
    TENURE_COL: 100,
    MONTHLY_COL: 512,
    TOTAL_COL: 2048,
    LAPTIME_COL: 512,
}
LOW_CARD_BIN_CARDINALITY = {
    TENURE_COL: 16,
    MONTHLY_COL: 16,
    TOTAL_COL: 32,
    LAPTIME_COL: 16,
}
LOW_BIN_NAME_MAP = {
    TENURE_COL: 'tenure_bin16',
    MONTHLY_COL: 'MonthlyCharges_bin16',
    TOTAL_COL: 'TotalCharges_bin32',
    LAPTIME_COL: 'LapTime_bin16',
}

# No Driver crosses, only modest-cardinality state crosses (keeps the dense one-hot matrix manageable)
CAT_CAT_CROSSES = [
    ('Race', 'Compound'),
    ('Race', 'PitStop'),
    ('Race', 'Stint'),
    ('Compound', 'Stint'),
    ('Compound', 'PitStop'),
    ('Compound', 'Position'),
    ('Compound', 'Position_Change'),
    ('Year', 'Race'),
    ('LapNumber', 'PitStop'),
    ('Position', 'PitStop'),
]
CAT_LOWNUM_CROSSES = [
    ('Compound', 'tenure_bin16'),
    ('Race', 'MonthlyCharges_bin16'),
    ('Compound', 'MonthlyCharges_bin16'),
    ('Race', 'TotalCharges_bin32'),
    ('Compound', 'TotalCharges_bin32'),
    ('PitStop', 'tenure_bin16'),
    ('Position', 'tenure_bin16'),
    ('Stint', 'LapTime_bin16'),
]
LOW_NUMGRID_CROSSES = [
    ('tenure_bin16', 'MonthlyCharges_bin16'),
    ('tenure_bin16', 'TotalCharges_bin32'),
    ('MonthlyCharges_bin16', 'TotalCharges_bin32'),
    ('LapTime_bin16', 'MonthlyCharges_bin16'),
]

GATED_NUMERIC_SPECS = [
    ('Compound', 'SOFT', TENURE_COL, 'tyrelife_if_soft'),
    ('Compound', 'MEDIUM', TENURE_COL, 'tyrelife_if_medium'),
    ('Compound', 'HARD', TENURE_COL, 'tyrelife_if_hard'),
    ('Compound', 'SOFT', MONTHLY_COL, 'lapdelta_if_soft'),
    ('Compound', 'MEDIUM', MONTHLY_COL, 'lapdelta_if_medium'),
    ('Compound', 'HARD', MONTHLY_COL, 'lapdelta_if_hard'),
    ('Compound', 'INTERMEDIATE', LAPTIME_COL, 'laptime_if_intermediate'),
    ('Compound', 'WET', LAPTIME_COL, 'laptime_if_wet'),
    ('PitStop', '1', TOTAL_COL, 'degradation_if_pitstop1'),
    ('PitStop', '0', TOTAL_COL, 'degradation_if_pitstop0'),
    ('PitStop', '1', 'tc_div_tenure', 'tcdivt_if_pitstop1'),
    ('PitStop', '1', 'log_tc', 'logtc_if_pitstop1'),
    ('Stint', '1', TENURE_COL, 'tyrelife_if_stint1'),
    ('Stint', '2', TENURE_COL, 'tyrelife_if_stint2'),
    ('Position_Change', '-1.0', MONTHLY_COL, 'lapdelta_if_lost1'),
    ('Position_Change', '1.0', MONTHLY_COL, 'lapdelta_if_gained1'),
]

GATED_HINGE_SPECS = [
    ('Compound', 'SOFT', TENURE_COL, TENURE_COL, 1, 'hinge_tyre_k1_if_soft'),
    ('Compound', 'HARD', TENURE_COL, TENURE_COL, 2, 'hinge_tyre_k2_if_hard'),
    ('Compound', 'SOFT', MONTHLY_COL, MONTHLY_COL, 2, 'hinge_delta_k2_if_soft'),
    ('Compound', 'MEDIUM', MONTHLY_COL, MONTHLY_COL, 2, 'hinge_delta_k2_if_medium'),
    ('PitStop', '1', TOTAL_COL, TOTAL_COL, 2, 'hinge_degradation_k2_if_pitstop1'),
    ('PitStop', '0', TOTAL_COL, TOTAL_COL, 2, 'hinge_degradation_k2_if_pitstop0'),
    ('Compound', 'SOFT', LAPTIME_COL, LAPTIME_COL, 2, 'hinge_laptime_k2_if_soft'),
    ('Compound', 'HARD', LAPTIME_COL, LAPTIME_COL, 2, 'hinge_laptime_k2_if_hard'),
]

GROUP_DENSE_GROUPS = [
    ('Race',),
    ('Compound',),
    ('PitStop',),
    ('Race', 'Compound'),
    ('Compound', 'Stint'),
]
GROUP_DENSE_NUMERIC_COLS = [
    TENURE_COL,
    MONTHLY_COL,
    TOTAL_COL,
    LAPTIME_COL,
    'mc_div_tenure',
    'tc_div_tenure',
    'log_tc',
]

LOGREG_PARAMS = {
    'penalty': 'l2',
    'C': 0.30,
    'fit_intercept': True,
    'max_iter': 3000,
    'tol': 1e-6,
    'linesearch_max_iter': 100,
    'output_type': 'numpy',
}
GPU_COPY_ROWS = 4096


def dedupe(seq):
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def signed_log1p(x):
    x = np.asarray(x, dtype=np.float32)
    return (np.sign(x) * np.log1p(np.abs(x))).astype(np.float32)


def reduce_mem_float32(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    return df


def first_decimal_digit(s):
    arr = pd.to_numeric(s, errors='coerce').fillna(0).to_numpy(dtype=np.float32, copy=False)
    out = np.floor(np.abs(arr) * 10.0 + 1e-6).astype(np.int32) % 10
    return pd.Series(out, index=s.index, dtype='int32')


def make_apply_bins_fn(value_to_bin, edges, col_name):
    ends = np.array([b for a, b in edges], dtype=np.float32)

    def apply_bins(s):
        arr = pd.to_numeric(s, errors='coerce').to_numpy(dtype=np.float32, copy=False)
        out = np.full(len(arr), -1, dtype=np.int32)
        for i, v in enumerate(arr):
            if not np.isfinite(v):
                out[i] = -1
            else:
                key = float(v)
                if key in value_to_bin:
                    out[i] = value_to_bin[key]
                else:
                    pos = np.searchsorted(ends, v, side='left')
                    out[i] = int(np.clip(pos, 0, len(ends) - 1))
        return pd.Series(out, index=s.index, name=f'{col_name}_bin', dtype='int32')

    return apply_bins


def build_bins_from_orig(train_df, test_df, orig_df, col, desired_card, out_name):
    """Equal-count chunks of the sorted unique original values; other values go to the first chunk whose upper end is
    >= the value."""
    uniq = np.sort(pd.to_numeric(orig_df[col], errors='coerce').dropna().unique())
    n_unique = len(uniq)
    if n_unique == 0:
        raise ValueError(f'No valid original values for {col}')
    actual_card = min(desired_card, n_unique)
    chunks = [arr for arr in np.array_split(uniq, actual_card) if len(arr)]
    value_to_bin, edges = {}, []
    for bin_id, arr in enumerate(chunks):
        for v in arr:
            value_to_bin[float(v)] = int(bin_id)
        edges.append((float(arr[0]), float(arr[-1])))
    apply_bins = make_apply_bins_fn(value_to_bin, edges, out_name)
    train_df[out_name] = apply_bins(train_df[col]).astype(str)
    test_df[out_name] = apply_bins(test_df[col]).astype(str)
    orig_df[out_name] = apply_bins(orig_df[col]).astype(str)


def compute_spline_knots(orig_df):
    knots = {}
    t = pd.to_numeric(orig_df[TENURE_COL], errors='coerce').dropna().to_numpy(dtype=np.float32)
    t_knots = np.array([1, 3, 6, 12, 24, 48], dtype=np.float32)
    if len(t):
        t_knots = t_knots[(t_knots > np.nanmin(t)) & (t_knots < np.nanmax(t))]
    knots[TENURE_COL] = np.unique(t_knots)
    for col in [MONTHLY_COL, TOTAL_COL, LAPTIME_COL]:
        arr = pd.to_numeric(orig_df[col], errors='coerce').dropna().to_numpy(dtype=np.float32)
        if len(arr):
            knots[col] = np.unique(np.quantile(arr, [0.2, 0.4, 0.6, 0.8]).astype(np.float32))
        else:
            knots[col] = np.array([], dtype=np.float32)
    return knots


def add_numeric_interaction_features(df):
    t = pd.to_numeric(df[TENURE_COL], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
    mc = pd.to_numeric(df[MONTHLY_COL], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
    tc = pd.to_numeric(df[TOTAL_COL], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
    lap = pd.to_numeric(df[LAPTIME_COL], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
    progress = pd.to_numeric(df[PROGRESS_COL], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
    denom = t + np.float32(1.0)
    logt = np.log1p(np.clip(t, 0, None)).astype(np.float32)
    logmc = signed_log1p(mc)
    logtc = signed_log1p(tc)
    loglap = signed_log1p(lap)

    out = pd.DataFrame(index=df.index)
    out['mc_x_tenure'] = (mc * t).astype(np.float32)
    out['mc_div_tenure'] = (mc / denom).astype(np.float32)
    out['tc_div_tenure'] = (tc / denom).astype(np.float32)
    out['lap_div_tenure'] = (lap / denom).astype(np.float32)
    out['mc_sq'] = (mc * mc).astype(np.float32)
    out['tenure_sq'] = (t * t).astype(np.float32)
    out['log_tc'] = logtc
    out['log_tenure'] = logt
    out['log_mc'] = logmc
    out['log_lap'] = loglap
    out['tc_over_mc1'] = (tc / (np.abs(mc) + 1.0)).astype(np.float32)
    out['mc_over_tc1'] = (mc / (np.abs(tc) + 1.0)).astype(np.float32)
    out['lap_over_tc1'] = (lap / (np.abs(tc) + 1.0)).astype(np.float32)
    out['mc_x_logt'] = (mc * logt).astype(np.float32)
    out['logmc_x_logt'] = (logmc * logt).astype(np.float32)
    out['progress_x_tenure'] = (progress * t).astype(np.float32)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out


def add_spline_features(df, knots_dict):
    out = pd.DataFrame(index=df.index)
    for col, knots in knots_dict.items():
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
        for i, k in enumerate(knots):
            out[f'{col}_hinge_{i}'] = np.maximum(x - np.float32(k), 0.0).astype(np.float32)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def add_gated_numeric_features(df, specs):
    out = pd.DataFrame(index=df.index)
    for col, level, src_num, new_col in specs:
        if col not in df.columns or src_num not in df.columns:
            continue
        mask = (df[col].astype(str).to_numpy(copy=False) == str(level)).astype(np.float32)
        vals = pd.to_numeric(df[src_num], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
        out[new_col] = (mask * vals).astype(np.float32)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def add_gated_hinge_features(df, spline_knots, specs):
    out = pd.DataFrame(index=df.index)
    for cat_col, cat_level, num_col, knot_col, knot_idx, new_col in specs:
        if cat_col not in df.columns or num_col not in df.columns or knot_col not in spline_knots:
            continue
        knots = spline_knots[knot_col]
        if knot_idx >= len(knots):
            continue
        knot = np.float32(knots[knot_idx])
        mask = (df[cat_col].astype(str).to_numpy(copy=False) == str(cat_level)).astype(np.float32)
        x = pd.to_numeric(df[num_col], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
        out[new_col] = (mask * np.maximum(x - knot, 0.0).astype(np.float32)).astype(np.float32)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def add_crosses(df, pairs):
    new_cols = []
    for a, b in pairs:
        if a not in df.columns or b not in df.columns:
            continue
        new_col = f'{a}__{b}'
        df[new_col] = df[a].astype(str) + '|' + df[b].astype(str)
        new_cols.append(new_col)
    return new_cols


def add_group_relative_features(train_df, test_df, orig_df, group_cols, numeric_cols):
    group_cols = list(group_cols)
    numeric_cols = [c for c in numeric_cols if c in train_df.columns and c in test_df.columns and c in orig_df.columns]
    if not numeric_cols or any(c not in train_df.columns for c in group_cols):
        return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index), pd.DataFrame(index=orig_df.index)
    use_cols = dedupe(group_cols + numeric_cols)
    combined = pd.concat([train_df[use_cols], test_df[use_cols], orig_df[use_cols]], axis=0, ignore_index=True).copy()
    for c in group_cols:
        combined[c] = combined[c].astype(str).fillna('__MISSING__')
    for c in numeric_cols:
        combined[c] = pd.to_numeric(combined[c], errors='coerce').astype('float32')
    grp = combined.groupby(group_cols, dropna=False)[numeric_cols].agg(['mean', 'std'])
    grp.columns = [f'{c}_{stat}' for c, stat in grp.columns]
    grp = grp.reset_index()

    def apply_one(df, name):
        tmp_in = df[use_cols].copy()
        for c in group_cols:
            tmp_in[c] = tmp_in[c].astype(str).fillna('__MISSING__')
        for c in numeric_cols:
            tmp_in[c] = pd.to_numeric(tmp_in[c], errors='coerce').astype('float32')
        tmp = tmp_in.merge(grp, on=group_cols, how='left')
        out = pd.DataFrame(index=df.index)
        eps = np.float32(1e-3)
        for c in numeric_cols:
            val = pd.to_numeric(tmp[c], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
            mean = pd.to_numeric(tmp[f'{c}_mean'], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
            std = pd.to_numeric(tmp[f'{c}_std'], errors='coerce').fillna(1).to_numpy(dtype=np.float32)
            std = np.where(np.abs(std) < eps, 1.0, std).astype(np.float32)
            out[f'{name}__{c}_minus_mean'] = (val - mean).astype(np.float32)
            out[f'{name}__{c}_over_mean'] = (val / (np.abs(mean) + eps)).astype(np.float32)
            out[f'{name}__{c}_z'] = ((val - mean) / std).astype(np.float32)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    name = '__'.join(group_cols)
    train_out = apply_one(train_df, name)
    test_out = apply_one(test_df, name)
    orig_out = apply_one(orig_df, name)
    del combined, grp
    gc.collect()
    return train_out, test_out, orig_out


def preprocess(train_df, test_df, orig_df):
    train_df[TARGET] = train_df[TARGET].astype(np.int8)
    orig_df[TARGET] = orig_df[TARGET].astype(np.int8)
    train_df = train_df.drop(columns=[ID])
    test_df = test_df.drop(columns=[ID])
    keep_orig = [c for c in train_df.columns if c in orig_df.columns]
    orig_df = orig_df[keep_orig].copy()

    all_numeric = dedupe(ALL_BASE_NUMERIC)
    for df in [train_df, test_df, orig_df]:
        reduce_mem_float32(df, all_numeric + ['LapNumber', 'Position', 'Position_Change', 'PitStop', 'Stint', 'Year'])
        for c in all_numeric:
            if c in df.columns:
                med = pd.to_numeric(df[c], errors='coerce').median()
                if not np.isfinite(med):
                    med = 0.0
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(med).astype('float32')

    # base categoricals: strings and numerics with < 100 values; TyreLife is kept both as a numeric and as a category
    both = pd.concat([train_df.drop(columns=[TARGET]), test_df], axis=0, ignore_index=True)
    base_cat_cols, base_num_cols = [], []
    for c in test_df.columns:
        if pd.api.types.is_object_dtype(both[c]) or pd.api.types.is_string_dtype(both[c]) or both[c].nunique(dropna=False) < 100:
            base_cat_cols.append(c)
        else:
            base_num_cols.append(c)
    base_cat_cols = [c for c in base_cat_cols if c in train_df.columns and c in test_df.columns and c in orig_df.columns]
    base_num_cols = [c for c in dedupe(all_numeric + base_num_cols)
                     if c in train_df.columns and c in test_df.columns and c in orig_df.columns]

    for df in [train_df, test_df, orig_df]:
        for c in base_cat_cols:
            df[c] = df[c].astype(str).fillna('__MISSING__')

    spline_knots = compute_spline_knots(orig_df)
    print('spline knots:', {k: [float(x) for x in v] for k, v in spline_knots.items()}, flush=True)

    train_num_int = add_numeric_interaction_features(train_df)
    test_num_int = add_numeric_interaction_features(test_df)
    orig_num_int = add_numeric_interaction_features(orig_df)
    train_spline = add_spline_features(train_df, spline_knots)
    test_spline = add_spline_features(test_df, spline_knots)
    orig_spline = add_spline_features(orig_df, spline_knots)
    train_df = pd.concat([train_df, train_num_int, train_spline], axis=1)
    test_df = pd.concat([test_df, test_num_int, test_spline], axis=1)
    orig_df = pd.concat([orig_df, orig_num_int, orig_spline], axis=1)
    extra_numeric_cols = train_num_int.columns.tolist() + train_spline.columns.tolist()

    for col, card in NUMERIC_BIN_CARDINALITY.items():
        if col in train_df.columns:
            build_bins_from_orig(train_df, test_df, orig_df, col, card, f'{col}_bin')
    for col, card in LOW_CARD_BIN_CARDINALITY.items():
        if col in train_df.columns:
            build_bins_from_orig(train_df, test_df, orig_df, col, card, LOW_BIN_NAME_MAP[col])
    digit_cols = []
    for col in DECIMAL_DIGIT_COLS:
        if col in train_df.columns:
            new_col = f'{col}_d1'
            train_df[new_col] = first_decimal_digit(train_df[col]).astype(str)
            test_df[new_col] = first_decimal_digit(test_df[col]).astype(str)
            orig_df[new_col] = first_decimal_digit(orig_df[col]).astype(str)
            digit_cols.append(new_col)

    cat_cat_cols = add_crosses(train_df, CAT_CAT_CROSSES)
    add_crosses(test_df, CAT_CAT_CROSSES)
    add_crosses(orig_df, CAT_CAT_CROSSES)
    cat_lownum_cols = add_crosses(train_df, CAT_LOWNUM_CROSSES)
    add_crosses(test_df, CAT_LOWNUM_CROSSES)
    add_crosses(orig_df, CAT_LOWNUM_CROSSES)
    low_numgrid_cols = add_crosses(train_df, LOW_NUMGRID_CROSSES)
    add_crosses(test_df, LOW_NUMGRID_CROSSES)
    add_crosses(orig_df, LOW_NUMGRID_CROSSES)

    train_gated = add_gated_numeric_features(train_df, GATED_NUMERIC_SPECS)
    test_gated = add_gated_numeric_features(test_df, GATED_NUMERIC_SPECS)
    orig_gated = add_gated_numeric_features(orig_df, GATED_NUMERIC_SPECS)
    train_gh = add_gated_hinge_features(train_df, spline_knots, GATED_HINGE_SPECS)
    test_gh = add_gated_hinge_features(test_df, spline_knots, GATED_HINGE_SPECS)
    orig_gh = add_gated_hinge_features(orig_df, spline_knots, GATED_HINGE_SPECS)
    train_df = pd.concat([train_df, train_gated, train_gh], axis=1)
    test_df = pd.concat([test_df, test_gated, test_gh], axis=1)
    orig_df = pd.concat([orig_df, orig_gated, orig_gh], axis=1)

    group_train, group_test, group_orig = [], [], []
    for group_cols in GROUP_DENSE_GROUPS:
        tr_add, te_add, or_add = add_group_relative_features(train_df, test_df, orig_df, group_cols,
                                                             GROUP_DENSE_NUMERIC_COLS)
        if tr_add.shape[1]:
            group_train.append(tr_add)
            group_test.append(te_add)
            group_orig.append(or_add)
    if group_train:
        train_groupdense = pd.concat(group_train, axis=1)
        test_groupdense = pd.concat(group_test, axis=1)
        orig_groupdense = pd.concat(group_orig, axis=1)
        train_df = pd.concat([train_df, train_groupdense], axis=1)
        test_df = pd.concat([test_df, test_groupdense], axis=1)
        orig_df = pd.concat([orig_df, orig_groupdense], axis=1)
        group_dense_cols = train_groupdense.columns.tolist()
    else:
        group_dense_cols = []

    final_numeric_cols = dedupe(base_num_cols + extra_numeric_cols + train_gated.columns.tolist()
                                + train_gh.columns.tolist() + group_dense_cols)
    added_cat_cols = dedupe([f'{c}_bin' for c in NUMERIC_BIN_CARDINALITY if f'{c}_bin' in train_df.columns]
                            + list(LOW_BIN_NAME_MAP.values()) + digit_cols + cat_cat_cols + cat_lownum_cols
                            + low_numgrid_cols)
    final_cat_cols = dedupe(base_cat_cols + added_cat_cols)
    final_cat_cols = [c for c in final_cat_cols if c in train_df.columns]
    final_numeric_cols = [c for c in final_numeric_cols if c in train_df.columns]
    print(f'base categoricals ({len(base_cat_cols)}): {base_cat_cols}', flush=True)
    print(f'{len(final_cat_cols)} categorical, {len(final_numeric_cols)} numeric columns', flush=True)
    return train_df, test_df, orig_df, final_cat_cols, final_numeric_cols


def cap_categories_for_ohe(combined, cat_cols, min_count=30, max_levels_per_col=2500):
    capped = pd.DataFrame(index=combined.index)
    kept = {}
    for c in cat_cols:
        s = combined[c].astype(str).fillna('__MISSING__')
        vc = s.value_counts(dropna=False)
        keep = vc[vc >= min_count].index.tolist()[:max_levels_per_col]
        if not keep:
            keep = vc.index.tolist()[:min(50, len(vc))]
        keep_set = set(keep)
        capped[c] = np.where(s.isin(keep_set), s, '__RARE__')
        kept[c] = len(keep_set)
    return capped, kept


def build_global_ohe(train_df, test_df, orig_df, cat_cols):
    combined = pd.concat([train_df[cat_cols], test_df[cat_cols], orig_df[cat_cols]], axis=0, ignore_index=True)
    combined, kept = cap_categories_for_ohe(combined, cat_cols)
    ohe_all = pd.get_dummies(combined, columns=cat_cols, prefix=cat_cols, prefix_sep='=', dtype=np.uint8)
    n_train, n_test = len(train_df), len(test_df)
    X_cat_train = np.ascontiguousarray(ohe_all.iloc[:n_train].to_numpy(dtype=np.float32, copy=False))
    X_cat_test = np.ascontiguousarray(ohe_all.iloc[n_train:n_train + n_test].to_numpy(dtype=np.float32, copy=False))
    del combined, ohe_all
    gc.collect()
    print(f'one-hot train matrix {X_cat_train.shape}', flush=True)
    return X_cat_train, X_cat_test


def standardize_fold(train_num, valid_num, test_num):
    mean = np.nanmean(train_num, axis=0, dtype=np.float64).astype(np.float32)
    std = np.nanstd(train_num, axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    train_scaled = np.nan_to_num((train_num - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    valid_scaled = np.nan_to_num((valid_num - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    test_scaled = np.nan_to_num((test_num - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.ascontiguousarray(train_scaled), np.ascontiguousarray(valid_scaled), np.ascontiguousarray(test_scaled)


def to_cupy_chunked(arr, dtype=cp.float32, rows_per_chunk=GPU_COPY_ROWS):
    arr = np.ascontiguousarray(arr)
    out = cp.empty(arr.shape, dtype=dtype)
    for start in range(0, arr.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, arr.shape[0])
        out[start:end] = cp.asarray(arr[start:end], dtype=dtype, blocking=True)
    cp.cuda.Stream.null.synchronize()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    out = model_dir(FOLDER)
    todo = [k for k in fold_list(args.folds) if not have_part(out, PAIR, k)]

    train_df, test_df, orig_df, cat_features, numeric_cols = preprocess(read_train(), read_test(), read_original())
    y = train_df[TARGET].astype(np.int32).values
    fold_id = folds(y)
    if todo:
        X_cat_train, X_cat_test = build_global_ohe(train_df, test_df, orig_df, cat_features)
        X_num_train = np.ascontiguousarray(train_df[numeric_cols].to_numpy(dtype=np.float32, copy=True))
        X_num_test = np.ascontiguousarray(test_df[numeric_cols].to_numpy(dtype=np.float32, copy=True))
        del train_df, test_df, orig_df
        gc.collect()

    for k in todo:
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        X_num_tr_sc, X_num_va_sc, X_num_te_sc = standardize_fold(X_num_train[tr_idx], X_num_train[va_idx], X_num_test)
        X_tr = np.ascontiguousarray(np.concatenate([X_num_tr_sc, X_cat_train[tr_idx]], axis=1).astype(np.float32, copy=False))
        X_va = np.ascontiguousarray(np.concatenate([X_num_va_sc, X_cat_train[va_idx]], axis=1).astype(np.float32, copy=False))
        X_te = np.ascontiguousarray(np.concatenate([X_num_te_sc, X_cat_test], axis=1).astype(np.float32, copy=False))
        y_tr = np.ascontiguousarray(y[tr_idx].astype(np.int32, copy=False))
        y_va = y[va_idx].astype(np.int32, copy=False)
        print(f'{PAIR} fold {k}: X_tr {X_tr.shape}, X_va {X_va.shape}', flush=True)

        X_tr_gpu = to_cupy_chunked(X_tr, dtype=cp.float32)
        X_va_gpu = to_cupy_chunked(X_va, dtype=cp.float32)
        X_te_gpu = to_cupy_chunked(X_te, dtype=cp.float32)
        y_tr_gpu = to_cupy_chunked(y_tr, dtype=cp.int32)
        model = cuLogisticRegression(**LOGREG_PARAMS)
        model.fit(X_tr_gpu, y_tr_gpu)
        p_va = model.predict_proba(X_va_gpu)[:, 1]
        p_te = model.predict_proba(X_te_gpu)[:, 1]
        if hasattr(p_va, 'get'):
            p_va = p_va.get(blocking=True)
        if hasattr(p_te, 'get'):
            p_te = p_te.get(blocking=True)
        p_va = np.clip(np.asarray(p_va, dtype=np.float32), 1e-7, 1 - 1e-7)
        p_te = np.clip(np.asarray(p_te, dtype=np.float32), 1e-7, 1 - 1e-7)
        save_part(out, PAIR, k, p_va, p_te)
        print(f'{PAIR} fold {k}: AUC {roc_auc_score(y_va, p_va):.6f}', flush=True)

        del X_num_tr_sc, X_num_va_sc, X_num_te_sc, X_tr, X_va, X_te, X_tr_gpu, X_va_gpu, X_te_gpu, y_tr_gpu, model
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    assemble(out, PAIR, fold_id, y)


if __name__ == '__main__':
    main()
