#!/usr/bin/env python
"""cuML logistic regression on one-hot categoricals / bins / crosses + dense numeric features
(pair logreg_ohe_crosses_hinges_group_dense).

Built once on train + test + original (the original rows shape bins, knots, group statistics and one-hot levels but are
never training rows; its 11 rows with a blank TotalCharges are dropped):
  - one-hot (4,362 columns; levels over train + test + original): the 16 categoricals; tenure / MonthlyCharges /
    TotalCharges binned into (at most) 100 / 512 / 2,048 and 16 / 16 / 32 bins of consecutive sorted unique original
    values (a value absent from the original data goes to the bin whose [first, last] original value range contains
    it, else to the first / last bin); first decimal digit of MonthlyCharges and TotalCharges; 10 categorical crosses
    (Contract x InternetService, ...), 8 categorical x coarse-bin crosses, 3 coarse-bin x coarse-bin crosses;
  - 148 dense columns, standardized per fold on the outer-training rows: the 3 numerics; 12 interactions (MC x tenure,
    MC / (tenure + 1), TC / (tenure + 1), MC^2, tenure^2, log1p of the 3 numerics, TC / (MC + 1), MC / (TC + 1),
    MC x log1p(tenure), log1p(MC) x log1p(tenure)); 13 spline hinges max(x - knot, 0) (tenure knots 3, 6, 12, 24, 48;
    MonthlyCharges / TotalCharges knots at the original 20 / 40 / 60 / 80% quantiles); 30 category-gated numerics; 18
    category-gated hinges; for 6 numerics (tenure, MC, TC, MC / (tenure + 1), TC / (tenure + 1), log1p(TC)) within
    Contract, InternetService, PaymentMethod and Contract x InternetService (mean / std over train + test + original):
    value - mean, value / (mean + 1e-3) and z-score (72 columns).
Model: cuML LogisticRegression (L2, C 0.3, max_iter 3000, tol 1e-6, linesearch_max_iter 100) on the 4,510 columns.
Test = mean of the 5 fold models.
Usage: python train_logreg.py"""
import gc
import sys
import time
import warnings
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

try:
    cp.cuda.set_pinned_memory_allocator(None)
except Exception:
    pass
try:
    cp.get_default_pinned_memory_pool().free_all_blocks()
except Exception:
    pass

PAIR = 'logreg_ohe_crosses_hinges_group_dense'
FOLDER = 'classic'
N_SPLITS = 5

NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

NUMERIC_BIN_CARDINALITY = {
    'tenure': 100,
    'MonthlyCharges': 256 * 2,
    'TotalCharges': 1024 * 2,
}

LOW_CARD_BIN_CARDINALITY = {
    'tenure': 16,
    'MonthlyCharges': 16,
    'TotalCharges': 32,
}

DECIMAL_DIGIT_COLS = ['MonthlyCharges', 'TotalCharges']

LOGREG_PARAMS = {
    'penalty': 'l2',
    'C': 0.30,
    'fit_intercept': True,
    'max_iter': 3000,
    'tol': 1e-6,
    'linesearch_max_iter': 100,
    'output_type': 'numpy',
}

CAT_CAT_CROSSES = [
    ('Contract', 'InternetService'),
    ('Contract', 'PaymentMethod'),
    ('InternetService', 'OnlineSecurity'),
    ('InternetService', 'TechSupport'),
    ('InternetService', 'StreamingTV'),
    ('InternetService', 'StreamingMovies'),
    ('PaperlessBilling', 'PaymentMethod'),
    ('PhoneService', 'MultipleLines'),
    ('Partner', 'Dependents'),
    ('SeniorCitizen', 'Contract'),
]

CAT_LOWNUM_CROSSES = [
    ('Contract', 'tenure_bin16'),
    ('Contract', 'MonthlyCharges_bin16'),
    ('InternetService', 'MonthlyCharges_bin16'),
    ('InternetService', 'TotalCharges_bin32'),
    ('PaymentMethod', 'MonthlyCharges_bin16'),
    ('SeniorCitizen', 'tenure_bin16'),
    ('PaperlessBilling', 'MonthlyCharges_bin16'),
    ('TechSupport', 'MonthlyCharges_bin16'),
]

LOW_NUMGRID_CROSSES = [
    ('tenure_bin16', 'MonthlyCharges_bin16'),
    ('tenure_bin16', 'TotalCharges_bin32'),
    ('MonthlyCharges_bin16', 'TotalCharges_bin32'),
]

# (categorical column, level, numeric column, new column): numeric value if the row has that level, else 0
GATED_NUMERIC_SPECS = [
    ('Contract', 'Month-to-month', 'tenure', 'tenure_if_month2month'),
    ('Contract', 'One year', 'tenure', 'tenure_if_oneyear'),
    ('Contract', 'Two year', 'tenure', 'tenure_if_twoyear'),

    ('InternetService', 'Fiber optic', 'MonthlyCharges', 'mc_if_fiber'),
    ('InternetService', 'DSL', 'MonthlyCharges', 'mc_if_dsl'),
    ('InternetService', 'No', 'MonthlyCharges', 'mc_if_nointernet'),

    ('PaymentMethod', 'Electronic check', 'MonthlyCharges', 'mc_if_echeck'),
    ('PaymentMethod', 'Electronic check', 'tc_div_tenure', 'tcdivt_if_echeck'),

    ('PaperlessBilling', 'Yes', 'MonthlyCharges', 'mc_if_paperless_yes'),
    ('PaperlessBilling', 'Yes', 'mc_div_tenure', 'mcdivt_if_paperless_yes'),

    ('SeniorCitizen', '1', 'tenure', 'tenure_if_senior'),
    ('SeniorCitizen', '1', 'MonthlyCharges', 'mc_if_senior'),

    ('OnlineSecurity', 'No', 'MonthlyCharges', 'mc_if_no_sec'),
    ('TechSupport', 'No', 'MonthlyCharges', 'mc_if_no_tech'),

    ('StreamingTV', 'Yes', 'MonthlyCharges', 'mc_if_streamtv_yes'),
    ('StreamingMovies', 'Yes', 'MonthlyCharges', 'mc_if_streammovies_yes'),

    ('Partner', 'Yes', 'tenure', 'tenure_if_partner_yes'),
    ('Dependents', 'Yes', 'tenure', 'tenure_if_dependents_yes'),

    ('Contract', 'Month-to-month', 'mc_div_tenure', 'mcdivt_if_month2month'),
    ('Contract', 'Month-to-month', 'log_tc', 'logtc_if_month2month'),
    ('Contract', 'Two year', 'log_tc', 'logtc_if_twoyear'),

    ('InternetService', 'Fiber optic', 'mc_div_tenure', 'mcdivt_if_fiber'),
    ('InternetService', 'Fiber optic', 'log_tc', 'logtc_if_fiber'),
    ('InternetService', 'DSL', 'mc_div_tenure', 'mcdivt_if_dsl'),

    ('PaymentMethod', 'Electronic check', 'log_tc', 'logtc_if_echeck'),
    ('PaymentMethod', 'Electronic check', 'mc_sq', 'mcsq_if_echeck'),

    ('PaperlessBilling', 'Yes', 'log_tc', 'logtc_if_paperless'),
    ('SeniorCitizen', '1', 'log_tc', 'logtc_if_senior'),

    ('OnlineSecurity', 'No', 'mc_div_tenure', 'mcdivt_if_no_sec'),
    ('TechSupport', 'No', 'mc_div_tenure', 'mcdivt_if_no_tech'),
]

# (categorical column, level, numeric column, knot column, knot index, new column): gated max(x - knot, 0)
GATED_HINGE_SPECS = [
    ('Contract', 'Month-to-month', 'tenure', 'tenure', 1, 'hinge_tenure_k1_if_m2m'),
    ('Contract', 'Month-to-month', 'tenure', 'tenure', 3, 'hinge_tenure_k3_if_m2m'),
    ('Contract', 'Two year', 'tenure', 'tenure', 3, 'hinge_tenure_k3_if_2yr'),

    ('InternetService', 'Fiber optic', 'MonthlyCharges', 'MonthlyCharges', 2, 'hinge_mc_k2_if_fiber'),
    ('InternetService', 'Fiber optic', 'MonthlyCharges', 'MonthlyCharges', 3, 'hinge_mc_k3_if_fiber'),
    ('InternetService', 'DSL', 'MonthlyCharges', 'MonthlyCharges', 2, 'hinge_mc_k2_if_dsl'),

    ('PaymentMethod', 'Electronic check', 'MonthlyCharges', 'MonthlyCharges', 2, 'hinge_mc_k2_if_echeck'),
    ('PaymentMethod', 'Electronic check', 'TotalCharges', 'TotalCharges', 2, 'hinge_tc_k2_if_echeck'),

    ('PaperlessBilling', 'Yes', 'MonthlyCharges', 'MonthlyCharges', 2, 'hinge_mc_k2_if_paperless'),
    ('SeniorCitizen', '1', 'MonthlyCharges', 'MonthlyCharges', 2, 'hinge_mc_k2_if_senior'),

    ('Contract', 'Month-to-month', 'tenure', 'tenure', 2, 'hinge_tenure_k2_if_m2m'),
    ('Contract', 'One year', 'tenure', 'tenure', 2, 'hinge_tenure_k2_if_1yr'),

    ('InternetService', 'Fiber optic', 'MonthlyCharges', 'MonthlyCharges', 1, 'hinge_mc_k1_if_fiber'),
    ('InternetService', 'Fiber optic', 'TotalCharges', 'TotalCharges', 2, 'hinge_tc_k2_if_fiber'),
    ('InternetService', 'DSL', 'MonthlyCharges', 'MonthlyCharges', 1, 'hinge_mc_k1_if_dsl'),

    ('PaymentMethod', 'Electronic check', 'MonthlyCharges', 'MonthlyCharges', 1, 'hinge_mc_k1_if_echeck'),
    ('PaperlessBilling', 'Yes', 'MonthlyCharges', 'MonthlyCharges', 1, 'hinge_mc_k1_if_paperless'),
    ('SeniorCitizen', '1', 'MonthlyCharges', 'MonthlyCharges', 1, 'hinge_mc_k1_if_senior'),
]

GROUP_DENSE_GROUPS = [
    'Contract',
    'InternetService',
    'PaymentMethod',
    'Contract__InternetService',
]

GROUP_DENSE_NUMERIC_COLS = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'mc_div_tenure',
    'tc_div_tenure',
    'log_tc',
]


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def reduce_mem_float32(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    return df


def first_decimal_digit(s):
    arr = s.to_numpy(dtype=np.float32, copy=False)
    out = np.full(len(arr), -1, dtype=np.int32)
    for i, v in enumerate(arr):
        if pd.isna(v):
            out[i] = -1
        else:
            out[i] = int(np.floor(np.abs(v) * 10.0 + 1e-6)) % 10
    return pd.Series(out, index=s.index, dtype='int32')


def make_apply_bins_fn(value_to_bin, edges, col_name):
    interval_starts = np.array([a for a, b in edges], dtype=np.float32)
    interval_ends = np.array([b for a, b in edges], dtype=np.float32)

    def apply_bins(s):
        arr = s.to_numpy(dtype=np.float32, copy=False)
        out = np.full(len(arr), -1, dtype=np.int32)
        for i, v in enumerate(arr):
            if pd.isna(v):
                out[i] = -1
            else:
                key = float(v)
                if key in value_to_bin:
                    out[i] = value_to_bin[key]
                else:
                    placed = False
                    for b in range(len(interval_starts)):
                        if interval_starts[b] <= v <= interval_ends[b]:
                            out[i] = b
                            placed = True
                            break
                    if not placed:
                        if v < interval_starts[0]:
                            out[i] = 0
                        else:
                            out[i] = len(interval_starts) - 1
        return pd.Series(out, index=s.index, name=f'{col_name}_bin', dtype='int32')

    return apply_bins


def build_bins_from_orig(train_df, test_df, orig_df, col, desired_card, out_name):
    """Splits the sorted unique original values of col into min(desired_card, n_unique) consecutive chunks."""
    uniq = np.sort(orig_df[col].dropna().unique())
    n_unique = len(uniq)
    if n_unique == 0:
        raise ValueError(f"No valid unique values found in original for column '{col}'")

    actual_card = min(desired_card, n_unique)
    chunks = np.array_split(uniq, actual_card)

    value_to_bin = {}
    edges = []
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

    t = orig_df['tenure'].dropna().to_numpy(dtype=np.float32)
    t_knots = np.array([1, 3, 6, 12, 24, 48], dtype=np.float32)
    t_knots = t_knots[(t_knots > np.nanmin(t)) & (t_knots < np.nanmax(t))]
    knots['tenure'] = np.unique(t_knots)

    for col in ['MonthlyCharges', 'TotalCharges']:
        arr = orig_df[col].dropna().to_numpy(dtype=np.float32)
        qs = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
        knots[col] = np.unique(np.quantile(arr, qs).astype(np.float32))

    return knots


def add_numeric_interaction_features(df):
    t = df['tenure'].to_numpy(dtype=np.float32, copy=False)
    mc = df['MonthlyCharges'].to_numpy(dtype=np.float32, copy=False)
    tc = df['TotalCharges'].to_numpy(dtype=np.float32, copy=False)

    denom = t + np.float32(1.0)
    logt = np.log1p(np.clip(t, a_min=0.0, a_max=None)).astype(np.float32)
    logmc = np.log1p(np.clip(mc, a_min=0.0, a_max=None)).astype(np.float32)
    logtc = np.log1p(np.clip(tc, a_min=0.0, a_max=None)).astype(np.float32)

    out = pd.DataFrame(index=df.index)

    out['mc_x_tenure'] = (mc * t).astype(np.float32)
    out['mc_div_tenure'] = (mc / denom).astype(np.float32)
    out['tc_div_tenure'] = (tc / denom).astype(np.float32)
    out['mc_sq'] = (mc * mc).astype(np.float32)
    out['tenure_sq'] = (t * t).astype(np.float32)
    out['log_tc'] = logtc

    out['log_tenure'] = logt
    out['log_mc'] = logmc
    out['tc_over_mc1'] = (tc / (mc + np.float32(1.0))).astype(np.float32)
    out['mc_over_tc1'] = (mc / (tc + np.float32(1.0))).astype(np.float32)
    out['mc_x_logt'] = (mc * logt).astype(np.float32)
    out['logmc_x_logt'] = (logmc * logt).astype(np.float32)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_spline_features(df, knots_dict):
    out = pd.DataFrame(index=df.index)
    for col, knots in knots_dict.items():
        x = df[col].to_numpy(dtype=np.float32, copy=False)
        for i, k in enumerate(knots):
            out[f'{col}_hinge_{i}'] = np.maximum(x - np.float32(k), 0.0).astype(np.float32)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_gated_numeric_features(df, specs):
    out = pd.DataFrame(index=df.index)
    for col, level, src_num, new_col in specs:
        mask = (df[col].astype(str).to_numpy(copy=False) == str(level)).astype(np.float32)
        vals = df[src_num].to_numpy(dtype=np.float32, copy=False)
        out[new_col] = (mask * vals).astype(np.float32)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_gated_hinge_features(df, spline_knots, specs):
    out = pd.DataFrame(index=df.index)
    for cat_col, cat_level, num_col, knot_col, knot_idx, new_col in specs:
        knots = spline_knots[knot_col]
        if knot_idx >= len(knots):
            continue
        knot = np.float32(knots[knot_idx])
        mask = (df[cat_col].astype(str).to_numpy(copy=False) == str(cat_level)).astype(np.float32)
        x = df[num_col].to_numpy(dtype=np.float32, copy=False)
        out[new_col] = (mask * np.maximum(x - knot, 0.0).astype(np.float32)).astype(np.float32)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_crosses(df, pairs):
    new_cols = []
    for a, b in pairs:
        new_col = f'{a}__{b}'
        df[new_col] = df[a].astype(str) + '|' + df[b].astype(str)
        new_cols.append(new_col)
    return new_cols


def add_group_relative_features(train_df, test_df, orig_df, group_cols, numeric_cols):
    """value - group mean, value / group mean and z-score within the group (stats over train + test + original)."""
    combined = pd.concat(
        [
            train_df[group_cols + numeric_cols],
            test_df[group_cols + numeric_cols],
            orig_df[group_cols + numeric_cols],
        ],
        axis=0,
        ignore_index=True,
    ).copy()

    grp = combined.groupby(group_cols, dropna=False)[numeric_cols].agg(['mean', 'std'])
    grp.columns = [f'{c}_{stat}' for c, stat in grp.columns]
    grp = grp.reset_index()

    def apply_one(df, name):
        tmp = df[group_cols + numeric_cols].merge(grp, on=group_cols, how='left')

        out = pd.DataFrame(index=df.index)
        eps = np.float32(1e-3)

        for c in numeric_cols:
            val = tmp[c].to_numpy(dtype=np.float32, copy=False)
            mean = tmp[f'{c}_mean'].to_numpy(dtype=np.float32, copy=False)
            std = tmp[f'{c}_std'].to_numpy(dtype=np.float32, copy=False)

            std = np.where(np.isfinite(std), std, 0.0).astype(np.float32)
            std = np.where(std < eps, 1.0, std).astype(np.float32)

            mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)

            out[f'{name}__{c}_minus_mean'] = (val - mean).astype(np.float32)
            out[f'{name}__{c}_over_mean'] = (val / (mean + eps)).astype(np.float32)
            out[f'{name}__{c}_z'] = ((val - mean) / std).astype(np.float32)

        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    train_out = apply_one(train_df, '__'.join(group_cols))
    test_out = apply_one(test_df, '__'.join(group_cols))
    orig_out = apply_one(orig_df, '__'.join(group_cols))

    del combined
    gc.collect()

    return train_out, test_out, orig_out


def preprocess_telco():
    log('Loading train/test/original...')
    train_df = read_train().drop(columns=[ID])
    test_df = read_test().drop(columns=[ID])
    orig_df = read_original()

    for df in [train_df, orig_df]:
        df[TARGET] = df[TARGET].map({'Yes': 1, 'No': 0}).astype('int8')

    orig_df = orig_df.loc[orig_df['TotalCharges'].notna()].copy()
    orig_df['TotalCharges'] = orig_df['TotalCharges'].astype('float32')

    train_df = reduce_mem_float32(train_df, NUMERIC_COLS)
    test_df = reduce_mem_float32(test_df, NUMERIC_COLS)
    orig_df = reduce_mem_float32(orig_df, NUMERIC_COLS)

    spline_knots = compute_spline_knots(orig_df)
    log(f"Spline knots = {{ {', '.join([f'{k}: {[float(x) for x in v]}' for k, v in spline_knots.items()])} }}")

    log('Adding numeric interaction and spline hinge features...')
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
    all_numeric_cols = NUMERIC_COLS + extra_numeric_cols

    feature_cols = [c for c in train_df.columns if c != TARGET]
    base_cat_cols = [c for c in feature_cols if c not in all_numeric_cols]

    for col in base_cat_cols:
        train_df[col] = train_df[col].astype(str).fillna('__MISSING__')
        test_df[col] = test_df[col].astype(str).fillna('__MISSING__')
        orig_df[col] = orig_df[col].astype(str).fillna('__MISSING__')

    for df in [train_df, test_df, orig_df]:
        df['Contract__InternetService'] = df['Contract'].astype(str) + '|' + df['InternetService'].astype(str)

    log('Adding numeric bins from the original values...')
    for col in NUMERIC_COLS:
        build_bins_from_orig(train_df, test_df, orig_df, col, NUMERIC_BIN_CARDINALITY[col], f'{col}_bin')

    low_bin_name_map = {
        'tenure': 'tenure_bin16',
        'MonthlyCharges': 'MonthlyCharges_bin16',
        'TotalCharges': 'TotalCharges_bin32',
    }
    for col in NUMERIC_COLS:
        build_bins_from_orig(train_df, test_df, orig_df, col, LOW_CARD_BIN_CARDINALITY[col], low_bin_name_map[col])

    for col in DECIMAL_DIGIT_COLS:
        new_col = f'{col}_d1'
        train_df[new_col] = first_decimal_digit(train_df[col]).astype(str)
        test_df[new_col] = first_decimal_digit(test_df[col]).astype(str)
        orig_df[new_col] = first_decimal_digit(orig_df[col]).astype(str)

    log('Adding crosses...')
    cat_cat_cols = add_crosses(train_df, CAT_CAT_CROSSES)
    add_crosses(test_df, CAT_CAT_CROSSES)
    add_crosses(orig_df, CAT_CAT_CROSSES)

    cat_lownum_cols = add_crosses(train_df, CAT_LOWNUM_CROSSES)
    add_crosses(test_df, CAT_LOWNUM_CROSSES)
    add_crosses(orig_df, CAT_LOWNUM_CROSSES)

    low_numgrid_cols = add_crosses(train_df, LOW_NUMGRID_CROSSES)
    add_crosses(test_df, LOW_NUMGRID_CROSSES)
    add_crosses(orig_df, LOW_NUMGRID_CROSSES)

    log('Adding gated numeric and gated hinge features...')
    train_gated = add_gated_numeric_features(train_df, GATED_NUMERIC_SPECS)
    test_gated = add_gated_numeric_features(test_df, GATED_NUMERIC_SPECS)
    orig_gated = add_gated_numeric_features(orig_df, GATED_NUMERIC_SPECS)

    train_gh = add_gated_hinge_features(train_df, spline_knots, GATED_HINGE_SPECS)
    test_gh = add_gated_hinge_features(test_df, spline_knots, GATED_HINGE_SPECS)
    orig_gh = add_gated_hinge_features(orig_df, spline_knots, GATED_HINGE_SPECS)

    train_df = pd.concat([train_df, train_gated, train_gh], axis=1)
    test_df = pd.concat([test_df, test_gated, test_gh], axis=1)
    orig_df = pd.concat([orig_df, orig_gated, orig_gh], axis=1)

    log('Adding group-relative dense features...')
    group_dense_frames_train = []
    group_dense_frames_test = []
    group_dense_frames_orig = []
    for g in GROUP_DENSE_GROUPS:
        gcols = g.split('__')
        tr_add, te_add, or_add = add_group_relative_features(
            train_df, test_df, orig_df,
            group_cols=gcols,
            numeric_cols=GROUP_DENSE_NUMERIC_COLS,
        )
        group_dense_frames_train.append(tr_add)
        group_dense_frames_test.append(te_add)
        group_dense_frames_orig.append(or_add)

    train_groupdense = pd.concat(group_dense_frames_train, axis=1)
    test_groupdense = pd.concat(group_dense_frames_test, axis=1)
    orig_groupdense = pd.concat(group_dense_frames_orig, axis=1)

    train_df = pd.concat([train_df, train_groupdense], axis=1)
    test_df = pd.concat([test_df, test_groupdense], axis=1)
    orig_df = pd.concat([orig_df, orig_groupdense], axis=1)

    gated_numeric_cols = train_gated.columns.tolist()
    gated_hinge_cols = train_gh.columns.tolist()
    group_dense_cols = train_groupdense.columns.tolist()

    final_numeric_cols = all_numeric_cols + gated_numeric_cols + gated_hinge_cols + group_dense_cols
    final_feature_cols = [c for c in train_df.columns if c != TARGET]
    final_cat_cols = [c for c in final_feature_cols if c not in final_numeric_cols]

    added_cross_cols = cat_cat_cols + cat_lownum_cols + low_numgrid_cols
    log(f'{len(added_cross_cols)} crosses, {len(gated_numeric_cols)} gated numerics, {len(gated_hinge_cols)} gated '
        f'hinges, {len(group_dense_cols)} group-relative columns; {len(final_cat_cols)} categorical and '
        f'{len(final_numeric_cols)} numeric columns')

    return train_df, test_df, orig_df, final_cat_cols, final_numeric_cols


def build_global_ohe(train_df, test_df, orig_df, cat_cols):
    combined = pd.concat(
        [train_df[cat_cols], test_df[cat_cols], orig_df[cat_cols]],
        axis=0,
        ignore_index=True,
    ).astype(str).fillna('__MISSING__')

    ohe_all = pd.get_dummies(
        combined,
        columns=cat_cols,
        prefix=cat_cols,
        prefix_sep='=',
        dtype=np.uint8,
    )

    n_train = len(train_df)
    n_test = len(test_df)

    X_cat_train = np.ascontiguousarray(ohe_all.iloc[:n_train].to_numpy(dtype=np.float32, copy=False))
    X_cat_test = np.ascontiguousarray(ohe_all.iloc[n_train:n_train + n_test].to_numpy(dtype=np.float32, copy=False))

    del combined, ohe_all
    gc.collect()

    log(f'One-hot columns = {X_cat_train.shape[1]}')
    return X_cat_train, X_cat_test


def standardize_fold(train_num, valid_num, test_num):
    mean = np.nanmean(train_num, axis=0, dtype=np.float64).astype(np.float32)
    std = np.nanstd(train_num, axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0

    train_scaled = np.nan_to_num((train_num - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    valid_scaled = np.nan_to_num((valid_num - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    test_scaled = np.nan_to_num((test_num - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return (
        np.ascontiguousarray(train_scaled),
        np.ascontiguousarray(valid_scaled),
        np.ascontiguousarray(test_scaled),
    )


def main():
    train_df, test_df, orig_df, cat_features, numeric_cols = preprocess_telco()

    X_cat_train, X_cat_test = build_global_ohe(train_df=train_df, test_df=test_df, orig_df=orig_df,
                                               cat_cols=cat_features)
    X_num_train = np.ascontiguousarray(train_df[numeric_cols].to_numpy(dtype=np.float32, copy=True))
    X_num_test = np.ascontiguousarray(test_df[numeric_cols].to_numpy(dtype=np.float32, copy=True))
    y = train_df[TARGET].astype(np.int32).values
    del train_df, test_df, orig_df
    gc.collect()

    fold_id = folds(y)
    oof = np.zeros(len(y), dtype=np.float32)
    pred_test = np.zeros(len(X_num_test), dtype=np.float32)
    for k in range(N_SPLITS):
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]

        X_num_tr_sc, X_num_va_sc, X_num_te_sc = standardize_fold(X_num_train[tr_idx], X_num_train[va_idx], X_num_test)

        X_tr = np.ascontiguousarray(
            np.concatenate([X_num_tr_sc, X_cat_train[tr_idx]], axis=1).astype(np.float32, copy=False))
        X_va = np.ascontiguousarray(
            np.concatenate([X_num_va_sc, X_cat_train[va_idx]], axis=1).astype(np.float32, copy=False))
        X_te = np.ascontiguousarray(
            np.concatenate([X_num_te_sc, X_cat_test], axis=1).astype(np.float32, copy=False))

        y_tr = np.ascontiguousarray(y[tr_idx].astype(np.int32, copy=False))
        y_va = y[va_idx].astype(np.int32, copy=False)

        model = cuLogisticRegression(**LOGREG_PARAMS)
        model.fit(X_tr, y_tr)

        p_va = model.predict_proba(X_va)[:, 1].astype(np.float32)
        p_te = model.predict_proba(X_te)[:, 1].astype(np.float32)

        oof[va_idx] = p_va
        pred_test += p_te / N_SPLITS
        log(f'fold {k} ({X_tr.shape[1]} columns): AUC {roc_auc_score(y_va, p_va):.6f}')

        del X_num_tr_sc, X_num_va_sc, X_num_te_sc
        del X_tr, X_va, X_te, y_tr, y_va, model, p_va, p_te
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        try:
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    log(f'{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}')
    save_pair(model_dir(FOLDER), PAIR, oof, pred_test)


if __name__ == '__main__':
    main()
