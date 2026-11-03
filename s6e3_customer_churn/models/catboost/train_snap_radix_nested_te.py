#!/usr/bin/env python
"""GPU CatBoost on a wide union of snap, digit, bin, radix, count and original-prior features plus fold-local nested
target encodings (pair catboost_snap_radix_union_nested_te).

MonthlyCharges / TotalCharges are read as strings; blanks get the train median and both are rewritten as 2-decimal
strings. Features (1,770 columns, 857 CatBoost-native categoricals; id and SeniorCitizen are numeric columns):
  - snap: every train / test charge value that does not occur in the original data is mapped to the nearest original
    value (*_snap string, *_snap_diff, *_snap_i = 100 x snap, *_snap_num); decimal strings of the charges (mc / tc),
    their first / second decimal digits (mc0, mc1, tc0, tc1) and lengths
  - frequency and percentile rank over train + test + original, log1p / sqrt / 1 / (1 + x) / missing flag of the 4
    numerics; charge relations (TotalCharges - tenure x MonthlyCharges, ratios, products, residual with the snapped
    MonthlyCharges); service Yes / No / other counts, has-internet / has-phone, Yes / No / other flags of 11 columns
  - tenure bins, 40 / 60 equal-count rank bins of MonthlyCharges / TotalCharges; 16 pair and 3 triple keys of
    categoricals, bins and snapped charges; per-frame 5-quantile, rounded (0 / 1 decimals) and halved categories of
    the numerics; integer and 2 decimal digit positions of the charges
  - radix keys: 100 x snapped MonthlyCharges / TotalCharges and tenure combined with the per-frame code of each of the
    16 categoricals (value + 100,000 x (code + 1))
  - counts and rare flags (<= 50) of the categorical-like columns, counts (and log1p) of the high-cardinality ones and
    of the 16 pairs, over train + test + original
  - original-data priors of every column shared with the original data: seen-in-original flag, original churn rate
    and count; original churn rate / count / seen flag of the 16 pairs; train-vs-original drift
    log1p(p_train / p_original) of the categorical-like columns
Per fold, nested target encodings (inner StratifiedKFold(5), seed 42 + fold number) of 19 columns (Contract,
InternetService, PaymentMethod, PaperlessBilling, snapped charges, decimal digits, bins and 6 pair keys) are added.
CatBoost (GPU): depth 7, lr 0.03, l2 8, Bernoulli subsample 0.8, min_data_in_leaf 24, random_strength 0.8, Cosine
score function, 10 Newton leaf-estimation iterations, border_count 254, up to 20,000 iterations with early stopping
(500) on the validation fold (use_best_model=True), seed 42. Test = mean of the fold models.

Usage: python train_snap_radix_nested_te.py"""
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import DATA_DIR, ORIGINAL_FILE, TARGET  # noqa: E402
from common.pairs import save_pair  # noqa: E402
from models.catboost.lib.cv import run_folds  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'catboost_snap_radix_union_nested_te'
SEED = 42
N_THREADS = 32
TEST_FOLDS = 15
STRING_COLS = {'MonthlyCharges': 'string', 'TotalCharges': 'string'}

BASE_CATS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]
BASE_NUMS = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
SERVICE_COLS = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
YN_FLAG_COLS = [
    'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'MultipleLines'
]
LOW_CARD_FOR_PAIRS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
PAIR_COLS = [
    ('Contract', 'InternetService'),
    ('PaymentMethod', 'Contract'),
    ('InternetService', 'OnlineSecurity'),
    ('PaymentMethod', 'PaperlessBilling'),
    ('Contract', 'PaperlessBilling'),
    ('InternetService', 'TechSupport'),
    ('tenure_bin', 'Contract'),
    ('tenure_bin', 'InternetService'),
    ('MonthlyCharges_bin', 'Contract'),
    ('MonthlyCharges_bin', 'InternetService'),
    ('TotalCharges_bin', 'Contract'),
    ('TotalCharges_bin', 'PaymentMethod'),
    ('MonthlyCharges_snap', 'Contract'),
    ('MonthlyCharges_snap', 'InternetService'),
    ('TotalCharges_snap', 'Contract'),
    ('TotalCharges_snap', 'InternetService'),
]
TRIPLE_COLS = [
    ('Contract', 'InternetService', 'PaymentMethod'),
    ('tenure_bin', 'Contract', 'InternetService'),
    ('MonthlyCharges_bin', 'Contract', 'PaymentMethod'),
]
RADIX_BASE_COLS = LOW_CARD_FOR_PAIRS[:]

MC_BINS = 40
TC_BINS = 60
CAT_RARE_THRESH = 50
HIGH_CARD_THRESHOLD = 7
DRIFT_EPS = 1e-6
RADIX = 100_000

NESTED_TE_COLS = [
    'Contract', 'InternetService', 'PaymentMethod', 'PaperlessBilling',
    'MonthlyCharges_snap', 'TotalCharges_snap', 'mc0', 'mc1', 'tc0', 'tc1',
    'tenure_bin', 'MonthlyCharges_bin', 'TotalCharges_bin',
    'MonthlyCharges_snap__Contract', 'MonthlyCharges_snap__InternetService',
    'TotalCharges_snap__Contract', 'TotalCharges_snap__InternetService',
    'Contract__InternetService', 'PaymentMethod__Contract',
]

CAT_PARAMS = dict(
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=SEED,
    iterations=20000,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=8.0,
    bootstrap_type='Bernoulli',
    subsample=0.8,
    grow_policy='SymmetricTree',
    min_data_in_leaf=24,
    random_strength=0.8,
    auto_class_weights=None,
    score_function='Cosine',
    leaf_estimation_iterations=10,
    leaf_estimation_method='Newton',
    border_count=254,
    bagging_temperature=None,
    od_type='Iter',
    od_wait=500,
    verbose=250,
    thread_count=N_THREADS,
    allow_writing_files=False,
    task_type='GPU',
)


def dedup_keep_order(seq):
    return list(dict.fromkeys(seq))


def map_target_binary(s: pd.Series) -> pd.Series:
    x = s.astype('string').str.strip().str.lower()
    mapped = x.map({'yes': 1, 'no': 0, '1': 1, '0': 0, 'true': 1, 'false': 0, 'y': 1, 'n': 0})
    if mapped.isna().any() and pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors='coerce').fillna(0).astype('int8')
    return mapped.astype('int8')


def clean_totalcharges_series(s: pd.Series) -> pd.Series:
    s = s.astype('string')
    s = s.str.strip()
    s = s.replace('', pd.NA)
    return s


def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            if c == 'TotalCharges':
                df[c] = clean_totalcharges_series(df[c])
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def safe_str_inplace(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype('string').fillna('missing').str.strip()


def make_triple_key(df: pd.DataFrame, c1: str, c2: str, c3: str) -> pd.Series:
    return (
        df[c1].astype('string').fillna('<NA>') + '|'
        + df[c2].astype('string').fillna('<NA>') + '|'
        + df[c3].astype('string').fillna('<NA>')
    )


def nested_target_encode(X_tr, y_tr, X_va, col, n_splits=5, seed=42):
    tr_col = X_tr[col].astype('string').fillna('<NA>')
    va_col = X_va[col].astype('string').fillna('<NA>')
    oof_te = np.zeros(len(X_tr), dtype=np.float32)
    inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = float(np.mean(y_tr))
    y_tr_series = pd.Series(y_tr, index=X_tr.index)

    for itr_idx, iva_idx in inner.split(X_tr, y_tr):
        sub_tr = X_tr.iloc[itr_idx]
        sub_y = y_tr_series.iloc[itr_idx]
        grp = sub_y.groupby(sub_tr[col].astype('string').fillna('<NA>')).mean()
        enc = X_tr.iloc[iva_idx][col].astype('string').fillna('<NA>').map(grp).fillna(global_mean)
        oof_te[iva_idx] = enc.values.astype(np.float32)

    full_grp = y_tr_series.groupby(tr_col).mean()
    va_te = va_col.map(full_grp).fillna(global_mean).values.astype(np.float32)
    return oof_te, va_te


# ------------------------------------------------------------------------------------------------------------------
# snap and decimal-string features
# ------------------------------------------------------------------------------------------------------------------
def _build_snap_map(combo_s: pd.Series, orig_s: pd.Series) -> dict:
    """Maps every train / test value string that is not an original value string to the nearest original value."""
    combo_str = combo_s.astype('string')
    orig_str = orig_s.astype('string')

    orig_unique = pd.Series(orig_str.dropna().unique(), dtype='string')
    orig_num = pd.to_numeric(orig_unique, errors='coerce')
    keep = orig_num.notna().values
    orig_unique = orig_unique[keep].reset_index(drop=True)
    orig_num = orig_num[keep].astype('float64').reset_index(drop=True)

    combo_unique = pd.Series(combo_str.dropna().unique(), dtype='string')
    combo_num = pd.to_numeric(combo_unique, errors='coerce')
    keep2 = combo_num.notna().values
    combo_unique = combo_unique[keep2].reset_index(drop=True)
    combo_num = combo_num[keep2].astype('float64').reset_index(drop=True)

    orig_set = set(orig_unique.tolist())
    missing_mask = ~combo_unique.isin(orig_set)
    missing_vals = combo_unique[missing_mask].reset_index(drop=True)
    missing_num = combo_num[missing_mask].reset_index(drop=True)
    if len(missing_vals) == 0:
        return {}

    orig_arr = orig_num.to_numpy()
    mapping = {}
    for s_val, n_val in zip(missing_vals.tolist(), missing_num.tolist()):
        j = int(np.argmin(np.abs(orig_arr - n_val)))
        mapping[s_val] = str(orig_unique.iloc[j])
    return mapping


def _apply_snap(combo_s: pd.Series, mapping: dict):
    s = combo_s.astype('string').copy()
    out = s.replace(mapping)
    diff = pd.to_numeric(out, errors='coerce') - pd.to_numeric(s, errors='coerce')
    return out, diff.astype('float32')


def add_snap_features(train_df, test_df, orig_df):
    out_tr, out_te, out_or = {}, {}, {}
    for col in ['MonthlyCharges', 'TotalCharges']:
        combo = pd.concat([train_df[col], test_df[col]], axis=0, ignore_index=True)
        mp = _build_snap_map(combo, orig_df[col])

        tr_snap, tr_diff = _apply_snap(train_df[col], mp)
        te_snap, te_diff = _apply_snap(test_df[col], mp)
        or_snap, or_diff = _apply_snap(orig_df[col], {})

        out_tr[f'{col}_snap'] = tr_snap
        out_te[f'{col}_snap'] = te_snap
        out_or[f'{col}_snap'] = or_snap
        out_tr[f'{col}_snap_diff'] = tr_diff
        out_te[f'{col}_snap_diff'] = te_diff
        out_or[f'{col}_snap_diff'] = or_diff
        out_tr[f'{col}_snap_i'] = pd.to_numeric(tr_snap, errors='coerce').round(2).mul(100).fillna(-1).astype('int32')
        out_te[f'{col}_snap_i'] = pd.to_numeric(te_snap, errors='coerce').round(2).mul(100).fillna(-1).astype('int32')
        out_or[f'{col}_snap_i'] = pd.to_numeric(or_snap, errors='coerce').round(2).mul(100).fillna(-1).astype('int32')

    train_df = pd.concat([train_df, pd.DataFrame(out_tr, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(out_te, index=test_df.index)], axis=1)
    orig_df = pd.concat([orig_df, pd.DataFrame(out_or, index=orig_df.index)], axis=1)
    return train_df, test_df, orig_df


def add_string_digit_features(train_df, test_df, orig_df):
    def build(df):
        out = {}
        s = df['MonthlyCharges'].astype('string')
        mc = s.str.split('.').str[1]
        out['mc'] = mc.fillna('-1')
        out['mc0'] = mc.str[0].fillna('-1')
        out['mc1'] = np.where(mc.str.len().fillna(0).eq(2), mc.str[1], '-1')
        out['mc_len'] = mc.str.len().fillna(-1).astype('int8')
        s = df['TotalCharges'].astype('string')
        tc = s.str.split('.').str[1]
        out['tc'] = tc.fillna('-1')
        out['tc0'] = tc.str[0].fillna('-1')
        out['tc1'] = np.where(tc.str.len().fillna(0).eq(2), tc.str[1], '-1')
        out['tc_len'] = tc.str.len().fillna(-1).astype('int8')
        return pd.DataFrame(out, index=df.index)

    train_df = pd.concat([train_df, build(train_df)], axis=1)
    test_df = pd.concat([test_df, build(test_df)], axis=1)
    orig_df = pd.concat([orig_df, build(orig_df)], axis=1)
    return train_df, test_df, orig_df


# ------------------------------------------------------------------------------------------------------------------
# frequencies, transforms, relations, flags, bins and crosses
# ------------------------------------------------------------------------------------------------------------------
def add_global_rank_and_freq(train_df, test_df, orig_df, num_cols):
    all_num = pd.concat([train_df[num_cols], test_df[num_cols], orig_df[num_cols]], axis=0, ignore_index=True)
    add_tr, add_te, add_or = {}, {}, {}
    n_tr, n_te = len(train_df), len(test_df)
    for c in num_cols:
        freq = all_num[c].value_counts(normalize=True, dropna=False)
        add_tr[f'FREQ_{c}'] = train_df[c].map(freq).fillna(0).astype('float32').values
        add_te[f'FREQ_{c}'] = test_df[c].map(freq).fillna(0).astype('float32').values
        add_or[f'FREQ_{c}'] = orig_df[c].map(freq).fillna(0).astype('float32').values

        r = all_num[c].rank(method='average', pct=True)
        add_tr[f'RANK_{c}'] = r.iloc[:n_tr].astype('float32').values
        add_te[f'RANK_{c}'] = r.iloc[n_tr:n_tr + n_te].astype('float32').values
        add_or[f'RANK_{c}'] = r.iloc[n_tr + n_te:].astype('float32').values

    train_df = pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1)
    orig_df = pd.concat([orig_df, pd.DataFrame(add_or, index=orig_df.index)], axis=1)
    return train_df, test_df, orig_df


def add_num_transforms(train_df, test_df, orig_df, num_cols):
    def build(df):
        out = {}
        for c in num_cols:
            v = pd.to_numeric(df[c], errors='coerce').astype('float32')
            out[f'LOG1P_{c}'] = np.log1p(v.clip(lower=0)).astype('float32')
            out[f'SQRT_{c}'] = np.sqrt(v.clip(lower=0)).astype('float32')
            out[f'INV1P_{c}'] = (1.0 / (1.0 + v.clip(lower=0))).astype('float32')
            out[f'MISS_{c}'] = pd.to_numeric(df[c], errors='coerce').isna().astype('int8')
        return pd.DataFrame(out, index=df.index)

    train_df = pd.concat([train_df, build(train_df)], axis=1)
    test_df = pd.concat([test_df, build(test_df)], axis=1)
    orig_df = pd.concat([orig_df, build(orig_df)], axis=1)
    return train_df, test_df, orig_df


def add_charge_relationships(train_df, test_df, orig_df):
    def build(df):
        out = {}
        t = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)
        m = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)
        tc = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        out['charges_deviation'] = (tc - t * m).astype('float32')
        out['abs_charges_dev'] = np.abs(out['charges_deviation']).astype('float32')
        out['monthly_to_total_ratio'] = (m / (tc + 1)).astype('float32')
        out['total_to_monthly_ratio'] = (tc / (m + 1)).astype('float32')
        out['avg_monthly_charges'] = (tc / (t + 1)).astype('float32')
        out['tenure_x_monthly'] = (t * m).astype('float32')
        out['tenure_x_total'] = (t * tc).astype('float32')
        mc_snap = pd.to_numeric(df.get('MonthlyCharges_snap', df['MonthlyCharges']), errors='coerce').fillna(0)
        out['tc_residual_snap'] = (tc - t * mc_snap).astype('float32')
        out['tc_residual_snap_abs'] = np.abs(out['tc_residual_snap']).astype('float32')
        return pd.DataFrame(out, index=df.index)

    train_df = pd.concat([train_df, build(train_df)], axis=1)
    test_df = pd.concat([test_df, build(test_df)], axis=1)
    orig_df = pd.concat([orig_df, build(orig_df)], axis=1)
    return train_df, test_df, orig_df


def add_service_counts(train_df, test_df, orig_df):
    for df in (train_df, test_df, orig_df):
        safe_str_inplace(df, SERVICE_COLS + ['InternetService'])

    def build(df):
        out = {}
        out['service_yes_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
        out['service_no_count'] = (df[SERVICE_COLS] == 'No').sum(axis=1).astype('float32')
        out['service_other_count'] = (
            df[SERVICE_COLS].isin(['No phone service', 'No internet service'])).sum(axis=1).astype('float32')
        out['service_count'] = out['service_yes_count'].astype('float32')
        out['has_internet'] = (df['InternetService'] != 'No').astype('float32')
        out['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
        return pd.DataFrame(out, index=df.index)

    train_df = pd.concat([train_df, build(train_df)], axis=1)
    test_df = pd.concat([test_df, build(test_df)], axis=1)
    orig_df = pd.concat([orig_df, build(orig_df)], axis=1)
    return train_df, test_df, orig_df


def add_yes_no_other_flags(train_df, test_df, orig_df):
    for df in (train_df, test_df, orig_df):
        safe_str_inplace(df, YN_FLAG_COLS)

    def build(df):
        out = {}
        for c in YN_FLAG_COLS:
            s = df[c].astype('string').fillna('missing')
            out[f'ISYES_{c}'] = (s == 'Yes').astype('float32')
            out[f'ISNO_{c}'] = (s == 'No').astype('float32')
            out[f'ISOTHER_{c}'] = (~s.isin(['Yes', 'No'])).astype('float32')
        return pd.DataFrame(out, index=df.index)

    train_df = pd.concat([train_df, build(train_df)], axis=1)
    test_df = pd.concat([test_df, build(test_df)], axis=1)
    orig_df = pd.concat([orig_df, build(orig_df)], axis=1)
    return train_df, test_df, orig_df


def add_bins_and_cross_cats(train_df, test_df, orig_df):
    tenure_bins = [0, 1, 3, 6, 12, 24, 36, 48, 60, 72, 10_000]
    for df in (train_df, test_df, orig_df):
        df['tenure_bin'] = pd.cut(pd.to_numeric(df['tenure'], errors='coerce'), bins=tenure_bins,
                                  include_lowest=True).astype(str)

    all_mc = pd.concat([pd.to_numeric(train_df['MonthlyCharges'], errors='coerce'),
                        pd.to_numeric(test_df['MonthlyCharges'], errors='coerce'),
                        pd.to_numeric(orig_df['MonthlyCharges'], errors='coerce')], axis=0)
    all_tc = pd.concat([pd.to_numeric(train_df['TotalCharges'], errors='coerce'),
                        pd.to_numeric(test_df['TotalCharges'], errors='coerce'),
                        pd.to_numeric(orig_df['TotalCharges'], errors='coerce')], axis=0)
    mc_bins = pd.qcut(all_mc.rank(method='first'), q=MC_BINS, retbins=True, duplicates='drop')[1]
    tc_bins = pd.qcut(all_tc.rank(method='first'), q=TC_BINS, retbins=True, duplicates='drop')[1]

    for df in (train_df, test_df, orig_df):
        mc_rank = pd.to_numeric(df['MonthlyCharges'], errors='coerce').rank(method='first')
        tc_rank = pd.to_numeric(df['TotalCharges'], errors='coerce').rank(method='first')
        df['MonthlyCharges_bin'] = pd.cut(mc_rank, bins=mc_bins, include_lowest=True).astype(str)
        df['TotalCharges_bin'] = pd.cut(tc_rank, bins=tc_bins, include_lowest=True).astype(str)

    for a, b in PAIR_COLS:
        for df in (train_df, test_df, orig_df):
            if a in df.columns and b in df.columns:
                df[f'{a}__{b}'] = df[a].astype('string').fillna('<NA>') + '|' + df[b].astype('string').fillna('<NA>')

    for a, b, c in TRIPLE_COLS:
        for df in (train_df, test_df, orig_df):
            if a in df.columns and b in df.columns and c in df.columns:
                df[f'{a}__{b}__{c}'] = make_triple_key(df, a, b, c)

    return train_df, test_df, orig_df


def add_qcut_round_digit_features(train_df, test_df, orig_df):
    def add_three_way(df, col, prefix):
        x = pd.to_numeric(df[col], errors='coerce')
        df[f'{prefix}_{col}_q5'] = pd.qcut(x.rank(method='first'), q=min(5, max(2, x.nunique(dropna=True))),
                                           duplicates='drop').astype(str)
        df[f'{prefix}_{col}_round0'] = x.round(0).fillna(-1).astype('int64').astype(str)
        if col in ['MonthlyCharges', 'TotalCharges']:
            df[f'{prefix}_{col}_round1'] = x.round(1).fillna(-1).astype(str)
            df[f'{prefix}_{col}_round_div2'] = (x.round(0).fillna(-1) // 2).astype('int64').astype(str)
        return df

    for col in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:
        train_df = add_three_way(train_df, col, 'A')
        test_df = add_three_way(test_df, col, 'A')
        orig_df = add_three_way(orig_df, col, 'A')

    def add_digit_positions(df, cols, max_dec_digits=2):
        out = {}
        for col in cols:
            v = pd.to_numeric(df[col], errors='coerce')
            fill_val = float(v.median()) if v.notna().any() else 0.0
            vv = v.fillna(fill_val)
            max_val = float(vv.abs().max()) if len(vv) else 0.0
            max_int_digits = 1 if max_val == 0 else int(np.log10(max_val)) + 1
            for i in range(max_int_digits):
                out[f'{col}_digit_pos_{i + 1}'] = ((vv // (10 ** i)) % 10).astype('int8')
            if bool((vv % 1 != 0).any()):
                for i in range(1, max_dec_digits + 1):
                    out[f'{col}_digit_dec_{i}'] = ((vv * (10 ** i)).round().astype(int) % 10).astype('int8')
        return pd.DataFrame(out, index=df.index)

    train_df = pd.concat([train_df, add_digit_positions(train_df, ['MonthlyCharges', 'TotalCharges'])], axis=1)
    test_df = pd.concat([test_df, add_digit_positions(test_df, ['MonthlyCharges', 'TotalCharges'])], axis=1)
    orig_df = pd.concat([orig_df, add_digit_positions(orig_df, ['MonthlyCharges', 'TotalCharges'])], axis=1)
    return train_df, test_df, orig_df


def add_radix_features(train_df, test_df, orig_df):
    def add_for_df(df):
        out = {}
        for num_col in ['MonthlyCharges_snap', 'TotalCharges_snap', 'tenure']:
            if num_col == 'tenure':
                base_i = pd.to_numeric(df[num_col], errors='coerce').fillna(-1).clip(-1, RADIX - 1).astype('int32')
            else:
                base_i = pd.to_numeric(df[num_col], errors='coerce').round(2).mul(100).fillna(-1).clip(
                    -1, RADIX - 1).astype('int32')
            for c in RADIX_BASE_COLS:
                codes = pd.factorize(df[c].astype('string').fillna('<NA>'), sort=False)[0].astype('int32')
                out[f'{num_col}_radix_{c}'] = (base_i + (codes + 1) * RADIX).astype('int32')
        return pd.DataFrame(out, index=df.index)

    train_df = pd.concat([train_df, add_for_df(train_df)], axis=1)
    test_df = pd.concat([test_df, add_for_df(test_df)], axis=1)
    orig_df = pd.concat([orig_df, add_for_df(orig_df)], axis=1)
    return train_df, test_df, orig_df


# ------------------------------------------------------------------------------------------------------------------
# counts over train + test + original, original-data priors and drift
# ------------------------------------------------------------------------------------------------------------------
def add_count_and_rare_features(train_df, test_df, orig_df, cat_cols, rare_thresh=50):
    all_cat = pd.concat([train_df[cat_cols], test_df[cat_cols], orig_df[cat_cols]], axis=0, ignore_index=True)
    add_tr, add_te, add_or = {}, {}, {}
    for c in cat_cols:
        vc = all_cat[c].astype('string').value_counts(dropna=False)
        tr_cnt = train_df[c].astype('string').map(vc).fillna(0).astype('float32')
        te_cnt = test_df[c].astype('string').map(vc).fillna(0).astype('float32')
        or_cnt = orig_df[c].astype('string').map(vc).fillna(0).astype('float32')
        add_tr[f'CAT_CNT_{c}'] = tr_cnt.values
        add_te[f'CAT_CNT_{c}'] = te_cnt.values
        add_or[f'CAT_CNT_{c}'] = or_cnt.values
        add_tr[f'CAT_RARE_{c}'] = (tr_cnt <= rare_thresh).astype('float32').values
        add_te[f'CAT_RARE_{c}'] = (te_cnt <= rare_thresh).astype('float32').values
        add_or[f'CAT_RARE_{c}'] = (or_cnt <= rare_thresh).astype('float32').values
    train_df = pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1)
    orig_df = pd.concat([orig_df, pd.DataFrame(add_or, index=orig_df.index)], axis=1)
    return train_df, test_df, orig_df


def add_count_features_combined(train_df, test_df, orig_df, cols):
    all_df = pd.concat([train_df[cols], test_df[cols], orig_df[cols]], axis=0, ignore_index=True)
    add_tr, add_te, add_or = {}, {}, {}
    for c in cols:
        s = all_df[c].astype('string').fillna('<NA>')
        vc = s.value_counts(dropna=False)
        add_tr[f'cnt_{c}'] = train_df[c].astype('string').fillna('<NA>').map(vc).astype('float32').values
        add_te[f'cnt_{c}'] = test_df[c].astype('string').fillna('<NA>').map(vc).astype('float32').values
        add_or[f'cnt_{c}'] = orig_df[c].astype('string').fillna('<NA>').map(vc).astype('float32').values
        add_tr[f'cnt_{c}_log1p'] = np.log1p(add_tr[f'cnt_{c}']).astype('float32')
        add_te[f'cnt_{c}_log1p'] = np.log1p(add_te[f'cnt_{c}']).astype('float32')
        add_or[f'cnt_{c}_log1p'] = np.log1p(add_or[f'cnt_{c}']).astype('float32')
    train_df = pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1)
    orig_df = pd.concat([orig_df, pd.DataFrame(add_or, index=orig_df.index)], axis=1)
    return train_df, test_df, orig_df


def add_pair_count_features(train_df, test_df, orig_df, pairs):
    def build(df_all, df_part):
        out = {}
        for c1, c2 in pairs:
            if c1 in df_part.columns and c2 in df_part.columns:
                name = f'{c1}_x_{c2}'
                key_all = df_all[c1].astype('string').fillna('<NA>') + '|' + df_all[c2].astype('string').fillna('<NA>')
                vc = key_all.value_counts(dropna=False)
                key = df_part[c1].astype('string').fillna('<NA>') + '|' + df_part[c2].astype('string').fillna('<NA>')
                feat = key.map(vc).fillna(0).astype('float32')
                out[f'pcnt_{name}'] = feat.values
                out[f'pcnt_{name}_log1p'] = np.log1p(feat).astype('float32').values
        return pd.DataFrame(out, index=df_part.index)

    all_df = pd.concat([train_df, test_df, orig_df], axis=0, ignore_index=True)
    train_df = pd.concat([train_df, build(all_df, train_df)], axis=1)
    test_df = pd.concat([test_df, build(all_df, test_df)], axis=1)
    orig_df = pd.concat([orig_df, build(all_df, orig_df)], axis=1)
    return train_df, test_df, orig_df


def add_orig_membership_priors(train_df, test_df, orig_df, target_col=TARGET):
    y_orig = map_target_binary(orig_df[target_col])
    tmp = orig_df.copy()
    tmp['_y_'] = y_orig.astype('float32')

    safe_cols = [c for c in train_df.columns if c in orig_df.columns and c != target_col]
    add_tr, add_te = {}, {}
    for c in safe_cols:
        orig_vals = set(orig_df[c].astype('string').fillna('<NA>').unique())
        add_tr[f'{c}_in_orig'] = train_df[c].astype('string').fillna('<NA>').isin(orig_vals).astype('int8').values
        add_te[f'{c}_in_orig'] = test_df[c].astype('string').fillna('<NA>').isin(orig_vals).astype('int8').values

        grp = tmp.groupby(c, dropna=False)['_y_']
        mean_map = grp.mean()
        cnt_map = grp.size()
        add_tr[f'orig_mean_{c}'] = train_df[c].map(mean_map).astype('float32').fillna(float(tmp['_y_'].mean())).values
        add_te[f'orig_mean_{c}'] = test_df[c].map(mean_map).astype('float32').fillna(float(tmp['_y_'].mean())).values
        add_tr[f'orig_count_{c}'] = train_df[c].map(cnt_map).astype('float32').fillna(0).values
        add_te[f'orig_count_{c}'] = test_df[c].map(cnt_map).astype('float32').fillna(0).values

    train_df = pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1)
    return train_df, test_df


def add_orig_pair_priors(train_df, test_df, orig_df, pairs, target_col=TARGET):
    y_orig = map_target_binary(orig_df[target_col])
    out_tr, out_te = {}, {}
    for c1, c2 in pairs:
        if c1 in orig_df.columns and c2 in orig_df.columns and c1 in train_df.columns and c2 in train_df.columns:
            name = f'{c1}_x_{c2}'
            orig_key = orig_df[c1].astype('string').fillna('<NA>') + '|' + orig_df[c2].astype('string').fillna('<NA>')
            tmp = pd.DataFrame({name: orig_key, target_col: y_orig.values})
            grp = tmp.groupby(name, dropna=False)[target_col]
            mean_map = grp.mean()
            cnt_map = grp.size()

            tr_key = train_df[c1].astype('string').fillna('<NA>') + '|' + train_df[c2].astype('string').fillna('<NA>')
            te_key = test_df[c1].astype('string').fillna('<NA>') + '|' + test_df[c2].astype('string').fillna('<NA>')
            out_tr[f'orig_mean_{name}'] = tr_key.map(mean_map).astype('float32').fillna(float(y_orig.mean())).values
            out_te[f'orig_mean_{name}'] = te_key.map(mean_map).astype('float32').fillna(float(y_orig.mean())).values
            out_tr[f'orig_count_{name}'] = tr_key.map(cnt_map).astype('float32').fillna(0).values
            out_te[f'orig_count_{name}'] = te_key.map(cnt_map).astype('float32').fillna(0).values
            seen = set(mean_map.index)
            out_tr[f'orig_seen_{name}'] = tr_key.isin(seen).astype('int8').values
            out_te[f'orig_seen_{name}'] = te_key.isin(seen).astype('int8').values
    train_df = pd.concat([train_df, pd.DataFrame(out_tr, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(out_te, index=test_df.index)], axis=1)
    return train_df, test_df


def add_drift_ratio(train_df, test_df, orig_df, cols, eps=1e-6):
    add_tr, add_te = {}, {}
    for c in cols:
        if c in train_df.columns and c in orig_df.columns:
            p_tr = train_df[c].value_counts(normalize=True, dropna=False)
            p_or = orig_df[c].value_counts(normalize=True, dropna=False)
            add_tr[f'DRIFT_{c}'] = np.log1p(
                train_df[c].map(p_tr).fillna(0) / (train_df[c].map(p_or).fillna(eps) + eps)).astype('float32').values
            add_te[f'DRIFT_{c}'] = np.log1p(
                test_df[c].map(p_tr).fillna(0) / (test_df[c].map(p_or).fillna(eps) + eps)).astype('float32').values
    train_df = pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1)
    return train_df, test_df


def finalize_categoricals(train_df, test_df, orig_df):
    """String / category / bool columns, plus small-cardinality bin / key / digit columns, become CatBoost-native
    categoricals (strings); every other column becomes float32."""
    cat_cols = []
    all_cols = [c for c in train_df.columns if c != TARGET]
    for c in all_cols:
        dt = train_df[c].dtype
        if str(dt).startswith('string') or dt == object or str(dt) == 'category' or str(dt) == 'bool':
            cat_cols.append(c)
    for c in all_cols:
        if c in cat_cols:
            continue
        if any(tok in c for tok in ['_bin', '__', '_q5', '_round0', '_round1', '_round_div2', 'mc', 'tc']) \
                and train_df[c].nunique(dropna=False) <= 5000:
            cat_cols.append(c)
    cat_cols = dedup_keep_order([c for c in cat_cols if c in train_df.columns and c in test_df.columns])

    for df in (train_df, test_df, orig_df):
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].astype('string').fillna('missing')

    num_cols = [c for c in train_df.columns if c not in cat_cols + [TARGET]]
    for df in (train_df, test_df, orig_df):
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    return train_df, test_df, orig_df, cat_cols, num_cols


def build_frames():
    train = pd.read_csv(DATA_DIR / 'train.csv', dtype=STRING_COLS)
    test = pd.read_csv(DATA_DIR / 'test.csv', dtype=STRING_COLS)
    orig = pd.read_csv(DATA_DIR / ORIGINAL_FILE, dtype=STRING_COLS)

    y = map_target_binary(train[TARGET]).values.astype('int8')
    train[TARGET] = y
    orig[TARGET] = map_target_binary(orig[TARGET]).astype('int8')

    for df in (train, test, orig):
        df['TotalCharges'] = clean_totalcharges_series(df['TotalCharges'])
        df['MonthlyCharges'] = df['MonthlyCharges'].astype('string').str.strip()
        safe_str_inplace(df, BASE_CATS)
    train = coerce_numeric(train, BASE_NUMS)
    test = coerce_numeric(test, BASE_NUMS)
    orig = coerce_numeric(orig, BASE_NUMS)
    for c in BASE_NUMS:
        med = float(pd.to_numeric(train[c], errors='coerce').median()) if train[c].notna().any() else 0.0
        train[c] = pd.to_numeric(train[c], errors='coerce').fillna(med).astype('float32')
        test[c] = pd.to_numeric(test[c], errors='coerce').fillna(med).astype('float32')
        orig[c] = pd.to_numeric(orig[c], errors='coerce').fillna(med).astype('float32')
    for df in (train, test, orig):
        df['MonthlyCharges'] = df['MonthlyCharges'].map(lambda x: f'{float(x):.2f}')
        df['TotalCharges'] = df['TotalCharges'].map(lambda x: f'{float(x):.2f}')

    print('feature engineering...', flush=True)
    train, test, orig = add_snap_features(train, test, orig)
    train, test, orig = add_string_digit_features(train, test, orig)
    for df in (train, test, orig):
        for c in BASE_NUMS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
        for c in ['MonthlyCharges_snap', 'TotalCharges_snap']:
            df[f'{c}_num'] = pd.to_numeric(df[c], errors='coerce').astype('float32')

    train, test, orig = add_global_rank_and_freq(train, test, orig, BASE_NUMS)
    train, test, orig = add_num_transforms(train, test, orig, BASE_NUMS)
    train, test, orig = add_charge_relationships(train, test, orig)
    train, test, orig = add_service_counts(train, test, orig)
    train, test, orig = add_yes_no_other_flags(train, test, orig)
    train, test, orig = add_bins_and_cross_cats(train, test, orig)
    train, test, orig = add_qcut_round_digit_features(train, test, orig)
    train, test, orig = add_radix_features(train, test, orig)

    candidate_cat_cols = []
    for c in train.columns:
        if c == TARGET:
            continue
        if train[c].dtype == object or str(train[c].dtype).startswith('string') or str(train[c].dtype) == 'category':
            candidate_cat_cols.append(c)
    for c in ['mc', 'mc0', 'mc1', 'tc', 'tc0', 'tc1', 'tenure_bin', 'MonthlyCharges_bin', 'TotalCharges_bin']:
        candidate_cat_cols.append(c)
    candidate_cat_cols = dedup_keep_order(candidate_cat_cols)

    train, test, orig = add_count_and_rare_features(train, test, orig, candidate_cat_cols, rare_thresh=CAT_RARE_THRESH)
    count_cols = dedup_keep_order(
        [c for c in candidate_cat_cols if train[c].nunique(dropna=False) > HIGH_CARD_THRESHOLD]
        + ['MonthlyCharges_snap', 'TotalCharges_snap', 'Contract', 'InternetService', 'PaymentMethod', 'tenure_bin',
           'MonthlyCharges_bin', 'TotalCharges_bin'])
    train, test, orig = add_count_features_combined(train, test, orig, count_cols)
    train, test, orig = add_pair_count_features(train, test, orig, PAIR_COLS)

    train, test = add_orig_membership_priors(train, test, orig, target_col=TARGET)
    train, test = add_orig_pair_priors(train, test, orig, PAIR_COLS, target_col=TARGET)
    train, test = add_drift_ratio(train, test, orig, [c for c in candidate_cat_cols if c in orig.columns],
                                  eps=DRIFT_EPS)

    train, test, orig, cat_cols, num_cols = finalize_categoricals(train, test, orig)
    print(f'features: {len(cat_cols) + len(num_cols)} ({len(cat_cols)} categorical)', flush=True)
    return train, test, y, cat_cols, num_cols


def main():
    train, test, y, cat_cols, num_cols = build_frames()
    features = cat_cols + num_cols

    def fit_fold(fold, tr_idx, va_idx, seed):
        X_tr = train.iloc[tr_idx][features].copy()
        X_va = train.iloc[va_idx][features].copy()
        X_te = test[features].copy()
        y_tr = y[tr_idx]
        for c in NESTED_TE_COLS:
            if c in X_tr.columns:
                te_tr, te_va = nested_target_encode(X_tr, y_tr, X_va, c, n_splits=5, seed=seed + fold)
                _, te_te = nested_target_encode(X_tr, y_tr, X_te, c, n_splits=5, seed=seed + fold)
                X_tr[f'NTE_{c}'] = te_tr
                X_va[f'NTE_{c}'] = te_va
                X_te[f'NTE_{c}'] = te_te

        fold_cat_cols = [c for c in cat_cols if c in X_tr.columns]
        tr_pool = Pool(X_tr, label=y_tr, cat_features=fold_cat_cols)
        va_pool = Pool(X_va, label=y[va_idx], cat_features=fold_cat_cols)
        te_pool = Pool(X_te, cat_features=fold_cat_cols)
        model = CatBoostClassifier(**CAT_PARAMS)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        p_va = model.predict_proba(va_pool)[:, 1].astype('float32')
        p_te = model.predict_proba(te_pool)[:, 1].astype('float32')
        print(f'fold {fold}: best iteration {model.get_best_iteration()}', flush=True)
        del X_tr, X_va, X_te, tr_pool, va_pool, te_pool, model
        gc.collect()
        return p_va, p_te

    oof, _ = run_folds(fit_fold, y, len(test), name=PAIR)
    _, test_pred = run_folds(fit_fold, y, len(test), n_folds=TEST_FOLDS, seed=SEED, name=PAIR)
    print(f'{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
