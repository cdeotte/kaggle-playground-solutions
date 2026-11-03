#!/usr/bin/env python
"""GPU XGBoost on a wide label-free union of snap, decimal-string, LGBM-port and radix features with nested target
encodings (pair xgb_snap_lgbm_port_radix_nested_te).

Reads the snap-feature table of the train + test rows written by ../snap_features (combo.csv); run that folder first.
The original rows (customerID kept as id; the 11 rows with a blank TotalCharges dropped) get their own decimal strings,
snap = value and snap difference 0. Label-free blocks over the train + test + original rows:
- the raw columns (id included), the decimal strings of the charges (mc, mc0, mc1, mc_len, tc, tc0, tc1, tc_len), the
  snapped charges and the snap differences;
- frequency and percentile rank of the 3 numerics; log1p, sqrt, 1 / (1 + x) and missing flags of the 3 numerics and
  the 2 snaps; charge relations (TotalCharges - tenure x MonthlyCharges and its absolute value, MonthlyCharges /
  (TotalCharges + 1), TotalCharges / (MonthlyCharges + 1), TotalCharges / (tenure + 1), tenure x MonthlyCharges,
  tenure x TotalCharges, tenure x each snap); service counts, has-internet / has-phone, Yes / No / other flags of 11
  columns;
- string bins: tenure (edges -1, 0, 1, 3, 6, 12, 24, 48, 72), 12 quantile bins of the charges and of the snaps, 6
  category pairs and Contract x InternetService x PaymentMethod; counts and rare flags (count <= 50) of the 16
  categoricals, the 6 decimal strings and the 3 bins; drift log1p(train share / original share) of the 16
  categoricals; original churn rate of the 16 categoricals, the 3 bins and the 2 snaps (unseen: 0.5);
- roundings (tenure to units, charges and snaps to units / 0.1, TotalCharges and its snap also to tens), integer and
  2-decimal digit positions of tenure, the charges and the snaps;
- LGBM-port block: every column with > 12 distinct values as a string key with its count, original churn rate,
  original count and count ratio; for the numerics and snaps round(x) // 2, integer, 10-quantile and log bins as string
  keys; 4 integer + 2 decimal digits (strings) of the charges and snaps and the 15 pairs of the first 6 decimal digits;
- radix keys: codes of the 16 categoricals, round(100 x snap) + 100,000 x code for both snaps and tenure + 100,000 x
  code;
- 7 pair keys (Contract x InternetService, PaymentMethod x Contract, InternetService x TechSupport, the MonthlyCharges /
  tenure / MonthlyCharges-snap bins x Contract, the TotalCharges-snap bin x PaymentMethod), capped at the 20,000 most
  frequent levels.
Per outer fold: every *_te key gets sklearn TargetEncoder means (smooth auto), out-of-fold over an inner
StratifiedKFold(5, shuffle=True, random_state=42) of the training rows and fitted on all training rows for the
validation / test rows; the remaining string columns are factorized to integer codes (training + validation rows
together, test rows separately) and everything else is cast to float32.
Model: depth 6, lr 0.03, subsample 0.8, colsample_bytree 0.45, min_child_weight 6, gamma 0.1, alpha 0.5, lambda 4,
max_bin 4096, up to 1,000,000 rounds, early stopping 500 on the validation fold, random_state 42.
Usage: python train_snap_lgbm_port_radix_union.py"""
import gc
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import DATA_DIR, ORIGINAL_FILE, TARGET, cache_dir, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'xgb_snap_lgbm_port_radix_nested_te'
SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5
TEST_FOLDS = 15
TEST_INNER_FOLDS = 10
N_THREADS = 32

HIGH_CARD_THRESHOLD = 12
PAIR_MAX_UNIQUE = 20000
RADIX = 100_000
CAT_RARE_THRESH = 50
MC_BINS = 12
TC_BINS = 12
TENURE_BINS = [-1, 0, 1, 3, 6, 12, 24, 48, 72]

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
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    enable_categorical=False,
    device='cuda',
    random_state=SEED,
    n_jobs=N_THREADS,
    early_stopping_rounds=500,
    verbosity=0,
)

BASE_CATS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
             'Contract', 'PaperlessBilling', 'PaymentMethod']
BASE_NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']
YN_COLS = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
CROSS_PAIRS = [('Contract', 'InternetService'), ('PaymentMethod', 'Contract'), ('InternetService', 'OnlineSecurity'),
               ('PaymentMethod', 'PaperlessBilling'), ('Contract', 'PaperlessBilling'),
               ('InternetService', 'TechSupport')]
TRIPLE = [('Contract', 'InternetService', 'PaymentMethod')]
LOW_CARD_COLS = BASE_CATS
PAIR_FEATURES = [('Contract', 'InternetService'), ('PaymentMethod', 'Contract'), ('InternetService', 'TechSupport'),
                 ('MonthlyCharges_bin', 'Contract'), ('tenure_bin', 'Contract'),
                 ('MonthlyCharges_snap_bin', 'Contract'), ('TotalCharges_snap_bin', 'PaymentMethod')]
ROUND_SPECS = {'tenure': [0], 'MonthlyCharges': [0, 1], 'TotalCharges': [0, 1, -1], 'MonthlyCharges_snap': [0, 1],
               'TotalCharges_snap': [0, 1, -1]}
DIGIT_BASE_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyCharges_snap', 'TotalCharges_snap']
DECIMAL_STRING_COLS = ['mc', 'mc0', 'mc1', 'tc', 'tc0', 'tc1']
COMBO_DTYPES = {'TotalCharges': 'string', 'MonthlyCharges': 'string', 'tc': 'object', 'tc0': 'object', 'tc1': 'object',
                'mc': 'object', 'mc0': 'object', 'mc1': 'object', 'MonthlyCharges_snap': 'string',
                'TotalCharges_snap': 'string'}


def log(msg):
    print(msg, flush=True)


def dedup_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def to_bin_target(s):
    if str(s.dtype).startswith(('object', 'string')):
        y = s.astype('string').str.strip().map({'No': 0, 'Yes': 1})
        return y.astype('int8').values
    return pd.to_numeric(s, errors='coerce').fillna(0).astype('int8').values


def clean_totalcharges_string_inplace(df):
    df['TotalCharges'] = df['TotalCharges'].astype('string').replace(r'^\s*$', pd.NA, regex=True)


def coerce_numeric_series(s):
    return pd.to_numeric(s.astype('string').replace(r'^\s*$', pd.NA, regex=True), errors='coerce').astype('float32')


def ensure_numeric_columns(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = coerce_numeric_series(df[c])


def safe_string_df(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype('string').fillna('missing').str.strip()


def add_decimal_parts_like_eda(df):
    mc = df['MonthlyCharges'].astype('string').str.split('.').str[1]
    df['mc'] = mc
    df['mc0'] = mc.str[0]
    df['mc1'] = '-1'
    df['mc_len'] = mc.str.len().fillna(0).astype('int16')
    idx = df['mc_len'] == 2
    df.loc[idx, 'mc1'] = df.loc[idx, 'mc'].str[1]
    tc = df['TotalCharges'].astype('string').str.split('.').str[1]
    df['tc'] = tc
    df['tc0'] = tc.str[0]
    df['tc1'] = '-1'
    df['tc_len'] = tc.str.len().fillna(0).astype('int16')
    idx = df['tc_len'] == 2
    df.loc[idx, 'tc1'] = df.loc[idx, 'tc'].str[1]


# ---------------------------------------------------------------------------------------------------------------------
# Label-free blocks
# ---------------------------------------------------------------------------------------------------------------------
def add_freq_rank_features(combined, num_cols):
    out = {}
    for c in num_cols:
        s = combined[c].astype('float32')
        vc = s.value_counts(normalize=True, dropna=False)
        out[f'FREQ_{c}'] = s.map(vc).fillna(0).astype('float32')
        out[f'RANK_{c}'] = s.rank(method='average', pct=True).astype('float32')
    return pd.DataFrame(out, index=combined.index)


def add_num_transforms(combined, num_cols):
    out = {}
    for c in num_cols:
        v = combined[c].astype('float32')
        out[f'LOG1P_{c}'] = np.log1p(v.clip(lower=0)).astype('float32')
        out[f'SQRT_{c}'] = np.sqrt(v.clip(lower=0)).astype('float32')
        out[f'INV1P_{c}'] = (1.0 / (1.0 + v.clip(lower=0))).astype('float32')
        out[f'MISS_{c}'] = v.isna().astype('int8')
    return pd.DataFrame(out, index=combined.index)


def add_charge_relationships(combined):
    ten = combined['tenure'].astype('float32')
    mc = combined['MonthlyCharges'].astype('float32')
    tc = combined['TotalCharges'].astype('float32')
    out = {}
    out['charges_deviation'] = (tc - ten * mc).astype('float32')
    out['abs_charges_dev'] = np.abs(out['charges_deviation']).astype('float32')
    out['monthly_to_total_ratio'] = (mc / (tc + 1.0)).astype('float32')
    out['total_to_monthly_ratio'] = (tc / (mc + 1.0)).astype('float32')
    out['avg_monthly_charges'] = (tc / (ten + 1.0)).astype('float32')
    out['tenure_x_monthly'] = (ten * mc).astype('float32')
    out['tenure_x_total'] = (ten * tc).astype('float32')
    out['tenure_x_monthly_snap'] = (ten * combined['MonthlyCharges_snap'].astype('float32')).astype('float32')
    out['tenure_x_total_snap'] = (ten * combined['TotalCharges_snap'].astype('float32')).astype('float32')
    return pd.DataFrame(out, index=combined.index)


def add_service_counts(combined):
    safe_string_df(combined, SERVICE_COLS + ['InternetService'])
    out = {}
    out['service_yes_count'] = (combined[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
    out['service_no_count'] = (combined[SERVICE_COLS] == 'No').sum(axis=1).astype('float32')
    out['service_other_count'] = (combined[SERVICE_COLS].isin(['No phone service', 'No internet service'])
                                  ).sum(axis=1).astype('float32')
    out['service_count'] = out['service_yes_count'].astype('float32')
    out['has_internet'] = (combined['InternetService'] != 'No').astype('float32')
    out['has_phone'] = (combined['PhoneService'] == 'Yes').astype('float32')
    return pd.DataFrame(out, index=combined.index)


def add_yes_no_other_flags(combined):
    safe_string_df(combined, YN_COLS)
    out = {}
    for c in YN_COLS:
        s = combined[c].astype('string')
        out[f'ISYES_{c}'] = (s == 'Yes').astype('float32')
        out[f'ISNO_{c}'] = (s == 'No').astype('float32')
        out[f'ISOTHER_{c}'] = (~s.isin(['Yes', 'No'])).astype('float32')
    return pd.DataFrame(out, index=combined.index)


def add_bins_and_cross_cats(combined):
    out = pd.DataFrame(index=combined.index)
    out['tenure_bin'] = pd.cut(combined['tenure'], bins=TENURE_BINS, include_lowest=True).astype('string')

    def qbins(s, q):
        return pd.qcut(s, q=min(q, max(2, s.nunique(dropna=True))), duplicates='drop', retbins=True)[1]

    mc_bins = qbins(combined['MonthlyCharges'], MC_BINS)
    tc_bins = qbins(combined['TotalCharges'].fillna(-1), TC_BINS)
    out['MonthlyCharges_bin'] = pd.cut(combined['MonthlyCharges'], bins=mc_bins, include_lowest=True).astype('string')
    out['TotalCharges_bin'] = pd.cut(combined['TotalCharges'], bins=tc_bins, include_lowest=True).astype('string')
    mcs_bins = qbins(combined['MonthlyCharges_snap'], MC_BINS)
    tcs_bins = qbins(combined['TotalCharges_snap'].fillna(-1), TC_BINS)
    out['MonthlyCharges_snap_bin'] = pd.cut(combined['MonthlyCharges_snap'], bins=mcs_bins,
                                            include_lowest=True).astype('string')
    out['TotalCharges_snap_bin'] = pd.cut(combined['TotalCharges_snap'], bins=tcs_bins,
                                          include_lowest=True).astype('string')

    for a, b in CROSS_PAIRS:
        out[f'{a}__{b}'] = combined[a].astype('string') + '|' + combined[b].astype('string')
    for a, b, c in TRIPLE:
        out[f'{a}__{b}__{c}'] = (combined[a].astype('string') + '|' + combined[b].astype('string') + '|'
                                 + combined[c].astype('string'))
    return out


def add_cat_counts_and_rare(combined, cat_cols, rare_thresh=50):
    out = {}
    for c in cat_cols:
        vc = combined[c].astype('string').value_counts(dropna=False)
        cnt = combined[c].astype('string').map(vc).fillna(0).astype('float32')
        out[f'CAT_CNT_{c}'] = cnt
        out[f'CAT_RARE_{c}'] = (cnt <= rare_thresh).astype('float32')
    return pd.DataFrame(out, index=combined.index)


def add_drift_ratio_from_orig(combined, n_train, n_test, cols, eps=1e-6):
    train_df = combined.iloc[:n_train]
    orig_df = combined.iloc[n_train + n_test:]
    out = {}
    for c in cols:
        p_tr = train_df[c].value_counts(normalize=True, dropna=False)
        p_or = orig_df[c].value_counts(normalize=True, dropna=False)
        num = combined[c].map(p_tr).fillna(0)
        den = combined[c].map(p_or).fillna(eps) + eps
        out[f'DRIFT_{c}'] = np.log1p(num / den).astype('float32')
    return pd.DataFrame(out, index=combined.index)


def add_orig_proba_features(combined, n_train, n_test, y_orig, cols):
    orig_df = combined.iloc[n_train + n_test:].copy()
    orig_df['_y_'] = y_orig.astype('float32')
    out = {}
    for c in cols:
        m = orig_df.groupby(c, dropna=False)['_y_'].mean()
        out[f'ORIG_proba_{c}'] = combined[c].map(m).fillna(0.5).astype('float32')
    return pd.DataFrame(out, index=combined.index)


def add_round_features(combined, round_specs):
    out = {}
    for c, rr in round_specs.items():
        s = pd.to_numeric(combined[c], errors='coerce')
        for r in rr:
            out[f'B_round_{c}_r{r}'] = s.round(r).astype('float32')
    return pd.DataFrame(out, index=combined.index)


def add_digit_position_features(combined, base_cols, max_dec_digits=2):
    out = {}
    for col in base_cols:
        vals = pd.to_numeric(combined[col], errors='coerce').fillna(0)
        abs_vals = vals.abs().round(max_dec_digits)
        max_val = int(abs_vals.max()) if len(abs_vals) else 0
        max_int_digits = 1 if max_val == 0 else int(np.log10(max_val)) + 1
        for i in range(max_int_digits):
            out[f'{col}_digit_pos_{i + 1}'] = ((abs_vals // (10 ** i)).astype(np.int64) % 10).astype('int16')
        if not np.allclose(vals, vals.round(), equal_nan=True):
            for i in range(1, max_dec_digits + 1):
                out[f'{col}_digit_dec_{i}'] = ((abs_vals * (10 ** i)).round().astype(np.int64) % 10).astype('int16')
    return pd.DataFrame(out, index=combined.index)


def add_string_digit_features(combined, cols):
    out = {}
    for col in cols:
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
    return pd.DataFrame(out, index=combined.index)


def create_count_features(X, cols_list):
    out = {}
    for cols in cols_list:
        out[f'count({",".join(cols)})'] = X.groupby(list(cols), dropna=False).transform('size').astype('float32')
    return pd.DataFrame(out, index=X.index)


def create_groupby_features_original(X_keys, inputs, orig_data, target_col):
    out = {}
    for cols, agg in inputs:
        key_cols = list(cols)
        xk = X_keys[key_cols].copy()
        od = orig_data[key_cols + [target_col]].copy()
        for c in key_cols:
            xk[c] = xk[c].astype('string')
            od[c] = od[c].astype('string')
        if agg == 'mean':
            mapping = od.groupby(key_cols, dropna=False)[target_col].mean().rename('v').reset_index()
        else:
            mapping = od.groupby(key_cols, dropna=False)[target_col].size().rename('v').reset_index()
        name = f'groupby({",".join(cols)})_{agg}_{target_col}_orig'
        out[name] = xk.merge(mapping, on=key_cols, how='left')['v'].astype('float32')
    return pd.DataFrame(out, index=X_keys.index)


def add_lgbm_port_features(combined, n_train, n_test, y_orig):
    out_blocks = []
    orig_df = combined.iloc[n_train + n_test:].copy()
    orig_df['_y_orig_'] = y_orig.astype('float32')

    base_cols = [c for c in combined.columns if c != TARGET]
    high_card = [c for c in base_cols if combined[c].astype('string').nunique(dropna=False) > HIGH_CARD_THRESHOLD]
    if high_card:
        keys = combined[high_card].astype('string')
        out_blocks.append(keys.add_prefix('L_').add_suffix('_te'))
        out_blocks.append(create_count_features(keys, [(c,) for c in high_card]).add_prefix('L_'))
        orig_keys = orig_df[high_card + ['_y_orig_']]
        out_blocks.append(create_groupby_features_original(keys, [((c,), 'mean') for c in high_card], orig_keys,
                                                           '_y_orig_').add_prefix('L_'))
        out_blocks.append(create_groupby_features_original(keys, [((c,), 'size') for c in high_card], orig_keys,
                                                           '_y_orig_').add_prefix('L_'))
        for c in high_card:
            vc_comb = combined[c].astype('string').value_counts(dropna=False)
            vc_orig = orig_df[c].astype('string').value_counts(dropna=False)
            ratio = (combined[c].astype('string').map(vc_comb).fillna(0)
                     / (combined[c].astype('string').map(vc_orig).fillna(0) + 1))
            out_blocks.append(pd.DataFrame({f'L_count_ratio_{c}': ratio.astype('float32')}, index=combined.index))

    def na_series():
        return pd.Series(['NA'] * len(combined), index=combined.index, dtype='string')

    num_bins = {}
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyCharges_snap', 'TotalCharges_snap']:
        s = pd.to_numeric(combined[col], errors='coerce')
        num_bins[f'L_{col}_round_div2_te'] = (s.round().fillna(-1).astype('int64') // 2).astype('string')
        num_bins[f'L_{col}_int_te'] = s.fillna(-1).astype('int64').astype('string')
        try:
            q = int(min(10, max(2, s.nunique(dropna=True))))
            num_bins[f'L_{col}_quantiled_te'] = pd.qcut(s, q=q, duplicates='drop').astype('string')
        except Exception:
            num_bins[f'L_{col}_quantiled_te'] = na_series()
        try:
            bins = np.unique(np.nanquantile(s.dropna(), np.linspace(0, 1, 11)))
            num_bins[f'L_{col}_binned_te'] = (pd.cut(s, bins=bins, duplicates='drop').astype('string')
                                              if len(bins) >= 2 else na_series())
        except Exception:
            num_bins[f'L_{col}_binned_te'] = na_series()
        try:
            slog = np.log1p(s.clip(lower=0))
            bins = np.unique(np.nanquantile(slog.dropna(), np.linspace(0, 1, 11)))
            num_bins[f'L_{col}_log_binned_te'] = (pd.cut(slog, bins=bins, duplicates='drop').astype('string')
                                                  if len(bins) >= 2 else na_series())
        except Exception:
            num_bins[f'L_{col}_log_binned_te'] = na_series()
    out_blocks.append(pd.DataFrame(num_bins, index=combined.index))

    digit_df = add_string_digit_features(combined, ['MonthlyCharges', 'TotalCharges', 'MonthlyCharges_snap',
                                                    'TotalCharges_snap'])
    out_blocks.append(digit_df)
    cand = [c for c in digit_df.columns if ('TotalCharges' in c or 'MonthlyCharges' in c) and 'decimal_digit' in c][:6]
    combs = {}
    for a, b in itertools.combinations(cand, 2):
        combs[f'L_digits_{a}__{b}_te'] = digit_df[a].astype('string') + '|' + digit_df[b].astype('string')
    out_blocks.append(pd.DataFrame(combs, index=combined.index))
    return pd.concat(out_blocks, axis=1)


def add_radix_features(combined):
    code_df = pd.DataFrame({f'code_{c}': pd.Series(pd.factorize(combined[c].astype('string'), sort=False)[0],
                                                   index=combined.index).astype('int32') for c in LOW_CARD_COLS},
                           index=combined.index)
    out2 = {}
    mc_i = np.round(combined['MonthlyCharges_snap'].fillna(0).astype('float32') * 100.0).astype('int32')
    for c in LOW_CARD_COLS:
        out2[f'mc_radix_{c}'] = (mc_i + code_df[f'code_{c}'] * RADIX).astype('int32')
    tc_i = np.round(combined['TotalCharges_snap'].fillna(0).astype('float32') * 100.0).astype('int32')
    for c in LOW_CARD_COLS:
        out2[f'tc_radix_{c}'] = (tc_i + code_df[f'code_{c}'] * RADIX).astype('int32')
    tenure_i = combined['tenure'].fillna(0).astype('float32').round().clip(lower=0, upper=RADIX - 1).astype('int32')
    for c in LOW_CARD_COLS:
        out2[f'tenure_radix_{c}'] = (tenure_i + code_df[f'code_{c}'] * RADIX).astype('int32')
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


def build_features():
    n_train, n_test = len(read_train()), len(read_test())

    orig = pd.read_csv(DATA_DIR / ORIGINAL_FILE, dtype={'TotalCharges': 'string', 'MonthlyCharges': 'string'})
    orig = orig.rename(columns={'customerID': 'id'})
    clean_totalcharges_string_inplace(orig)
    orig = orig.loc[orig['TotalCharges'].notna()].reset_index(drop=True)
    add_decimal_parts_like_eda(orig)
    ensure_numeric_columns(orig, ['tenure', 'MonthlyCharges', 'TotalCharges'])
    for c in ['MonthlyCharges', 'TotalCharges']:
        orig[f'{c}_snap'] = coerce_numeric_series(orig[c])
        orig[f'{c}_snap_diff'] = np.float32(0.0)

    combo_path = cache_dir('snap_features') / 'combo.csv'
    assert combo_path.exists(), f'{combo_path} not found: run ../snap_features first'
    combo = pd.read_csv(combo_path, dtype=COMBO_DTYPES)
    assert len(combo) == n_train + n_test, 'combo.csv must hold the train + test rows'
    ensure_numeric_columns(combo, ['tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyCharges_snap',
                                   'TotalCharges_snap'])

    train = combo.iloc[:n_train].copy().reset_index(drop=True)
    test = combo.iloc[n_train:n_train + n_test].copy().reset_index(drop=True)
    train[TARGET] = to_bin_target(train[TARGET]).astype('int8')
    y = train[TARGET].values
    y_orig = to_bin_target(orig[TARGET])

    for df in (train, test, orig):
        safe_string_df(df, BASE_CATS + DECIMAL_STRING_COLS)
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyCharges_snap', 'TotalCharges_snap',
                    'MonthlyCharges_snap_diff', 'TotalCharges_snap_diff']
    ensure_numeric_columns(train, numeric_cols)
    ensure_numeric_columns(test, numeric_cols)

    combined = pd.concat([train.drop(columns=[TARGET]), test.drop(columns=[TARGET]), orig.drop(columns=[TARGET])],
                         axis=0, ignore_index=True)
    log(f'combined rows / columns: {combined.shape}')

    blocks = [add_freq_rank_features(combined, BASE_NUMS),
              add_num_transforms(combined, BASE_NUMS + ['MonthlyCharges_snap', 'TotalCharges_snap']),
              add_charge_relationships(combined),
              add_service_counts(combined),
              add_yes_no_other_flags(combined)]
    block_bins = add_bins_and_cross_cats(combined)
    blocks.append(block_bins)
    with_bins = pd.concat([combined, block_bins], axis=1)

    cat_for_counts = dedup_keep_order(BASE_CATS + DECIMAL_STRING_COLS
                                      + ['tenure_bin', 'MonthlyCharges_bin', 'TotalCharges_bin'])
    blocks.append(add_cat_counts_and_rare(with_bins, cat_for_counts, rare_thresh=CAT_RARE_THRESH))
    blocks.append(add_drift_ratio_from_orig(with_bins, n_train, n_test,
                                            dedup_keep_order(BASE_CATS + ['Contract', 'PaymentMethod',
                                                                          'InternetService'])))
    orig_proba_cols = dedup_keep_order(BASE_CATS + ['tenure_bin', 'MonthlyCharges_bin', 'TotalCharges_bin',
                                                    'MonthlyCharges_snap', 'TotalCharges_snap'])
    blocks.append(add_orig_proba_features(with_bins, n_train, n_test, y_orig, orig_proba_cols))
    blocks.append(add_round_features(combined, ROUND_SPECS))
    blocks.append(add_digit_position_features(combined, DIGIT_BASE_COLS, max_dec_digits=2))
    blocks.append(add_lgbm_port_features(with_bins, n_train, n_test, y_orig))
    blocks.append(add_radix_features(combined))
    pair_block = {}
    for cols in PAIR_FEATURES:
        s = make_pair_feature_from_combined(with_bins, cols, pair_max_unique=PAIR_MAX_UNIQUE)
        pair_block[s.name] = s
    blocks.append(pd.DataFrame(pair_block, index=combined.index))

    X_all = pd.concat([combined] + blocks, axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()].copy()
    log(f'X_all shape after feature engineering: {X_all.shape}; *_te keys: '
        f'{sum(c.endswith("_te") for c in X_all.columns)}')
    X_train_raw = X_all.iloc[:n_train].reset_index(drop=True)
    X_test_raw = X_all.iloc[n_train:n_train + n_test].reset_index(drop=True)
    return X_train_raw, X_test_raw, y


# ---------------------------------------------------------------------------------------------------------------------
# Fold encodings
# ---------------------------------------------------------------------------------------------------------------------
def nested_target_encode(train_raw, others, y, te_cols, inner_folds):
    """TargetEncoder means of every te_cols column: out-of-fold over inner folds for the training rows, fitted on all
    training rows for each frame in `others`."""
    y = np.asarray(y)
    new_train = pd.DataFrame(index=train_raw.index)
    new_others = [pd.DataFrame(index=o.index) for o in others]
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=SEED)
    for c in te_cols:
        oof = np.zeros(len(train_raw), dtype=np.float32)
        s_train = train_raw[c].astype('string').fillna('NA')
        for itr, iva in inner_cv.split(s_train, y):
            enc = TargetEncoder(cv=2, smooth='auto', target_type='binary')
            enc.fit(s_train.iloc[itr].to_frame(c), y[itr])
            oof[iva] = np.asarray(enc.transform(s_train.iloc[iva].to_frame(c)), dtype=np.float32)[:, 0]
        enc = TargetEncoder(cv=2, smooth='auto', target_type='binary')
        enc.fit(s_train.to_frame(c), y)
        new_name = c.replace('_te', '_TE') if c.endswith('_te') else f'{c}_TE'
        new_train[new_name] = oof
        for o, n in zip(others, new_others):
            n[new_name] = np.asarray(enc.transform(o[c].astype('string').fillna('NA').to_frame(c)),
                                     dtype=np.float32)[:, 0]
    return new_train, new_others


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


def run_cv(X_train_raw, X_test_raw, y, n_folds, inner_folds):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_train_raw), dtype=np.float32)
    pred = np.zeros(len(X_test_raw), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_raw, y), 1):
        X_tr = X_train_raw.iloc[tr_idx].reset_index(drop=True).copy()
        X_va = X_train_raw.iloc[va_idx].reset_index(drop=True).copy()
        X_te = X_test_raw.reset_index(drop=True).copy()
        y_tr, y_va = y[tr_idx], y[va_idx]

        te_cols = [c for c in X_tr.columns if c.endswith('_te')]
        if te_cols:
            tr_te, (va_te, te_te) = nested_target_encode(X_tr[te_cols], [X_va[te_cols], X_te[te_cols]], y_tr,
                                                         te_cols, inner_folds)
            X_tr = pd.concat([X_tr.drop(columns=te_cols), tr_te], axis=1)
            X_va = pd.concat([X_va.drop(columns=te_cols), va_te], axis=1)
            X_te = pd.concat([X_te.drop(columns=te_cols), te_te], axis=1)

        X_tr, X_va = final_factorize(X_tr, X_va)
        _, X_te = final_factorize(X_tr, X_te)
        X_tr = X_tr.loc[:, ~X_tr.columns.duplicated()].copy()
        X_va = X_va[X_tr.columns].copy()
        X_te = X_te[X_tr.columns].copy()

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=500)
        p_va = model.predict_proba(X_va)[:, 1].astype(np.float32)
        oof[va_idx] = p_va
        pred += model.predict_proba(X_te)[:, 1].astype(np.float32) / n_folds
        log(f'{n_folds} folds, fold {fold}: AUC {roc_auc_score(y_va, p_va):.6f}, best_iteration '
            f'{getattr(model, "best_iteration", None)}, {X_tr.shape[1]} features')
        del X_tr, X_va, X_te, model
        gc.collect()
    log(f'{n_folds} folds: OOF AUC {roc_auc_score(y, oof):.6f}')
    return oof, pred


def main():
    X_train_raw, X_test_raw, y = build_features()
    oof, _ = run_cv(X_train_raw, X_test_raw, y, N_FOLDS, INNER_FOLDS)
    _, test_pred = run_cv(X_train_raw, X_test_raw, y, TEST_FOLDS, TEST_INNER_FOLDS)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
