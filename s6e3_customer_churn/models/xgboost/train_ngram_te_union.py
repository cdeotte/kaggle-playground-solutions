#!/usr/bin/env python
"""GPU XGBoost with n-gram / binned-key target encodings and a wide label-free feature union
(pair xgb_ngram_te_binned_keys_drift).

Label-free numeric features (train + test rows; missing TotalCharges get the frame's median):
- frequency over train + original + test rows, percentile rank over train + test + original rows, log1p / sqrt /
  1 / (1 + x) / missing flag of the 3 numerics;
- charge relations: TotalCharges - tenure x MonthlyCharges and its absolute value, monthly / total and total / monthly
  ratios, TotalCharges / (tenure + 1), tenure x MonthlyCharges, tenure x TotalCharges;
- service counts (Yes / No / no-service), has-internet / has-phone, Yes / No / other flags of 11 service columns;
- original churn rate of every value of the 16 categoricals and the 3 numerics (unseen: 0.5);
- TotalCharges against the original rows (percentile ranks among churners / non-churners / all and within the same
  InternetService / Contract, z-scores and gaps, MonthlyCharges minus the original InternetService mean, distances to
  the original churner / non-churner 25 / 50 / 75 % quantiles and their gaps);
- 35 digit features (first / second / last digits, mod 10 / 12 / 100, multiples, roundings, fractional parts) and the
  digit positions of tenure, MonthlyCharges and TotalCharges written with 6 decimals (the zero-padded integer digits,
  at most 4, and the first 4 decimals);
- counts over train + test of the 3 numerics with their original counts, original churn rates and count ratios;
  counts and rare flags (count <= 50) of the 16 categoricals (with the original rows) and of every key below;
- codes of the 16 categoricals, the 3 numerics, MonthlyCharges / TotalCharges rounded to 1 / 0 / -1 decimals and their
  2-digit fractions (factorized over train + test + original rows): drift log1p(train share / original share),
  original churn rate and original count of each; mean of each numeric per value of each categorical (train rows).
58 string keys for target encoding: the 3 numerics as strings, 15 bigrams of Contract, InternetService,
PaymentMethod, OnlineSecurity, TechSupport, PaperlessBilling and 4 trigrams of the first four; per numeric (and
SeniorCitizen): round / 2, integer, 5,000-quantile bin, equal-width bin (5,000 / 8,000 / 200 bins) and log-scale bin;
tenure bins, 40 / 60 quantile bins of the charges (edges over train + test + original rows), 6 category pairs, the
Contract x InternetService x PaymentMethod triple, and the 6 pairs of the first 4 decimal digits of TotalCharges.
Per fold: sklearn TargetEncoder means (smooth auto) of every key, out-of-fold over an inner StratifiedKFold(5, shuffle,
random_state 42) of the training rows and fitted on all training rows for the validation / test rows; std / min / max of
the target per value of the 16 categoricals and the 3 numerics as strings (same inner folds). The 16 categoricals stay
as native XGBoost categories, the key strings are dropped.
Model: depth 5, lr 0.0063, subsample 0.81, colsample_bytree 0.32, min_child_weight 6, alpha 3.5017, lambda 1.2925,
gamma 0.79, up to 50,000 rounds, early stopping 500 on the validation fold, random_state 42.
Usage: python train_ngram_te_union.py"""
import gc
import itertools
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

PAIR = 'xgb_ngram_te_binned_keys_drift'
ID_COL = ID
N_FOLDS = 5
INNER_FOLDS = 5
TEST_FOLDS = 15
TEST_INNER_FOLDS = 10
RANDOM_SEED = 42

HIGH_CARD_THRESHOLD = 7
QCUT_BINS = 5000
CUT_BINS_MONTHLY = 5000
CUT_BINS_TOTAL = 8000

XGB_PARAMS = {
    'n_estimators': 50000,
    'learning_rate': 0.0063,
    'max_depth': 5,
    'subsample': 0.81,
    'colsample_bytree': 0.32,
    'min_child_weight': 6,
    'reg_alpha': 3.5017,
    'reg_lambda': 1.2925,
    'gamma': 0.790,
    'random_state': RANDOM_SEED,
    'early_stopping_rounds': 500,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'enable_categorical': True,
    'device': 'cuda',
    'verbosity': 0,
}

TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling',
]
CATS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
]
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
SERVICE_COLS = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
]
YN_COLS = [
    'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'MultipleLines',
]
STATS = ['std', 'min', 'max']


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ helpers
def sanitize_series(s, fill='missing'):
    return s.astype('string').fillna(fill).str.strip()


def sanitize_object_series(s, fill='missing'):
    s = s.astype('object')
    return s.where(pd.notna(s), fill)


def to_xgb_category(s, fill='missing'):
    return sanitize_object_series(s, fill=fill).astype('category')


def pctrank_against(values, reference):
    reference = np.asarray(reference)
    if len(reference) == 0:
        return np.zeros(len(values), dtype='float32')
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')


def zscore_against(values, reference):
    reference = np.asarray(reference)
    if len(reference) == 0:
        return np.zeros(len(values), dtype='float32')
    mu = np.mean(reference)
    sigma = np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype='float32')
    return ((values - mu) / sigma).astype('float32')


def safe_qcut_str(series, q, default='NA'):
    x = pd.to_numeric(series, errors='coerce')
    try:
        q_eff = int(min(q, max(2, x.nunique(dropna=True))))
        return pd.qcut(x, q=q_eff, duplicates='drop').astype(str)
    except Exception:
        return pd.Series([default] * len(series), index=series.index, dtype='object')


def safe_cut_str(series, bins, default='NA'):
    x = pd.to_numeric(series, errors='coerce')
    try:
        return pd.cut(x, bins=bins, duplicates='drop').astype(str)
    except Exception:
        return pd.Series([default] * len(series), index=series.index, dtype='object')


def safe_log_cut_str(series, bins, default='NA'):
    x = pd.to_numeric(series, errors='coerce').clip(lower=0)
    try:
        return pd.cut(np.log1p(x), bins=max(200, bins // 2), duplicates='drop').astype(str)
    except Exception:
        return pd.Series([default] * len(series), index=series.index, dtype='object')


def factorize_on_all(train_s, test_s, orig_s):
    comb = pd.concat([train_s, test_s, orig_s], axis=0, ignore_index=True)
    codes, uniques = pd.factorize(comb, sort=False)
    n_tr = len(train_s)
    n_te = len(test_s)
    return (
        pd.Series(codes[:n_tr], index=train_s.index),
        pd.Series(codes[n_tr:n_tr + n_te], index=test_s.index),
        pd.Series(codes[n_tr + n_te:], index=orig_s.index),
        uniques,
    )


def coerce_numeric_except_categories(df, cat_cols):
    cat_cols = [c for c in cat_cols if c in df.columns]
    for c in [c for c in df.columns if c not in cat_cols]:
        if str(df[c].dtype) in ('object', 'string'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


# ------------------------------------------------------------------ data
def load_data():
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = labels(train[TARGET])
    orig[TARGET] = labels(orig[TARGET])
    for df in [train, test, orig]:
        for c in NUMS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        for c in CATS:
            df[c] = sanitize_object_series(df[c]).astype(str)
    return train, test, orig


# ------------------------------------------------------------------ features
def build_features(train, test, orig):
    train_base = train.drop(columns=[TARGET, ID_COL], errors='ignore').copy()
    test_base = test.drop(columns=[ID_COL], errors='ignore').copy()
    n_train = len(train_base)
    n_test = len(test_base)
    combined_tt = pd.concat([train_base.reset_index(drop=True), test_base.reset_index(drop=True)], axis=0,
                            ignore_index=True)

    log('[1/8] numeric features...')
    new_num_blocks = []

    freq_block = {}
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]], axis=0).value_counts(normalize=True)
        freq_block[f'FREQ_{col}'] = combined_tt[col].map(freq).fillna(0).astype('float32').values
    new_num_blocks.append(pd.DataFrame(freq_block))

    rank_block = {}
    all_num = pd.concat([train[NUMS], test[NUMS], orig[NUMS]], axis=0, ignore_index=True)
    for col in NUMS:
        r = all_num[col].rank(method='average', pct=True)
        rank_block[f'RANK_{col}'] = r.iloc[:n_train + n_test].astype('float32').values
        v = pd.to_numeric(combined_tt[col], errors='coerce').fillna(0).astype('float32')
        rank_block[f'LOG1P_{col}'] = np.log1p(v.clip(lower=0)).astype('float32').values
        rank_block[f'SQRT_{col}'] = np.sqrt(v.clip(lower=0)).astype('float32').values
        rank_block[f'INV1P_{col}'] = (1.0 / (1.0 + v.clip(lower=0))).astype('float32').values
        rank_block[f'MISS_{col}'] = pd.to_numeric(combined_tt[col], errors='coerce').isna().astype('int8').values
    new_num_blocks.append(pd.DataFrame(rank_block))

    tc = combined_tt['TotalCharges'].astype('float32')
    ten = combined_tt['tenure'].astype('float32')
    mc = combined_tt['MonthlyCharges'].astype('float32')
    dev = (tc - ten * mc).astype('float32')
    new_num_blocks.append(pd.DataFrame({
        'charges_deviation': dev,
        'abs_charges_deviation': np.abs(dev).astype('float32'),
        'monthly_to_total_ratio': (mc / (tc + 1)).astype('float32'),
        'total_to_monthly_ratio': (tc / (mc + 1)).astype('float32'),
        'avg_monthly_charges': (tc / (ten + 1)).astype('float32'),
        'tenure_x_monthly': (ten * mc).astype('float32'),
        'tenure_x_total': (ten * tc).astype('float32'),
    }))

    svc = combined_tt[SERVICE_COLS + ['InternetService']].copy()
    for c in svc.columns:
        svc[c] = sanitize_series(svc[c])
    new_num_blocks.append(pd.DataFrame({
        'service_count': (svc[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32'),
        'service_no_count': (svc[SERVICE_COLS] == 'No').sum(axis=1).astype('float32'),
        'service_other_count': svc[SERVICE_COLS].isin(['No phone service', 'No internet service']).sum(axis=1).astype('float32'),
        'has_internet': (svc['InternetService'] != 'No').astype('float32'),
        'has_phone': (svc['PhoneService'] == 'Yes').astype('float32'),
    }))

    yn_block = {}
    for c in YN_COLS:
        s = sanitize_series(combined_tt[c])
        yn_block[f'ISYES_{c}'] = (s == 'Yes').astype('float32').values
        yn_block[f'ISNO_{c}'] = (s == 'No').astype('float32').values
        yn_block[f'ISOTHER_{c}'] = (~s.isin(['Yes', 'No'])).astype('float32').values
    new_num_blocks.append(pd.DataFrame(yn_block))

    orig_prob_block = {}
    for col in CATS + NUMS:
        if col in orig.columns:
            tmp = orig.groupby(col, observed=False)[TARGET].mean()
            orig_prob_block[f'ORIG_proba_{col}'] = combined_tt[col].map(tmp).fillna(0.5).astype('float32').values
    new_num_blocks.append(pd.DataFrame(orig_prob_block))

    log('[2/8] distribution / quantile features...')
    orig_churner_tc = orig.loc[orig[TARGET] == 1, 'TotalCharges'].values
    orig_nonchurner_tc = orig.loc[orig[TARGET] == 0, 'TotalCharges'].values
    orig_tc = orig['TotalCharges'].values
    orig_is_mc_mean = orig.groupby('InternetService', observed=False)['MonthlyCharges'].mean()

    tc_tt = combined_tt['TotalCharges'].values
    dist_block = {
        'pctrank_nonchurner_TC': pctrank_against(tc_tt, orig_nonchurner_tc),
        'pctrank_churner_TC': pctrank_against(tc_tt, orig_churner_tc),
        'pctrank_orig_TC': pctrank_against(tc_tt, orig_tc),
        'zscore_churn_gap_TC': (
            np.abs(zscore_against(tc_tt, orig_churner_tc)) - np.abs(zscore_against(tc_tt, orig_nonchurner_tc))
        ).astype('float32'),
        'zscore_nonchurner_TC': zscore_against(tc_tt, orig_nonchurner_tc),
        'pctrank_churn_gap_TC': (
            pctrank_against(tc_tt, orig_churner_tc) - pctrank_against(tc_tt, orig_nonchurner_tc)
        ).astype('float32'),
        'resid_IS_MC': (
            combined_tt['MonthlyCharges'].values - combined_tt['InternetService'].map(orig_is_mc_mean).fillna(0).values
        ).astype('float32'),
    }
    cond_is = np.zeros(len(combined_tt), dtype='float32')
    for cat_val in orig['InternetService'].dropna().unique():
        mask = (combined_tt['InternetService'] == cat_val).values
        ref = orig.loc[orig['InternetService'] == cat_val, 'TotalCharges'].values
        if len(ref) > 0 and mask.sum() > 0:
            cond_is[mask] = pctrank_against(combined_tt.loc[mask, 'TotalCharges'].values, ref)
    dist_block['cond_pctrank_IS_TC'] = cond_is
    cond_c = np.zeros(len(combined_tt), dtype='float32')
    for cat_val in orig['Contract'].dropna().unique():
        mask = (combined_tt['Contract'] == cat_val).values
        ref = orig.loc[orig['Contract'] == cat_val, 'TotalCharges'].values
        if len(ref) > 0 and mask.sum() > 0:
            cond_c[mask] = pctrank_against(combined_tt.loc[mask, 'TotalCharges'].values, ref)
    dist_block['cond_pctrank_C_TC'] = cond_c

    qdist_block = {}
    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        ch_q = np.quantile(orig_churner_tc, q_val)
        nc_q = np.quantile(orig_nonchurner_tc, q_val)
        qdist_block[f'dist_To_ch_{q_label}'] = np.abs(combined_tt['TotalCharges'] - ch_q).astype('float32').values
        qdist_block[f'dist_To_nc_{q_label}'] = np.abs(combined_tt['TotalCharges'] - nc_q).astype('float32').values
        qdist_block[f'qdist_gap_To_{q_label}'] = (
            np.abs(combined_tt['TotalCharges'] - nc_q) - np.abs(combined_tt['TotalCharges'] - ch_q)
        ).astype('float32').values
    new_num_blocks.append(pd.DataFrame(dist_block))
    new_num_blocks.append(pd.DataFrame(qdist_block))

    log('[3/8] digit features...')
    t_str = combined_tt['tenure'].fillna(0).astype(int).astype(str)
    mc_floor = np.floor(combined_tt['MonthlyCharges'].fillna(0)).astype(int)
    tc_floor = np.floor(combined_tt['TotalCharges'].fillna(0)).astype(int)
    mc_str = combined_tt['MonthlyCharges'].fillna(0).astype(str).str.replace('.', '', regex=False)
    tc_str = combined_tt['TotalCharges'].fillna(0).astype(str).str.replace('.', '', regex=False)
    mcs, tcs, tens = combined_tt['MonthlyCharges'], combined_tt['TotalCharges'], combined_tt['tenure']
    new_num_blocks.append(pd.DataFrame({
        'tenure_first_digit': t_str.str[0].astype(int),
        'tenure_last_digit': t_str.str[-1].astype(int),
        'tenure_second_digit': t_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0).astype('int16'),
        'tenure_mod10': (tens % 10).astype('float32'),
        'tenure_mod12': (tens % 12).astype('float32'),
        'tenure_num_digits': t_str.str.len().astype('int16'),
        'tenure_is_multiple_10': (tens % 10 == 0).astype('float32'),
        'tenure_rounded_10': (np.round(tens / 10) * 10).astype('float32'),
        'tenure_dev_from_round10': np.abs(tens - np.round(tens / 10) * 10).astype('float32'),
        'mc_first_digit': mc_str.str[0].replace('-', '0').fillna('0').astype(int),
        'mc_last_digit': mc_str.str[-1].replace('-', '0').fillna('0').astype(int),
        'mc_second_digit': mc_str.apply(lambda x: int(x[1]) if len(x) > 1 and x[1].isdigit() else 0).astype('int16'),
        'mc_mod10': (mc_floor % 10).astype('float32'),
        'mc_mod100': (mc_floor % 100).astype('float32'),
        'mc_num_digits': mc_floor.astype(str).str.len().astype('int16'),
        'mc_is_multiple_10': (mc_floor % 10 == 0).astype('float32'),
        'mc_is_multiple_50': (mc_floor % 50 == 0).astype('float32'),
        'mc_rounded_10': (np.round(mcs / 10) * 10).astype('float32'),
        'mc_fractional': (mcs - np.floor(mcs)).astype('float32'),
        'mc_dev_from_round10': np.abs(mcs - np.round(mcs / 10) * 10).astype('float32'),
        'tc_first_digit': tc_str.str[0].replace('-', '0').fillna('0').astype(int),
        'tc_last_digit': tc_str.str[-1].replace('-', '0').fillna('0').astype(int),
        'tc_second_digit': tc_str.apply(lambda x: int(x[1]) if len(x) > 1 and x[1].isdigit() else 0).astype('int16'),
        'tc_mod10': (tc_floor % 10).astype('float32'),
        'tc_mod100': (tc_floor % 100).astype('float32'),
        'tc_num_digits': tc_floor.astype(str).str.len().astype('int16'),
        'tc_is_multiple_10': (tc_floor % 10 == 0).astype('float32'),
        'tc_is_multiple_100': (tc_floor % 100 == 0).astype('float32'),
        'tc_rounded_100': (np.round(tcs / 100) * 100).astype('float32'),
        'tc_fractional': (tcs - np.floor(tcs)).astype('float32'),
        'tc_dev_from_round100': np.abs(tcs - np.round(tcs / 100) * 100).astype('float32'),
        'tenure_years': (tens // 12).astype('float32'),
        'tenure_months_in_year': (tens % 12).astype('float32'),
        'mc_per_digit': (mcs / (mc_floor.astype(str).str.len().astype(float) + 0.001)).astype('float32'),
        'tc_per_digit': (tcs / (tc_floor.astype(str).str.len().astype(float) + 0.001)).astype('float32'),
    }))

    log('[4/8] digit positions...')
    digits_more = {}
    float_cols = [c for c in combined_tt.columns if pd.api.types.is_float_dtype(combined_tt[c])]
    for col in float_cols:
        ss = pd.to_numeric(combined_tt[col], errors='coerce').fillna(0).astype('float64').map(lambda x: f'{x:.6f}')
        splitted = ss.str.split('.', expand=True)
        non_decimal_part = splitted[0]
        decimal_part = splitted[1] if splitted.shape[1] > 1 else pd.Series(['0'] * len(ss), index=ss.index)
        max_len_non_decimal = int(non_decimal_part.str.len().max())
        max_len_decimal = int(decimal_part.str.len().max())
        non_decimal_part_padded = non_decimal_part.str.rjust(max_len_non_decimal, '0')
        decimal_part_padded = decimal_part.str.ljust(max_len_decimal, '0')
        for i in range(min(max_len_non_decimal, 4)):
            digits_more[f'{col}_digit{i}'] = non_decimal_part_padded.str[i].astype('int8').values
        for i in range(min(max_len_decimal, 4)):
            digits_more[f'{col}_decimal_digit{i}'] = decimal_part_padded.str[i].astype('int8').values
    if len(digits_more) > 0:
        new_num_blocks.append(pd.DataFrame(digits_more))

    log('[5/8] target-encoding keys...')
    te_source_blocks = {}
    raw_cat_keep = combined_tt[CATS].copy()

    for col in NUMS:
        te_source_blocks[f'CAT_{col}'] = combined_tt[col].astype(str)
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        te_source_blocks[f'BG_{c1}_{c2}'] = (
            sanitize_series(combined_tt[c1]) + '_' + sanitize_series(combined_tt[c2])
        ).astype(str)
    for c1, c2, c3 in combinations(TOP_CATS_FOR_NGRAM[:4], 3):
        te_source_blocks[f'TG_{c1}_{c2}_{c3}'] = (
            sanitize_series(combined_tt[c1]) + '_' + sanitize_series(combined_tt[c2]) + '_'
            + sanitize_series(combined_tt[c3])
        ).astype(str)

    for col in ['MonthlyCharges', 'TotalCharges', 'tenure', 'SeniorCitizen']:
        if col not in combined_tt.columns:
            continue
        s = combined_tt[col]
        te_source_blocks[f'{col}_round_div2_te'] = (pd.to_numeric(s, errors='coerce').round().fillna(-1) // 2).astype('int64').astype(str)
        te_source_blocks[f'{col}_int_te'] = pd.to_numeric(s, errors='coerce').fillna(-1).astype('int64').astype(str)
        te_source_blocks[f'{col}_quantiled_te'] = safe_qcut_str(s, QCUT_BINS).astype(str)
        bins = CUT_BINS_MONTHLY if col == 'MonthlyCharges' else (CUT_BINS_TOTAL if col == 'TotalCharges' else 200)
        te_source_blocks[f'{col}_binned_te'] = safe_cut_str(s, bins).astype(str)
        te_source_blocks[f'{col}_log_binned_te'] = safe_log_cut_str(s, bins).astype(str)

    ten = pd.to_numeric(combined_tt['tenure'], errors='coerce')
    te_source_blocks['tenure_bin_te'] = pd.cut(
        ten, bins=[0, 1, 3, 6, 12, 24, 36, 48, 60, 72, 10000], include_lowest=True
    ).astype(str)

    mc_all = pd.concat([train['MonthlyCharges'], test['MonthlyCharges'], orig['MonthlyCharges']], axis=0, ignore_index=True)
    try:
        mc_bins = pd.qcut(mc_all, q=40, retbins=True, duplicates='drop')[1]
        te_source_blocks['MonthlyCharges_bin_te'] = pd.cut(combined_tt['MonthlyCharges'], bins=mc_bins, include_lowest=True).astype(str)
    except Exception:
        te_source_blocks['MonthlyCharges_bin_te'] = pd.Series(['NA'] * len(combined_tt)).astype(str)
    tc_all = pd.concat([train['TotalCharges'], test['TotalCharges'], orig['TotalCharges']], axis=0, ignore_index=True)
    try:
        tc_bins = pd.qcut(tc_all, q=60, retbins=True, duplicates='drop')[1]
        te_source_blocks['TotalCharges_bin_te'] = pd.cut(combined_tt['TotalCharges'], bins=tc_bins, include_lowest=True).astype(str)
    except Exception:
        te_source_blocks['TotalCharges_bin_te'] = pd.Series(['NA'] * len(combined_tt)).astype(str)

    cross_pairs = [
        ('Contract', 'InternetService'),
        ('PaymentMethod', 'Contract'),
        ('InternetService', 'OnlineSecurity'),
        ('PaymentMethod', 'PaperlessBilling'),
        ('Contract', 'PaperlessBilling'),
        ('InternetService', 'TechSupport'),
    ]
    for a, b in cross_pairs:
        te_source_blocks[f'{a}__{b}_te'] = (sanitize_series(combined_tt[a]) + '|' + sanitize_series(combined_tt[b])).astype(str)
    te_source_blocks['Contract__InternetService__PaymentMethod_te'] = (
        sanitize_series(combined_tt['Contract']) + '|' + sanitize_series(combined_tt['InternetService']) + '|'
        + sanitize_series(combined_tt['PaymentMethod'])
    ).astype(str)

    candidate_digit_cols = [c for c in digits_more.keys() if c.startswith('TotalCharges_') and 'decimal_digit' in c][:6]
    if len(candidate_digit_cols) < 2:
        candidate_digit_cols = [c for c in digits_more.keys() if c.startswith('MonthlyCharges_') and 'decimal_digit' in c][:6]
    if len(candidate_digit_cols) >= 2:
        digits_df_tmp = pd.DataFrame(digits_more)[candidate_digit_cols].astype(str)
        for a, b in itertools.combinations(candidate_digit_cols, 2):
            te_source_blocks[f'DIGPAIR_{a}__{b}_te'] = (digits_df_tmp[a] + '|' + digits_df_tmp[b]).astype(str)

    te_source_df = pd.DataFrame(te_source_blocks).fillna('missing').astype(object)

    log('[6/8] counts, original sizes, ratios, rare flags...')
    count_num_blocks = []
    high_cardinality = [
        col for col in combined_tt.columns
        if col != ID_COL and combined_tt[col].nunique(dropna=False) > HIGH_CARD_THRESHOLD
    ]
    for col in high_cardinality:
        vc = combined_tt[col].value_counts(dropna=False)
        feat_name = f'count({col})'
        count_num_blocks.append(pd.DataFrame({feat_name: combined_tt[col].map(vc).fillna(0).astype('float32')}))
        if col in orig.columns:
            tmp_size = orig.groupby([col], dropna=False).size()
            count_num_blocks.append(pd.DataFrame({
                f'groupby({col})_size_original': combined_tt[col].map(tmp_size).fillna(0).astype('float32')}))
            tmp_mean = orig.groupby([col], dropna=False)[TARGET].mean()
            count_num_blocks.append(pd.DataFrame({
                f'groupby({col})_mean_churn_original': combined_tt[col].map(tmp_mean).fillna(0.5).astype('float32')}))
            numer = combined_tt[col].map(vc).fillna(0).astype('float32')
            denom = combined_tt[col].map(tmp_size).fillna(0).astype('float32').replace(0, np.nan)
            count_num_blocks.append(pd.DataFrame({f'{feat_name}_ratio': (numer / denom).fillna(0).astype('float32')}))

    cat_count_block = {}
    orig_cat_base = pd.DataFrame({c: sanitize_object_series(orig[c]) for c in CATS if c in orig.columns})
    for c in CATS:
        tt_s = sanitize_object_series(combined_tt[c])
        if c in orig_cat_base.columns:
            all_s = pd.concat([tt_s, orig_cat_base[c]], axis=0, ignore_index=True)
        else:
            all_s = tt_s.reset_index(drop=True)
        vc = all_s.value_counts(dropna=False)
        cnt = tt_s.map(vc).fillna(0).astype('float32')
        cat_count_block[f'CAT_CNT_{c}'] = cnt.values
        cat_count_block[f'CAT_RARE_{c}'] = (cnt <= 50).astype('float32').values
    for c in te_source_df.columns:
        tt_s = sanitize_object_series(te_source_df[c])
        vc = tt_s.value_counts(dropna=False)
        cnt = tt_s.map(vc).fillna(0).astype('float32')
        cat_count_block[f'CAT_CNT_{c}'] = cnt.values
        cat_count_block[f'CAT_RARE_{c}'] = (cnt <= 50).astype('float32').values
    count_num_blocks.append(pd.DataFrame(cat_count_block))

    log('[7/8] drift / original count / group mean features...')
    b_num_blocks = []
    b_train = train.drop(columns=[ID_COL], errors='ignore').copy()
    b_test = test.drop(columns=[ID_COL], errors='ignore').copy()
    b_orig = orig.copy()
    for df in [b_train, b_test, b_orig]:
        for c in CATS:
            df[c] = sanitize_object_series(df[c])

    b_numcat_cols = []
    for c in NUMS:
        n = f'B_NUMCAT_{c}'
        for df in [b_train, b_test, b_orig]:
            df[n] = pd.to_numeric(df[c], errors='coerce')
        b_numcat_cols.append(n)
    for r in [1, 0, -1]:
        n = f'B_MonthlyCharges_round_{r}'
        for df in [b_train, b_test, b_orig]:
            df[n] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').round(r)
        b_numcat_cols.append(n)
    for r in [1, 0, -1]:
        n = f'B_TotalCharges_round_{r}'
        for df in [b_train, b_test, b_orig]:
            df[n] = pd.to_numeric(df['TotalCharges'], errors='coerce').round(r)
        b_numcat_cols.append(n)
    for col in ['MonthlyCharges', 'TotalCharges']:
        n = f'B_{col}_frac2'
        for df in [b_train, b_test, b_orig]:
            x = pd.to_numeric(df[col], errors='coerce').astype('float64')
            df[n] = (np.floor((x * 100.0) % 100.0)).fillna(-1).astype('int16')
        b_numcat_cols.append(n)

    b_fact_cols = CATS + b_numcat_cols
    for c in b_fact_cols:
        tr_code, te_code, or_code, _ = factorize_on_all(b_train[c].astype(str), b_test[c].astype(str), b_orig[c].astype(str))
        b_train[c] = tr_code.values
        b_test[c] = te_code.values
        b_orig[c] = or_code.values

    for c in b_fact_cols:
        p_tr = b_train[c].value_counts(normalize=True)
        p_or = b_orig[c].value_counts(normalize=True)
        trv = np.log1p(b_train[c].map(p_tr).fillna(0) / (b_train[c].map(p_or).fillna(1e-6) + 1e-6)).astype('float32')
        tev = np.log1p(b_test[c].map(p_tr).fillna(0) / (b_test[c].map(p_or).fillna(1e-6) + 1e-6)).astype('float32')
        b_num_blocks.append(pd.DataFrame({f'B_DRIFT_{c}': pd.concat([trv, tev], axis=0, ignore_index=True)}))

    tmp_orig = b_orig.copy()
    tmp_orig['_y_'] = b_orig[TARGET].astype('float32')
    for c in b_fact_cols:
        m = tmp_orig.groupby(c, dropna=False)['_y_'].mean().astype('float32')
        vc = tmp_orig[c].value_counts(dropna=False)
        trv = b_train[c].map(m)
        tev = b_test[c].map(m)
        fillv = float(np.nanmean(trv.values))
        b_num_blocks.append(pd.DataFrame({f'B_TEORIG_{c}': pd.concat([
            trv.fillna(fillv).astype('float32'), tev.fillna(fillv).astype('float32')], axis=0, ignore_index=True)}))
        b_num_blocks.append(pd.DataFrame({f'B_CEORIG_{c}': pd.concat([
            b_train[c].map(vc).fillna(0).astype('float32'), b_test[c].map(vc).fillna(0).astype('float32')],
            axis=0, ignore_index=True)}))

    for g in CATS:
        for n in NUMS:
            m = train.groupby(g, dropna=False)[n].mean()
            trv = train[g].map(m)
            tev = test[g].map(m)
            fillv = float(np.nanmean(trv.values))
            b_num_blocks.append(pd.DataFrame({f'B_MEAN_{n}_BY_{g}': pd.concat([
                trv.fillna(fillv).astype('float32'), tev.fillna(fillv).astype('float32')], axis=0, ignore_index=True)}))

    log('[8/8] assembly...')
    num_df = pd.concat(new_num_blocks + count_num_blocks + b_num_blocks, axis=1)
    num_df = num_df.loc[:, ~num_df.columns.duplicated()].copy()
    train_num = num_df.iloc[:n_train].reset_index(drop=True)
    test_num = num_df.iloc[n_train:].reset_index(drop=True)
    train_cat = raw_cat_keep.iloc[:n_train].reset_index(drop=True).copy()
    test_cat = raw_cat_keep.iloc[n_train:].reset_index(drop=True).copy()
    train_te_src = te_source_df.iloc[:n_train].reset_index(drop=True).copy()
    test_te_src = te_source_df.iloc[n_train:].reset_index(drop=True).copy()
    for df in [train_cat, test_cat]:
        for c in df.columns:
            df[c] = to_xgb_category(df[c])

    train_final = pd.concat([train_num, train_cat, train_te_src, train[[TARGET]].reset_index(drop=True)], axis=1)
    test_final = pd.concat([test_num, test_cat, test_te_src], axis=1)
    num_features = train_num.columns.tolist()
    raw_cat_features = train_cat.columns.tolist()
    te_source_columns = train_te_src.columns.tolist()
    features = num_features + raw_cat_features + te_source_columns
    log(f'{len(num_features)} numeric, {len(raw_cat_features)} categorical, {len(te_source_columns)} key columns')
    return train_final, test_final, features, raw_cat_features, te_source_columns


# ------------------------------------------------------------------ training
def train_model(train_df, test_df, features, raw_cat_features, te_source_columns, n_folds, inner_folds):
    np.random.seed(RANDOM_SEED)
    skf_outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_SEED)

    y_all = train_df[TARGET].values
    xgb_oof = np.zeros(len(train_df), dtype='float32')
    xgb_pred = np.zeros(len(test_df), dtype='float32')
    for fold, (train_idx, val_idx) in enumerate(skf_outer.split(train_df, y_all), start=1):
        X_tr = train_df.loc[train_idx, features].reset_index(drop=True).copy()
        y_tr = train_df.loc[train_idx, TARGET].values
        X_val = train_df.loc[val_idx, features].reset_index(drop=True).copy()
        y_val = train_df.loc[val_idx, TARGET].values
        X_te = test_df[features].reset_index(drop=True).copy()

        # nested target-mean encoding of the keys
        te_mean_cols = [f'TE_{c}' for c in te_source_columns]
        X_tr_te = np.zeros((len(X_tr), len(te_source_columns)), dtype=np.float32)
        for in_tr, in_va in skf_inner.split(X_tr, y_tr):
            enc = TargetEncoder(cv=inner_folds, shuffle=True, smooth='auto', target_type='binary',
                                random_state=RANDOM_SEED)
            enc.fit(X_tr.iloc[in_tr][te_source_columns], y_tr[in_tr])
            X_tr_te[in_va, :] = enc.transform(X_tr.iloc[in_va][te_source_columns]).astype('float32')
        X_tr[te_mean_cols] = X_tr_te

        enc_full = TargetEncoder(cv=inner_folds, shuffle=True, smooth='auto', target_type='binary',
                                 random_state=RANDOM_SEED)
        enc_full.fit(X_tr[te_source_columns], y_tr)
        X_val[te_mean_cols] = enc_full.transform(X_val[te_source_columns]).astype('float32')
        X_te[te_mean_cols] = enc_full.transform(X_te[te_source_columns]).astype('float32')

        # nested std / min / max of the target on the categoricals and numerics-as-strings
        stat_te_cols = CATS + [c for c in te_source_columns if c.startswith('CAT_')]
        for in_tr, in_va in skf_inner.split(X_tr, y_tr):
            X_tr2 = X_tr.loc[in_tr, stat_te_cols].copy()
            X_va2 = X_tr.loc[in_va, stat_te_cols].copy()
            tmp_df = X_tr2.copy()
            tmp_df[TARGET] = y_tr[in_tr]
            for col in stat_te_cols:
                tmp = tmp_df.groupby(col, observed=False)[TARGET].agg(STATS)
                tmp.columns = [f'TE1_{col}_{s}' for s in STATS]
                X_m = X_va2.merge(tmp, on=col, how='left')
                for c in tmp.columns:
                    X_tr.loc[in_va, c] = X_m[c].fillna(0).astype('float32').values

        tmp_full_df = X_tr[stat_te_cols].copy()
        tmp_full_df[TARGET] = y_tr
        for col in stat_te_cols:
            tmp = tmp_full_df.groupby(col, observed=False)[TARGET].agg(STATS)
            tmp.columns = [f'TE1_{col}_{s}' for s in STATS]
            tmp = tmp.astype('float32')
            X_val = X_val.merge(tmp, on=col, how='left')
            X_te = X_te.merge(tmp, on=col, how='left')
            for c in tmp.columns:
                X_tr[c] = X_tr[c].fillna(0).astype('float32')
                X_val[c] = X_val[c].fillna(0).astype('float32')
                X_te[c] = X_te[c].fillna(0).astype('float32')

        for df in [X_tr, X_val, X_te]:
            for c in raw_cat_features:
                if c in df.columns:
                    df[c] = to_xgb_category(df[c])
        X_tr.drop(columns=te_source_columns, inplace=True, errors='ignore')
        X_val.drop(columns=te_source_columns, inplace=True, errors='ignore')
        X_te.drop(columns=te_source_columns, inplace=True, errors='ignore')
        X_tr = coerce_numeric_except_categories(X_tr, raw_cat_features)
        X_val = coerce_numeric_except_categories(X_val, raw_cat_features)
        X_te = coerce_numeric_except_categories(X_te, raw_cat_features)
        cols_xgb = X_tr.columns.tolist()

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=1000)
        p_val = model.predict_proba(X_val)[:, 1].astype('float32')
        p_te = model.predict_proba(X_te[cols_xgb])[:, 1].astype('float32')
        xgb_oof[val_idx] = p_val
        xgb_pred += p_te / n_folds
        log(f'{n_folds} folds, fold {fold} ({len(cols_xgb)} features): AUC {roc_auc_score(y_val, p_val):.6f}')

        del X_tr, X_val, X_te, y_tr, y_val, model, p_val, p_te
        gc.collect()

    log(f'{n_folds} folds: OOF AUC {roc_auc_score(y_all, xgb_oof):.6f}')
    return xgb_oof, xgb_pred


def main():
    train, test, orig = load_data()
    train_df, test_df, features, raw_cat_features, te_source_columns = build_features(train, test, orig)
    oof, _ = train_model(train_df, test_df, features, raw_cat_features, te_source_columns, N_FOLDS, INNER_FOLDS)
    _, test_pred = train_model(train_df, test_df, features, raw_cat_features, te_source_columns,
                               TEST_FOLDS, TEST_INNER_FOLDS)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
