#!/usr/bin/env python
"""GPU XGBoost on the union of two label-free feature pipelines (A, B) with fold-wise target encodings (4 presets).

Label-free features (built on the train, test and original rows; only the competition rows are trained on):
- Pipeline A (A_*): tenure / MonthlyCharges / TotalCharges as float32 (missing TotalCharges: the frame's median); their
  frequency over train + original + test rows, percentile rank over train + test + original rows, log1p, sqrt,
  1 / (1 + x) and missing flags; charge relations (TotalCharges - tenure x MonthlyCharges and its absolute value,
  MonthlyCharges / (TotalCharges + 1), TotalCharges / (MonthlyCharges + 1), TotalCharges / (tenure + 1),
  tenure x MonthlyCharges, tenure x TotalCharges); service counts (Yes / No / no-service), has-internet / has-phone,
  Yes / No / other flags of 11 columns; string keys: tenure bins (edges 0, 1, 3, 6, 12, 24, 36, 48, 60, 72), 40 / 60
  quantile bins of the charges, 6 category pairs and Contract x InternetService x PaymentMethod; counts and rare flags
  (count <= 50) of the 16 categoricals and these 10 keys; original churn rate of every value of the 16 categoricals,
  the 3 numerics and the 10 keys (unseen: 0.5); the 3 numerics as strings.
- Pipeline B (B_*): 5 quantile bins of the numerics (edges from the train rows), MonthlyCharges / TotalCharges rounded
  to tens and units and their digits from the hundreds down to the first / second decimal; 27 keys factorized over
  train + test + original rows (the 16 categoricals, the 3 numerics, the charges rounded to 1 / 0 / -1 decimals and
  their 2-digit fractions); drift log1p(train share / original share) of each key; original churn rate and original
  count of each key and digit; mean of each numeric per categorical (train rows); pairs of the 27 keys and
  MonthlyCharges rounded to tens, kept when they have <= 20,000 levels.
Per outer fold (out-of-fold over an inner StratifiedKFold(5, shuffle=True, random_state=42) for the training rows, all
training rows for the validation / test rows): target std / min / max per value of the 29 A keys (the 3 numeric
strings, the 16 categoricals, the 10 keys; unseen: 0); sklearn TargetEncoder means (cv 5, shuffle, smooth auto,
random_state 42) of the A keys and of the B keys and pairs, with the row-wise mean / std / min / max of the B encodings;
counts of the B pairs on the training rows. The raw keys are dropped after encoding.
Presets:
  xgb_union_ab_te                   pipelines A + B
  xgb_union_ab_total_anchor_te      + TotalCharges anchor block: TotalCharges in cents, the 16 category x anchor keys,
                                    6 anchor digits and the 10 pairs of the first 5, the 120 category pairs; per fold
                                    (inner KFold(5, shuffle=True, random_state=42)) target mean / std / count /
                                    nunique / median / min / max / skew and 5, 10, 40, 45, 55, 60, 90, 95 % quantiles
                                    per anchor value, the target mean of the 146 keys (unseen: 0), count / nunique and
                                    std / count ratios; colsample_bytree 0.8, up to 50,000 rounds
  xgb_union_ab_lgbm_port_te         + LGBM-port block: tenure / MonthlyCharges / TotalCharges as strings with their
                                    counts over train + test, original churn rate and count and count ratios; for the
                                    3 numerics and SeniorCitizen: round(x) // 2, integer, quantile bins (<= 5,000),
                                    equal-width bins (5,000 / 8,000 / 200) and log bins as strings; up to 4 integer and
                                    4 decimal digits of the charges (6-decimal strings) and the 15 pairs of the first 6
                                    decimal digits; per fold TargetEncoder means of these 38 string keys with their
                                    row-wise mean / std / min / max
  xgb_union_ab_lgbm_port_prebinned  as xgb_union_ab_lgbm_port_te, then every column except MonthlyCharges / TotalCharges
                                    with > 256 distinct training values is replaced by its 256-quantile bin (edges from
                                    the fold's training rows); max_bin 16,000
Model: depth 6, lr 0.03, subsample 0.8, colsample_bytree 0.4, min_child_weight 5, alpha 0.1, lambda 1, gamma 0.05, up to
1,000,000 rounds, early stopping 500 on the validation fold, random_state 42.
Usage: python train_union_pipelines_ab.py {xgb_union_ab_te,xgb_union_ab_total_anchor_te,xgb_union_ab_lgbm_port_te,
                                           xgb_union_ab_lgbm_port_prebinned}"""
import argparse
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PRESETS = {
    'xgb_union_ab_te': dict(test_learning_rate=0.01),
    'xgb_union_ab_total_anchor_te': dict(total_anchor=True, xgb=dict(n_estimators=50_000, colsample_bytree=0.8)),
    'xgb_union_ab_lgbm_port_te': dict(lgbm_port=True, test_learning_rate=0.005),
    'xgb_union_ab_lgbm_port_prebinned': dict(lgbm_port=True, prebin=True, xgb=dict(max_bin=16000)),
}

N_FOLDS = 5
INNER_FOLDS = 5
TEST_FOLDS = 15
TEST_INNER_FOLDS = 10
SEED = 42
LEARNING_RATE = 0.03

XGB_PARAMS = {
    'n_estimators': 1_000_000,
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
    'verbosity': 0,
    'random_state': SEED,
}

# Pipeline A
A_CATS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
          'Contract', 'PaperlessBilling', 'PaymentMethod']
A_NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']
TENURE_BINS = [0, 1, 3, 6, 12, 24, 36, 48, 60, 72, 10_000]
MC_BINS = 40
TC_BINS = 60
YN_COLS = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
CROSS_PAIRS = [('Contract', 'InternetService'), ('PaymentMethod', 'Contract'), ('InternetService', 'OnlineSecurity'),
               ('PaymentMethod', 'PaperlessBilling'), ('Contract', 'PaperlessBilling'),
               ('InternetService', 'TechSupport')]
TRIPLE = [('Contract', 'InternetService', 'PaymentMethod')]
A_STATS = ['std', 'min', 'max']

# Pipeline B
NUMS_CAND = ['tenure', 'MonthlyCharges', 'TotalCharges']
Q_LIST = [5]
DIGIT_SPECS = {'MonthlyCharges': list(range(-2, 2)), 'TotalCharges': list(range(-2, 3))}
ROUND_SPECS = {'MonthlyCharges': [-1, 0], 'TotalCharges': [-1, 0]}
DRIFT_EPS = 1e-6
PAIR_MAX_UNIQUE = 20000

# LGBM-port block
HIGH_CARD_THRESHOLD = 7
QCUT_BINS = 5000
CUT_BINS_MONTHLY = 5000
CUT_BINS_TOTAL = 8000

# TotalCharges anchor block
ANCHOR_COL = 'TotalCharges'
ANCHOR_CENTS = 100
MAX_PAIR_CATS = 25
ANCHOR_DIGITS_K = 6
STATS_TE_ANCHOR = ['mean', 'std', 'count', 'nunique', 'median', 'min', 'max', 'skew']
ANCHOR_QUANTILES = [5, 10, 40, 45, 55, 60, 90, 95]
TE2_MAX_COMBOS = 250

# Pre-binning
BIN_SMALL = 256
BIN_EXCLUDE_COLS = {'MonthlyCharges', 'TotalCharges'}


def log(msg):
    print(msg, flush=True)


def clean_total_charges(df):
    out = df.copy()
    if out['TotalCharges'].dtype == 'object' or str(out['TotalCharges'].dtype).startswith('string'):
        s = out['TotalCharges'].astype(str)
        out.loc[s.str.strip() == '', 'TotalCharges'] = np.nan
        out['TotalCharges'] = pd.to_numeric(out['TotalCharges'], errors='coerce')
    out['TotalCharges'] = out['TotalCharges'].astype('float32')
    return out


def safe_map(keys, mapping):
    return keys.map(mapping).astype('float32').to_numpy()


def attach(df, block):
    return pd.concat([df, pd.DataFrame(block, index=df.index)], axis=1)


# ---------------------------------------------------------------------------------------------------------------------
# Pipeline A
# ---------------------------------------------------------------------------------------------------------------------
def add_pipeline_a(train, test, orig):
    for df in (train, test, orig):
        for c in A_CATS:
            df[c] = df[c].astype(str).fillna('missing').str.strip()
        for c in A_NUMS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    for df in (train, test, orig):
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median()).astype('float32')

    new_cats = []
    for col in A_NUMS:
        freq = pd.concat([train[col], orig[col], test[col]], axis=0).value_counts(normalize=True)
        for df in (train, test, orig):
            df[f'A_FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')

    all_num = pd.concat([train[A_NUMS], test[A_NUMS], orig[A_NUMS]], axis=0, ignore_index=True)
    n_tr, n_te = len(train), len(test)
    for col in A_NUMS:
        r = all_num[col].rank(method='average', pct=True)
        train[f'A_RANK_{col}'] = r.iloc[:n_tr].astype('float32').values
        test[f'A_RANK_{col}'] = r.iloc[n_tr:n_tr + n_te].astype('float32').values
        orig[f'A_RANK_{col}'] = r.iloc[n_tr + n_te:].astype('float32').values

    for df in (train, test, orig):
        for col in A_NUMS:
            v = df[col].astype('float32')
            df[f'A_LOG1P_{col}'] = np.log1p(v.clip(lower=0)).astype('float32')
            df[f'A_SQRT_{col}'] = np.sqrt(v.clip(lower=0)).astype('float32')
            df[f'A_INV1P_{col}'] = (1.0 / (1.0 + v.clip(lower=0))).astype('float32')

    for df in (train, test, orig):
        df['A_charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['A_abs_charges_dev'] = np.abs(df['A_charges_deviation']).astype('float32')
        df['A_monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['A_total_to_monthly_ratio'] = (df['TotalCharges'] / (df['MonthlyCharges'] + 1)).astype('float32')
        df['A_avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
        df['A_tenure_x_monthly'] = (df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['A_tenure_x_total'] = (df['tenure'] * df['TotalCharges']).astype('float32')

    for df in (train, test, orig):
        for c in SERVICE_COLS + ['InternetService']:
            df[c] = df[c].astype(str).fillna('missing').str.strip()
        df['A_service_yes_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
        df['A_service_no_count'] = (df[SERVICE_COLS] == 'No').sum(axis=1).astype('float32')
        df['A_service_other_count'] = (df[SERVICE_COLS].isin(['No phone service', 'No internet service'])
                                       ).sum(axis=1).astype('float32')
        df['A_service_count'] = df['A_service_yes_count'].astype('float32')
        df['A_has_internet'] = (df['InternetService'] != 'No').astype('float32')
        df['A_has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')

    for df in (train, test, orig):
        df['A_tenure_bin'] = pd.cut(df['tenure'], bins=TENURE_BINS, include_lowest=True).astype(str)
    mc_bins = pd.qcut(pd.concat([train['MonthlyCharges'], test['MonthlyCharges'], orig['MonthlyCharges']]),
                      q=MC_BINS, retbins=True, duplicates='drop')[1]
    tc_bins = pd.qcut(pd.concat([train['TotalCharges'], test['TotalCharges'], orig['TotalCharges']]),
                      q=TC_BINS, retbins=True, duplicates='drop')[1]
    for df in (train, test, orig):
        df['A_MonthlyCharges_bin'] = pd.cut(df['MonthlyCharges'], bins=mc_bins, include_lowest=True).astype(str)
        df['A_TotalCharges_bin'] = pd.cut(df['TotalCharges'], bins=tc_bins, include_lowest=True).astype(str)
    new_cats += ['A_tenure_bin', 'A_MonthlyCharges_bin', 'A_TotalCharges_bin']

    for df in (train, test, orig):
        for col in A_NUMS:
            df[f'A_MISS_{col}'] = pd.to_numeric(df[col], errors='coerce').isna().astype('int8')

    for df in (train, test, orig):
        for c in YN_COLS:
            s = df[c].astype(str)
            df[f'A_ISYES_{c}'] = (s == 'Yes').astype('float32')
            df[f'A_ISNO_{c}'] = (s == 'No').astype('float32')
            df[f'A_ISOTHER_{c}'] = (~s.isin(['Yes', 'No'])).astype('float32')

    for a, b in CROSS_PAIRS:
        name = f'A_{a}__{b}'
        for df in (train, test, orig):
            df[name] = (df[a].astype(str) + '|' + df[b].astype(str)).astype(str)
        new_cats.append(name)
    for a, b, c in TRIPLE:
        name = f'A_{a}__{b}__{c}'
        for df in (train, test, orig):
            df[name] = (df[a].astype(str) + '|' + df[b].astype(str) + '|' + df[c].astype(str)).astype(str)
        new_cats.append(name)

    count_cols = A_CATS + new_cats
    all_cat = pd.concat([train[count_cols], test[count_cols], orig[count_cols]], axis=0, ignore_index=True)
    for c in count_cols:
        vc = all_cat[c].value_counts(dropna=False)
        for df in (train, test, orig):
            cnt = df[c].astype('string').map(vc)
            df[f'A_CAT_CNT_{c}'] = pd.to_numeric(cnt, errors='coerce').fillna(0).astype('float32')
            df[f'A_CAT_RARE_{c}'] = (df[f'A_CAT_CNT_{c}'] <= 50).astype('float32')

    for col in A_CATS + A_NUMS + new_cats:
        tmp = orig.groupby(col, observed=False)[TARGET].mean()
        name = f'A_ORIG_proba_{col}'
        train = train.merge(tmp.rename(name), on=col, how='left')
        test = test.merge(tmp.rename(name), on=col, how='left')
        orig = orig.merge(tmp.rename(name), on=col, how='left')
        for df in (train, test, orig):
            df[name] = df[name].fillna(0.5).astype('float32')

    num_as_cat = []
    for col in A_NUMS:
        n = f'A_CAT_{col}'
        for df in (train, test, orig):
            df[n] = df[col].astype(str)
        num_as_cat.append(n)
    return train, test, orig, num_as_cat + A_CATS + new_cats


# ---------------------------------------------------------------------------------------------------------------------
# Pipeline B
# ---------------------------------------------------------------------------------------------------------------------
def add_qcut_bins(train, test, orig, num_cols, q_list):
    for c in num_cols:
        for q in q_list:
            name = f'B_qcut_{c}_q{q}'
            try:
                b_train, bins = pd.qcut(train[c], q=q, labels=False, retbins=True, duplicates='drop')
                train[name] = b_train.astype('float32')
                test[name] = pd.cut(test[c], bins=bins, labels=False, include_lowest=True).astype('float32')
                orig[name] = pd.cut(orig[c], bins=bins, labels=False, include_lowest=True).astype('float32')
            except Exception:
                train[name] = 0.0
                test[name] = 0.0
                orig[name] = 0.0


def add_round_features(train, test, orig, round_specs):
    round_cols = []
    for c, rr in round_specs.items():
        for r in rr:
            n = f'B_round_{c}_r{r}'
            train[n] = train[c].round(r)
            test[n] = test[c].round(r)
            orig[n] = orig[c].round(r)
            round_cols.append(n)
    return round_cols


def add_digit_features(train, test, orig, digit_specs):
    digits = []
    for c, ks in digit_specs.items():
        for k in ks:
            n = f'B_digit_{c}_d{k}'
            for df in (train, test, orig):
                df[n] = (((df[c] * (10 ** k)) % 10).fillna(-1)).astype('int8')
            digits.append(n)
    return digits


def factorize_across_all(train, test, orig, cols):
    for c in cols:
        combined = pd.concat([train[c], test[c], orig[c]], ignore_index=True)
        codes, _ = combined.factorize(sort=False)
        train[c] = codes[:len(train)]
        test[c] = codes[len(train):len(train) + len(test)]
        orig[c] = codes[len(train) + len(test):]


def make_pair_feature(train, test, orig, cols, pair_max_unique=20000):
    name = 'B_pair_' + '__'.join(cols)
    for df in (train, test, orig):
        s = df[cols[0]].astype(str)
        for col in cols[1:]:
            s = s + '_' + df[col].astype(str)
        df[name] = s
    combined = pd.concat([train[name], test[name], orig[name]], ignore_index=True)
    codes, _ = combined.factorize(sort=False)
    if pd.Series(codes).nunique() > pair_max_unique:
        for df in (train, test, orig):
            df.drop(columns=[name], inplace=True)
        return None
    train[name] = codes[:len(train)]
    test[name] = codes[len(train):len(train) + len(test)]
    orig[name] = codes[len(train) + len(test):]
    return name


def add_orig_te_ce(train, test, orig, cols, y_orig):
    orig_tmp = orig.copy()
    orig_tmp['_y_orig_'] = y_orig.astype('float32')
    for c in cols:
        te_name = f'B_TEORIG_{c}'
        m = orig_tmp.groupby(c, observed=False)['_y_orig_'].mean().astype('float32')
        m.name = te_name
        train = train.merge(m, on=c, how='left')
        test = test.merge(m, on=c, how='left')
        fillv = float(train[te_name].mean())
        train[te_name] = train[te_name].fillna(fillv).astype('float32')
        test[te_name] = test[te_name].fillna(fillv).astype('float32')

        ce_name = f'B_CEORIG_{c}'
        vc = orig_tmp[c].value_counts()
        train[ce_name] = train[c].map(vc).fillna(0).astype('float32')
        test[ce_name] = test[c].map(vc).fillna(0).astype('float32')
    return train, test


def add_group_means(train, test, group_cols, num_cols):
    for g in group_cols:
        for n in num_cols:
            if g == n:
                continue
            name = f'B_MEAN_{n}_BY_{g}'
            tmp = train.groupby(g, observed=False)[n].mean().astype('float32')
            tmp.name = name
            train = train.merge(tmp, on=g, how='left')
            test = test.merge(tmp, on=g, how='left')
            fillv = float(train[name].mean())
            train[name] = train[name].fillna(fillv).astype('float32')
            test[name] = test[name].fillna(fillv).astype('float32')
    return train, test


def add_drift_ratio(train, test, orig, cols, eps=1e-6):
    for c in cols:
        p_tr = train[c].value_counts(normalize=True)
        p_or = orig[c].value_counts(normalize=True)
        name = f'B_DRIFT_{c}'
        for df in (train, test):
            df[name] = np.log1p(df[c].map(p_tr).fillna(0) / (df[c].map(p_or).fillna(eps) + eps)).astype('float32')


def add_pipeline_b(train, test, orig):
    y_orig = orig[TARGET].values.astype(np.int32)
    b_nums = [c for c in NUMS_CAND if c in train.columns and c in test.columns and c in orig.columns]
    b_cats = [c for c in train.columns if c not in [TARGET] + b_nums and c != ID]
    b_cats = [c for c in b_cats if not c.startswith('A_') and not c.startswith('B_')]

    add_qcut_bins(train, test, orig, b_nums, Q_LIST)
    b_round = add_round_features(train, test, orig, ROUND_SPECS)
    b_digits = add_digit_features(train, test, orig, DIGIT_SPECS)

    b_num_cat = []
    for c in b_nums:
        n = f'B_NUMCAT_{c}'
        train[n] = train[c].copy()
        test[n] = test[c].copy()
        orig[n] = orig[c].copy()
        b_num_cat.append(n)
    for c in ['MonthlyCharges', 'TotalCharges']:
        for r in [1, 0, -1]:
            n = f'B_{c}_round_{r}'
            train[n] = train[c].round(r)
            test[n] = test[c].round(r)
            orig[n] = orig[c].round(r)
            b_num_cat.append(n)
    for c in ['MonthlyCharges', 'TotalCharges']:
        n = f'B_{c}_frac2'
        for df in (train, test, orig):
            x = df[c].astype('float64')
            df[n] = (np.floor((x * 100.0) % 100.0)).fillna(-1).astype('int16')
        b_num_cat.append(n)

    for df in (train, test, orig):
        for c in b_cats:
            df[c] = df[c].astype(str).fillna('missing').str.strip()
    factorize_across_all(train, test, orig, b_cats + b_num_cat)
    add_drift_ratio(train, test, orig, b_cats + b_num_cat, eps=DRIFT_EPS)
    train, test = add_orig_te_ce(train, test, orig, b_cats + b_num_cat + b_digits, y_orig)
    train, test = add_group_means(train, test, group_cols=b_cats, num_cols=b_nums)

    pair_base = b_cats + b_num_cat + [b_round[0]]
    log(f'Building B pair features from {len(pair_base)} columns ...')
    b_pairs = []
    for cols in combinations(pair_base, 2):
        nm = make_pair_feature(train, test, orig, list(cols), pair_max_unique=PAIR_MAX_UNIQUE)
        if nm is not None:
            b_pairs.append(nm)
    log(f'Kept B pair features: {len(b_pairs)}')
    return train, test, orig, b_cats, b_num_cat, b_pairs


# ---------------------------------------------------------------------------------------------------------------------
# LGBM-port block
# ---------------------------------------------------------------------------------------------------------------------
def create_categorical_combs(X, list_of_col_tuples, delimiter='|'):
    combs = {}
    for cols in list_of_col_tuples:
        combs[delimiter.join(cols)] = X[list(cols)].astype(str).agg(delimiter.join, axis=1)
    return pd.DataFrame(combs)


def create_count_features(X, cols_list):
    out = {}
    for cols in [[c] if isinstance(c, str) else c for c in cols_list]:
        out[f"L_count({','.join(cols)})"] = X.groupby(list(cols), dropna=False).transform('size')
    return pd.DataFrame(out)


def create_groupby_features_original(X_keys, inputs, orig_data, target_col):
    out = {}
    for cols, agg in inputs:
        if agg == 'mean':
            mapping = orig_data.groupby(list(cols), dropna=False)[target_col].mean()
        else:
            mapping = orig_data.groupby(list(cols), dropna=False)[target_col].size()
        mapping = mapping.rename(agg)
        name = f"L_groupby({','.join(cols)})_{agg}_{target_col}_orig"
        out[name] = X_keys[list(cols)].merge(mapping, on=list(cols), how='left')[agg]
    return pd.DataFrame(out)


def add_lgbm_port_features(train, test, orig):
    base_cols = [c for c in train.columns
                 if c not in [TARGET, ID] and not c.startswith('A_') and not c.startswith('B_') and c in test.columns]
    comb = pd.concat([train[base_cols].reset_index(drop=True), test[base_cols].reset_index(drop=True)],
                     axis=0, ignore_index=True)
    n_tr, n_all = len(train), len(comb)
    for c in base_cols:
        if comb[c].dtype == 'object' or str(comb[c].dtype).startswith('string'):
            comb[c] = comb[c].astype(str).fillna('missing').str.strip()
    high_card = [c for c in base_cols if comb[c].nunique(dropna=False) > HIGH_CARD_THRESHOLD]

    blocks = {}
    if high_card:
        blocks['te1'] = comb[high_card].astype(str).add_prefix('L_').add_suffix('_te').reset_index(drop=True)
        blocks['ce1'] = create_count_features(comb, high_card).reset_index(drop=True)
        orig_tmp = orig.copy()
        for c in high_card:
            if c in orig_tmp.columns and (orig_tmp[c].dtype == 'object'
                                          or str(orig_tmp[c].dtype).startswith('string')):
                orig_tmp[c] = orig_tmp[c].astype(str).fillna('missing').str.strip()
        mean_inputs = [([c], 'mean') for c in high_card if c in orig_tmp.columns]
        size_inputs = [([c], 'size') for c in high_card if c in orig_tmp.columns]
        if mean_inputs:
            blocks['gmean'] = create_groupby_features_original(comb[high_card], mean_inputs, orig_tmp,
                                                               TARGET).reset_index(drop=True)
        if size_inputs:
            blocks['gsize'] = create_groupby_features_original(comb[high_card], size_inputs, orig_tmp,
                                                               TARGET).reset_index(drop=True)
        if 'ce1' in blocks and 'gsize' in blocks:
            ratios = {}
            for j, ccol in enumerate(blocks['ce1'].columns):
                denom = blocks['gsize'].iloc[:, j].astype('float32')
                numer = blocks['ce1'].iloc[:, j].astype('float32')
                ratios[f'{ccol}_ratio'] = (numer / denom.replace(0, np.nan)).fillna(0.0)
            blocks['ratios'] = pd.DataFrame(ratios).reset_index(drop=True)

    num_bins = {}
    for col in ['MonthlyCharges', 'TotalCharges', 'tenure', 'SeniorCitizen']:
        if col not in comb.columns:
            continue
        s = pd.to_numeric(comb[col], errors='coerce')
        num_bins[f'L_{col}_round_div2_te'] = (s.round().fillna(-1).astype('int64') // 2).astype(str)
        num_bins[f'L_{col}_int_te'] = s.fillna(-1).astype('int64').astype(str)
        try:
            q = int(min(QCUT_BINS, max(2, s.nunique(dropna=True))))
            num_bins[f'L_{col}_quantiled_te'] = pd.qcut(s, q=q, duplicates='drop').astype(str)
        except Exception:
            num_bins[f'L_{col}_quantiled_te'] = pd.Series(['NA'] * n_all)
        bins = CUT_BINS_MONTHLY if col == 'MonthlyCharges' else (CUT_BINS_TOTAL if col == 'TotalCharges' else 200)
        try:
            num_bins[f'L_{col}_binned_te'] = pd.cut(s, bins=bins, duplicates='drop').astype(str)
        except Exception:
            num_bins[f'L_{col}_binned_te'] = pd.Series(['NA'] * n_all)
        try:
            num_bins[f'L_{col}_log_binned_te'] = pd.cut(np.log1p(s.clip(lower=0)), bins=max(200, bins // 2),
                                                        duplicates='drop').astype(str)
        except Exception:
            num_bins[f'L_{col}_log_binned_te'] = pd.Series(['NA'] * n_all)
    if num_bins:
        blocks['num_bins'] = pd.DataFrame(num_bins).reset_index(drop=True)

    digits = {}
    for col in [c for c in ['MonthlyCharges', 'TotalCharges'] if c in comb.columns]:
        s = pd.to_numeric(comb[col], errors='coerce').fillna(0).astype('float64')
        sp = s.map(lambda x: f'{x:.6f}').str.split('.', expand=True)
        non = sp[0].astype(str)
        dec = sp[1].astype(str) if sp.shape[1] > 1 else pd.Series(['0'] * len(s))
        max_non, max_dec = int(non.str.len().max()), int(dec.str.len().max())
        nonp, decp = non.str.rjust(max_non, '0'), dec.str.ljust(max_dec, '0')
        for i in range(min(max_non, 4)):
            digits[f'L_{col}_digit{i}'] = nonp.str[i].astype(int)
        for i in range(min(max_dec, 4)):
            digits[f'L_{col}_decimal_digit{i}'] = decp.str[i].astype(int)
    digits = pd.DataFrame(digits).reset_index(drop=True)
    if digits.shape[1] > 0:
        blocks['digits'] = digits
        cand = [c for c in digits.columns if 'decimal_digit' in c][:6]
        if len(cand) >= 2:
            combs = create_categorical_combs(digits[cand].astype(str), list(combinations(cand, 2)))
            blocks['digits_combs'] = combs.add_prefix('L_').add_suffix('_te').reset_index(drop=True)

    comb_fe = pd.concat(list(blocks.values()), axis=1)
    tr_block = comb_fe.iloc[:n_tr].copy()
    te_block = comb_fe.iloc[n_tr:].copy()
    for c in tr_block.columns:
        if c.endswith('_te'):
            tr_block[c] = tr_block[c].astype(str).fillna('NA')
            te_block[c] = te_block[c].astype(str).fillna('NA')
        else:
            tr_block[c] = pd.to_numeric(tr_block[c], errors='coerce').fillna(0).astype('float32')
            te_block[c] = pd.to_numeric(te_block[c], errors='coerce').fillna(0).astype('float32')
    return tr_block, te_block, [c for c in tr_block.columns if c.endswith('_te')]


# ---------------------------------------------------------------------------------------------------------------------
# TotalCharges anchor block
# ---------------------------------------------------------------------------------------------------------------------
def make_anchor_int(s, cents=100):
    v = pd.to_numeric(s, errors='coerce').astype('float64')
    return pd.Series(np.rint(v * cents), index=s.index).fillna(-1).astype('int32')


def make_cat_anchor_key(cat_code, anchor_int, scale=5_000_000):
    offset = 2_000_000
    return ((cat_code.astype('int64') + 1) * scale + (anchor_int.astype('int64') + offset)).astype('int64')


def add_anchor_digit_cols(df, anchor_float_col, k=6):
    out_cols = []
    v = pd.to_numeric(df[anchor_float_col], errors='coerce').astype('float64').fillna(0.0)
    for i in range(1, k + 1):
        name = f'X_digit{i}'
        df[name] = ((v * (10 ** i)) % 10).astype('int8')
        out_cols.append(name)
    return out_cols


def add_digit_pair_cols(df, k_pairs_from=5):
    out = []
    for i in range(1, k_pairs_from):
        for j in range(i + 1, k_pairs_from + 1):
            name = f'X_digit_{i}_{j}'
            df[name] = ((df[f'X_digit{i}'].astype('int16') + 1) * 11 + (df[f'X_digit{j}'].astype('int16') + 1)
                        ).astype('int16')
            out.append(name)
    return out


def add_cat_pair_id_cols(train, test, cat_cols, max_pair_cats=25):
    use = cat_cols[:max_pair_cats]
    new_cols = []
    for i, c1 in enumerate(use[:-1]):
        c1_tr, c1_te = train[c1].astype('int64'), test[c1].astype('int64')
        for c2 in use[i + 1:]:
            name = f'X_pair_{c1}__{c2}'
            c2_tr, c2_te = train[c2].astype('int64'), test[c2].astype('int64')
            m2 = int(c2_tr.max() + 2)
            train[name] = ((c1_tr + 1) * m2 + (c2_tr + 1)).astype('int64')
            test[name] = ((c1_te + 1) * m2 + (c2_te + 1)).astype('int64')
            new_cols.append(name)
    return new_cols


def add_total_anchor_keys(train, test, cat_cols):
    train['X_anchor_int'] = make_anchor_int(train[ANCHOR_COL], cents=ANCHOR_CENTS)
    test['X_anchor_int'] = make_anchor_int(test[ANCHOR_COL], cents=ANCHOR_CENTS)
    combo_keys = []
    for c in cat_cols:
        name = f'X_{c}_anc'
        train[name] = make_cat_anchor_key(train[c], train['X_anchor_int'])
        test[name] = make_cat_anchor_key(test[c], test['X_anchor_int'])
        combo_keys.append(name)
    add_anchor_digit_cols(train, ANCHOR_COL, k=ANCHOR_DIGITS_K)
    add_anchor_digit_cols(test, ANCHOR_COL, k=ANCHOR_DIGITS_K)
    digit_pairs = add_digit_pair_cols(train, k_pairs_from=min(5, ANCHOR_DIGITS_K))
    add_digit_pair_cols(test, k_pairs_from=min(5, ANCHOR_DIGITS_K))
    catpair_keys = add_cat_pair_id_cols(train, test, cat_cols, max_pair_cats=MAX_PAIR_CATS)
    return (combo_keys + digit_pairs + catpair_keys)[:TE2_MAX_COMBOS]


def encode_total_anchor(X_tr, X_va, X_te, y_tr, te2_keys, inner_folds):
    anchor_key_tr = X_tr['X_anchor_int'].astype('int32')
    n = len(X_tr)
    te_anchor_store = {f'X_TE1_anchor_{s}': np.full(n, np.nan, dtype='float32') for s in STATS_TE_ANCHOR}
    te_q_store = {f'X_Q_{q}': np.full(n, np.nan, dtype='float32') for q in ANCHOR_QUANTILES}
    te2_store = {f'X_TE2_{c}_mean': np.full(n, np.nan, dtype='float32') for c in te2_keys}

    inner = KFold(n_splits=inner_folds, shuffle=True, random_state=SEED)
    for in_tr, in_va in inner.split(X_tr):
        y_in = y_tr[in_tr]
        g = pd.DataFrame({'k': anchor_key_tr.iloc[in_tr].values, '_y_': y_in})
        agg = g.groupby('k')['_y_'].agg(STATS_TE_ANCHOR)
        keys_va = anchor_key_tr.iloc[in_va]
        for s in STATS_TE_ANCHOR:
            te_anchor_store[f'X_TE1_anchor_{s}'][in_va] = safe_map(keys_va, agg[s])
        for q in ANCHOR_QUANTILES:
            te_q_store[f'X_Q_{q}'][in_va] = safe_map(keys_va, g.groupby('k')['_y_'].quantile(q / 100.0))
        for c in te2_keys:
            m = pd.DataFrame({'k': X_tr[c].values[in_tr], '_y_': y_in}).groupby('k')['_y_'].mean()
            te2_store[f'X_TE2_{c}_mean'][in_va] = safe_map(X_tr[c].iloc[in_va], m)
        del g, agg
        gc.collect()
    X_tr = attach(attach(attach(X_tr, te_anchor_store), te_q_store), te2_store)

    gfull = pd.DataFrame({'k': anchor_key_tr.values, '_y_': y_tr})
    agg_full = gfull.groupby('k')['_y_'].agg(STATS_TE_ANCHOR)
    va_block, te_block = {}, {}
    for s in STATS_TE_ANCHOR:
        va_block[f'X_TE1_anchor_{s}'] = safe_map(X_va['X_anchor_int'], agg_full[s])
        te_block[f'X_TE1_anchor_{s}'] = safe_map(X_te['X_anchor_int'], agg_full[s])
    for q in ANCHOR_QUANTILES:
        qmap = gfull.groupby('k')['_y_'].quantile(q / 100.0)
        va_block[f'X_Q_{q}'] = safe_map(X_va['X_anchor_int'], qmap)
        te_block[f'X_Q_{q}'] = safe_map(X_te['X_anchor_int'], qmap)
    for c in te2_keys:
        m = pd.DataFrame({'k': X_tr[c].values, '_y_': y_tr}).groupby('k')['_y_'].mean()
        va_block[f'X_TE2_{c}_mean'] = safe_map(X_va[c], m)
        te_block[f'X_TE2_{c}_mean'] = safe_map(X_te[c], m)
    X_va = attach(X_va, va_block)
    X_te = attach(X_te, te_block)

    frames = []
    for df in (X_tr, X_va, X_te):
        xcols = [c for c in df.columns if c.startswith('X_TE1_') or c.startswith('X_Q_') or c.startswith('X_TE2_')]
        df[xcols] = df[xcols].fillna(0).astype('float32')
        ratios = {
            'X_TE1_count_per_nunique': (df['X_TE1_anchor_count'] / df['X_TE1_anchor_nunique'].replace(0, np.nan)
                                        ).astype('float32').to_numpy(),
            'X_TE1_std_per_count': (df['X_TE1_anchor_std'] / df['X_TE1_anchor_count'].replace(0, np.nan)
                                    ).astype('float32').to_numpy(),
        }
        frames.append(attach(df, ratios))
    return frames


# ---------------------------------------------------------------------------------------------------------------------
# Pre-binning
# ---------------------------------------------------------------------------------------------------------------------
def make_bin_edges_from_train(x, n_bins):
    x = x.astype(np.float64, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0 or np.unique(x).size <= n_bins:
        return None
    edges = np.unique(np.nanquantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        return None
    return edges


def apply_edges(s, edges):
    x = pd.to_numeric(s, errors='coerce').to_numpy(dtype=np.float64, copy=False)
    out = np.full(x.shape, -1, dtype=np.int32)
    m = np.isfinite(x)
    out[m] = np.searchsorted(edges[1:-1], x[m], side='right').astype(np.int32)
    return out


def prebin_matrices(X_tr, X_va, X_te, n_bins, exclude_cols):
    for c in [c for c in X_tr.columns if c not in exclude_cols]:
        edges = make_bin_edges_from_train(X_tr[c].to_numpy(dtype=np.float64, copy=False), n_bins)
        for df in (X_tr, X_va, X_te):
            if edges is None:
                if df[c].dtype == 'float64':
                    df[c] = df[c].astype('float32')
            else:
                df[c] = apply_edges(df[c], edges).astype('int16')
    return X_tr, X_va, X_te


# ---------------------------------------------------------------------------------------------------------------------
# Features, folds
# ---------------------------------------------------------------------------------------------------------------------
def build_features(cfg):
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = train[TARGET].astype(str).str.strip().map({'No': 0, 'Yes': 1}).fillna(0).astype('int8')
    orig[TARGET] = orig[TARGET].astype(str).str.strip().map({'No': 0, 'Yes': 1}).fillna(0).astype('int8')
    y = train[TARGET].values.astype(np.int32)
    orig = clean_total_charges(orig)
    train = clean_total_charges(train)
    test = clean_total_charges(test)

    train, test, orig, a_te = add_pipeline_a(train, test, orig)
    train, test, orig, b_cats, b_num_cat, b_pairs = add_pipeline_b(train, test, orig)

    l_te, x_keys, x_te2 = [], [], []
    if cfg.get('lgbm_port'):
        tr_block, te_block, l_te = add_lgbm_port_features(train, test, orig)
        for c in tr_block.columns:
            train[c] = tr_block[c].values
            test[c] = te_block[c].values
        log(f'LGBM-port columns: {tr_block.shape[1]} ({len(l_te)} string keys)')
    if cfg.get('total_anchor'):
        x_te2 = add_total_anchor_keys(train, test, list(b_cats))
        x_keys = x_te2 + ['X_anchor_int']
        log(f'TotalCharges anchor keys: {len(x_te2)}')

    b_te = b_num_cat + b_cats + b_pairs
    features = [c for c in train.columns if c not in [TARGET, ID]]
    drop_keys = sorted(set(a_te + b_te + l_te + x_keys))
    log(f'{len(features)} base union features; A keys {len(a_te)}, B keys {len(b_te)}, B pairs {len(b_pairs)}')
    return dict(train=train, test=test, y=y, features=features, a_te=a_te, b_te=b_te, b_ce=list(b_pairs), l_te=l_te,
                x_te2=x_te2, drop_keys=drop_keys)


def target_encoder_block(prefix, cols, X_tr, X_va, X_te, y_tr, inner_folds):
    names = [f'{prefix}_{c}' for c in cols]
    enc = TargetEncoder(cv=inner_folds, shuffle=True, smooth='auto', target_type='binary', random_state=SEED)
    z_tr = enc.fit_transform(X_tr[cols], y_tr)
    z_va = enc.transform(X_va[cols])
    z_te = enc.transform(X_te[cols])
    return [attach(df, dict(zip(names, z.T))) for df, z in ((X_tr, z_tr), (X_va, z_va), (X_te, z_te))], names


def add_row_aggregates(frames, names, prefix):
    out = []
    for df in frames:
        block = {f'{prefix}_{agg}': df[names].agg(agg, axis=1).astype('float32')
                 for agg in ['mean', 'std', 'min', 'max']}
        out.append(attach(df, block))
    return out


def encode_fold(data, cfg, tr_idx, va_idx, inner_folds):
    train, test, y, feats = data['train'], data['test'], data['y'], data['features']
    X_tr = train.iloc[tr_idx][feats].reset_index(drop=True)
    X_va = train.iloc[va_idx][feats].reset_index(drop=True)
    X_te = test[feats].reset_index(drop=True)
    y_tr = y[tr_idx]

    # A keys: target std / min / max per value
    a_cols = data['a_te']
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=SEED)
    store = {f'A_TE1_{c}_{s}': np.full(len(X_tr), np.nan, dtype='float32') for c in a_cols for s in A_STATS}
    for in_tr, in_va in inner.split(X_tr, y_tr):
        for c in a_cols:
            frame = pd.DataFrame({c: X_tr[c].values[in_tr], '_y_': y_tr[in_tr]})
            st = frame.groupby(c, observed=False)['_y_'].agg(A_STATS)
            keys_va = X_tr[c].iloc[in_va]
            for s in A_STATS:
                store[f'A_TE1_{c}_{s}'][in_va] = safe_map(keys_va, st[s])
    va_block, te_block = {}, {}
    for c in a_cols:
        st = pd.DataFrame({c: X_tr[c].values, '_y_': y_tr}).groupby(c, observed=False)['_y_'].agg(A_STATS)
        for s in A_STATS:
            va_block[f'A_TE1_{c}_{s}'] = safe_map(X_va[c], st[s])
            te_block[f'A_TE1_{c}_{s}'] = safe_map(X_te[c], st[s])
    X_tr = attach(X_tr, {k: np.nan_to_num(v, nan=0.0) for k, v in store.items()})
    X_va = attach(X_va, {k: np.nan_to_num(v, nan=0.0) for k, v in va_block.items()})
    X_te = attach(X_te, {k: np.nan_to_num(v, nan=0.0) for k, v in te_block.items()})

    # A keys: target means
    (X_tr, X_va, X_te), _ = target_encoder_block('A_TE', a_cols, X_tr, X_va, X_te, y_tr, inner_folds)

    if cfg.get('total_anchor'):
        X_tr, X_va, X_te = encode_total_anchor(X_tr, X_va, X_te, y_tr, data['x_te2'], inner_folds)

    # B pairs: counts on the training rows
    ce_tr, ce_va, ce_te = {}, {}, {}
    for c in data['b_ce']:
        vc = X_tr[c].value_counts()
        ce_tr[f'B_CE_{c}'] = X_tr[c].map(vc).astype('float32')
        ce_va[f'B_CE_{c}'] = X_va[c].map(vc).fillna(0).astype('float32')
        ce_te[f'B_CE_{c}'] = X_te[c].map(vc).fillna(0).astype('float32')
    X_tr, X_va, X_te = attach(X_tr, ce_tr), attach(X_va, ce_va), attach(X_te, ce_te)

    # B keys and pairs: target means and their row-wise aggregates
    frames, names = target_encoder_block('B_TE', data['b_te'], X_tr, X_va, X_te, y_tr, inner_folds)
    X_tr, X_va, X_te = add_row_aggregates(frames, names, 'B_TEagg')

    if cfg.get('lgbm_port'):
        frames, names = target_encoder_block('L_TE', data['l_te'], X_tr, X_va, X_te, y_tr, inner_folds)
        X_tr, X_va, X_te = add_row_aggregates(frames, names, 'L_TEagg')

    drop_now = [c for c in data['drop_keys'] if c in X_tr.columns]
    X_tr = X_tr.drop(columns=drop_now)
    X_va = X_va.drop(columns=drop_now)
    X_te = X_te.drop(columns=drop_now)

    obj_cols = X_tr.select_dtypes(include=['object', 'string']).columns.tolist()
    for c in obj_cols:
        codes, _ = pd.concat([X_tr[c], X_va[c], X_te[c]], axis=0).astype(str).factorize(sort=False)
        X_tr[c] = codes[:len(X_tr)].astype('int32')
        X_va[c] = codes[len(X_tr):len(X_tr) + len(X_va)].astype('int32')
        X_te[c] = codes[len(X_tr) + len(X_va):].astype('int32')
    for df in (X_tr, X_va, X_te):
        for c in df.columns:
            if df[c].dtype == 'float64':
                df[c] = df[c].astype('float32')
            if df[c].dtype == 'int64':
                df[c] = df[c].astype('int32')

    if cfg.get('prebin'):
        keep_unbinned = {c for c in BIN_EXCLUDE_COLS if c in X_tr.columns}
        X_tr, X_va, X_te = prebin_matrices(X_tr, X_va, X_te, BIN_SMALL, keep_unbinned)
    return X_tr, X_va, X_te


def run_cv(data, cfg, n_folds, inner_folds, learning_rate):
    train, test, y = data['train'], data['test'], data['y']
    params = dict(XGB_PARAMS, **cfg.get('xgb', {}), learning_rate=learning_rate)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float64)
    pred = np.zeros(len(test), dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), start=1):
        X_tr, X_va, X_te = encode_fold(data, cfg, tr_idx, va_idx, inner_folds)
        model = XGBClassifier(**params)
        model.fit(X_tr, y[tr_idx], eval_set=[(X_va, y[va_idx])], verbose=False)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        pred += model.predict_proba(X_te[X_tr.columns])[:, 1]
        log(f'{n_folds} folds, fold {fold}: AUC {roc_auc_score(y[va_idx], oof[va_idx]):.6f}, '
            f'best_iteration {getattr(model, "best_iteration", None)}, {X_tr.shape[1]} features')
        del X_tr, X_va, X_te, model
        gc.collect()
    pred /= n_folds
    log(f'{n_folds} folds: OOF AUC {roc_auc_score(y, oof):.6f}')
    return oof, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preset', choices=sorted(PRESETS))
    args = ap.parse_args()
    cfg = PRESETS[args.preset]

    data = build_features(cfg)
    oof, _ = run_cv(data, cfg, N_FOLDS, INNER_FOLDS, LEARNING_RATE)
    _, test_pred = run_cv(data, cfg, TEST_FOLDS, TEST_INNER_FOLDS, cfg.get('test_learning_rate', LEARNING_RATE))
    save_pair(HERE, args.preset, oof, test_pred)


if __name__ == '__main__':
    main()
