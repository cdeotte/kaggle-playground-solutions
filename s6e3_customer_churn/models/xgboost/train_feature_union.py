#!/usr/bin/env python
"""GPU XGBoost on a union of label-free feature blocks plus nested target encodings
(pair xgb_union_snap_priors_drift_te).

Reads the snap columns of ../snap_features (combo.csv, merged on id); run that folder first.
Label-free blocks (built on the train, test and original rows; missing numerics get the frame's median):
- snaps: MonthlyCharges / TotalCharges snapped to the nearest value present in the original data (as strings) and
  snap - value (original rows: the value itself, difference 0);
- frequency and percentile rank over train + test + original rows, log1p, sqrt, 1 / (1 + x) and a missing flag of
  tenure, the two charges and the two snap differences;
- charge relations: TotalCharges - tenure x MonthlyCharges and its absolute value, MonthlyCharges / (TotalCharges + 1),
  TotalCharges / (MonthlyCharges + 1), TotalCharges / (tenure + 1), tenure x MonthlyCharges, tenure x TotalCharges;
- service counts (Yes / No / no-service), has-internet / has-phone, Yes / No / other flags of 11 service columns;
- tenure bins (edges 0, 1, 3, 6, 12, 24, 36, 48, 60, 72), 40 / 60 rank-quantile bins of MonthlyCharges / TotalCharges,
  6 category pairs and the Contract x InternetService x PaymentMethod triple;
- roundings (tenure to 0, MonthlyCharges to 0 / 1, TotalCharges to 0 / -1 decimals), 16 / 32 / 64-quantile bins (edges
  from the train rows), the last integer digit and the 2 decimal digits of the 3 numerics;
- counts over train + test + original rows and rare flags (count <= 25) of every string column;
- original data: seen-in-original flag, churn rate and count of 10 columns (Contract, InternetService, PaymentMethod,
  OnlineSecurity, TechSupport, the 3 bins, the 2 snaps), churn rate / count / seen flag of the 4 pairs (each snap x
  Contract / InternetService), drift log1p(train share / original share) of 8 columns.
Per fold: sklearn TargetEncoder (smooth auto) of 18 columns (5 categoricals, 2 snaps, 3 bins, 3 pairs and the triple,
2 roundings, 2 quantile bins), out-of-fold over an inner StratifiedKFold(5, shuffle, random_state 42 + fold number) for
the training rows and fitted on all training rows for the validation / test rows; then all string columns are
factorized and missing values set to -1.
Model: depth 5, lr 0.025, min_child_weight 5, subsample 0.8, colsample_bytree 0.45, alpha 0.1, lambda 1, gamma 0.05,
max_bin 16,000, up to 20,000 rounds, early stopping 300 on the validation fold, random_state 42.
Usage: python train_feature_union.py"""
import gc
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
from common.data import ID, TARGET, cache_dir, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'xgb_union_snap_priors_drift_te'
ID_COL = ID
SEED = 42
N_SPLITS = 5
INNER_FOLDS = 5

RARE_MIN_COUNT = 25
QCUT_BINS_SMALL = [16, 32, 64]
MC_BINS = 40
TC_BINS = 60

XGB_PARAMS = dict(
    n_estimators=20000,
    learning_rate=0.025,
    max_depth=5,
    min_child_weight=5,
    subsample=0.80,
    colsample_bytree=0.45,
    reg_alpha=0.10,
    reg_lambda=1.00,
    gamma=0.05,
    max_bin=16000,
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    device='cuda',
    early_stopping_rounds=300,
    random_state=SEED,
    enable_categorical=False,
    verbosity=0,
)

CROSS_PAIRS = [
    ('Contract', 'InternetService'),
    ('PaymentMethod', 'Contract'),
    ('InternetService', 'OnlineSecurity'),
    ('PaymentMethod', 'PaperlessBilling'),
    ('Contract', 'PaperlessBilling'),
    ('InternetService', 'TechSupport'),
]

TRIPLE_CATS = [
    ('Contract', 'InternetService', 'PaymentMethod'),
]

SERVICE_COLS = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
]

YN_COLS = [
    'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'MultipleLines',
]

PAIR_PRIOR_COLS = [
    ('MonthlyCharges_snap', 'Contract'),
    ('MonthlyCharges_snap', 'InternetService'),
    ('TotalCharges_snap', 'Contract'),
    ('TotalCharges_snap', 'InternetService'),
]

TE_COLS_BASE = [
    'Contract',
    'InternetService',
    'PaymentMethod',
    'OnlineSecurity',
    'TechSupport',
    'MonthlyCharges_snap',
    'TotalCharges_snap',
    'tenure_bin',
    'MonthlyCharges_bin',
    'TotalCharges_bin',
    'Contract__InternetService',
    'PaymentMethod__Contract',
    'InternetService__OnlineSecurity',
    'Contract__InternetService__PaymentMethod',
    'MonthlyCharges_round0',
    'TotalCharges_round0',
    'MonthlyCharges_qcut_32',
    'TotalCharges_qcut_32',
]


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ helpers
def safe_str_series(s):
    return s.astype('string').fillna('<NA>').str.strip()


def safe_str_inplace(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = safe_str_series(df[c])


def dedupe_columns_keep_first(df):
    return df.loc[:, ~df.columns.duplicated()].copy()


def factorize_train_valid_test(tr, va, te, cols):
    tr = tr.copy()
    va = va.copy()
    te = te.copy()
    for c in cols:
        allv = pd.concat([tr[c].astype('string'), va[c].astype('string'), te[c].astype('string')], ignore_index=True)
        codes, _ = pd.factorize(allv, sort=False)
        n_tr = len(tr)
        n_va = len(va)
        tr[c] = codes[:n_tr].astype('int32')
        va[c] = codes[n_tr:n_tr + n_va].astype('int32')
        te[c] = codes[n_tr + n_va:].astype('int32')
    return tr, va, te


def make_pair_key(df, c1, c2):
    s1 = df[c1].astype('string').fillna('<NA>')
    s2 = df[c2].astype('string').fillna('<NA>')
    return (s1 + '||' + s2).astype('string')


def _te_to_1d_float32(x):
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0].astype(np.float32).to_numpy()
    if isinstance(x, pd.Series):
        return x.astype(np.float32).to_numpy()
    x = np.asarray(x)
    if x.ndim == 1:
        return x.astype(np.float32)
    if x.ndim == 2:
        return x[:, 0].astype(np.float32)
    raise ValueError(f'unexpected transform output ndim={x.ndim}')


# ------------------------------------------------------------------ snap features
def add_snap_features(train_df, test_df):
    use_cols = [ID_COL, 'MonthlyCharges_snap', 'MonthlyCharges_snap_diff', 'TotalCharges_snap', 'TotalCharges_snap_diff']
    combo = pd.read_csv(cache_dir('snap_features') / 'combo.csv', usecols=lambda c: c in use_cols)
    train_df = train_df.merge(combo[use_cols], on=ID_COL, how='left')
    test_df = test_df.merge(combo[use_cols], on=ID_COL, how='left')
    return train_df, test_df


# ------------------------------------------------------------------ feature blocks
def add_global_rank_and_freq(train_df, test_df, orig_df, num_cols):
    all_num = pd.concat([train_df[num_cols], test_df[num_cols], orig_df[num_cols]], axis=0, ignore_index=True)

    add_tr, add_te, add_or = {}, {}, {}
    for c in num_cols:
        freq = all_num[c].value_counts(normalize=True, dropna=False)
        add_tr[f'FREQ_{c}'] = train_df[c].map(freq).fillna(0).astype('float32').values
        add_te[f'FREQ_{c}'] = test_df[c].map(freq).fillna(0).astype('float32').values
        add_or[f'FREQ_{c}'] = orig_df[c].map(freq).fillna(0).astype('float32').values

    for c in num_cols:
        r = all_num[c].rank(method='average', pct=True)
        n_tr, n_te = len(train_df), len(test_df)
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
            v = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('float32')
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
        t = pd.to_numeric(df['tenure'], errors='coerce').fillna(0).astype('float32')
        mc = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0).astype('float32')
        tc = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0).astype('float32')
        out = {}
        out['charges_deviation'] = (tc - t * mc).astype('float32')
        out['abs_charges_dev'] = np.abs(out['charges_deviation']).astype('float32')
        out['monthly_to_total_ratio'] = (mc / (tc + 1.0)).astype('float32')
        out['total_to_monthly_ratio'] = (tc / (mc + 1.0)).astype('float32')
        out['avg_monthly_charges'] = (tc / (t + 1.0)).astype('float32')
        out['tenure_x_monthly'] = (t * mc).astype('float32')
        out['tenure_x_total'] = (t * tc).astype('float32')
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
            df[SERVICE_COLS].isin(['No phone service', 'No internet service'])
        ).sum(axis=1).astype('float32')
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
        safe_str_inplace(df, YN_COLS)

    def build(df):
        out = {}
        for c in YN_COLS:
            s = df[c].astype('string').fillna('<NA>')
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
        df['tenure_bin'] = pd.cut(
            pd.to_numeric(df['tenure'], errors='coerce').fillna(-1),
            bins=tenure_bins,
            include_lowest=True,
        ).astype('string')

    all_mc = pd.concat([train_df['MonthlyCharges'], test_df['MonthlyCharges'], orig_df['MonthlyCharges']], axis=0)
    all_tc = pd.concat([train_df['TotalCharges'], test_df['TotalCharges'], orig_df['TotalCharges']], axis=0)
    mc_bins = pd.qcut(all_mc.rank(method='first'), q=MC_BINS, duplicates='drop', retbins=True)[1]
    tc_bins = pd.qcut(all_tc.rank(method='first'), q=TC_BINS, duplicates='drop', retbins=True)[1]

    def rank_bin(series, bins):
        r = series.rank(method='first')
        return pd.cut(r, bins=bins, include_lowest=True).astype('string')

    for df in (train_df, test_df, orig_df):
        df['MonthlyCharges_bin'] = rank_bin(pd.to_numeric(df['MonthlyCharges'], errors='coerce'), mc_bins)
        df['TotalCharges_bin'] = rank_bin(pd.to_numeric(df['TotalCharges'], errors='coerce'), tc_bins)

    for a, b in CROSS_PAIRS:
        for df in (train_df, test_df, orig_df):
            df[f'{a}__{b}'] = df[a].astype('string') + '|' + df[b].astype('string')

    for a, b, c in TRIPLE_CATS:
        for df in (train_df, test_df, orig_df):
            df[f'{a}__{b}__{c}'] = (
                df[a].astype('string') + '|' + df[b].astype('string') + '|' + df[c].astype('string')
            )
    return train_df, test_df, orig_df


def add_round_features(train_df, test_df, orig_df):
    round_specs = [
        ('tenure', 0),
        ('MonthlyCharges', 0),
        ('MonthlyCharges', 1),
        ('TotalCharges', 0),
        ('TotalCharges', -1),
    ]
    made = []
    for c, r in round_specs:
        name = f'{c}_round{r}'.replace('-', 'm')
        for df in (train_df, test_df, orig_df):
            df[name] = pd.to_numeric(df[c], errors='coerce').round(r).astype('float32')
        made.append(name)
    return train_df, test_df, orig_df, made


def add_qcut_bins(train_df, test_df, orig_df):
    made = []
    for c in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        train_s = pd.to_numeric(train_df[c], errors='coerce')
        for q in QCUT_BINS_SMALL:
            name = f'{c}_qcut_{q}'
            try:
                _, bins = pd.qcut(train_s, q=q, labels=False, retbins=True, duplicates='drop')
                train_df[name] = pd.cut(pd.to_numeric(train_df[c], errors='coerce'), bins=bins, labels=False,
                                        include_lowest=True)
                test_df[name] = pd.cut(pd.to_numeric(test_df[c], errors='coerce'), bins=bins, labels=False,
                                       include_lowest=True)
                orig_df[name] = pd.cut(pd.to_numeric(orig_df[c], errors='coerce'), bins=bins, labels=False,
                                       include_lowest=True)
            except Exception:
                train_df[name] = 0
                test_df[name] = 0
                orig_df[name] = 0

            for df in (train_df, test_df, orig_df):
                df[name] = pd.to_numeric(df[name], errors='coerce').fillna(-1).astype('int32')
            made.append(name)
    return train_df, test_df, orig_df, made


def add_digit_features(train_df, test_df, orig_df):
    made = []
    for c in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        allv = pd.concat([
            pd.to_numeric(train_df[c], errors='coerce'),
            pd.to_numeric(test_df[c], errors='coerce'),
            pd.to_numeric(orig_df[c], errors='coerce'),
        ], axis=0, ignore_index=True).fillna(0)

        scaled = (allv * 100).round().astype(np.int64)
        n_tr = len(train_df)
        n_te = len(test_df)
        for k, (a, b) in {
            'int_last': (scaled // 100, 10),
            'dec0': (scaled // 10, 10),
            'dec1': (scaled, 10),
        }.items():
            vals = (a % b).astype(np.int16)
            name = f'{c}_digit_{k}'
            train_df[name] = vals.iloc[:n_tr].values
            test_df[name] = vals.iloc[n_tr:n_tr + n_te].values
            orig_df[name] = vals.iloc[n_tr + n_te:].values
            made.append(name)
    return train_df, test_df, orig_df, made


def add_cat_counts_and_rare(train_df, test_df, orig_df, cat_cols, rare_thresh=RARE_MIN_COUNT):
    all_cat = pd.concat([train_df[cat_cols], test_df[cat_cols], orig_df[cat_cols]], axis=0, ignore_index=True)

    add_tr, add_te, add_or = {}, {}, {}
    for c in cat_cols:
        vc = all_cat[c].astype('string').value_counts(dropna=False)
        tr_cnt = pd.to_numeric(train_df[c].astype('string').map(vc), errors='coerce').fillna(0).astype('float32')
        te_cnt = pd.to_numeric(test_df[c].astype('string').map(vc), errors='coerce').fillna(0).astype('float32')
        or_cnt = pd.to_numeric(orig_df[c].astype('string').map(vc), errors='coerce').fillna(0).astype('float32')

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


def add_orig_membership(df, orig, cols):
    out = {}
    for c in cols:
        if c in df.columns and c in orig.columns:
            orig_vals = set(orig[c].astype('string').fillna('<NA>').unique())
            out[f'{c}_in_orig'] = df[c].astype('string').fillna('<NA>').isin(orig_vals).astype('float32')
    return pd.DataFrame(out, index=df.index)


def add_orig_priors(df, orig, target, cols):
    out = {}
    for c in cols:
        if c in df.columns and c in orig.columns:
            grp = orig.groupby(c, dropna=False)[target]
            mean_map = grp.mean()
            cnt_map = grp.size()
            out[f'orig_mean_{c}'] = df[c].map(mean_map).fillna(orig[target].mean()).astype('float32')
            out[f'orig_count_{c}'] = df[c].map(cnt_map).fillna(0).astype('float32')
    return pd.DataFrame(out, index=df.index)


def add_orig_pair_priors(df, orig, target, pairs):
    out = {}
    for c1, c2 in pairs:
        if c1 in df.columns and c2 in df.columns and c1 in orig.columns and c2 in orig.columns:
            name = f'{c1}_x_{c2}'
            orig_key = make_pair_key(orig, c1, c2)
            tmp = pd.DataFrame({name: orig_key, target: orig[target].values})
            grp = tmp.groupby(name, dropna=False)[target]
            mean_map = grp.mean()
            cnt_map = grp.size()

            df_key = make_pair_key(df, c1, c2)
            out[f'orig_mean_{name}'] = df_key.map(mean_map).fillna(orig[target].mean()).astype('float32')
            out[f'orig_count_{name}'] = df_key.map(cnt_map).fillna(0).astype('float32')
            out[f'orig_seen_{name}'] = df_key.isin(set(mean_map.index)).astype('float32')
    return pd.DataFrame(out, index=df.index)


def add_drift_ratio(train_df, test_df, orig_df, cols, eps=1e-6):
    add_tr, add_te = {}, {}
    for c in cols:
        if c not in train_df.columns or c not in orig_df.columns:
            continue
        p_tr = train_df[c].value_counts(normalize=True, dropna=False)
        p_or = orig_df[c].value_counts(normalize=True, dropna=False)
        name = f'DRIFT_{c}'
        add_tr[name] = np.log1p(
            train_df[c].map(p_tr).fillna(0) / (train_df[c].map(p_or).fillna(eps) + eps)
        ).astype('float32').values
        add_te[name] = np.log1p(
            test_df[c].map(p_tr).fillna(0) / (test_df[c].map(p_or).fillna(eps) + eps)
        ).astype('float32').values

    train_df = pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1)
    return train_df, test_df


# ------------------------------------------------------------------ nested target encoding
def target_encode_nested(X_tr, y_tr, X_va, X_te, te_cols, inner_folds=5, seed=42):
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    X_te = X_te.copy()

    made_cols = []
    skf_inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    for c in te_cols:
        if c not in X_tr.columns:
            continue
        tr_col = X_tr[c].astype('string').fillna('<NA>')
        va_col = X_va[c].astype('string').fillna('<NA>')
        te_col = X_te[c].astype('string').fillna('<NA>')

        oof_enc = np.zeros(len(X_tr), dtype=np.float32)
        for itr, iva in skf_inner.split(tr_col, y_tr):
            enc = TargetEncoder(target_type='binary', smooth='auto')
            enc.fit(tr_col.iloc[itr].to_frame(name=c), y_tr.iloc[itr])
            oof_enc[iva] = _te_to_1d_float32(enc.transform(tr_col.iloc[iva].to_frame(name=c)))

        enc_full = TargetEncoder(target_type='binary', smooth='auto')
        enc_full.fit(tr_col.to_frame(name=c), y_tr)
        va_enc = _te_to_1d_float32(enc_full.transform(va_col.to_frame(name=c)))
        te_enc = _te_to_1d_float32(enc_full.transform(te_col.to_frame(name=c)))

        newc = f'TE_{c}'
        X_tr[newc] = oof_enc
        X_va[newc] = va_enc
        X_te[newc] = te_enc
        made_cols.append(newc)
    return X_tr, X_va, X_te, made_cols


# ------------------------------------------------------------------ pipeline
def build_features():
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = labels(train[TARGET]).astype(np.int32)
    orig[TARGET] = labels(orig[TARGET]).astype(np.int32)

    base_nums = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for df in (train, test, orig):
        for c in base_nums:
            med = pd.to_numeric(df[c], errors='coerce').median()
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(med).astype('float32')

    base_cat_cols = [c for c in train.columns if c not in [TARGET, ID_COL] + base_nums]
    for df in (train, test, orig):
        safe_str_inplace(df, [c for c in base_cat_cols if c in df.columns])

    log('Snap features...')
    train, test = add_snap_features(train, test)
    safe_str_inplace(train, ['MonthlyCharges_snap', 'TotalCharges_snap'])
    safe_str_inplace(test, ['MonthlyCharges_snap', 'TotalCharges_snap'])
    orig['MonthlyCharges_snap'] = orig['MonthlyCharges'].astype('string')
    orig['TotalCharges_snap'] = orig['TotalCharges'].astype('string')
    orig['MonthlyCharges_snap_diff'] = np.zeros(len(orig), dtype=np.float32)
    orig['TotalCharges_snap_diff'] = np.zeros(len(orig), dtype=np.float32)

    log('Frequency / rank, transforms, charge relations, service counts, flags...')
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyCharges_snap_diff', 'TotalCharges_snap_diff']
    train, test, orig = add_global_rank_and_freq(train, test, orig, num_cols)
    train, test, orig = add_num_transforms(train, test, orig, num_cols)
    train, test, orig = add_charge_relationships(train, test, orig)
    train, test, orig = add_service_counts(train, test, orig)
    train, test, orig = add_yes_no_other_flags(train, test, orig)

    log('Bins, crosses, roundings, quantile bins, digits...')
    train, test, orig = add_bins_and_cross_cats(train, test, orig)
    train, test, orig, _ = add_round_features(train, test, orig)
    train, test, orig, _ = add_qcut_bins(train, test, orig)
    train, test, orig, _ = add_digit_features(train, test, orig)

    cat_cols = [c for c in train.columns if c not in [TARGET, ID_COL] and train[c].dtype.name in ['object', 'string']]
    cat_cols += [c for c in ['tenure_bin', 'MonthlyCharges_bin', 'TotalCharges_bin'] if c in train.columns]
    cat_cols = list(dict.fromkeys(cat_cols))

    log('Category counts / rare flags, original priors, drift...')
    train, test, orig = add_cat_counts_and_rare(train, test, orig, cat_cols, rare_thresh=RARE_MIN_COUNT)

    orig_prior_cols = [
        'Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity', 'TechSupport',
        'tenure_bin', 'MonthlyCharges_bin', 'TotalCharges_bin',
        'MonthlyCharges_snap', 'TotalCharges_snap',
    ]
    train = pd.concat([train, add_orig_membership(train, orig, orig_prior_cols)], axis=1)
    test = pd.concat([test, add_orig_membership(test, orig, orig_prior_cols)], axis=1)
    train = pd.concat([train, add_orig_priors(train, orig, TARGET, orig_prior_cols)], axis=1)
    test = pd.concat([test, add_orig_priors(test, orig, TARGET, orig_prior_cols)], axis=1)
    train = pd.concat([train, add_orig_pair_priors(train, orig, TARGET, PAIR_PRIOR_COLS)], axis=1)
    test = pd.concat([test, add_orig_pair_priors(test, orig, TARGET, PAIR_PRIOR_COLS)], axis=1)

    drift_cols = [
        'Contract', 'InternetService', 'PaymentMethod',
        'OnlineSecurity', 'TechSupport',
        'tenure_bin', 'MonthlyCharges_bin', 'TotalCharges_bin',
    ]
    train, test = add_drift_ratio(train, test, orig, drift_cols)

    train = dedupe_columns_keep_first(train)
    test = dedupe_columns_keep_first(test)
    log(f'after feature engineering: train {train.shape}, test {test.shape}')

    y = train[TARGET].astype(np.int32)
    X = train.drop(columns=[TARGET, ID_COL]).copy()
    X_test = test.drop(columns=[ID_COL], errors='ignore').copy()
    cat_cols = list(dict.fromkeys([c for c in X.columns if X[c].dtype.name in ['object', 'string']]))
    te_cols = [c for c in TE_COLS_BASE if c in X.columns]
    log(f'{len(cat_cols)} string columns, {X.shape[1] - len(cat_cols)} numeric columns, {len(te_cols)} TE columns')
    return X, y, X_test, cat_cols, te_cols


def main():
    X, y, X_test, cat_cols, te_cols = build_features()

    oof = np.zeros(len(X), dtype=np.float32)
    pred_test = np.zeros(len(X_test), dtype=np.float32)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr = X.iloc[tr_idx].reset_index(drop=True).copy()
        X_va = X.iloc[va_idx].reset_index(drop=True).copy()
        y_tr = y.iloc[tr_idx].reset_index(drop=True)
        y_va = y.iloc[va_idx].reset_index(drop=True)
        X_te = X_test.copy()

        X_tr, X_va, X_te, _ = target_encode_nested(X_tr, y_tr, X_va, X_te, te_cols=te_cols,
                                                   inner_folds=INNER_FOLDS, seed=SEED + fold)
        cats_fold = [c for c in cat_cols if c in X_tr.columns]
        X_tr, X_va, X_te = factorize_train_valid_test(X_tr, X_va, X_te, cats_fold)
        for df in (X_tr, X_va, X_te):
            for c in df.columns:
                if pd.api.types.is_bool_dtype(df[c]):
                    df[c] = df[c].astype(np.int8)
                else:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(-1).astype(np.float32)

        features = X_tr.columns.tolist()
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr[features], y_tr, eval_set=[(X_va[features], y_va)], verbose=500)
        p_va = model.predict_proba(X_va[features])[:, 1].astype(np.float32)
        p_te = model.predict_proba(X_te[features])[:, 1].astype(np.float32)
        oof[va_idx] = p_va
        pred_test += p_te / N_SPLITS
        log(f'fold {fold}: AUC {roc_auc_score(y_va, p_va):.6f}, best_iteration {getattr(model, "best_iteration", None)}')

        del X_tr, X_va, X_te, y_tr, y_va, model, p_va, p_te
        gc.collect()

    log(f'{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}')
    save_pair(HERE, PAIR, oof, pred_test)


if __name__ == '__main__':
    main()
