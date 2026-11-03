#!/usr/bin/env python
"""GPU XGBoost with bigram / trigram target encodings and original-data distribution features
(pair xgb_bigram_trigram_te_stats).

Label-free features (missing original TotalCharges get the original median):
- the 3 numerics; their frequency over train + original + test rows;
- TotalCharges - tenure x MonthlyCharges, MonthlyCharges / (TotalCharges + 1), TotalCharges / (tenure + 1);
- service count (Yes among 8 service columns), has-internet, has-phone;
- original churn rate of every value of the 16 categoricals and the 3 numerics (unseen: 0.5);
- TotalCharges against the original rows: percentile rank among the original churners / non-churners / all rows,
  z-scores and their churner / non-churner gaps, percentile rank within the original rows of the same InternetService
  and of the same Contract, MonthlyCharges minus the original mean of its InternetService, and 8 absolute distances to
  the 25 / 50 / 75 % quantiles of the original churners / non-churners and their gaps;
- 35 digit features of tenure / MonthlyCharges / TotalCharges (first / second / last digits, mod 10 / 12 / 100, digit
  counts, multiples, roundings and deviations, fractional parts, tenure years);
- 15 bigrams of Contract, InternetService, PaymentMethod, OnlineSecurity, TechSupport, PaperlessBilling and the 4
  trigrams of the first four.
Per fold (inner StratifiedKFold(5, shuffle, random_state 42) of the training rows; the training rows get out-of-fold
values, validation / test rows the statistics of all training rows):
- std / min / max of the target per value of the 16 categoricals and the 3 numerics as strings;
- target mean of every bigram / trigram (unseen: 0.5);
- sklearn TargetEncoder means (cv 5, smooth auto) of the 16 categoricals and the 3 numerics as strings.
The string sources are then dropped (numeric features only).
Model: depth 5, lr 0.0063, subsample 0.81, colsample_bytree 0.32, min_child_weight 6, alpha 3.5017, lambda 1.2925,
gamma 0.79, up to 50,000 rounds, early stopping 500 on the validation fold, random_state 42.
Usage: python train_ngram_te.py"""
import gc
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
from common.data import TARGET, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'xgb_bigram_trigram_te_stats'
N_FOLDS = 5
INNER_FOLDS = 5
TEST_FOLDS = 15
TEST_INNER_FOLDS = 10
RANDOM_SEED = 42

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
STATS = ['std', 'min', 'max']


def log(msg):
    print(msg, flush=True)


def load_data():
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = labels(train[TARGET])
    orig[TARGET] = labels(orig[TARGET])
    orig['TotalCharges'] = orig['TotalCharges'].fillna(orig['TotalCharges'].median())
    for df in [train, test, orig]:
        for c in NUMS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
        for c in CATS:
            df[c] = df[c].astype(str)
    return train, test, orig


def add_block(df, cols_dict):
    return pd.concat([df, pd.DataFrame(cols_dict, index=df.index)], axis=1)


def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')


def zscore_against(values, reference):
    mu = np.mean(reference)
    sigma = np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype='float32')
    return ((values - mu) / sigma).astype('float32')


def build_features(train, test, orig):
    new_nums = []
    num_as_cat = []

    # frequency encoding
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]], axis=0).value_counts(normalize=True)
        for df in (train, test, orig):
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        new_nums.append(f'FREQ_{col}')

    # arithmetic interactions
    def arithmetic(df):
        return add_block(df, {
            'charges_deviation': (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32'),
            'monthly_to_total_ratio': (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32'),
            'avg_monthly_charges': (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32'),
        })
    train, test, orig = arithmetic(train), arithmetic(test), arithmetic(orig)
    new_nums += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']

    # service counts
    def services(df):
        return add_block(df, {
            'service_count': (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32'),
            'has_internet': (df['InternetService'] != 'No').astype('float32'),
            'has_phone': (df['PhoneService'] == 'Yes').astype('float32'),
        })
    train, test, orig = services(train), services(test), services(orig)
    new_nums += ['service_count', 'has_internet', 'has_phone']

    # original churn rate per value
    for col in CATS + NUMS:
        tmp = orig.groupby(col, observed=False)[TARGET].mean()
        new_col = f'ORIG_proba_{col}'
        train[new_col] = train[col].map(tmp).fillna(0.5).astype('float32')
        test[new_col] = test[col].map(tmp).fillna(0.5).astype('float32')
        new_nums.append(new_col)

    # TotalCharges against the original distribution
    orig_churner_tc = orig.loc[orig[TARGET] == 1, 'TotalCharges'].values
    orig_nonchurner_tc = orig.loc[orig[TARGET] == 0, 'TotalCharges'].values
    orig_tc = orig['TotalCharges'].values
    orig_is_mc_mean = orig.groupby('InternetService', observed=False)['MonthlyCharges'].mean()

    def distribution(df):
        tc = df['TotalCharges'].values
        vals = {
            'pctrank_nonchurner_TC': pctrank_against(tc, orig_nonchurner_tc),
            'pctrank_churner_TC': pctrank_against(tc, orig_churner_tc),
            'pctrank_orig_TC': pctrank_against(tc, orig_tc),
            'zscore_churn_gap_TC': (
                np.abs(zscore_against(tc, orig_churner_tc)) - np.abs(zscore_against(tc, orig_nonchurner_tc))
            ).astype('float32'),
            'zscore_nonchurner_TC': zscore_against(tc, orig_nonchurner_tc),
            'pctrank_churn_gap_TC': (
                pctrank_against(tc, orig_churner_tc) - pctrank_against(tc, orig_nonchurner_tc)
            ).astype('float32'),
            'resid_IS_MC': (
                df['MonthlyCharges'].values - df['InternetService'].map(orig_is_mc_mean).fillna(0).values
            ).astype('float32'),
        }
        cond_is = np.zeros(len(df), dtype='float32')
        for cat_val in orig['InternetService'].unique():
            mask = (df['InternetService'] == cat_val).values
            ref = orig.loc[orig['InternetService'] == cat_val, 'TotalCharges'].values
            if len(ref) > 0 and mask.sum() > 0:
                cond_is[mask] = pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
        vals['cond_pctrank_IS_TC'] = cond_is
        cond_c = np.zeros(len(df), dtype='float32')
        for cat_val in orig['Contract'].unique():
            mask = (df['Contract'] == cat_val).values
            ref = orig.loc[orig['Contract'] == cat_val, 'TotalCharges'].values
            if len(ref) > 0 and mask.sum() > 0:
                cond_c[mask] = pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
        vals['cond_pctrank_C_TC'] = cond_c
        return add_block(df, vals)
    train, test = distribution(train), distribution(test)
    new_nums += [
        'pctrank_nonchurner_TC', 'zscore_churn_gap_TC', 'pctrank_churn_gap_TC',
        'resid_IS_MC', 'cond_pctrank_IS_TC', 'zscore_nonchurner_TC',
        'pctrank_orig_TC', 'pctrank_churner_TC', 'cond_pctrank_C_TC',
    ]

    # distances to the original churner / non-churner quantiles
    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        ch_q = np.quantile(orig_churner_tc, q_val)
        nc_q = np.quantile(orig_nonchurner_tc, q_val)

        def qdist(df):
            return add_block(df, {
                f'dist_To_ch_{q_label}': np.abs(df['TotalCharges'] - ch_q).astype('float32'),
                f'dist_To_nc_{q_label}': np.abs(df['TotalCharges'] - nc_q).astype('float32'),
                f'qdist_gap_To_{q_label}': (
                    np.abs(df['TotalCharges'] - nc_q) - np.abs(df['TotalCharges'] - ch_q)
                ).astype('float32'),
            })
        train, test = qdist(train), qdist(test)
    new_nums += [
        'qdist_gap_To_q50', 'dist_To_ch_q50', 'dist_To_nc_q50',
        'dist_To_nc_q25', 'qdist_gap_To_q25',
        'dist_To_nc_q75', 'dist_To_ch_q75', 'qdist_gap_To_q75',
    ]

    # numerics as categories
    for col in NUMS:
        new_col = f'CAT_{col}'
        train[new_col] = train[col].astype(str).astype('category')
        test[new_col] = test[col].astype(str).astype('category')
        num_as_cat.append(new_col)

    # digit features
    digit_features = [
        'tenure_first_digit', 'tenure_last_digit', 'tenure_second_digit',
        'tenure_mod10', 'tenure_mod12', 'tenure_num_digits',
        'tenure_is_multiple_10', 'tenure_rounded_10', 'tenure_dev_from_round10',
        'mc_first_digit', 'mc_last_digit', 'mc_second_digit',
        'mc_mod10', 'mc_mod100', 'mc_num_digits',
        'mc_is_multiple_10', 'mc_is_multiple_50',
        'mc_rounded_10', 'mc_fractional', 'mc_dev_from_round10',
        'tc_first_digit', 'tc_last_digit', 'tc_second_digit',
        'tc_mod10', 'tc_mod100', 'tc_num_digits',
        'tc_is_multiple_10', 'tc_is_multiple_100',
        'tc_rounded_100', 'tc_fractional', 'tc_dev_from_round100',
        'tenure_years', 'tenure_months_in_year',
        'mc_per_digit', 'tc_per_digit',
    ]

    def digits(df):
        t_str = df['tenure'].fillna(0).astype(int).astype(str)
        mc_floor = np.floor(df['MonthlyCharges'].fillna(0)).astype(int)
        tc_floor = np.floor(df['TotalCharges'].fillna(0)).astype(int)
        mc_str = df['MonthlyCharges'].fillna(0).astype(str).str.replace('.', '', regex=False)
        tc_str = df['TotalCharges'].fillna(0).astype(str).str.replace('.', '', regex=False)
        return add_block(df, {
            'tenure_first_digit': t_str.str[0].astype(int),
            'tenure_last_digit': t_str.str[-1].astype(int),
            'tenure_second_digit': t_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0).astype('int16'),
            'tenure_mod10': (df['tenure'] % 10).astype('float32'),
            'tenure_mod12': (df['tenure'] % 12).astype('float32'),
            'tenure_num_digits': t_str.str.len().astype('int16'),
            'tenure_is_multiple_10': (df['tenure'] % 10 == 0).astype('float32'),
            'tenure_rounded_10': (np.round(df['tenure'] / 10) * 10).astype('float32'),
            'tenure_dev_from_round10': np.abs(df['tenure'] - np.round(df['tenure'] / 10) * 10).astype('float32'),
            'mc_first_digit': mc_str.str[0].replace('-', '0').fillna('0').astype(int),
            'mc_last_digit': mc_str.str[-1].replace('-', '0').fillna('0').astype(int),
            'mc_second_digit': mc_str.apply(lambda x: int(x[1]) if len(x) > 1 and x[1].isdigit() else 0).astype('int16'),
            'mc_mod10': (mc_floor % 10).astype('float32'),
            'mc_mod100': (mc_floor % 100).astype('float32'),
            'mc_num_digits': mc_floor.astype(str).str.len().astype('int16'),
            'mc_is_multiple_10': (mc_floor % 10 == 0).astype('float32'),
            'mc_is_multiple_50': (mc_floor % 50 == 0).astype('float32'),
            'mc_rounded_10': (np.round(df['MonthlyCharges'] / 10) * 10).astype('float32'),
            'mc_fractional': (df['MonthlyCharges'] - np.floor(df['MonthlyCharges'])).astype('float32'),
            'mc_dev_from_round10': np.abs(df['MonthlyCharges'] - np.round(df['MonthlyCharges'] / 10) * 10).astype('float32'),
            'tc_first_digit': tc_str.str[0].replace('-', '0').fillna('0').astype(int),
            'tc_last_digit': tc_str.str[-1].replace('-', '0').fillna('0').astype(int),
            'tc_second_digit': tc_str.apply(lambda x: int(x[1]) if len(x) > 1 and x[1].isdigit() else 0).astype('int16'),
            'tc_mod10': (tc_floor % 10).astype('float32'),
            'tc_mod100': (tc_floor % 100).astype('float32'),
            'tc_num_digits': tc_floor.astype(str).str.len().astype('int16'),
            'tc_is_multiple_10': (tc_floor % 10 == 0).astype('float32'),
            'tc_is_multiple_100': (tc_floor % 100 == 0).astype('float32'),
            'tc_rounded_100': (np.round(df['TotalCharges'] / 100) * 100).astype('float32'),
            'tc_fractional': (df['TotalCharges'] - np.floor(df['TotalCharges'])).astype('float32'),
            'tc_dev_from_round100': np.abs(df['TotalCharges'] - np.round(df['TotalCharges'] / 100) * 100).astype('float32'),
            'tenure_years': (df['tenure'] // 12).astype('float32'),
            'tenure_months_in_year': (df['tenure'] % 12).astype('float32'),
            'mc_per_digit': (df['MonthlyCharges'] / (mc_floor.astype(str).str.len().astype(float) + 0.001)).astype('float32'),
            'tc_per_digit': (df['TotalCharges'] / (tc_floor.astype(str).str.len().astype(float) + 0.001)).astype('float32'),
        })
    train, test = digits(train), digits(test)
    new_nums += digit_features

    # bigrams / trigrams
    bigram_cols, trigram_cols = [], []
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f'BG_{c1}_{c2}'
        train[col_name] = (train[c1].astype(str) + '_' + train[c2].astype(str)).astype('category')
        test[col_name] = (test[c1].astype(str) + '_' + test[c2].astype(str)).astype('category')
        bigram_cols.append(col_name)
    for c1, c2, c3 in combinations(TOP_CATS_FOR_NGRAM[:4], 3):
        col_name = f'TG_{c1}_{c2}_{c3}'
        train[col_name] = (
            train[c1].astype(str) + '_' + train[c2].astype(str) + '_' + train[c3].astype(str)
        ).astype('category')
        test[col_name] = (
            test[c1].astype(str) + '_' + test[c2].astype(str) + '_' + test[c3].astype(str)
        ).astype('category')
        trigram_cols.append(col_name)
    ngram_cols = bigram_cols + trigram_cols

    features = NUMS + CATS + new_nums + num_as_cat + ngram_cols
    te_columns = num_as_cat + CATS
    to_remove = num_as_cat + CATS + ngram_cols
    log(f'{len(features)} features before encoding ({len(new_nums)} engineered numerics, {len(ngram_cols)} n-grams)')
    return train, test, features, te_columns, ngram_cols, to_remove


def train_model(train, test, features, te_columns, te_ngram_columns, to_remove, n_folds, inner_folds):
    np.random.seed(RANDOM_SEED)
    skf_outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_SEED)

    xgb_oof = np.zeros(len(train), dtype='float32')
    xgb_pred = np.zeros(len(test), dtype='float32')
    for fold, (train_idx, val_idx) in enumerate(skf_outer.split(train, train[TARGET]), start=1):
        X_tr = train.loc[train_idx, features + [TARGET]].reset_index(drop=True).copy()
        y_tr = train.loc[train_idx, TARGET].values
        X_val = train.loc[val_idx, features].reset_index(drop=True).copy()
        y_val = train.loc[val_idx, TARGET].values
        X_te = test[features].reset_index(drop=True).copy()

        # inner-fold std / min / max of the target for the categoricals and numerics-as-categories
        for in_tr, in_va in skf_inner.split(X_tr, y_tr):
            X_tr2 = X_tr.loc[in_tr, features + [TARGET]].copy()
            X_va2 = X_tr.loc[in_va, features].copy()
            for col in te_columns:
                tmp = X_tr2.groupby(col, observed=False)[TARGET].agg(STATS)
                tmp.columns = [f'TE1_{col}_{s}' for s in STATS]
                X_va2 = X_va2.merge(tmp, on=col, how='left')
                for c in tmp.columns:
                    X_tr.loc[in_va, c] = X_va2[c].values.astype('float32')

        for col in te_columns:
            tmp = X_tr.groupby(col, observed=False)[TARGET].agg(STATS)
            tmp.columns = [f'TE1_{col}_{s}' for s in STATS]
            tmp = tmp.astype('float32')
            X_val = X_val.merge(tmp, on=col, how='left')
            X_te = X_te.merge(tmp, on=col, how='left')
            for c in tmp.columns:
                X_tr[c] = X_tr[c].fillna(0).astype('float32')
                X_val[c] = X_val[c].fillna(0).astype('float32')
                X_te[c] = X_te[c].fillna(0).astype('float32')

        # inner-fold target mean of the bigrams / trigrams
        for in_tr, in_va in skf_inner.split(X_tr, y_tr):
            X_tr2 = X_tr.loc[in_tr].copy()
            X_va2 = X_tr.loc[in_va].copy()
            for col in te_ngram_columns:
                ng_te = X_tr2.groupby(col, observed=False)[TARGET].mean()
                ng_name = f'TE_ng_{col}'
                mapped = X_va2[col].astype(str).map(ng_te)
                X_tr.loc[in_va, ng_name] = pd.to_numeric(mapped, errors='coerce').fillna(0.5).astype('float32').values

        for col in te_ngram_columns:
            ng_te = X_tr.groupby(col, observed=False)[TARGET].mean()
            ng_name = f'TE_ng_{col}'
            X_val[ng_name] = pd.to_numeric(X_val[col].astype(str).map(ng_te), errors='coerce').fillna(0.5).astype('float32')
            X_te[ng_name] = pd.to_numeric(X_te[col].astype(str).map(ng_te), errors='coerce').fillna(0.5).astype('float32')
            if ng_name in X_tr.columns:
                X_tr[ng_name] = pd.to_numeric(X_tr[ng_name], errors='coerce').fillna(0.5).astype('float32')
            else:
                X_tr[ng_name] = np.float32(0.5)

        # sklearn TargetEncoder means of the categoricals and numerics-as-categories
        te_mean_cols = [f'TE_{col}' for col in te_columns]
        te = TargetEncoder(cv=inner_folds, shuffle=True, smooth='auto', target_type='binary',
                           random_state=RANDOM_SEED)
        X_tr[te_mean_cols] = te.fit_transform(X_tr[te_columns], y_tr)
        X_val[te_mean_cols] = te.transform(X_val[te_columns])
        X_te[te_mean_cols] = te.transform(X_te[te_columns])

        for df in [X_tr, X_val, X_te]:
            for c in CATS + [f'CAT_{x}' for x in NUMS]:
                if c in df.columns:
                    df[c] = df[c].astype(str).astype('category')
            df.drop(columns=[c for c in to_remove if c in df.columns], inplace=True, errors='ignore')
        X_tr.drop(columns=[TARGET], inplace=True, errors='ignore')
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

    log(f'{n_folds} folds: OOF AUC {roc_auc_score(train[TARGET], xgb_oof):.6f}')
    return xgb_oof, xgb_pred


def main():
    train, test, orig = load_data()
    train, test, features, te_columns, ngram_cols, to_remove = build_features(train, test, orig)
    oof, _ = train_model(train, test, features, te_columns, ngram_cols, to_remove, N_FOLDS, INNER_FOLDS)
    _, test_pred = train_model(train, test, features, te_columns, ngram_cols, to_remove, TEST_FOLDS, TEST_INNER_FOLDS)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
