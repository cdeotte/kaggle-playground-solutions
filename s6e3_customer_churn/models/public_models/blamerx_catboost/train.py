#!/usr/bin/env python
"""Public notebook "S6E3 | 0.91902 Optimized CatBoost" run on our 5 folds.
https://www.kaggle.com/code/blamerx/s6e3-0-91902-optimized-catboost

Features (the notebook's, built once on train / test): frequencies of the 3 numerics over train + original + test;
charges deviation, monthly / total and total / tenure ratios; service count, has-internet and has-phone flags; the
original dataset's churn rate of every categorical and numeric value (19, 0.5 if unseen); TotalCharges percentile ranks
/ z-scores against the original churners, non-churners and all rows, conditional percentile ranks within
InternetService / Contract, the MonthlyCharges residual from the original InternetService mean and distances to the
original churner / non-churner quartiles; 35 digit and rounding features of tenure, MonthlyCharges and TotalCharges;
15 bigrams of Contract, InternetService, PaymentMethod, OnlineSecurity, TechSupport, PaperlessBilling and 4 trigrams of
the first four.
Encodings per fold (5 inner stratified folds on the training rows; validation / test rows get the training-row
statistics): std / min / max of the target per value of the 16 categoricals and the 3 numerics as strings (unseen: 0),
the target mean per n-gram value (unseen: 0.5) and a scikit-learn TargetEncoder (cv 5, smooth 'auto') of the 19
categorical / numeric-string columns. The raw categoricals, numeric strings and n-grams are then dropped: CatBoost sees
numeric columns only.
Model: CatBoost (GPU) with the notebook's Optuna-tuned parameters: SymmetricTree depth 7, lr 0.00984, up to 20,000
iterations with early stopping (200) on the validation fold's AUC (best model kept), l2_leaf_reg 5.333,
random_strength 2.877, bagging_temperature 0.264, border_count 254, min_data_in_leaf 14, seed 42.
Test = mean of the fold models.
Usage: python train.py"""
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, read_original, read_test, read_train  # noqa: E402
from lib.runner import run_folds, work_dir, working_directory  # noqa: E402

warnings.filterwarnings('ignore')

FOLDER = 'blamerx_catboost'
INNER_FOLDS = 5
RANDOM_SEED = 42

TOP_CATS_FOR_NGRAM = ['Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity', 'TechSupport',
                      'PaperlessBilling']
CATS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod']
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']
STATS = ['std', 'min', 'max']

CAT_PARAMS = {
    'iterations': 20000,
    'learning_rate': 0.00984,
    'depth': 7,
    'l2_leaf_reg': 5.333,
    'random_strength': 2.877,
    'bagging_temperature': 0.264,
    'border_count': 254,
    'min_data_in_leaf': 14,
    'grow_policy': 'SymmetricTree',
    'random_seed': RANDOM_SEED,
    'early_stopping_rounds': 200,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'task_type': 'GPU',
    'verbose': 0,
    'use_best_model': True,
}

DIGIT_FEATURES = [
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
    'tenure_years', 'tenure_months_in_year', 'mc_per_digit', 'tc_per_digit',
]


def as_object_strings(s):
    """Python strings, NaN kept."""
    return s.map(lambda v: np.nan if pd.isna(v) else str(v)).astype('object')


def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')


def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    return np.zeros(len(values), dtype='float32') if sigma == 0 else ((values - mu) / sigma).astype('float32')


def load_data():
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = train[TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    y = train[TARGET].values.astype(np.int32)
    orig[TARGET] = orig[TARGET].map({'No': 0, 'Yes': 1}).astype(int)

    X_train = train.drop(columns=[TARGET, ID]).copy()
    X_test = test.drop(columns=[ID]).copy()
    for df in (X_train, X_test):
        for c in ['tenure', 'MonthlyCharges']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0).astype('float32')

    all_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    for c in CATS:
        all_df[c] = as_object_strings(all_df[c])
    X_train = all_df.iloc[:len(X_train)].reset_index(drop=True)
    X_test = all_df.iloc[len(X_train):].reset_index(drop=True)

    train = X_train.copy()
    train[TARGET] = y
    test = X_test.copy()

    orig['TotalCharges'] = orig['TotalCharges'].fillna(orig['TotalCharges'].median())
    for c in CATS:
        orig[c] = as_object_strings(orig[c])
    for c in ['tenure', 'MonthlyCharges']:
        orig[c] = pd.to_numeric(orig[c], errors='coerce')
    return train, test, orig


def build_features(train, test, orig):
    new_nums, num_as_cat = [], []

    # 1. frequency encoding
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for df in (train, test):
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        new_nums.append(f'FREQ_{col}')

    # 2. arithmetic interactions
    for df in (train, test):
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    new_nums += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']

    # 3. service counts
    for df in (train, test):
        df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
    new_nums += ['service_count', 'has_internet', 'has_phone']

    # 4. churn rate of each value in the original dataset
    for col in CATS + NUMS:
        tmp = orig.groupby(col, observed=False)[TARGET].mean()
        name = f'ORIG_proba_{col}'
        train = train.merge(tmp.rename(name), on=col, how='left')
        test = test.merge(tmp.rename(name), on=col, how='left')
        for df in (train, test):
            df[name] = pd.to_numeric(df[name], errors='coerce').fillna(0.5).astype('float32')
        new_nums.append(name)

    # 5. distribution features against the original dataset
    orig_churner_tc = orig.loc[orig[TARGET] == 1, 'TotalCharges'].values
    orig_nonchurner_tc = orig.loc[orig[TARGET] == 0, 'TotalCharges'].values
    orig_tc = orig['TotalCharges'].values
    orig_is_mc_mean = orig.groupby('InternetService', observed=False)['MonthlyCharges'].mean()
    for df in (train, test):
        tc = df['TotalCharges'].values.astype('float32')
        df['pctrank_nonchurner_TC'] = pctrank_against(tc, orig_nonchurner_tc)
        df['pctrank_churner_TC'] = pctrank_against(tc, orig_churner_tc)
        df['pctrank_orig_TC'] = pctrank_against(tc, orig_tc)
        df['zscore_churn_gap_TC'] = (np.abs(zscore_against(tc, orig_churner_tc))
                                     - np.abs(zscore_against(tc, orig_nonchurner_tc))).astype('float32')
        df['zscore_nonchurner_TC'] = zscore_against(tc, orig_nonchurner_tc)
        df['pctrank_churn_gap_TC'] = (pctrank_against(tc, orig_churner_tc)
                                      - pctrank_against(tc, orig_nonchurner_tc)).astype('float32')
        is_mean = pd.to_numeric(df['InternetService'].astype('object').map(orig_is_mc_mean),
                                errors='coerce').fillna(0.0).astype('float32')
        df['resid_IS_MC'] = (pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0).astype('float32')
                             - is_mean).astype('float32')
        for cat_col, out_col in [('InternetService', 'cond_pctrank_IS_TC'), ('Contract', 'cond_pctrank_C_TC')]:
            vals = np.zeros(len(df), dtype='float32')
            for cat_val in orig[cat_col].dropna().unique():
                mask = df[cat_col].astype('object') == cat_val
                ref = orig.loc[orig[cat_col].astype('object') == cat_val, 'TotalCharges'].values
                if len(ref) > 0 and mask.sum() > 0:
                    vals[np.where(mask.values)[0]] = pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
            df[out_col] = vals
    new_nums += ['pctrank_nonchurner_TC', 'zscore_churn_gap_TC', 'pctrank_churn_gap_TC', 'resid_IS_MC',
                 'cond_pctrank_IS_TC', 'zscore_nonchurner_TC', 'pctrank_orig_TC', 'pctrank_churner_TC',
                 'cond_pctrank_C_TC']

    # 6. quantile distance features
    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        ch_q = np.quantile(orig_churner_tc, q_val) if len(orig_churner_tc) else 0.0
        nc_q = np.quantile(orig_nonchurner_tc, q_val) if len(orig_nonchurner_tc) else 0.0
        for df in (train, test):
            df[f'dist_To_ch_{q_label}'] = np.abs(df['TotalCharges'] - ch_q).astype('float32')
            df[f'dist_To_nc_{q_label}'] = np.abs(df['TotalCharges'] - nc_q).astype('float32')
            df[f'qdist_gap_To_{q_label}'] = (df[f'dist_To_nc_{q_label}'] - df[f'dist_To_ch_{q_label}']).astype(
                'float32')
    new_nums += ['qdist_gap_To_q50', 'dist_To_ch_q50', 'dist_To_nc_q50', 'dist_To_nc_q25', 'qdist_gap_To_q25',
                 'dist_To_nc_q75', 'dist_To_ch_q75', 'qdist_gap_To_q75']

    # 7. digit features
    for df in (train, test):
        t = pd.to_numeric(df['tenure'], errors='coerce').fillna(0).astype(int)
        t_str = t.astype(str)
        df['tenure_first_digit'] = t_str.str[0].fillna('0').astype(int)
        df['tenure_last_digit'] = t_str.str[-1].fillna('0').astype(int)
        df['tenure_second_digit'] = t_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tenure_mod10'] = (t % 10).astype('float32')
        df['tenure_mod12'] = (t % 12).astype('float32')
        df['tenure_num_digits'] = t_str.str.len().astype('float32')
        df['tenure_is_multiple_10'] = (t % 10 == 0).astype('float32')
        df['tenure_rounded_10'] = (np.round(t / 10) * 10).astype('float32')
        df['tenure_dev_from_round10'] = np.abs(t - df['tenure_rounded_10']).astype('float32')

        mc = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0).astype('float32')
        mc_str = mc.astype(str).str.replace('.', '', regex=False)
        df['mc_first_digit'] = mc_str.str[0].fillna('0').astype(int)
        df['mc_last_digit'] = mc_str.str[-1].fillna('0').astype(int)
        df['mc_second_digit'] = mc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['mc_mod10'] = (np.floor(mc) % 10).astype('float32')
        df['mc_mod100'] = (np.floor(mc) % 100).astype('float32')
        df['mc_num_digits'] = np.floor(mc).astype(int).astype(str).str.len().astype('float32')
        df['mc_is_multiple_10'] = (np.floor(mc) % 10 == 0).astype('float32')
        df['mc_is_multiple_50'] = (np.floor(mc) % 50 == 0).astype('float32')
        df['mc_rounded_10'] = (np.round(mc / 10) * 10).astype('float32')
        df['mc_fractional'] = (mc - np.floor(mc)).astype('float32')
        df['mc_dev_from_round10'] = np.abs(mc - df['mc_rounded_10']).astype('float32')

        tc = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0).astype('float32')
        tc_str = tc.astype(str).str.replace('.', '', regex=False)
        df['tc_first_digit'] = tc_str.str[0].fillna('0').astype(int)
        df['tc_last_digit'] = tc_str.str[-1].fillna('0').astype(int)
        df['tc_second_digit'] = tc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tc_mod10'] = (np.floor(tc) % 10).astype('float32')
        df['tc_mod100'] = (np.floor(tc) % 100).astype('float32')
        df['tc_num_digits'] = np.floor(tc).astype(int).astype(str).str.len().astype('float32')
        df['tc_is_multiple_10'] = (np.floor(tc) % 10 == 0).astype('float32')
        df['tc_is_multiple_100'] = (np.floor(tc) % 100 == 0).astype('float32')
        df['tc_rounded_100'] = (np.round(tc / 100) * 100).astype('float32')
        df['tc_fractional'] = (tc - np.floor(tc)).astype('float32')
        df['tc_dev_from_round100'] = np.abs(tc - df['tc_rounded_100']).astype('float32')

        df['tenure_years'] = (t // 12).astype('float32')
        df['tenure_months_in_year'] = (t % 12).astype('float32')
        df['mc_per_digit'] = (mc / (df['mc_num_digits'] + 0.001)).astype('float32')
        df['tc_per_digit'] = (tc / (df['tc_num_digits'] + 0.001)).astype('float32')
    new_nums += DIGIT_FEATURES

    # 8. numerics as categories
    for col in NUMS:
        name = f'CAT_{col}'
        num_as_cat.append(name)
        for df in (train, test):
            df[name] = as_object_strings(df[col])

    # n-gram composite categoricals
    ngram_cols = []
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        name = f'BG_{c1}_{c2}'
        for df in (train, test):
            df[name] = (df[c1].astype(str) + '_' + df[c2].astype(str)).astype('object')
        ngram_cols.append(name)
    for c1, c2, c3 in combinations(TOP_CATS_FOR_NGRAM[:4], 3):
        name = f'TG_{c1}_{c2}_{c3}'
        for df in (train, test):
            df[name] = (df[c1].astype(str) + '_' + df[c2].astype(str) + '_' + df[c3].astype(str)).astype('object')
        ngram_cols.append(name)
    return train, test, new_nums, num_as_cat, ngram_cols


def main():
    train, test, orig = load_data()
    train, test, new_nums, num_as_cat, ngram_cols = build_features(train, test, orig)
    features = NUMS + CATS + new_nums + num_as_cat + ngram_cols
    te_columns = num_as_cat + CATS
    to_remove = num_as_cat + CATS + ngram_cols
    print(f'{len(features)} features before encoding ({len(new_nums)} engineered numerics, {len(ngram_cols)} n-grams)',
          flush=True)
    np.random.seed(RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    def fit_fold(fold, tr_idx, va_idx, n_folds):
        X_tr = train.loc[tr_idx, features + [TARGET]].reset_index(drop=True).copy()
        y_tr = train.loc[tr_idx, TARGET].values
        X_va = train.loc[va_idx, features].reset_index(drop=True).copy()
        y_va = train.loc[va_idx, TARGET].values
        X_te = test[features].reset_index(drop=True).copy()

        # inner-fold std / min / max target statistics of the categoricals and numeric strings
        for in_tr, in_va in skf_inner.split(X_tr, y_tr):
            X_tr2 = X_tr.loc[in_tr, features + [TARGET]].copy()
            X_va2 = X_tr.loc[in_va, features].copy()
            for col in te_columns:
                tmp = X_tr2.groupby(col, observed=False)[TARGET].agg(STATS)
                tmp.columns = [f'TE1_{col}_{s}' for s in STATS]
                X_va2 = X_va2.merge(tmp, on=col, how='left')
                for c in tmp.columns:
                    X_tr.loc[in_va, c] = pd.to_numeric(X_va2[c], errors='coerce').astype('float32').values
        for col in te_columns:
            tmp = X_tr.groupby(col, observed=False)[TARGET].agg(STATS)
            tmp.columns = [f'TE1_{col}_{s}' for s in STATS]
            tmp = tmp.astype('float32')
            X_va = X_va.merge(tmp, on=col, how='left')
            X_te = X_te.merge(tmp, on=col, how='left')
            for c in tmp.columns:
                for df in (X_tr, X_va, X_te):
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).astype('float32')

        # inner-fold target means of the n-grams
        for in_tr, in_va in skf_inner.split(X_tr, y_tr):
            X_tr2 = X_tr.loc[in_tr].copy()
            X_va2 = X_tr.loc[in_va].copy()
            for col in ngram_cols:
                ng_te = X_tr2.groupby(col, observed=False)[TARGET].mean()
                mapped = X_va2[col].astype('object').map(ng_te)
                X_tr.loc[in_va, f'TE_ng_{col}'] = pd.to_numeric(mapped, errors='coerce').fillna(0.5).astype(
                    'float32').values
        for col in ngram_cols:
            ng_te = X_tr.groupby(col, observed=False)[TARGET].mean()
            ng_name = f'TE_ng_{col}'
            X_va[ng_name] = pd.to_numeric(X_va[col].astype('object').map(ng_te), errors='coerce').fillna(0.5).astype(
                'float32')
            X_te[ng_name] = pd.to_numeric(X_te[col].astype('object').map(ng_te), errors='coerce').fillna(0.5).astype(
                'float32')
            if ng_name in X_tr.columns:
                X_tr[ng_name] = pd.to_numeric(X_tr[ng_name], errors='coerce').fillna(0.5).astype('float32')
            else:
                X_tr[ng_name] = np.float32(0.5)

        # scikit-learn TargetEncoder (mean)
        te_mean_cols = [f'TE_{col}' for col in te_columns]
        te = TargetEncoder(cv=INNER_FOLDS, shuffle=True, smooth='auto', target_type='binary',
                           random_state=RANDOM_SEED)
        X_tr[te_mean_cols] = te.fit_transform(X_tr[te_columns], y_tr)
        X_va[te_mean_cols] = te.transform(X_va[te_columns])
        X_te[te_mean_cols] = te.transform(X_te[te_columns])

        for df in (X_tr, X_va, X_te):
            df.drop(columns=[c for c in to_remove if c in df.columns], inplace=True, errors='ignore')
        X_tr.drop(columns=[TARGET], inplace=True, errors='ignore')
        cols = X_tr.columns.tolist()
        if fold == 0:
            print(f'  features for CatBoost: {len(cols)}', flush=True)
        for df in (X_tr, X_va, X_te):
            for c in df.columns:
                if df[c].dtype == 'float64':
                    df[c] = df[c].astype('float32')

        train_pool = Pool(data=X_tr, label=y_tr)
        val_pool = Pool(data=X_va[cols], label=y_va)
        with working_directory(work_dir(FOLDER)):
            model = CatBoostClassifier(**CAT_PARAMS)
            model.fit(train_pool, eval_set=val_pool, verbose=0)
        p_va = model.predict_proba(val_pool)[:, 1].astype(np.float32)
        p_te = model.predict_proba(X_te[cols])[:, 1].astype(np.float32)
        del X_tr, X_va, X_te, model, train_pool, val_pool
        gc.collect()
        return p_va, p_te

    run_folds(FOLDER, FOLDER, fit_fold, y=train[TARGET].values)


if __name__ == '__main__':
    main()
