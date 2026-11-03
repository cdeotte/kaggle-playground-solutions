#!/usr/bin/env python
"""GPU CatBoost (depth 5, Balanced) on original-data probability / distribution / digit features and fold-local target
statistics; every model input is numeric (pair catboost_orig_dist_te_stats_ngrams_balanced).

Adapted from the CatBoost pipeline of Artem's (artemevstafyev) public notebook xgb-cb-best-cv-auc-0-91930-blend.
Features:
  - frequency of tenure / MonthlyCharges / TotalCharges over train + test + original
  - TotalCharges - tenure x MonthlyCharges, MonthlyCharges / (TotalCharges + 1), TotalCharges / (tenure + 1); number
    of Yes services, has-internet / has-phone flags
  - original churn rate of every value of the 19 columns (0.5 when unseen)
  - TotalCharges percentile among the original churners / non-churners / all rows, churner minus non-churner
    percentile and |z-score| gaps, non-churner z-score, percentile within the original InternetService / Contract
    group, MonthlyCharges minus the original InternetService mean; distances to the original churner / non-churner
    TotalCharges quartiles and their gaps
  - digit features: first / second / last digit, mod 10 / 12 / 100, number of digits, multiple-of-10 / 50 / 100
    flags, rounding deviations, fractional parts, tenure years / months, value per digit
Per outer fold, with an inner StratifiedKFold (seed 42) of the fold's training rows: std / min / max of the target per
value of the 16 categoricals and the 3 numerics as categories (inner out-of-fold for the training rows, full-fold
statistics for the validation / test rows) and a scikit-learn TargetEncoder (binary, smooth 'auto') mean of the same
19 columns; target means of 15 bigrams and 4 trigrams of Contract, InternetService, PaymentMethod, OnlineSecurity,
TechSupport, PaperlessBilling (0.5 when unseen). The raw categories and n-grams are then dropped (178 features).
CatBoost (GPU): depth 5, auto_class_weights Balanced, up to 50,000 iterations with early stopping on the validation
fold (use_best_model), seed 42. Test = mean of the fold models.

Usage: python train_orig_dist_te_stats.py"""
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402
from models.catboost.lib.cv import run_folds  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'catboost_orig_dist_te_stats_ngrams_balanced'
RANDOM_SEED = 42
OOF_RUN = dict(n_folds=5, inner_folds=5, learning_rate=0.05, early_stopping_rounds=100)
TEST_RUN = dict(n_folds=15, inner_folds=10, learning_rate=0.025, early_stopping_rounds=200)

TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'PaperlessBilling',
]
CATS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
SERVICE_COLS = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
STATS = ['std', 'min', 'max']


def cb_params(learning_rate, early_stopping_rounds):
    return {
        'n_estimators': 50000,
        'learning_rate': learning_rate,
        'eval_metric': 'AUC',
        'max_depth': 5,
        'auto_class_weights': 'Balanced',
        'random_state': RANDOM_SEED,
        'early_stopping_rounds': early_stopping_rounds,
        'verbose': False,
        'task_type': 'GPU',
    }


def reduce_mem_float_int(df, skip_cols=None):
    skip_cols = skip_cols or []
    for col in df.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_integer_dtype(df[col]):
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype(np.uint8)
                elif df[col].max() < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif df[col].max() < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if np.iinfo(np.int8).min <= df[col].min() and df[col].max() <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif np.iinfo(np.int16).min <= df[col].min() and df[col].max() <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif np.iinfo(np.int32).min <= df[col].min() and df[col].max() <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
    return df


def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')


def zscore_against(values, reference):
    mu = np.mean(reference)
    sigma = np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype='float32')
    return ((values - mu) / sigma).astype('float32')


def build_features():
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = train[TARGET].map({'No': 0, 'Yes': 1}).astype(np.int8)
    orig[TARGET] = orig[TARGET].map({'No': 0, 'Yes': 1}).astype(np.int8)
    orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce')
    orig['TotalCharges'] = orig['TotalCharges'].fillna(orig['TotalCharges'].median())

    new_nums = []
    num_as_cat = []
    for df in [train, test, orig]:
        for c in CATS:
            df[c] = df[c].astype(str)

    # 1. frequency encoding
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]], axis=0).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        new_nums.append(f'FREQ_{col}')

    # 2. arithmetic interactions
    for df in [train, test, orig]:
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    new_nums += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']

    # 3. service counts
    for df in [train, test, orig]:
        df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
    new_nums += ['service_count', 'has_internet', 'has_phone']

    # 4. original churn rate of every value
    for col in CATS + NUMS:
        tmp = orig.groupby(col)[TARGET].mean()
        name = f'ORIG_proba_{col}'
        train = train.merge(tmp.rename(name), on=col, how='left')
        test = test.merge(tmp.rename(name), on=col, how='left')
        train[name] = train[name].fillna(0.5).astype('float32')
        test[name] = test[name].fillna(0.5).astype('float32')
        new_nums.append(name)

    # 5. TotalCharges distribution against the original churners / non-churners
    orig_churner_tc = orig.loc[orig[TARGET] == 1, 'TotalCharges'].values
    orig_nonchurner_tc = orig.loc[orig[TARGET] == 0, 'TotalCharges'].values
    orig_tc = orig['TotalCharges'].values
    orig_is_mc_mean = orig.groupby('InternetService')['MonthlyCharges'].mean()
    for df in [train, test]:
        tc = df['TotalCharges'].values
        df['pctrank_nonchurner_TC'] = pctrank_against(tc, orig_nonchurner_tc)
        df['pctrank_churner_TC'] = pctrank_against(tc, orig_churner_tc)
        df['pctrank_orig_TC'] = pctrank_against(tc, orig_tc)
        df['zscore_churn_gap_TC'] = (
            np.abs(zscore_against(tc, orig_churner_tc)) - np.abs(zscore_against(tc, orig_nonchurner_tc))
        ).astype('float32')
        df['zscore_nonchurner_TC'] = zscore_against(tc, orig_nonchurner_tc)
        df['pctrank_churn_gap_TC'] = (
            pctrank_against(tc, orig_churner_tc) - pctrank_against(tc, orig_nonchurner_tc)
        ).astype('float32')
        df['resid_IS_MC'] = (
            df['MonthlyCharges'] - df['InternetService'].map(orig_is_mc_mean).fillna(0)
        ).astype('float32')
        for cat_col, name in [('InternetService', 'cond_pctrank_IS_TC'), ('Contract', 'cond_pctrank_C_TC')]:
            vals = np.zeros(len(df), dtype='float32')
            for cat_val in orig[cat_col].unique():
                mask = df[cat_col] == cat_val
                ref = orig.loc[orig[cat_col] == cat_val, 'TotalCharges'].values
                if len(ref) > 0 and mask.sum() > 0:
                    vals[mask] = pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
            df[name] = vals
    new_nums += [
        'pctrank_nonchurner_TC', 'zscore_churn_gap_TC', 'pctrank_churn_gap_TC',
        'resid_IS_MC', 'cond_pctrank_IS_TC', 'zscore_nonchurner_TC',
        'pctrank_orig_TC', 'pctrank_churner_TC', 'cond_pctrank_C_TC'
    ]

    # 6. distances to the original quartiles
    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        ch_q = np.quantile(orig_churner_tc, q_val)
        nc_q = np.quantile(orig_nonchurner_tc, q_val)
        for df in [train, test]:
            df[f'dist_To_ch_{q_label}'] = np.abs(df['TotalCharges'] - ch_q).astype('float32')
            df[f'dist_To_nc_{q_label}'] = np.abs(df['TotalCharges'] - nc_q).astype('float32')
            df[f'qdist_gap_To_{q_label}'] = (
                df[f'dist_To_nc_{q_label}'] - df[f'dist_To_ch_{q_label}']
            ).astype('float32')
    new_nums += [
        'qdist_gap_To_q50', 'dist_To_ch_q50', 'dist_To_nc_q50',
        'dist_To_nc_q25', 'qdist_gap_To_q25',
        'dist_To_nc_q75', 'dist_To_ch_q75', 'qdist_gap_To_q75'
    ]

    # 7. numerics as categories
    for col in NUMS:
        new_col = f'CAT_{col}'
        num_as_cat.append(new_col)
        for df in [train, test]:
            df[new_col] = df[col].astype(str).astype('category')

    # 8. digit features
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
    for df in [train, test]:
        t_str = df['tenure'].astype(int).astype(str)
        df['tenure_first_digit'] = t_str.str[0].astype(np.int16)
        df['tenure_last_digit'] = t_str.str[-1].astype(np.int16)
        df['tenure_second_digit'] = t_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0).astype(np.int16)
        df['tenure_mod10'] = (df['tenure'] % 10).astype(np.float32)
        df['tenure_mod12'] = (df['tenure'] % 12).astype(np.float32)
        df['tenure_num_digits'] = t_str.str.len().astype(np.int16)
        df['tenure_is_multiple_10'] = (df['tenure'] % 10 == 0).astype('float32')
        df['tenure_rounded_10'] = (np.round(df['tenure'] / 10) * 10).astype(np.float32)
        df['tenure_dev_from_round10'] = np.abs(df['tenure'] - df['tenure_rounded_10']).astype(np.float32)

        mc_str = df['MonthlyCharges'].astype(str).str.replace('.', '', regex=False)
        df['mc_first_digit'] = mc_str.str[0].astype(np.int16)
        df['mc_last_digit'] = mc_str.str[-1].astype(np.int16)
        df['mc_second_digit'] = mc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0).astype(np.int16)
        df['mc_mod10'] = (np.floor(df['MonthlyCharges']) % 10).astype(np.float32)
        df['mc_mod100'] = (np.floor(df['MonthlyCharges']) % 100).astype(np.float32)
        df['mc_num_digits'] = np.floor(df['MonthlyCharges']).astype(int).astype(str).str.len().astype(np.int16)
        df['mc_is_multiple_10'] = (np.floor(df['MonthlyCharges']) % 10 == 0).astype('float32')
        df['mc_is_multiple_50'] = (np.floor(df['MonthlyCharges']) % 50 == 0).astype('float32')
        df['mc_rounded_10'] = (np.round(df['MonthlyCharges'] / 10) * 10).astype(np.float32)
        df['mc_fractional'] = (df['MonthlyCharges'] - np.floor(df['MonthlyCharges'])).astype(np.float32)
        df['mc_dev_from_round10'] = np.abs(df['MonthlyCharges'] - df['mc_rounded_10']).astype(np.float32)

        tc_str = df['TotalCharges'].astype(str).str.replace('.', '', regex=False)
        df['tc_first_digit'] = tc_str.str[0].astype(np.int16)
        df['tc_last_digit'] = tc_str.str[-1].astype(np.int16)
        df['tc_second_digit'] = tc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0).astype(np.int16)
        df['tc_mod10'] = (np.floor(df['TotalCharges']) % 10).astype(np.float32)
        df['tc_mod100'] = (np.floor(df['TotalCharges']) % 100).astype(np.float32)
        df['tc_num_digits'] = np.floor(df['TotalCharges']).astype(int).astype(str).str.len().astype(np.int16)
        df['tc_is_multiple_10'] = (np.floor(df['TotalCharges']) % 10 == 0).astype('float32')
        df['tc_is_multiple_100'] = (np.floor(df['TotalCharges']) % 100 == 0).astype('float32')
        df['tc_rounded_100'] = (np.round(df['TotalCharges'] / 100) * 100).astype(np.float32)
        df['tc_fractional'] = (df['TotalCharges'] - np.floor(df['TotalCharges'])).astype(np.float32)
        df['tc_dev_from_round100'] = np.abs(df['TotalCharges'] - df['tc_rounded_100']).astype(np.float32)

        df['tenure_years'] = (df['tenure'] // 12).astype(np.float32)
        df['tenure_months_in_year'] = (df['tenure'] % 12).astype(np.float32)
        df['mc_per_digit'] = (df['MonthlyCharges'] / (df['mc_num_digits'] + 0.001)).astype(np.float32)
        df['tc_per_digit'] = (df['TotalCharges'] / (df['tc_num_digits'] + 0.001)).astype(np.float32)
    new_nums += digit_features

    # bigrams / trigrams
    ngram_cols = []
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f'BG_{c1}_{c2}'
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + '_' + df[c2].astype(str)).astype('category')
        ngram_cols.append(col_name)
    for c1, c2, c3 in combinations(TOP_CATS_FOR_NGRAM[:4], 3):
        col_name = f'TG_{c1}_{c2}_{c3}'
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + '_' + df[c2].astype(str) + '_' + df[c3].astype(str)).astype('category')
        ngram_cols.append(col_name)

    features = NUMS + CATS + new_nums + num_as_cat + ngram_cols
    train = reduce_mem_float_int(train, skip_cols=CATS + num_as_cat + ngram_cols + [TARGET])
    test = reduce_mem_float_int(test, skip_cols=CATS + num_as_cat + ngram_cols)
    return train, test, features, num_as_cat, ngram_cols


def make_fit_fold(train, test, features, num_as_cat, ngram_cols, inner_folds, learning_rate, early_stopping_rounds):
    te_columns = num_as_cat + CATS
    to_remove = num_as_cat + CATS + ngram_cols
    skf_inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_SEED)

    def fit_fold(fold, train_idx, val_idx, seed):
        X_tr = train.loc[train_idx, features + [TARGET]].reset_index(drop=True).copy()
        y_tr = train.loc[train_idx, TARGET].values
        X_val = train.loc[val_idx, features].reset_index(drop=True).copy()
        y_val = train.loc[val_idx, TARGET].values
        X_te = test[features].reset_index(drop=True).copy()

        # inner-fold std / min / max of the target per value of the categoricals and numerics-as-categories
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
                X_tr[c] = pd.to_numeric(X_tr[c], errors='coerce').fillna(0).astype('float32')
                X_val[c] = pd.to_numeric(X_val[c], errors='coerce').fillna(0).astype('float32')
                X_te[c] = pd.to_numeric(X_te[c], errors='coerce').fillna(0).astype('float32')

        # inner-fold target mean of the n-grams
        for in_tr, in_va in skf_inner.split(X_tr, y_tr):
            X_tr2 = X_tr.loc[in_tr].copy()
            X_va2 = X_tr.loc[in_va].copy()
            for col in ngram_cols:
                ng_te = X_tr2.groupby(col, observed=False)[TARGET].mean()
                mapped = X_va2[col].astype(str).map(ng_te)
                X_tr.loc[in_va, f'TE_ng_{col}'] = (
                    pd.Series(mapped).pipe(pd.to_numeric, errors='coerce').fillna(0.5).astype('float32').values
                )
        for col in ngram_cols:
            ng_te = X_tr.groupby(col, observed=False)[TARGET].mean()
            ng_name = f'TE_ng_{col}'
            X_val[ng_name] = pd.to_numeric(
                X_val[col].astype(str).map(ng_te), errors='coerce').fillna(0.5).astype('float32')
            X_te[ng_name] = pd.to_numeric(
                X_te[col].astype(str).map(ng_te), errors='coerce').fillna(0.5).astype('float32')
            if ng_name in X_tr.columns:
                X_tr[ng_name] = pd.to_numeric(X_tr[ng_name], errors='coerce').fillna(0.5).astype('float32')
            else:
                X_tr[ng_name] = np.float32(0.5)

        # scikit-learn TargetEncoder means of the categoricals and numerics-as-categories
        te_mean_cols = [f'TE_{col}' for col in te_columns]
        te = TargetEncoder(cv=inner_folds, shuffle=True, smooth='auto', target_type='binary', random_state=RANDOM_SEED)
        X_tr[te_mean_cols] = te.fit_transform(X_tr[te_columns], y_tr)
        X_val[te_mean_cols] = te.transform(X_val[te_columns])
        X_te[te_mean_cols] = te.transform(X_te[te_columns])

        # drop the raw categories and n-grams; everything left is numeric
        for df in [X_tr, X_val, X_te]:
            for c in CATS + num_as_cat:
                if c in df.columns:
                    df[c] = df[c].astype(str).astype('category')
            df.drop(columns=[c for c in to_remove if c in df.columns], inplace=True, errors='ignore')
        X_tr.drop(columns=[TARGET], inplace=True, errors='ignore')
        cols_model = X_tr.columns.tolist()
        X_val = X_val[cols_model].copy()
        X_te = X_te[cols_model].copy()
        for df in [X_tr, X_val, X_te]:
            for c in df.columns:
                if str(df[c].dtype) == 'category':
                    df[c] = df[c].astype(str)
                elif not pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('float32')
                else:
                    df[c] = df[c].astype('float32')
        if fold == 1:
            print(f'model features: {len(cols_model)}', flush=True)

        model = CatBoostClassifier(**cb_params(learning_rate, early_stopping_rounds))
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        p_val = model.predict_proba(X_val)[:, 1].astype(np.float32)
        p_te = model.predict_proba(X_te)[:, 1].astype(np.float32)
        print(f'fold {fold}: best iteration {model.get_best_iteration()}', flush=True)
        del model, X_tr, X_val, X_te
        gc.collect()
        return p_val, p_te

    return fit_fold


def main():
    np.random.seed(RANDOM_SEED)
    train, test, features, num_as_cat, ngram_cols = build_features()
    y = train[TARGET].values

    def run(n_folds, inner_folds, learning_rate, early_stopping_rounds):
        fit_fold = make_fit_fold(train, test, features, num_as_cat, ngram_cols, inner_folds, learning_rate,
                                 early_stopping_rounds)
        return run_folds(fit_fold, y, len(test), n_folds=n_folds, seed=RANDOM_SEED, name=PAIR)

    oof, _ = run(**OOF_RUN)
    _, test_pred = run(**TEST_RUN)
    print(f'{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
