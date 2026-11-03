#!/usr/bin/env python
"""Public notebook "S6E3 | TabM + Advanced Features | 0.91898 CV" run on our 5 folds.
https://www.kaggle.com/code/blamerx/s6e3-tabm-advanced-features-0-91898-cv

Features (the notebook's, label-free except the original dataset's churn rates; built once on train / test):
frequencies of the 3 numerics over train + original + test; charges deviation, monthly / total and total / tenure
ratios; service count, has-internet and has-phone flags; the original dataset's churn rate of every categorical and
numeric value (19, 0.5 if unseen); TotalCharges percentile ranks / z-scores against the original churners,
non-churners and all rows, conditional percentile ranks within InternetService / Contract, the MonthlyCharges residual
from the original InternetService mean and distances to the original churner / non-churner quartiles; 35 digit and
rounding features of tenure, MonthlyCharges and TotalCharges; 15 bigrams of Contract, InternetService, PaymentMethod,
OnlineSecurity, TechSupport, OnlineBackup and 4 trigrams of the first four.
Encodings per fold: the 3 numerics as strings and the 16 categoricals are mean-target-encoded with inner stratified
folds on the training rows (the validation / test rows get the training-row means), the 19 n-grams with the training-row
means; unseen values get 0.5. Numerics are standardized, the 16 categoricals ordinal-encoded.
Model: pytabkit TabM_D_Classifier (tabm-mini-normal, k=32, piecewise-linear embeddings of size 24, 3 blocks of 256,
dropout 0.2, lr 1e-3, weight decay 1e-3, batch 512, up to 50 epochs with patience 10 on the validation fold),
seed 42. The notebook seeds once (42) and trains its folds in one loop, so the folds run in order in one process.
Test = mean of the fold models.
Usage: python train.py"""
import gc
import random
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import TabM_D_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import TARGET, read_original, read_test, read_train  # noqa: E402
from lib.runner import kfold, run_all_folds, split, work_dir, working_directory  # noqa: E402

warnings.filterwarnings('ignore')

FOLDER = 'blamerx_tabm'
SEED = 42
TEST_FOLDS = 7

TABM_PARAMS = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'verbosity': 0,
    'arch_type': 'tabm-mini-normal',
    'tabm_k': 32,
    'num_emb_type': 'pwl',
    'd_embedding': 24,
    'batch_size': 512,
    'lr': 1e-3,
    'n_epochs': 50,
    'dropout': 0.2,
    'd_block': 256,
    'n_blocks': 3,
    'patience': 10,
    'weight_decay': 1e-3,
    'random_state': 42,
}

TOP_CATS_FOR_NGRAM = ['Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'OnlineBackup']
CATS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod']
SVC = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies']
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
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


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')


def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype='float32')
    return ((values - mu) / sigma).astype('float32')


def load_data():
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = train[TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    orig[TARGET] = orig[TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    for df in [train, test, orig]:
        for col in NUMS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    orig['TotalCharges'] = orig['TotalCharges'].fillna(orig['TotalCharges'].median())
    train_num_medians = train[NUMS].median()
    for df in [train, test]:
        for col in NUMS:
            df[col] = df[col].fillna(train_num_medians[col])
    print(f'Train: {train.shape} | Test: {test.shape} | Orig: {orig.shape} | target mean '
          f'{train[TARGET].mean():.6f}', flush=True)
    return train, test, orig


def add_base_features(train, test, orig):
    new_nums = []

    # 1. frequency encoding
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]], axis=0).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        new_nums.append(f'FREQ_{col}')

    # 2. arithmetic features
    for df in [train, test, orig]:
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    new_nums += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']

    # 3. service counts
    for df in [train, test, orig]:
        df['service_count'] = (df[SVC] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
    new_nums += ['service_count', 'has_internet', 'has_phone']

    # 4. churn rate of each value in the original dataset
    for col in CATS + NUMS:
        tmp = orig.groupby(col)[TARGET].mean()
        name = f'ORIG_proba_{col}'
        train = train.merge(tmp.rename(name), on=col, how='left')
        test = test.merge(tmp.rename(name), on=col, how='left')
        for df in [train, test]:
            df[name] = df[name].fillna(0.5).astype('float32')
        new_nums.append(name)

    # 5. distribution features against the original dataset
    orig_ch_tc = orig.loc[orig[TARGET] == 1, 'TotalCharges'].values
    orig_nc_tc = orig.loc[orig[TARGET] == 0, 'TotalCharges'].values
    orig_tc = orig['TotalCharges'].values
    orig_is_mc = orig.groupby('InternetService')['MonthlyCharges'].mean()
    for df in [train, test]:
        tc = df['TotalCharges'].values
        df['pctrank_nonchurner_TC'] = pctrank_against(tc, orig_nc_tc)
        df['pctrank_churner_TC'] = pctrank_against(tc, orig_ch_tc)
        df['pctrank_orig_TC'] = pctrank_against(tc, orig_tc)
        df['zscore_churn_gap_TC'] = (np.abs(zscore_against(tc, orig_ch_tc))
                                     - np.abs(zscore_against(tc, orig_nc_tc))).astype('float32')
        df['zscore_nonchurner_TC'] = zscore_against(tc, orig_nc_tc)
        df['pctrank_churn_gap_TC'] = (pctrank_against(tc, orig_ch_tc)
                                      - pctrank_against(tc, orig_nc_tc)).astype('float32')
        df['resid_IS_MC'] = (df['MonthlyCharges'] - df['InternetService'].map(orig_is_mc).fillna(0)).astype('float32')
        for cat_col, out_col in [('InternetService', 'cond_pctrank_IS_TC'), ('Contract', 'cond_pctrank_C_TC')]:
            vals = np.zeros(len(df), dtype='float32')
            for cv in orig[cat_col].unique():
                mask = df[cat_col] == cv
                ref = orig.loc[orig[cat_col] == cv, 'TotalCharges'].values
                if len(ref) > 0 and mask.sum() > 0:
                    vals[mask] = pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
            df[out_col] = vals
    new_nums += ['pctrank_nonchurner_TC', 'zscore_churn_gap_TC', 'pctrank_churn_gap_TC', 'resid_IS_MC',
                 'cond_pctrank_IS_TC', 'zscore_nonchurner_TC', 'pctrank_orig_TC', 'pctrank_churner_TC',
                 'cond_pctrank_C_TC']
    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        ch_q = np.quantile(orig_ch_tc, q_val)
        nc_q = np.quantile(orig_nc_tc, q_val)
        for df in [train, test]:
            df[f'dist_To_ch_{q_label}'] = np.abs(df['TotalCharges'] - ch_q).astype('float32')
            df[f'dist_To_nc_{q_label}'] = np.abs(df['TotalCharges'] - nc_q).astype('float32')
            df[f'qdist_gap_To_{q_label}'] = (df[f'dist_To_nc_{q_label}'] - df[f'dist_To_ch_{q_label}']).astype(
                'float32')
    new_nums += ['qdist_gap_To_q50', 'dist_To_ch_q50', 'dist_To_nc_q50', 'dist_To_nc_q25', 'qdist_gap_To_q25',
                 'dist_To_nc_q75', 'dist_To_ch_q75', 'qdist_gap_To_q75']

    # 6. digit features
    for df in [train, test]:
        t_int = df['tenure'].round().astype(int)
        t_str = t_int.astype(str)
        df['tenure_first_digit'] = t_str.str[0].astype(int)
        df['tenure_last_digit'] = t_str.str[-1].astype(int)
        df['tenure_second_digit'] = t_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tenure_mod10'] = t_int % 10
        df['tenure_mod12'] = t_int % 12
        df['tenure_num_digits'] = t_str.str.len()
        df['tenure_is_multiple_10'] = (t_int % 10 == 0).astype('float32')
        df['tenure_rounded_10'] = np.round(df['tenure'] / 10) * 10
        df['tenure_dev_from_round10'] = np.abs(df['tenure'] - df['tenure_rounded_10'])

        mc_str = df['MonthlyCharges'].astype(str).str.replace('.', '', regex=False)
        df['mc_first_digit'] = mc_str.str[0].astype(int)
        df['mc_last_digit'] = mc_str.str[-1].astype(int)
        df['mc_second_digit'] = mc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['mc_mod10'] = np.floor(df['MonthlyCharges']) % 10
        df['mc_mod100'] = np.floor(df['MonthlyCharges']) % 100
        df['mc_num_digits'] = np.floor(df['MonthlyCharges']).astype(int).astype(str).str.len()
        df['mc_is_multiple_10'] = (np.floor(df['MonthlyCharges']) % 10 == 0).astype('float32')
        df['mc_is_multiple_50'] = (np.floor(df['MonthlyCharges']) % 50 == 0).astype('float32')
        df['mc_rounded_10'] = np.round(df['MonthlyCharges'] / 10) * 10
        df['mc_fractional'] = df['MonthlyCharges'] - np.floor(df['MonthlyCharges'])
        df['mc_dev_from_round10'] = np.abs(df['MonthlyCharges'] - df['mc_rounded_10'])

        tc_str = df['TotalCharges'].astype(str).str.replace('.', '', regex=False)
        df['tc_first_digit'] = tc_str.str[0].astype(int)
        df['tc_last_digit'] = tc_str.str[-1].astype(int)
        df['tc_second_digit'] = tc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tc_mod10'] = np.floor(df['TotalCharges']) % 10
        df['tc_mod100'] = np.floor(df['TotalCharges']) % 100
        df['tc_num_digits'] = np.floor(df['TotalCharges']).astype(int).astype(str).str.len()
        df['tc_is_multiple_10'] = (np.floor(df['TotalCharges']) % 10 == 0).astype('float32')
        df['tc_is_multiple_100'] = (np.floor(df['TotalCharges']) % 100 == 0).astype('float32')
        df['tc_rounded_100'] = np.round(df['TotalCharges'] / 100) * 100
        df['tc_fractional'] = df['TotalCharges'] - np.floor(df['TotalCharges'])
        df['tc_dev_from_round100'] = np.abs(df['TotalCharges'] - df['tc_rounded_100'])

        df['tenure_years'] = t_int // 12
        df['tenure_months_in_year'] = t_int % 12
        df['mc_per_digit'] = df['MonthlyCharges'] / (df['mc_num_digits'] + 0.001)
        df['tc_per_digit'] = df['TotalCharges'] / (df['tc_num_digits'] + 0.001)
        for c in DIGIT_FEATURES:
            df[c] = df[c].astype('float32')
    new_nums += DIGIT_FEATURES

    # 7. n-gram features
    bigram_cols, trigram_cols = [], []
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f'BG_{c1}_{c2}'
        for df in [train, test]:
            df[col_name] = df[c1].astype(str) + '_' + df[c2].astype(str)
        bigram_cols.append(col_name)
    for c1, c2, c3 in combinations(TOP_CATS_FOR_NGRAM[:4], 3):
        col_name = f'TG_{c1}_{c2}_{c3}'
        for df in [train, test]:
            df[col_name] = df[c1].astype(str) + '_' + df[c2].astype(str) + '_' + df[c3].astype(str)
        trigram_cols.append(col_name)
    ngram_cols = bigram_cols + trigram_cols

    num_as_cat = []
    for col in NUMS:
        new_col = f'CAT_{col}'
        num_as_cat.append(new_col)
        for df in [train, test]:
            df[new_col] = df[col].astype(str)
    te_columns = num_as_cat + CATS

    print(f'numerical features: {len(NUMS + new_nums)} | target-encoded columns: {len(te_columns)} | n-grams: '
          f'{len(ngram_cols)}', flush=True)
    return train, test, new_nums, te_columns, ngram_cols


def main():
    train, test, orig = load_data()
    train, test, new_nums, te_columns, ngram_cols = add_base_features(train, test, orig)
    y_all = train[TARGET].values

    def cross_validate(n_folds):
        seed_everything(SEED)
        fold_ids = kfold(y_all, n_folds)
        skf_inner = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        oof = np.zeros(len(train), dtype='float32')
        pred = np.zeros(len(test), dtype='float32')

        for fold in range(n_folds):
            train_idx, val_idx = split(fold_ids, fold)
            X_tr = train.iloc[train_idx].reset_index(drop=True).copy()
            y_tr = y_all[train_idx]
            X_val = train.iloc[val_idx].reset_index(drop=True).copy()
            y_val = y_all[val_idx]
            X_te = test.copy()

            te_feat_names = [f'TE1_{col}_mean' for col in te_columns]
            for df in [X_tr, X_val, X_te]:
                for c in te_feat_names:
                    df[c] = 0.0

            X_tr[TARGET] = y_tr
            for in_tr, in_va in skf_inner.split(X_tr, y_tr):
                X_tr2 = X_tr.iloc[in_tr]
                for col in te_columns:
                    tmp = X_tr2.groupby(col)[TARGET].mean().rename(f'TE1_{col}_mean')
                    merged = X_tr.iloc[in_va][[col]].merge(tmp, on=col, how='left')
                    X_tr.loc[X_tr.index[in_va], f'TE1_{col}_mean'] = merged[f'TE1_{col}_mean'].values
            for col in te_columns:
                tmp = X_tr.groupby(col)[TARGET].mean().rename(f'TE1_{col}_mean')
                X_val[f'TE1_{col}_mean'] = X_val[[col]].merge(tmp, on=col, how='left')[f'TE1_{col}_mean'].values
                X_te[f'TE1_{col}_mean'] = X_te[[col]].merge(tmp, on=col, how='left')[f'TE1_{col}_mean'].values
            X_tr.drop(columns=[TARGET], inplace=True)
            for df in [X_tr, X_val, X_te]:
                for c in te_feat_names:
                    df[c] = df[c].fillna(0.5).astype('float32')

            ng_te_feat_names = [f'TE_ng_{col}' for col in ngram_cols]
            X_tr[TARGET] = y_tr
            for col in ngram_cols:
                ng_te = X_tr.groupby(col)[TARGET].mean()
                ng_n = f'TE_ng_{col}'
                X_tr[ng_n] = X_tr[col].map(ng_te).fillna(0.5).astype('float32')
                X_val[ng_n] = X_val[col].map(ng_te).fillna(0.5).astype('float32')
                X_te[ng_n] = X_te[col].map(ng_te).fillna(0.5).astype('float32')
            X_tr.drop(columns=[TARGET], inplace=True)

            all_nums_final = NUMS + new_nums + te_feat_names + ng_te_feat_names
            all_cats_final = CATS
            if fold == 0:
                print(f'numeric features: {len(all_nums_final)} | categorical features: {len(all_cats_final)}',
                      flush=True)

            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoder.fit(X_tr[all_cats_final].astype(str))
            X_tr_cat = encoder.transform(X_tr[all_cats_final].astype(str))
            X_val_cat = encoder.transform(X_val[all_cats_final].astype(str))
            X_te_cat = encoder.transform(X_te[all_cats_final].astype(str))

            for df in [X_tr, X_val, X_te]:
                df[all_nums_final] = df[all_nums_final].fillna(0).astype('float32')
            scaler = StandardScaler()
            X_tr_num = scaler.fit_transform(X_tr[all_nums_final])
            X_val_num = scaler.transform(X_val[all_nums_final])
            X_te_num = scaler.transform(X_te[all_nums_final])

            all_cols = all_nums_final + all_cats_final
            X_tr_final = pd.DataFrame(np.hstack([X_tr_num, X_tr_cat]), columns=all_cols)
            X_val_final = pd.DataFrame(np.hstack([X_val_num, X_val_cat]), columns=all_cols)
            X_te_final = pd.DataFrame(np.hstack([X_te_num, X_te_cat]), columns=all_cols)
            for c in all_cats_final:
                X_tr_final[c] = X_tr_final[c].astype(int)
                X_val_final[c] = X_val_final[c].astype(int)
                X_te_final[c] = X_te_final[c].astype(int)

            with working_directory(work_dir(FOLDER)):
                model = TabM_D_Classifier(**TABM_PARAMS)
                model.fit(X_tr_final, y_tr, X_val=X_val_final, y_val=y_val, cat_col_names=all_cats_final)
                val_probs = model.predict_proba(X_val_final)[:, 1]
                test_probs = model.predict_proba(X_te_final)[:, 1]
            oof[val_idx] = val_probs.astype('float32')
            pred += test_probs.astype('float32') / n_folds
            print(f'fold {fold}: AUC {roc_auc_score(y_val, val_probs):.6f}', flush=True)

            del model, X_tr, X_val, X_te, X_tr_final, X_val_final, X_te_final
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return oof, pred

    run_all_folds(FOLDER, FOLDER, cross_validate, test_folds=TEST_FOLDS, y=y_all)


if __name__ == '__main__':
    main()
