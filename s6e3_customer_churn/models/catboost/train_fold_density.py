#!/usr/bin/env python
"""GPU CatBoost on row-level artifacts, fold-local frequency / density / group statistics and native categories.

Recipes (pair name):
  catboost_fold_density_knn_orig_priors   the original data is used for lookups; fold statistics come from the fold's
                                          training rows (486 columns: 202 row-level, 234 fold, 50 categorical)
  catboost_fold_density_knn_orig_rows     no original-data lookups; the 7,043 original rows are appended to the
                                          training rows of every fold and every fold statistic comes from these rows
                                          (304 columns: 110 row-level, 144 fold, 50 categorical)

Numerics (float32): tenure, MonthlyCharges, TotalCharges (blank TotalCharges get the frame's median); the 16 other
columns are strings.
Row-level features (both recipes):
  - TotalCharges - tenure x MonthlyCharges (and absolute value), charge ratios, TotalCharges / (tenure + 1),
    tenure x MonthlyCharges, tenure x TotalCharges; log1p / sqrt / 1 / (1 + x) / missing flag of the numerics
  - number of Yes / No / no-service answers of the 8 services, has-internet / has-phone flags, Yes / No / other flags
    of 11 Yes / No columns
  - string artifacts of the numerics: string length, number of decimals, trailing zeros, significant decimals,
    exact at 0 / 1 / 2 decimals, Python string hash mod 997, 4 integer and 4 decimal digit positions
  - consistency of TotalCharges with tenure x MonthlyCharges: residual, |residual|, relative error, exact flag
Row-level features with original-data lookups (first recipe only):
  - original churn rate of every value of the 19 columns
  - TotalCharges percentile ranks / z-scores among all original rows, churners and non-churners, the churner minus
    non-churner gaps, percentile within the original InternetService / Contract group, distances to the original
    churner / non-churner TotalCharges quartiles, MonthlyCharges minus the original InternetService mean
  - value seen in the original data, nearest original value, its distance (and log), value / nearest value, for each
    numeric; (tenure, MonthlyCharges, TotalCharges) triple seen in the original data and its original count
  - residual minus the original mean residual per Contract / InternetService / PaymentMethod / MonthlyCharges value
    and its z-score; z-scores of the numerics within the original Contract / InternetService / PaymentMethod groups
  - in the numeric space standardized with the original means / stds: distance and per-numeric noise to the nearest
    original row, mean distance to 5; churn rate of the 1 / 3 / 5 / 10 nearest original rows, distance-weighted rate,
    variance, gradient and range, distances to the nearest original churner / non-churner and their ratio
Fold features (statistics of the reference rows mapped onto the training, validation and test rows):
  - frequency and percentile rank of the numerics; joint counts / frequencies / PMI of the 3 numeric pairs and the
    triple (singleton flag); frequency of each numeric given Contract / InternetService / PaymentMethod and its ratio
    to the marginal frequency; counts of tenure, MonthlyCharges, TotalCharges; counts and rare flags (<= 50) of the 16
    categoricals (the reference is the fold's training rows + the original rows in both recipes)
  - ball counts (radius 0.05 / 0.2 / 0.5) and distances to the 5th / 10th / 20th nearest neighbours (+ means, log)
    among the fold's training rows in the standardized numeric space
  - means of the 3 numerics per categorical value, and the churn rate per categorical value smoothed toward the fold
    mean (weight 20), from the fold's training rows
  - first recipe only: original row count, original churn rate and count ratio of the high-cardinality values;
    train-vs-original drift log1p(p_fold / p_original), original churn rate and original count of the 16
    categoricals, the numerics, MonthlyCharges / TotalCharges rounded to 1 / 0 / -1 decimals and their cents
CatBoost-native categoricals (50): the raw categoricals; every numeric as an exact / rounded (0 and 1 decimals) /
integer / 50-quantile bin (per frame) / even floor string; 15 bigrams and a trigram of Contract, InternetService,
PaymentMethod, OnlineSecurity, TechSupport, PaperlessBilling.
CatBoost (GPU): depth 8, lr 0.03, l2 3, random_strength 0.5, Bayesian bagging_temperature 0.7, border_count 254,
one_hot_max_size 4, max_ctr_complexity 2, up to 5000 iterations with early stopping (300) on the validation fold
(use_best_model=True). Test = mean of the fold models.

Usage: python train_fold_density.py <pair>"""
import argparse
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402
from models.catboost.lib.cv import run_folds  # noqa: E402

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

CATS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod',
]
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'PaperlessBilling',
]
SERVICE_COLS = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies',
]
YN_COLS = [
    'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines',
]
GROUP_CATS = ['Contract', 'InternetService', 'PaymentMethod']
HIGH_CARD_THRESHOLD = 7
N_INT_DIGITS = 4
N_DEC_DIGITS = 4
K_ORIG = 10
TE_SMOOTH = 20

RECIPES = {
    'catboost_fold_density_knn_orig_priors': dict(orig_rows=False, test_folds=25, test_seed=123),
    'catboost_fold_density_knn_orig_rows': dict(orig_rows=True, test_folds=25, test_seed=42),
}


def cb_params(seed):
    return dict(
        iterations=5000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3.0,
        random_strength=0.5,
        bagging_temperature=0.7,
        border_count=254,
        grow_policy='SymmetricTree',
        bootstrap_type='Bayesian',
        eval_metric='AUC',
        loss_function='Logloss',
        task_type='GPU',
        random_seed=seed,
        verbose=200,
        early_stopping_rounds=300,
        use_best_model=True,
        one_hot_max_size=4,
        max_ctr_complexity=2,
    )


def sanitize_series(s, fill='missing'):
    return s.astype('string').fillna(fill).str.strip()


def sanitize_object_series(s, fill='missing'):
    s = s.astype('object')
    return s.where(pd.notna(s), fill)


def pctrank_against(values, reference):
    reference = np.asarray(reference)
    if len(reference) == 0:
        return np.zeros(len(values), dtype='float32')
    return (np.searchsorted(np.sort(reference), values) / len(reference)).astype('float32')


def zscore_against(values, reference):
    reference = np.asarray(reference)
    if len(reference) == 0:
        return np.zeros(len(values), dtype='float32')
    mu, sigma = np.mean(reference), np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype='float32')
    return ((values - mu) / sigma).astype('float32')


def load_data():
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = labels(train[TARGET])
    orig[TARGET] = labels(orig[TARGET])
    for df in (train, test, orig):
        for c in NUMS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        for c in CATS:
            df[c] = sanitize_object_series(df[c]).astype(str)
    return train, test, orig


# ------------------------------------------------------------------------------------------------------------------
# row-level features (own values, plus original-data lookups when `orig` is given)
# ------------------------------------------------------------------------------------------------------------------
def build_safe_features(df, orig=None):
    feats = {}
    n = len(df)
    tc_vals = df['TotalCharges'].astype('float64').values
    mc_vals = df['MonthlyCharges'].astype('float64').values
    ten_vals = df['tenure'].astype('float64').values
    tc = df['TotalCharges'].astype('float32').values
    ten = df['tenure'].astype('float32').values
    mc = df['MonthlyCharges'].astype('float32').values

    dev = tc - ten * mc
    feats['charges_deviation'] = dev.astype('float32')
    feats['abs_charges_deviation'] = np.abs(dev).astype('float32')
    feats['monthly_to_total_ratio'] = (mc / (tc + 1)).astype('float32')
    feats['total_to_monthly_ratio'] = (tc / (mc + 1)).astype('float32')
    feats['avg_monthly_charges'] = (tc / (ten + 1)).astype('float32')
    feats['tenure_x_monthly'] = (ten * mc).astype('float32')
    feats['tenure_x_total'] = (ten * tc).astype('float32')
    for col in NUMS:
        v = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float32').values
        feats[f'LOG1P_{col}'] = np.log1p(np.clip(v, 0, None)).astype('float32')
        feats[f'SQRT_{col}'] = np.sqrt(np.clip(v, 0, None)).astype('float32')
        feats[f'INV1P_{col}'] = (1.0 / (1.0 + np.clip(v, 0, None))).astype('float32')
        feats[f'MISS_{col}'] = pd.to_numeric(df[col], errors='coerce').isna().astype('int8').values

    svc = df[SERVICE_COLS + ['InternetService']].copy()
    for c in svc.columns:
        svc[c] = sanitize_series(svc[c])
    feats['service_count'] = (svc[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32').values
    feats['service_no_count'] = (svc[SERVICE_COLS] == 'No').sum(axis=1).astype('float32').values
    feats['service_other_count'] = svc[SERVICE_COLS].isin(
        ['No phone service', 'No internet service']).sum(axis=1).astype('float32').values
    feats['has_internet'] = (svc['InternetService'] != 'No').astype('float32').values
    feats['has_phone'] = (svc['PhoneService'] == 'Yes').astype('float32').values
    for c in YN_COLS:
        s = sanitize_series(df[c])
        feats[f'ISYES_{c}'] = (s == 'Yes').astype('float32').values
        feats[f'ISNO_{c}'] = (s == 'No').astype('float32').values
        feats[f'ISOTHER_{c}'] = (~s.isin(['Yes', 'No'])).astype('float32').values

    if orig is not None:
        for col in CATS + NUMS:
            tmp = orig.groupby(col, observed=False)[TARGET].mean()
            feats[f'ORIG_proba_{col}'] = df[col].map(tmp).fillna(0.5).astype('float32').values

        orig_churner_tc = orig.loc[orig[TARGET] == 1, 'TotalCharges'].values
        orig_nonchurner_tc = orig.loc[orig[TARGET] == 0, 'TotalCharges'].values
        orig_tc = orig['TotalCharges'].values
        orig_is_mc_mean = orig.groupby('InternetService', observed=False)['MonthlyCharges'].mean()
        feats['pctrank_nonchurner_TC'] = pctrank_against(tc_vals, orig_nonchurner_tc)
        feats['pctrank_churner_TC'] = pctrank_against(tc_vals, orig_churner_tc)
        feats['pctrank_orig_TC'] = pctrank_against(tc_vals, orig_tc)
        feats['zscore_churn_gap_TC'] = (
            np.abs(zscore_against(tc_vals, orig_churner_tc)) - np.abs(zscore_against(tc_vals, orig_nonchurner_tc))
        ).astype('float32')
        feats['zscore_nonchurner_TC'] = zscore_against(tc_vals, orig_nonchurner_tc)
        feats['pctrank_churn_gap_TC'] = (
            pctrank_against(tc_vals, orig_churner_tc) - pctrank_against(tc_vals, orig_nonchurner_tc)
        ).astype('float32')
        feats['resid_IS_MC'] = (
            df['MonthlyCharges'].values - df['InternetService'].map(orig_is_mc_mean).fillna(0).values
        ).astype('float32')
        for cat_col, cat_label in [('InternetService', 'IS'), ('Contract', 'C')]:
            cond = np.zeros(n, dtype='float32')
            for cv in orig[cat_col].dropna().unique():
                mask = (df[cat_col] == cv).values
                ref = orig.loc[orig[cat_col] == cv, 'TotalCharges'].values
                if len(ref) > 0 and mask.sum() > 0:
                    cond[mask] = pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
            feats[f'cond_pctrank_{cat_label}_TC'] = cond
        for ql, qv in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
            ch_q = np.quantile(orig_churner_tc, qv)
            nc_q = np.quantile(orig_nonchurner_tc, qv)
            feats[f'dist_To_ch_{ql}'] = np.abs(tc_vals - ch_q).astype('float32')
            feats[f'dist_To_nc_{ql}'] = np.abs(tc_vals - nc_q).astype('float32')

        for col in NUMS:
            combined_vals = df[col].values
            feats[f'EXISTS_IN_ORIG_{col}'] = pd.Series(combined_vals).isin(
                set(orig[col].dropna().values)).astype('float32').values
            orig_arr = np.sort(orig[col].dropna().values)
            idx = np.clip(np.searchsorted(orig_arr, combined_vals), 0, len(orig_arr) - 1)
            dist_right = np.abs(combined_vals - orig_arr[idx])
            idx_left = np.clip(idx - 1, 0, len(orig_arr) - 1)
            dist_left = np.abs(combined_vals - orig_arr[idx_left])
            nearest_dist = np.minimum(dist_right, dist_left)
            nearest_val = np.where(dist_right < dist_left, orig_arr[idx], orig_arr[idx_left])
            feats[f'NEAREST_ORIG_DIST_{col}'] = nearest_dist.astype('float32')
            feats[f'NEAREST_ORIG_LOG_DIST_{col}'] = np.log1p(nearest_dist).astype('float32')
            feats[f'NEAREST_ORIG_VAL_{col}'] = nearest_val.astype('float32')
            feats[f'NEAREST_ORIG_RATIO_{col}'] = (combined_vals / (nearest_val + 1e-8)).astype('float32')
        orig_triples = set(zip(orig['tenure'].values, orig['MonthlyCharges'].values, orig['TotalCharges'].values))
        feats['EXISTS_TRIPLE_IN_ORIG'] = pd.Series(list(zip(
            df['tenure'].values, df['MonthlyCharges'].values, df['TotalCharges'].values
        ))).isin(orig_triples).astype('float32').values
        feats['N_NUMS_EXIST_IN_ORIG'] = (
            feats['EXISTS_IN_ORIG_tenure'] + feats['EXISTS_IN_ORIG_MonthlyCharges']
            + feats['EXISTS_IN_ORIG_TotalCharges']
        ).astype('float32')
        feats['TOTAL_NEAREST_DIST'] = (
            feats['NEAREST_ORIG_DIST_tenure'] + feats['NEAREST_ORIG_DIST_MonthlyCharges']
            + feats['NEAREST_ORIG_DIST_TotalCharges']
        ).astype('float32')

    # string artifacts and digit positions of the numerics
    for col in NUMS:
        str_vals = df[col].astype(str)
        feats[f'STRLEN_{col}'] = str_vals.str.len().astype('int8').values
        dec_len = str_vals.str.split('.', expand=True)
        if dec_len.shape[1] > 1:
            feats[f'N_DECIMALS_{col}'] = dec_len[1].fillna('').str.len().astype('int8').values
            dec_part = dec_len[1].fillna('')
            trailing_zeros = dec_part.str.rstrip('0').str.len()
            feats[f'TRAILING_ZEROS_{col}'] = (dec_part.str.len() - trailing_zeros).astype('int8').values
            feats[f'SIG_DECIMALS_{col}'] = trailing_zeros.astype('int8').values
        else:
            feats[f'N_DECIMALS_{col}'] = np.zeros(n, dtype='int8')
            feats[f'TRAILING_ZEROS_{col}'] = np.zeros(n, dtype='int8')
            feats[f'SIG_DECIMALS_{col}'] = np.zeros(n, dtype='int8')
        v = df[col].values
        for n_dec in [0, 1, 2]:
            feats[f'IS_EXACT_{n_dec}dec_{col}'] = (np.abs(v - np.round(v, n_dec)) < 1e-6).astype('float32')
        feats[f'STRHASH_{col}_mod997'] = str_vals.apply(lambda x: hash(x) % 997).astype('int16').values
    for col in NUMS:
        split = df[col].astype(str).str.split('.', expand=True)
        non_dec = split[0].fillna('0')
        dec = split[1].fillna('0') if split.shape[1] > 1 else pd.Series(['0'] * n, index=df.index)
        non_dec_pad = non_dec.str.rjust(N_INT_DIGITS, '0')
        dec_pad = dec.str.ljust(N_DEC_DIGITS, '0')
        for i in range(N_INT_DIGITS):
            feats[f'{col}_digit{i}'] = (
                pd.to_numeric(non_dec_pad.str[i], errors='coerce').fillna(0).astype('int8').values)
        for i in range(N_DEC_DIGITS):
            feats[f'{col}_decimal_digit{i}'] = (
                pd.to_numeric(dec_pad.str[i], errors='coerce').fillna(0).astype('int8').values)

    # consistency of TotalCharges with tenure x MonthlyCharges
    if orig is None:
        expected_tc = ten * mc
        residual = tc - expected_tc
    else:
        expected_tc = ten_vals * mc_vals
        residual = tc_vals - expected_tc
    feats['CONSISTENCY_residual'] = residual.astype('float32')
    feats['CONSISTENCY_abs_residual'] = np.abs(residual).astype('float32')
    feats['CONSISTENCY_pct_error'] = (residual / (expected_tc + 1)).astype('float32')
    feats['CONSISTENCY_abs_pct_error'] = np.abs(residual / (expected_tc + 1)).astype('float32')
    feats['CONSISTENCY_is_exact'] = (np.abs(residual) < 0.01).astype('float32')

    if orig is not None:
        orig_residual = orig['TotalCharges'].values - orig['tenure'].values * orig['MonthlyCharges'].values
        for cat in GROUP_CATS:
            orig_group = orig.assign(_r=orig_residual).groupby(cat)['_r']
            g_mean = df[cat].map(orig_group.mean()).fillna(0).values
            g_std = df[cat].map(orig_group.std().fillna(1)).fillna(1).values
            feats[f'CONSISTENCY_resid_vs_{cat}_mean'] = (residual - g_mean).astype('float32')
            feats[f'CONSISTENCY_resid_zscore_{cat}'] = ((residual - g_mean) / (g_std + 1e-6)).astype('float32')
        orig_mc_mean_resid = orig.assign(_r=orig_residual).groupby('MonthlyCharges')['_r'].mean()
        feats['CONSISTENCY_resid_vs_orig_mc_mean'] = (
            residual - df['MonthlyCharges'].map(orig_mc_mean_resid).fillna(0).values
        ).astype('float32')

        # nearest original rows in the numeric space standardized with the original means / stds
        orig_nums = np.column_stack([
            orig['tenure'].values.astype('float64'),
            orig['MonthlyCharges'].values.astype('float64'),
            orig['TotalCharges'].values.astype('float64'),
        ])
        comb_nums = np.column_stack([ten_vals, mc_vals, tc_vals])
        o_mean = orig_nums.mean(axis=0)
        o_std = orig_nums.std(axis=0) + 1e-8
        orig_knn_f = (orig_nums - o_mean) / o_std
        comb_knn_f = (comb_nums - o_mean) / o_std
        nn_orig = NearestNeighbors(n_neighbors=5, metric='euclidean', n_jobs=-1).fit(orig_knn_f)
        dists_orig, idxs_orig = nn_orig.kneighbors(comb_knn_f)
        nearest_orig_nums = orig_nums[idxs_orig[:, 0]]
        feats['CLUSTER_dist_to_nearest_orig'] = dists_orig[:, 0].astype('float32')
        feats['CLUSTER_dist_mean_5_orig'] = dists_orig.mean(axis=1).astype('float32')
        for d, col_name in enumerate(NUMS):
            feats[f'CLUSTER_noise_{col_name}'] = (comb_nums[:, d] - nearest_orig_nums[:, d]).astype('float32')
            feats[f'CLUSTER_abs_noise_{col_name}'] = np.abs(comb_nums[:, d] - nearest_orig_nums[:, d]).astype('float32')
        del nn_orig, dists_orig, idxs_orig
        gc.collect()

        for num_col in NUMS:
            for cat_col in GROUP_CATS:
                g_mean = df[cat_col].map(orig.groupby(cat_col)[num_col].mean()).fillna(0).values
                g_std = df[cat_col].map(orig.groupby(cat_col)[num_col].std()).fillna(1).values
                feats[f'ZSCORE_{num_col}_within_{cat_col}'] = (
                    (df[num_col].values - g_mean) / (g_std + 1e-6)
                ).astype('float32')

        nn_grad = NearestNeighbors(n_neighbors=K_ORIG, metric='euclidean', n_jobs=-1).fit(orig_knn_f)
        dists_grad, idxs_grad = nn_grad.kneighbors(comb_knn_f)
        neighbor_labels = orig[TARGET].values.astype('float32')[idxs_grad]
        feats['KNN_ORIG_churn_1'] = neighbor_labels[:, 0]
        feats['KNN_ORIG_churn_3'] = neighbor_labels[:, :3].mean(axis=1).astype('float32')
        feats['KNN_ORIG_churn_5'] = neighbor_labels[:, :5].mean(axis=1).astype('float32')
        feats['KNN_ORIG_churn_10'] = neighbor_labels.mean(axis=1).astype('float32')
        w = 1.0 / (dists_grad + 1e-8)
        feats['KNN_ORIG_churn_weighted'] = ((neighbor_labels * w).sum(axis=1) / w.sum(axis=1)).astype('float32')
        feats['KNN_ORIG_churn_var_5'] = neighbor_labels[:, :5].var(axis=1).astype('float32')
        feats['KNN_ORIG_churn_var_10'] = neighbor_labels.var(axis=1).astype('float32')
        feats['KNN_ORIG_churn_gradient'] = (
            neighbor_labels[:, :3].mean(axis=1) - neighbor_labels[:, 7:].mean(axis=1)
        ).astype('float32')
        feats['KNN_ORIG_churn_range'] = (neighbor_labels.max(axis=1) - neighbor_labels.min(axis=1)).astype('float32')
        dist_to_churner = np.full(n, np.inf, dtype='float32')
        dist_to_nonchurner = np.full(n, np.inf, dtype='float32')
        for k in range(K_ORIG):
            is_churner = neighbor_labels[:, k] == 1
            is_nonchurner = neighbor_labels[:, k] == 0
            dist_to_churner = np.where(
                is_churner & (dists_grad[:, k] < dist_to_churner), dists_grad[:, k], dist_to_churner)
            dist_to_nonchurner = np.where(
                is_nonchurner & (dists_grad[:, k] < dist_to_nonchurner), dists_grad[:, k], dist_to_nonchurner)
        feats['KNN_ORIG_dist_nearest_churner'] = dist_to_churner
        feats['KNN_ORIG_dist_nearest_nonchurner'] = dist_to_nonchurner
        feats['KNN_ORIG_dist_ratio_ch_nc'] = (dist_to_churner / (dist_to_nonchurner + 1e-8)).astype('float32')
        feats['KNN_ORIG_dist_1'] = dists_grad[:, 0].astype('float32')
        feats['KNN_ORIG_dist_mean_5'] = dists_grad[:, :5].mean(axis=1).astype('float32')
        feats['KNN_ORIG_dist_ratio_1_5'] = (
            dists_grad[:, 0] / (dists_grad[:, :5].mean(axis=1) + 1e-8)
        ).astype('float32')

        orig_triple_key = (
            orig['tenure'].astype(str) + '|' + orig['MonthlyCharges'].astype(str) + '|'
            + orig['TotalCharges'].astype(str)
        )
        df_triple_key = (
            df['tenure'].astype(str) + '|' + df['MonthlyCharges'].astype(str) + '|' + df['TotalCharges'].astype(str)
        )
        feats['JOINT_COUNT_TRIPLE_IN_ORIG'] = df_triple_key.map(
            orig_triple_key.value_counts()).fillna(0).astype('float32').values
        del nn_grad, dists_grad, idxs_grad
        gc.collect()

    return feats


# ------------------------------------------------------------------------------------------------------------------
# fold features: statistics of the fold's training rows (fold_df, labels fold_y) mapped onto apply_df
# ------------------------------------------------------------------------------------------------------------------
def build_fold_features(fold_df, apply_df, fold_y, orig=None):
    feats = {}
    if orig is None:
        ref_df = fold_df.copy()
    else:
        ref_df = pd.concat([fold_df, orig.drop(columns=[TARGET])], axis=0, ignore_index=True)

    for col in NUMS:
        freq = ref_df[col].value_counts(normalize=True)
        feats[f'FREQ_{col}'] = apply_df[col].map(freq).fillna(0).astype('float32').values
    for col in NUMS:
        ref_sorted = np.sort(ref_df[col].dropna().values)
        feats[f'RANK_{col}'] = (
            np.searchsorted(ref_sorted, apply_df[col].values) / max(len(ref_sorted), 1)
        ).astype('float32')

    for c1, c2 in combinations(NUMS, 2):
        ref_pair = ref_df[c1].astype(str) + '|' + ref_df[c2].astype(str)
        vc_pair = ref_pair.value_counts(normalize=True)
        vc_pair_count = ref_pair.value_counts()
        apply_pair = apply_df[c1].astype(str) + '|' + apply_df[c2].astype(str)
        feats[f'JOINT_FREQ_{c1}_{c2}'] = apply_pair.map(vc_pair).fillna(0).astype('float32').values
        feats[f'JOINT_COUNT_{c1}_{c2}'] = apply_pair.map(vc_pair_count).fillna(0).astype('float32').values
        freq1 = apply_df[c1].map(ref_df[c1].value_counts(normalize=True)).fillna(1e-10).values
        freq2 = apply_df[c2].map(ref_df[c2].value_counts(normalize=True)).fillna(1e-10).values
        observed = apply_pair.map(vc_pair).fillna(0).values
        feats[f'JOINT_PMI_{c1}_{c2}'] = np.log((observed + 1e-10) / (freq1 * freq2 + 1e-10)).astype('float32')
    ref_triple = (
        ref_df['tenure'].astype(str) + '|' + ref_df['MonthlyCharges'].astype(str) + '|'
        + ref_df['TotalCharges'].astype(str)
    )
    apply_triple = (
        apply_df['tenure'].astype(str) + '|' + apply_df['MonthlyCharges'].astype(str) + '|'
        + apply_df['TotalCharges'].astype(str)
    )
    feats['JOINT_FREQ_TRIPLE'] = (
        apply_triple.map(ref_triple.value_counts(normalize=True)).fillna(0).astype('float32').values)
    feats['JOINT_COUNT_TRIPLE'] = apply_triple.map(ref_triple.value_counts()).fillna(0).astype('float32').values
    feats['IS_SINGLETON_TRIPLE'] = (feats['JOINT_COUNT_TRIPLE'] == 0).astype('float32')

    for num_col in NUMS:
        for cat_col in GROUP_CATS:
            ref_key = ref_df[cat_col].astype(str) + '|' + ref_df[num_col].astype(str)
            vc = ref_key.value_counts(normalize=True)
            apply_key = apply_df[cat_col].astype(str) + '|' + apply_df[num_col].astype(str)
            conditional = apply_key.map(vc).fillna(0)
            feats[f'CONDF_{num_col}_given_{cat_col}'] = conditional.astype('float32').values
            marginal = apply_df[num_col].map(ref_df[num_col].value_counts(normalize=True)).fillna(0).values
            feats[f'CONDF_ratio_{num_col}_given_{cat_col}'] = (
                conditional.values / (marginal + 1e-10)
            ).astype('float32')

    for col in apply_df.columns:
        if col not in ref_df.columns or ref_df[col].nunique(dropna=False) <= HIGH_CARD_THRESHOLD:
            continue
        vc = ref_df[col].value_counts(dropna=False)
        feats[f'count({col})'] = apply_df[col].map(vc).fillna(0).astype('float32').values
        if orig is not None and col in orig.columns:
            tmp_size = orig.groupby([col], dropna=False).size()
            tmp_mean = orig.groupby([col], dropna=False)[TARGET].mean()
            feats[f'groupby({col})_size_original'] = apply_df[col].map(tmp_size).fillna(0).astype('float32').values
            feats[f'groupby({col})_mean_churn_original'] = (
                apply_df[col].map(tmp_mean).fillna(0.5).astype('float32').values
            )
            numer = apply_df[col].map(vc).fillna(0).astype('float32').values
            denom = apply_df[col].map(tmp_size).fillna(0).astype('float32').values
            denom = np.where(denom == 0, np.nan, denom)
            feats[f'count({col})_ratio'] = np.nan_to_num(numer / denom, 0).astype('float32')

    for c in CATS:
        vc = sanitize_object_series(ref_df[c]).value_counts(dropna=False)
        cnt = sanitize_object_series(apply_df[c]).map(vc).fillna(0).astype('float32')
        feats[f'CAT_CNT_{c}'] = cnt.values
        feats[f'CAT_RARE_{c}'] = (cnt <= 50).astype('float32').values

    # density among the fold's training rows in the standardized numeric space
    tf_ten = fold_df['tenure'].astype('float64').values
    tf_mc = fold_df['MonthlyCharges'].astype('float64').values
    tf_tc = fold_df['TotalCharges'].astype('float64').values
    ap_ten = apply_df['tenure'].astype('float64').values
    ap_mc = apply_df['MonthlyCharges'].astype('float64').values
    ap_tc = apply_df['TotalCharges'].astype('float64').values
    num_mean = np.array([tf_ten.mean(), tf_mc.mean(), tf_tc.mean()])
    num_std = np.array([tf_ten.std(), tf_mc.std(), tf_tc.std()]) + 1e-8
    ref_pts = np.column_stack([
        (tf_ten - num_mean[0]) / num_std[0], (tf_mc - num_mean[1]) / num_std[1], (tf_tc - num_mean[2]) / num_std[2],
    ])
    apply_pts = np.column_stack([
        (ap_ten - num_mean[0]) / num_std[0], (ap_mc - num_mean[1]) / num_std[1], (ap_tc - num_mean[2]) / num_std[2],
    ])
    tree = cKDTree(ref_pts)
    for radius_label, radius in [('tight', 0.05), ('medium', 0.2), ('wide', 0.5)]:
        counts = tree.query_ball_point(apply_pts, r=radius, workers=-1, return_length=True)
        feats[f'DENSITY_{radius_label}'] = np.array(counts, dtype='float32')
    nn_tt = NearestNeighbors(n_neighbors=min(21, len(ref_pts)), metric='euclidean', n_jobs=-1).fit(ref_pts)
    dists_tt, _ = nn_tt.kneighbors(apply_pts)
    for k in [5, 10, 20]:
        feats[f'KNN_TT_dist_{k}'] = dists_tt[:, k].astype('float32')
    feats['KNN_TT_mean_dist_5'] = dists_tt[:, 1:6].mean(axis=1).astype('float32')
    feats['KNN_TT_mean_dist_20'] = dists_tt[:, 1:21].mean(axis=1).astype('float32')
    feats['KNN_TT_dist_5_log'] = np.log1p(dists_tt[:, 5]).astype('float32')
    del tree, nn_tt, dists_tt
    gc.collect()

    if orig is not None:
        # train-vs-original drift, original churn rates and counts of categories, numerics and rounded charges
        b_fold = fold_df.copy()
        b_orig = orig.drop(columns=[TARGET]).copy()
        b_apply = apply_df.copy()
        frames = (b_fold, b_orig, b_apply)
        for df_b in frames:
            for c in CATS:
                df_b[c] = sanitize_object_series(df_b[c])
        b_numcat_cols = []
        for c in NUMS:
            nm = f'B_NUMCAT_{c}'
            for df_b in frames:
                df_b[nm] = pd.to_numeric(df_b[c], errors='coerce')
            b_numcat_cols.append(nm)
        for base_col in ['MonthlyCharges', 'TotalCharges']:
            for r in [1, 0, -1]:
                nm = f'B_{base_col}_round_{r}'
                for df_b in frames:
                    df_b[nm] = pd.to_numeric(df_b[base_col], errors='coerce').round(r)
                b_numcat_cols.append(nm)
            nm = f'B_{base_col}_frac2'
            for df_b in frames:
                x = pd.to_numeric(df_b[base_col], errors='coerce').astype('float64')
                df_b[nm] = (np.floor((x * 100.0) % 100.0)).fillna(-1).astype('int16')
            b_numcat_cols.append(nm)
        b_fact_cols = CATS + b_numcat_cols
        for c in b_fact_cols:
            ref_all = pd.concat([b_fold[c].astype(str), b_orig[c].astype(str)], ignore_index=True)
            _, uniques = pd.factorize(ref_all, sort=False)
            mapping = dict(zip(uniques, range(len(uniques))))
            for df_b in frames:
                df_b[c] = df_b[c].astype(str).map(mapping).fillna(-1).astype(int)
        for c in b_fact_cols:
            p_tr = b_fold[c].value_counts(normalize=True)
            p_or = b_orig[c].value_counts(normalize=True)
            feats[f'B_DRIFT_{c}'] = np.log1p(
                b_apply[c].map(p_tr).fillna(0) / (b_apply[c].map(p_or).fillna(1e-6) + 1e-6)
            ).astype('float32').values
        tmp_orig = b_orig.copy()
        tmp_orig['_y_'] = orig[TARGET].astype('float32').values
        for c in b_fact_cols:
            m = tmp_orig.groupby(c, dropna=False)['_y_'].mean().astype('float32')
            vc = tmp_orig[c].value_counts(dropna=False)
            fillv = float(m.mean()) if len(m) > 0 else 0.5
            feats[f'B_TEORIG_{c}'] = b_apply[c].map(m).fillna(fillv).astype('float32').values
            feats[f'B_CEORIG_{c}'] = b_apply[c].map(vc).fillna(0).astype('float32').values

    # group means of the numerics and smoothed churn rate per categorical value
    fold_with_y = fold_df.copy()
    fold_with_y[TARGET] = fold_y
    global_mean = float(fold_y.mean())
    for g in CATS:
        for num_col in NUMS:
            m = fold_with_y.groupby(g, dropna=False)[num_col].mean()
            fillv = float(np.nanmean(fold_with_y[num_col].values))
            feats[f'B_MEAN_{num_col}_BY_{g}'] = apply_df[g].map(m).fillna(fillv).astype('float32').values
        target_mean = fold_with_y.groupby(g, dropna=False)[TARGET].mean()
        group_count = fold_with_y.groupby(g, dropna=False)[TARGET].count()
        smoothed = (target_mean * group_count + global_mean * TE_SMOOTH) / (group_count + TE_SMOOTH)
        feats[f'TE_TRAIN_{TARGET}_BY_{g}'] = apply_df[g].map(smoothed).fillna(global_mean).astype('float32').values
    return feats


def build_cat_native(df):
    cat_native = {}
    for c in CATS:
        cat_native[f'RAW_{c}'] = df[c].astype(str).fillna('__NA__').values
    for c in NUMS:
        cat_native[f'NUMCAT_{c}_exact'] = df[c].astype(str).values
        cat_native[f'NUMCAT_{c}_round0'] = np.round(df[c], 0).astype(str).values
        cat_native[f'NUMCAT_{c}_round1'] = np.round(df[c], 1).astype(str).values
        cat_native[f'NUMCAT_{c}_int'] = np.floor(df[c].values).astype(int).astype(str)
        try:
            cat_native[f'NUMCAT_{c}_qbin50'] = pd.qcut(df[c], q=50, duplicates='drop').astype(str).values
        except Exception:
            cat_native[f'NUMCAT_{c}_qbin50'] = pd.cut(df[c], bins=50).astype(str).values
        cat_native[f'NUMCAT_{c}_div2'] = (np.floor(df[c].values / 2) * 2).astype(int).astype(str)
    for i, c1 in enumerate(TOP_CATS_FOR_NGRAM):
        for c2 in TOP_CATS_FOR_NGRAM[i + 1:]:
            cat_native[f'BI_{c1}_{c2}'] = (df[c1].astype(str) + '_' + df[c2].astype(str)).values
    cat_native[f"TRI_{'_'.join(TOP_CATS_FOR_NGRAM[:3])}"] = (
        df[TOP_CATS_FOR_NGRAM[0]].astype(str) + '_' + df[TOP_CATS_FOR_NGRAM[1]].astype(str) + '_'
        + df[TOP_CATS_FOR_NGRAM[2]].astype(str)
    ).values
    return pd.DataFrame(cat_native, index=df.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=list(RECIPES))
    pair = ap.parse_args().pair
    cfg = RECIPES[pair]
    orig_rows = cfg['orig_rows']

    train, test, orig = load_data()
    y, y_orig = train[TARGET].values, orig[TARGET].values
    train_base = train.drop(columns=[TARGET, ID]).copy()
    test_base = test.drop(columns=[ID]).copy()
    orig_base = orig.drop(columns=[TARGET]).copy()
    lookup = None if orig_rows else orig

    print('building row-level features...', flush=True)
    train_safe = pd.DataFrame(build_safe_features(train_base, lookup))
    test_safe = pd.DataFrame(build_safe_features(test_base, lookup))
    safe_cols = sorted(train_safe.columns)
    train_safe, test_safe = train_safe[safe_cols], test_safe[safe_cols]
    train_cat, test_cat = build_cat_native(train_base), build_cat_native(test_base)
    cat_cols = sorted(train_cat.columns)
    train_cat, test_cat = train_cat[cat_cols], test_cat[cat_cols]
    if orig_rows:
        orig_safe = pd.DataFrame(build_safe_features(orig_base))[safe_cols]
        orig_cat = build_cat_native(orig_base)[cat_cols]

    def fit_fold(fold, tr_idx, va_idx, seed):
        fold_df = train_base.iloc[tr_idx].reset_index(drop=True)
        fold_y = y[tr_idx]
        fold_safe = train_safe.iloc[tr_idx].reset_index(drop=True)
        fold_cat = train_cat.iloc[tr_idx].reset_index(drop=True)
        if orig_rows:
            fold_df = pd.concat([fold_df, orig_base], axis=0, ignore_index=True)
            fold_y = np.concatenate([fold_y, y_orig])
            fold_safe = pd.concat([fold_safe, orig_safe], ignore_index=True)
            fold_cat = pd.concat([fold_cat, orig_cat], ignore_index=True)
        val_df = train_base.iloc[va_idx].reset_index(drop=True)

        f_tr = pd.DataFrame(build_fold_features(fold_df, fold_df, fold_y, lookup))
        f_va = pd.DataFrame(build_fold_features(fold_df, val_df, fold_y, lookup))
        f_te = pd.DataFrame(build_fold_features(fold_df, test_base, fold_y, lookup))
        fold_cols = sorted(f_tr.columns)
        features = safe_cols + fold_cols + cat_cols
        X_tr = pd.concat([fold_safe, f_tr, fold_cat], axis=1)[features]
        X_val = pd.concat([
            train_safe.iloc[va_idx].reset_index(drop=True), f_va, train_cat.iloc[va_idx].reset_index(drop=True),
        ], axis=1)[features]
        X_te = pd.concat([test_safe, f_te, test_cat], axis=1)[features]
        if fold == 1:
            print(f'features: {len(features)} (row-level {len(safe_cols)}, fold {len(fold_cols)}, '
                  f'categorical {len(cat_cols)})', flush=True)

        cat_idx = [features.index(c) for c in cat_cols]
        train_pool = Pool(data=X_tr, label=fold_y, cat_features=cat_idx, feature_names=features)
        val_pool = Pool(data=X_val, label=y[va_idx], cat_features=cat_idx, feature_names=features)
        test_pool = Pool(data=X_te, cat_features=cat_idx, feature_names=features)
        model = CatBoostClassifier(**cb_params(seed))
        model.fit(train_pool, eval_set=val_pool)
        valid_pred = model.predict_proba(val_pool)[:, 1]
        test_pred = model.predict_proba(test_pool)[:, 1]
        del X_tr, X_val, X_te, f_tr, f_va, f_te, model
        gc.collect()
        return valid_pred, test_pred

    oof, _ = run_folds(fit_fold, y, len(test), name=pair)
    _, test_pred = run_folds(fit_fold, y, len(test), n_folds=cfg['test_folds'], seed=cfg['test_seed'], name=pair)
    print(f'{pair}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(HERE, pair, oof, test_pred)


if __name__ == '__main__':
    main()
