#!/usr/bin/env python
"""GPU CatBoost on string-precision artifacts, category-profile matches, fold-local frequencies and native categories.

Recipes (pair name):
  catboost_fold_artifacts_orig_priors   the original data is used for lookups; fold statistics come from the fold's
                                        training rows (243 columns: 122 row-level, 81 fold, 40 categorical)
  catboost_fold_artifacts_orig_rows     no original-data lookups; the 7,043 original rows are appended to the training
                                        rows of every fold and every fold statistic comes from these rows (191
                                        columns: 86 row-level, 65 fold, 40 categorical)

Numerics (float32): tenure, MonthlyCharges, TotalCharges (blank TotalCharges get the frame's median); the 16 other
columns are strings.
Row-level features (both recipes):
  - TotalCharges - tenure x MonthlyCharges (and absolute value), charge ratios, TotalCharges / (tenure + 1),
    tenure x MonthlyCharges, tenure x TotalCharges; log1p / sqrt / 1 / (1 + x) / missing flag of the numerics
  - number of Yes / No answers of the 8 services, has-internet / has-phone flags, Yes / No flags of 11 Yes / No columns
  - string artifacts of the numerics: string length (and the row total), number of decimals, more-than-2-decimals
    flag, integer flag, 4 integer and 4 decimal digit positions
  - TotalCharges vs tenure x MonthlyCharges: residual, |relative error|, exact flag (|residual| < 0.01), ratio
Row-level features with original-data lookups (first recipe only):
  - original churn rate of every value of the 19 columns
  - TotalCharges percentile among the original rows, churner minus non-churner |z-score| gap, MonthlyCharges minus
    the original InternetService mean
  - value seen in the original data, distance to and ratio with the nearest original value, for each numeric
  - the 16-categorical profile seen in the original data and its original count
  - churn rate of the 3 / 10 nearest original rows and its variance over 5 (numerics standardized with the original
    means / stds)
Fold features (statistics of the reference rows mapped onto the training, validation and test rows):
  - frequency of every numeric value and of its decimal string, percentile rank, frequency of each numeric given
    Contract / InternetService (the reference is the fold's training rows + the original rows in both recipes)
  - distance to the 5th nearest neighbour and the mean distance to neighbours 1-5 among the fold's training rows in
    the standardized numeric space; means of the 3 numerics per categorical value of the fold's training rows
  - first recipe only: train-vs-original drift log1p(p_fold / p_original) of the 16 categoricals
CatBoost-native categoricals (40): the raw categoricals; every numeric as an exact / rounded (0 and 1 decimals)
string; 15 bigrams of Contract, InternetService, PaymentMethod, OnlineSecurity, TechSupport, PaperlessBilling.
CatBoost (GPU): depth 8, lr 0.03, l2 3, random_strength 0.5, Bayesian bagging_temperature 0.7, border_count 254,
one_hot_max_size 4, max_ctr_complexity 2, up to 5000 iterations with early stopping (300) on the validation fold
(use_best_model=True). Test = mean of the 5 fold models.

Usage: python train_fold_artifacts.py <pair>"""
import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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
N_INT_DIGITS = 4
N_DEC_DIGITS = 4

RECIPES = {
    'catboost_fold_artifacts_orig_priors': dict(orig_rows=False),
    'catboost_fold_artifacts_orig_rows': dict(orig_rows=True),
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


def build_safe_features(df, orig=None):
    """Row-level features (own values, plus original-data lookups when `orig` is given)."""
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
    feats['has_internet'] = (svc['InternetService'] != 'No').astype('float32').values
    feats['has_phone'] = (svc['PhoneService'] == 'Yes').astype('float32').values
    for c in YN_COLS:
        s = sanitize_series(df[c])
        feats[f'ISYES_{c}'] = (s == 'Yes').astype('float32').values
        feats[f'ISNO_{c}'] = (s == 'No').astype('float32').values

    if orig is not None:
        for col in CATS + NUMS:
            tmp = orig.groupby(col, observed=False)[TARGET].mean()
            feats[f'ORIG_proba_{col}'] = df[col].map(tmp).fillna(0.5).astype('float32').values
        orig_churner_tc = orig.loc[orig[TARGET] == 1, 'TotalCharges'].values
        orig_nonchurner_tc = orig.loc[orig[TARGET] == 0, 'TotalCharges'].values
        orig_is_mc_mean = orig.groupby('InternetService', observed=False)['MonthlyCharges'].mean()
        feats['pctrank_orig_TC'] = pctrank_against(tc_vals, orig['TotalCharges'].values)
        feats['zscore_churn_gap_TC'] = (
            np.abs(zscore_against(tc_vals, orig_churner_tc)) - np.abs(zscore_against(tc_vals, orig_nonchurner_tc))
        ).astype('float32')
        feats['resid_IS_MC'] = (
            df['MonthlyCharges'].values - df['InternetService'].map(orig_is_mc_mean).fillna(0).values
        ).astype('float32')
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
            feats[f'NEAREST_ORIG_RATIO_{col}'] = (combined_vals / (nearest_val + 1e-8)).astype('float32')

    # string precision artifacts and digit positions of the numerics
    total_str_len = np.zeros(n, dtype='int16')
    for col in NUMS:
        str_vals = df[col].astype(str)
        str_len = str_vals.str.len().astype('int8').values
        feats[f'STRLEN_{col}'] = str_len
        total_str_len += str_len
        dec_len = str_vals.str.split('.', expand=True)
        if dec_len.shape[1] > 1:
            n_dec = dec_len[1].fillna('').str.len().astype('int8').values
            feats[f'N_DECIMALS_{col}'] = n_dec
            feats[f'HAS_WEIRD_PRECISION_{col}'] = (n_dec > 2).astype('float32')
        else:
            feats[f'N_DECIMALS_{col}'] = np.zeros(n, dtype='int8')
            feats[f'HAS_WEIRD_PRECISION_{col}'] = np.zeros(n, dtype='float32')
        feats[f'IS_INTEGER_{col}'] = (df[col].values % 1 == 0).astype('float32')
        non_dec = dec_len[0].fillna('0')
        dec = dec_len[1].fillna('0') if dec_len.shape[1] > 1 else pd.Series(['0'] * n, index=df.index)
        non_dec_pad = non_dec.str.rjust(N_INT_DIGITS, '0')
        dec_pad = dec.str.ljust(N_DEC_DIGITS, '0')
        for i in range(N_INT_DIGITS):
            feats[f'{col}_digit{i}'] = (
                pd.to_numeric(non_dec_pad.str[i], errors='coerce').fillna(0).astype('int8').values)
        for i in range(N_DEC_DIGITS):
            feats[f'{col}_decimal_digit{i}'] = (
                pd.to_numeric(dec_pad.str[i], errors='coerce').fillna(0).astype('int8').values)
    feats['TOTAL_NUMERIC_STRLEN'] = total_str_len.astype('int16')

    # consistency of TotalCharges with tenure x MonthlyCharges
    expected_tc = ten_vals * mc_vals
    residual = tc_vals - expected_tc
    feats['CONSISTENCY_residual'] = residual.astype('float32')
    feats['CONSISTENCY_abs_pct_error'] = np.abs(residual / (expected_tc + 1)).astype('float32')
    feats['CONSISTENCY_is_exact'] = (np.abs(residual) < 0.01).astype('float32')
    feats['CONSISTENCY_ratio'] = (tc_vals / (expected_tc + 1e-5)).astype('float32')

    if orig is not None:
        # exact match of the 16-categorical profile with the original rows
        orig_cat_str = orig[CATS].astype(str).agg('|'.join, axis=1)
        df_cat_str = df[CATS].astype(str).agg('|'.join, axis=1)
        feats['CAT_PERSONA_EXISTS_IN_ORIG'] = df_cat_str.isin(orig_cat_str.unique()).astype('float32').values
        feats['CAT_PERSONA_ORIG_COUNT'] = df_cat_str.map(orig_cat_str.value_counts()).fillna(0).astype('float32').values

        # churn rate of the nearest original rows (numerics standardized with the original means / stds)
        orig_nums = np.column_stack([
            orig['tenure'].values.astype('float64'),
            orig['MonthlyCharges'].values.astype('float64'),
            orig['TotalCharges'].values.astype('float64'),
        ])
        comb_nums = np.column_stack([ten_vals, mc_vals, tc_vals])
        o_mean = orig_nums.mean(axis=0)
        o_std = orig_nums.std(axis=0) + 1e-8
        nn_grad = NearestNeighbors(n_neighbors=10, metric='euclidean', n_jobs=-1).fit((orig_nums - o_mean) / o_std)
        _, idxs_grad = nn_grad.kneighbors((comb_nums - o_mean) / o_std)
        neighbor_labels = orig[TARGET].values.astype('float32')[idxs_grad]
        feats['KNN_ORIG_churn_3'] = neighbor_labels[:, :3].mean(axis=1).astype('float32')
        feats['KNN_ORIG_churn_10'] = neighbor_labels.mean(axis=1).astype('float32')
        feats['KNN_ORIG_churn_var_5'] = neighbor_labels[:, :5].var(axis=1).astype('float32')
        del nn_grad, idxs_grad
        gc.collect()
    return feats


def build_fold_features(fold_df, apply_df, orig=None):
    """Statistics of the fold's training rows (fold_df) mapped onto apply_df; with `orig`, the frequency / rank
    references also include the original rows and the train-vs-original drift is added."""
    feats = {}
    n_apply = len(apply_df)
    if orig is None:
        ref_df = fold_df.copy()
    else:
        ref_df = pd.concat([fold_df, orig.drop(columns=[TARGET])], axis=0, ignore_index=True)

    for col in NUMS:
        freq = ref_df[col].value_counts(normalize=True)
        feats[f'FREQ_{col}'] = apply_df[col].map(freq).fillna(0).astype('float32').values
        ref_frac = ref_df[col].astype(str).str.split('.', expand=True)
        ref_frac = ref_frac[1] if ref_frac.shape[1] > 1 else pd.Series(['0'] * len(ref_df))
        app_frac = apply_df[col].astype(str).str.split('.', expand=True)
        app_frac = app_frac[1] if app_frac.shape[1] > 1 else pd.Series(['0'] * n_apply)
        vc_frac = ref_frac.value_counts(normalize=True)
        feats[f'FREQ_FRAC_{col}'] = app_frac.map(vc_frac).fillna(0).astype('float32').values
    for col in NUMS:
        ref_sorted = np.sort(ref_df[col].dropna().values)
        feats[f'RANK_{col}'] = (
            np.searchsorted(ref_sorted, apply_df[col].values) / max(len(ref_sorted), 1)
        ).astype('float32')
    for num_col in NUMS:
        for cat_col in ['Contract', 'InternetService']:
            ref_key = ref_df[cat_col].astype(str) + '|' + ref_df[num_col].astype(str)
            apply_key = apply_df[cat_col].astype(str) + '|' + apply_df[num_col].astype(str)
            feats[f'CONDF_{num_col}_given_{cat_col}'] = apply_key.map(
                ref_key.value_counts(normalize=True)).fillna(0).astype('float32').values

    # neighbour distances among the fold's training rows in the standardized numeric space
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
    nn_tt = NearestNeighbors(n_neighbors=min(21, len(ref_pts)), metric='euclidean', n_jobs=-1).fit(ref_pts)
    dists_tt, _ = nn_tt.kneighbors(apply_pts)
    feats['KNN_TT_dist_5'] = dists_tt[:, 5].astype('float32')
    feats['KNN_TT_mean_dist_5'] = dists_tt[:, 1:6].mean(axis=1).astype('float32')
    del nn_tt, dists_tt
    gc.collect()

    if orig is not None:
        b_fold = fold_df.copy()
        b_orig = orig.drop(columns=[TARGET]).copy()
        b_apply = apply_df.copy()
        frames = (b_fold, b_orig, b_apply)
        for df_b in frames:
            for c in CATS:
                df_b[c] = sanitize_object_series(df_b[c])
        for c in CATS:
            ref_all = pd.concat([b_fold[c].astype(str), b_orig[c].astype(str)], ignore_index=True)
            _, uniques = pd.factorize(ref_all, sort=False)
            mapping = dict(zip(uniques, range(len(uniques))))
            for df_b in frames:
                df_b[c] = df_b[c].astype(str).map(mapping).fillna(-1).astype(int)
        for c in CATS:
            p_tr = b_fold[c].value_counts(normalize=True)
            p_or = b_orig[c].value_counts(normalize=True)
            feats[f'B_DRIFT_{c}'] = np.log1p(
                b_apply[c].map(p_tr).fillna(0) / (b_apply[c].map(p_or).fillna(1e-6) + 1e-6)
            ).astype('float32').values

    for g in CATS:
        for num_col in NUMS:
            m = fold_df.groupby(g, dropna=False)[num_col].mean()
            fillv = float(np.nanmean(fold_df[num_col].values))
            feats[f'B_MEAN_{num_col}_BY_{g}'] = apply_df[g].map(m).fillna(fillv).astype('float32').values
    return feats


def build_cat_native(df):
    cat_native = {}
    for c in CATS:
        cat_native[f'RAW_{c}'] = df[c].astype(str).fillna('__NA__').values
    for c in NUMS:
        cat_native[f'NUMCAT_{c}_exact'] = df[c].astype(str).values
        cat_native[f'NUMCAT_{c}_round0'] = np.round(df[c], 0).astype(str).values
        cat_native[f'NUMCAT_{c}_round1'] = np.round(df[c], 1).astype(str).values
    for i, c1 in enumerate(TOP_CATS_FOR_NGRAM):
        for c2 in TOP_CATS_FOR_NGRAM[i + 1:]:
            cat_native[f'BI_{c1}_{c2}'] = (df[c1].astype(str) + '_' + df[c2].astype(str)).values
    return pd.DataFrame(cat_native, index=df.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=list(RECIPES))
    pair = ap.parse_args().pair
    orig_rows = RECIPES[pair]['orig_rows']

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

        f_tr = pd.DataFrame(build_fold_features(fold_df, fold_df, lookup))
        f_va = pd.DataFrame(build_fold_features(fold_df, val_df, lookup))
        f_te = pd.DataFrame(build_fold_features(fold_df, test_base, lookup))
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

    oof, test_pred = run_folds(fit_fold, y, len(test), name=pair)
    print(f'{pair}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(HERE, pair, oof, test_pred)


if __name__ == '__main__':
    main()
