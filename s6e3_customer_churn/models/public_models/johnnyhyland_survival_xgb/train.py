#!/usr/bin/env python
"""Bayesian survival features + XGBoost, after the public notebook "s6e3 - Reframing as Bayesian Survival - PyMC".
https://www.kaggle.com/code/johnnyhyland/s6e3-reframing-as-bayesian-survival-pymc

Everything is fitted inside each fold on the fold's training rows only:
1. A Bayesian piecewise-exponential proportional-hazards model (Poisson likelihood on the monthly / quarterly /
   half-yearly tenure intervals) with gender-stratified log baseline hazards and a shared Gamma frailty per
   InternetService x Contract group, on 9 billing / demographic predictors. PyMC NUTS (2 chains, 1,000 tuning + 2,000
   draws, target_accept 0.9, seed 43 + fold) on 2,000 rows drawn from a 5,000-row sample of the training rows.
2. From the posterior means, survival-curve features for every row (the curve is read at interval index = tenure,
   capped at the last interval, as in the notebook): churn probability (1 - survival), frailty, survival velocity
   (drop to the next interval), hazard ratio vs the baseline, survival at 6 and 36 months and their ratio.
3. Further features: 22 binary / standardized encodings, 16 engineered numerics (charge ratios, logs, square roots,
   service counts), the original dataset's churn rate of each of the 16 categorical values, ranks of the 3 numerics
   and frequencies of 10 categoricals among the training rows, and 174 interaction codes (3 numerics in deciles and all
   171 pairs of the 16 categoricals + 3 deciles) replaced by their scikit-learn TargetEncoder values, plus the mean /
   std / min / max of those 174 encodings. 252 features.
Model: XGBoost (GPU hist), depth 7, lr 0.005, up to 50,000 trees with early stopping (500 rounds) on the validation
fold's AUC, colsample_bytree 0.35, subsample 0.9, min_child_weight 6, lambda 5, alpha 0.05, max_bin 1024, seed 42.
Test = mean of the fold models.
Usage: python train.py"""
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import read_original, read_test, read_train  # noqa: E402
from lib.runner import run_folds  # noqa: E402

warnings.filterwarnings('ignore')

FOLDER = 'johnnyhyland_survival_xgb'
SEED = 42

SURVIVAL_SAMPLE_SIZE = 5000
FRAILTY_SAMPLE_SIZE = 2000
PYMC_DRAWS = 2000
PYMC_TUNE = 1000
PYMC_CHAINS = 2
PYMC_TARGET_ACCEPT = 0.90

INTERNET_SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                     'StreamingMovies']
BASE_CAT_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
RANK_NUM_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
FREQ_CAT_COLS = ['InternetService', 'Contract', 'PaymentMethod', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
PRED_COLS = ['SeniorCitizen', 'Male', 'HasPartner', 'HasDependents', 'HasPhoneService', 'MultipleLines_Yes',
             'Contract_OneYear', 'Contract_TwoYear', 'Internet_DSL', 'Internet_Fiber', 'OnlineSecurity_Yes',
             'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes',
             'PaperlessBill', 'Pay_CreditCard', 'Pay_Electronic', 'Pay_Mailed', 'MonthlyCharges_std',
             'TotalCharges_std']
FRAILTY_PREDS = ['Male', 'SeniorCitizen', 'HasPartner', 'HasDependents', 'PaperlessBill', 'Pay_CreditCard',
                 'Pay_Mailed', 'MonthlyCharges_std', 'TotalCharges_std']

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'n_estimators': 50000,
    'max_depth': 7,
    'learning_rate': 0.005,
    'n_jobs': 32,
    'reg_lambda': 5.0,
    'reg_alpha': 0.05,
    'colsample_bytree': 0.35,
    'subsample': 0.90,
    'min_child_weight': 6,
    'max_bin': 1024,
    'random_state': 42,
    'verbosity': 0,
    'early_stopping_rounds': 500,
    'device': 'cuda',
}


# ------------------------------------------------------------------ preprocessing
def preprocess_survival_frame(df_raw, train_ref_for_std):
    df = df_raw.copy()
    df['Male'] = (df['gender'] == 'Male').astype(int)
    df['HasPartner'] = (df['Partner'] == 'Yes').astype(int)
    df['HasDependents'] = (df['Dependents'] == 'Yes').astype(int)
    df['HasPhoneService'] = (df['PhoneService'] == 'Yes').astype(int)
    df['PaperlessBill'] = (df['PaperlessBilling'] == 'Yes').astype(int)
    df['Contract_OneYear'] = (df['Contract'] == 'One year').astype(int)
    df['Contract_TwoYear'] = (df['Contract'] == 'Two year').astype(int)
    df['Internet_DSL'] = (df['InternetService'] == 'DSL').astype(int)
    df['Internet_Fiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['Pay_CreditCard'] = (df['PaymentMethod'] == 'Credit card (automatic)').astype(int)
    df['Pay_Electronic'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['Pay_Mailed'] = (df['PaymentMethod'] == 'Mailed check').astype(int)
    df['Internet_Contract'] = df['InternetService'].astype(str) + '_' + df['Contract'].astype(str)
    for col in INTERNET_SERVICES:
        df[col + '_Yes'] = (df[col] == 'Yes').astype(int)
    df['MultipleLines_Yes'] = (df['MultipleLines'] == 'Yes').astype(int)

    monthly_mean = train_ref_for_std['MonthlyCharges'].mean()
    monthly_std = train_ref_for_std['MonthlyCharges'].std()
    total_ref = pd.to_numeric(train_ref_for_std['TotalCharges'], errors='coerce').fillna(0)
    total_mean, total_std = total_ref.mean(), total_ref.std()
    df['MonthlyCharges_std'] = (df['MonthlyCharges'] - monthly_mean) / (monthly_std + 1e-12)
    total_num = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['TotalCharges_std'] = (total_num - total_mean) / (total_std + 1e-12)
    return df


# ------------------------------------------------------------------ survival / frailty model
def build_interval_structures(df_preprocessed, rng):
    sample_size = min(SURVIVAL_SAMPLE_SIZE, len(df_preprocessed))
    sample_idx = rng.choice(len(df_preprocessed), size=sample_size, replace=False)
    df_sample = df_preprocessed.iloc[sample_idx].reset_index(drop=True)

    max_tenure = int(df_sample['tenure'].max())
    interval_list = []
    interval_list.extend(np.arange(0, min(6, max_tenure) + 1, 1))
    if max_tenure > 6:
        interval_list.extend(np.arange(9, min(24, max_tenure) + 1, 3))
    if max_tenure > 24:
        interval_list.extend(np.arange(30, max_tenure + 1, 6))
    interval_bounds = np.unique(np.array(interval_list))
    interval_bounds.sort()
    if interval_bounds[0] > 0:
        interval_bounds = np.insert(interval_bounds, 0, 0)
    if interval_bounds[-1] < max_tenure:
        interval_bounds = np.append(interval_bounds, max_tenure + 1)

    n_intervals = len(interval_bounds) - 1
    intervals = np.arange(n_intervals)
    customers = np.arange(len(df_sample))
    last_period = np.searchsorted(interval_bounds[1:], df_sample['tenure'].values, side='left')
    last_period = np.clip(last_period, 0, n_intervals - 1)

    quit_matrix = np.zeros((len(df_sample), n_intervals), dtype=np.float32)
    quit_matrix[customers, last_period] = df_sample['Churn_binary'].values.astype(np.float32)
    exposure = np.zeros((len(df_sample), n_intervals), dtype=np.float32)
    for k in range(n_intervals):
        lo, hi = interval_bounds[k], interval_bounds[k + 1]
        exposure[:, k] = np.clip(df_sample['tenure'].values - lo, 0, hi - lo)
    assert not np.any((quit_matrix == 1) & (exposure == 0)), 'Event with zero exposure'

    return {'df_sample': df_sample, 'interval_bounds': interval_bounds, 'intervals': intervals,
            'quit_matrix': quit_matrix, 'exposure': exposure, 'n_intervals': n_intervals}


def make_coxph_frailty(df_input, pred_cols, quit_mat, exposure_mat, intervals, factor_col, seed):
    frailty_idx, frailty_labels = pd.factorize(df_input[factor_col])
    gender_idx = df_input['Male'].values.astype(int)
    coords = {
        'intervals': np.arange(len(intervals)),
        'preds': pred_cols,
        'frailty_id': frailty_labels,
        'gender': ['Female', 'Male'],
        'obs': np.arange(len(df_input)),
    }
    with pm.Model(coords=coords) as frailty_model:
        X_data = pm.Data('X_data_obs', df_input[pred_cols].values, dims=('obs', 'preds'))
        lambda0_log = pm.Normal('lambda0_log', mu=-4.5, sigma=0.5, dims=('intervals', 'gender'))
        sigma_frailty = pm.HalfNormal('sigma_frailty', 1.0)
        frailty = pm.Gamma('frailty', alpha=1.0 / sigma_frailty ** 2, beta=1.0 / sigma_frailty ** 2, dims='frailty_id')
        beta = pm.Normal('beta', 0, sigma=0.5, dims='preds')

        lambda0_per_obs = pt.exp(lambda0_log[:, gender_idx]).T
        hazard = pt.exp(pm.math.dot(X_data, beta))[:, None] * lambda0_per_obs
        lambda_ = frailty[frailty_idx, None] * hazard
        mu = pt.maximum(exposure_mat * lambda_, 1e-12)
        pm.Poisson('outcome', mu, observed=quit_mat, dims=('obs', 'intervals'))

        prior_pred = pm.sample_prior_predictive(random_seed=seed)
        idata = pm.sample(
            draws=PYMC_DRAWS,
            tune=PYMC_TUNE,
            chains=PYMC_CHAINS,
            target_accept=PYMC_TARGET_ACCEPT,
            random_seed=seed,
            idata_kwargs={'log_likelihood': True},
            progressbar=False,
        )
        idata.extend(prior_pred)
    return idata, frailty_model, frailty_labels


def fit_fold_frailty_model(df_train_pre, seed):
    rng = np.random.default_rng(seed)
    structures = build_interval_structures(df_train_pre, rng)
    df_sample = structures['df_sample']
    quit_matrix = structures['quit_matrix']
    exposure = structures['exposure']
    intervals = structures['intervals']
    interval_bounds = structures['interval_bounds']

    frailty_sample_size = min(FRAILTY_SAMPLE_SIZE, len(df_sample))
    frailty_idx_sample = rng.choice(len(df_sample), size=frailty_sample_size, replace=False)
    df_frailty = df_sample.iloc[frailty_idx_sample].reset_index(drop=True)
    quit_matrix_frailty = quit_matrix[frailty_idx_sample]
    exposure_frailty = exposure[frailty_idx_sample]

    idata, model, frailty_labels = make_coxph_frailty(
        df_frailty, FRAILTY_PREDS, quit_matrix_frailty, exposure_frailty, intervals,
        factor_col='Internet_Contract', seed=seed,
    )
    post = idata.posterior
    beta_fr = post['beta'].mean(('chain', 'draw')).values
    lambda0_fr = np.exp(post['lambda0_log'].mean(('chain', 'draw')).values)
    frailty_v = post['frailty'].mean(('chain', 'draw')).values
    frailty_l = np.array(frailty_labels)
    del post, idata, model
    gc.collect()
    return beta_fr, lambda0_fr, frailty_v, frailty_l, interval_bounds


def generate_frailty_features(df_input, frailty_preds, beta, lambda0, frailty_vals, frailty_labels, interval_bounds):
    frailty_map = {label: i for i, label in enumerate(frailty_labels)}
    X = df_input[frailty_preds].values
    frailty_idx = df_input['Internet_Contract'].map(frailty_map).fillna(0).values.astype(int)
    frailty_z = frailty_vals[frailty_idx]
    gender_col = (df_input['gender'] == 'Male').astype(int).values
    baseline = np.where(gender_col[:, None] == 1, lambda0[None, :, 1], lambda0[None, :, 0])

    risk = frailty_z * np.exp(X @ beta)
    hazard = risk[:, None] * baseline
    cum_haz = np.cumsum(hazard, axis=1)
    surv = np.exp(-cum_haz)

    rows = np.arange(len(df_input))
    tenure_idx = np.clip(df_input['tenure'].values.astype(int), 0, lambda0.shape[0] - 1)
    surv_current = surv[rows, tenure_idx]
    surv_next_idx = np.clip(tenure_idx + 1, 0, lambda0.shape[0] - 1)
    surv_next = surv[rows, surv_next_idx]
    surv_velocity = surv_current - surv_next
    baseline_hz_current = baseline[rows, tenure_idx]
    hazard_current = hazard[rows, tenure_idx]
    hazard_ratio = hazard_current / (baseline_hz_current + 1e-8)

    idx_6 = np.clip(np.searchsorted(interval_bounds[1:], 6, side='left'), 0, lambda0.shape[0] - 1)
    idx_36 = np.clip(np.searchsorted(interval_bounds[1:], 36, side='left'), 0, lambda0.shape[0] - 1)
    surv_6m = surv[:, idx_6]
    surv_36m = surv[:, idx_36]
    surv_drop_ratio = surv_36m / (surv_6m + 1e-8)

    return {
        'surv_frailty_churn_prob': 1.0 - surv_current,
        'surv_frailty_z': frailty_z,
        'surv_velocity': surv_velocity,
        'surv_hazard_ratio': hazard_ratio,
        'surv_prob_6m': surv_6m,
        'surv_prob_36m': surv_36m,
        'surv_drop_ratio': surv_drop_ratio,
    }


# ------------------------------------------------------------------ deterministic features
def engineer_extra_features(df_input):
    feats = {}
    tenure = df_input['tenure'].values.astype(float)
    monthly = df_input['MonthlyCharges'].values.astype(float)
    total = pd.to_numeric(df_input['TotalCharges'], errors='coerce').fillna(0).values
    feats['tenure'] = tenure
    feats['MonthlyCharges'] = monthly
    feats['TotalCharges'] = total

    tenure_safe = np.clip(tenure, 1, None)
    feats['AvgChargePerMonth'] = total / tenure_safe
    feats['ChargeRatio'] = monthly * tenure_safe / np.clip(total, 1, None)
    feats['charges_deviation'] = total - tenure * monthly
    feats['tenure_x_monthly'] = tenure * monthly
    feats['log1p_tenure'] = np.log1p(tenure)
    feats['log1p_MonthlyCharges'] = np.log1p(monthly)
    feats['log1p_TotalCharges'] = np.log1p(total)
    feats['sqrt_tenure'] = np.sqrt(tenure)
    feats['sqrt_TotalCharges'] = np.sqrt(total)

    svc_count = np.zeros(len(df_input), dtype=np.float32)
    for col in INTERNET_SERVICES:
        svc_count += (df_input[col] == 'Yes').values.astype(np.float32)
    feats['NumServices'] = svc_count
    feats['tenure_x_services'] = tenure * svc_count
    feats['HasAnySecurity'] = ((df_input['OnlineSecurity'] == 'Yes') | (df_input['OnlineBackup'] == 'Yes') |
                               (df_input['DeviceProtection'] == 'Yes') |
                               (df_input['TechSupport'] == 'Yes')).astype(float).values
    feats['HasStreaming'] = ((df_input['StreamingTV'] == 'Yes') |
                             (df_input['StreamingMovies'] == 'Yes')).astype(float).values
    return feats


def compute_orig_proba(df_apply_raw, orig_df):
    orig = orig_df.copy()
    orig['SeniorCitizen'] = orig['SeniorCitizen'].astype(str)
    global_mean = orig['Churn_binary'].mean()
    out = {}
    for col in BASE_CAT_COLS:
        proba_map = orig.groupby(col)['Churn_binary'].mean().to_dict()
        out[f'ORIG_proba_{col}'] = df_apply_raw[col].astype(str).map(proba_map).fillna(global_mean).values
    return out


def compute_rank_features_train_apply(df_train_raw, df_apply_raw):
    out = {}
    for col in RANK_NUM_COLS:
        if col == 'TotalCharges':
            train_vals = pd.to_numeric(df_train_raw[col], errors='coerce').fillna(0).values.astype(float)
            apply_vals = pd.to_numeric(df_apply_raw[col], errors='coerce').fillna(0).values.astype(float)
        else:
            train_vals = df_train_raw[col].values.astype(float)
            apply_vals = df_apply_raw[col].values.astype(float)
        sorted_train = np.sort(train_vals)
        out[f'rank_{col}'] = np.searchsorted(sorted_train, apply_vals, side='right') / max(len(sorted_train), 1)
    return out


def compute_freq_features_train_apply(df_train_raw, df_apply_raw):
    out = {}
    for col in FREQ_CAT_COLS:
        freq = df_train_raw[col].value_counts(normalize=True)
        out[f'freq_{col}'] = df_apply_raw[col].map(freq).fillna(0.0).values
    return out


class InteractionBuilder:
    """Decile codes of the numerics (edges from the training rows) and codes of every pair of the categoricals +
    deciles; values unseen in the training rows map to -1."""

    def __init__(self, cat_cols, num_cols_to_bin, q=10):
        self.cat_cols = cat_cols
        self.num_cols_to_bin = num_cols_to_bin
        self.q = q
        self.bin_edges_ = {}
        self.interaction_cols_ = []
        self.maps_ = {}
        self.twin_maps_ = {}

    def _fit_bins(self, series):
        vals = pd.to_numeric(series, errors='coerce').fillna(0).values.astype(float)
        edges = np.unique(np.quantile(vals, np.linspace(0, 1, self.q + 1)))
        if len(edges) <= 1:
            return np.array([-np.inf, np.inf], dtype=float)
        edges = edges.astype(float)
        edges[0], edges[-1] = -np.inf, np.inf
        return edges

    def _apply_bins(self, series, edges):
        vals = pd.to_numeric(series, errors='coerce').fillna(0).values.astype(float)
        return np.digitize(vals, edges[1:-1], right=True).astype(str)

    def _build_string_frame(self, df_raw, local_cat_cols):
        feats = pd.DataFrame(index=df_raw.index)
        for col in self.cat_cols:
            feats[col] = df_raw[col].astype(str).fillna('missing')
        for col in self.num_cols_to_bin:
            feats[f'C{col}'] = self._apply_bins(df_raw[col], self.bin_edges_[col])
        return feats[local_cat_cols]

    def fit(self, df_train_raw):
        local_cat_cols = self.cat_cols.copy()
        for col in self.num_cols_to_bin:
            self.bin_edges_[col] = self._fit_bins(df_train_raw[col])
            local_cat_cols.append(f'C{col}')
        train_feats = self._build_string_frame(df_train_raw, local_cat_cols)
        for col in self.num_cols_to_bin:
            twin_name = f'C{col}'
            uniq = pd.Series(train_feats[twin_name].astype(str).unique())
            self.twin_maps_[twin_name] = {v: i for i, v in enumerate(uniq.tolist())}
        for col1, col2 in combinations(local_cat_cols, 2):
            name = f'{col1}-{col2}'
            combo = (train_feats[col1].astype(str) + '-' + train_feats[col2].astype(str)).values
            self.maps_[name] = {v: i for i, v in enumerate(pd.Series(combo).unique().tolist())}
            self.interaction_cols_.append(name)
        return self

    def transform(self, df_raw):
        local_cat_cols = self.cat_cols.copy() + [f'C{c}' for c in self.num_cols_to_bin]
        feats = self._build_string_frame(df_raw, local_cat_cols)
        out = {}
        for col in self.num_cols_to_bin:
            twin_name = f'C{col}'
            out[twin_name] = feats[twin_name].map(self.twin_maps_[twin_name]).fillna(-1).astype(np.int32).values
        for name in self.interaction_cols_:
            col1, col2 = name.split('-', 1)
            combo = feats[col1].astype(str) + '-' + feats[col2].astype(str)
            out[name] = combo.map(self.maps_[name]).fillna(-1).astype(np.int32).values
        return pd.DataFrame(out, index=df_raw.index)


def assemble_fold_features(df_train_raw, df_val_raw, df_test_raw, orig_df, fold_seed):
    tr_pre = preprocess_survival_frame(df_train_raw, df_train_raw)
    va_pre = preprocess_survival_frame(df_val_raw, df_train_raw)
    te_pre = preprocess_survival_frame(df_test_raw, df_train_raw)

    print('  fitting the frailty model ...', flush=True)
    beta_fr, lambda0_fr, frailty_v, frailty_l, bounds = fit_fold_frailty_model(tr_pre, seed=fold_seed)

    def survival_frame(pre):
        return pd.DataFrame({
            **generate_frailty_features(pre, FRAILTY_PREDS, beta_fr, lambda0_fr, frailty_v, frailty_l, bounds),
            **engineer_extra_features(pre),
        }, index=pre.index).reset_index(drop=True)

    def train_stat_frames(fn):
        return [pd.DataFrame(fn(df_train_raw, d)).reset_index(drop=True) for d in (df_train_raw, df_val_raw, df_test_raw)]

    surv = [survival_frame(p) for p in (tr_pre, va_pre, te_pre)]
    orig_feats = [pd.DataFrame(compute_orig_proba(d, orig_df)).reset_index(drop=True)
                  for d in (df_train_raw, df_val_raw, df_test_raw)]
    rank = train_stat_frames(compute_rank_features_train_apply)
    freq = train_stat_frames(compute_freq_features_train_apply)
    builder = InteractionBuilder(cat_cols=BASE_CAT_COLS, num_cols_to_bin=RANK_NUM_COLS, q=10).fit(df_train_raw)
    inter = [builder.transform(d).reset_index(drop=True) for d in (df_train_raw, df_val_raw, df_test_raw)]

    frames = []
    for i, pre in enumerate((tr_pre, va_pre, te_pre)):
        frames.append(pd.concat([pre[PRED_COLS].reset_index(drop=True), surv[i], orig_feats[i], rank[i], freq[i],
                                 inter[i]], axis=1))
    return frames[0], frames[1], frames[2], tr_pre['Churn_binary'].values, va_pre['Churn_binary'].values, \
        list(inter[0].columns)


def main():
    train_df, test_df, orig_df = read_train(), read_test(), read_original()
    orig_df['TotalCharges'] = orig_df['TotalCharges'].fillna(0)
    orig_df['Churn_binary'] = (orig_df['Churn'] == 'Yes').astype(int)
    train_df['Churn_binary'] = (train_df['Churn'] == 'Yes').astype(int)
    for df_ in (train_df, test_df):
        df_['TotalCharges'] = pd.to_numeric(df_['TotalCharges'], errors='coerce').fillna(0)
    print(f'train churn rate {train_df["Churn_binary"].mean():.6f} | original churn rate '
          f'{orig_df["Churn_binary"].mean():.6f}', flush=True)
    y_all = train_df['Churn_binary'].values

    def fit_fold(fold, tr_idx, val_idx, n_folds):
        df_tr_raw = train_df.iloc[tr_idx].reset_index(drop=True).copy()
        df_va_raw = train_df.iloc[val_idx].reset_index(drop=True).copy()
        df_te_raw = test_df.reset_index(drop=True).copy()
        X_tr, X_va, X_te, y_tr, y_va, te_cols = assemble_fold_features(df_tr_raw, df_va_raw, df_te_raw, orig_df,
                                                                       fold_seed=SEED + fold + 1)
        if fold == 0:
            print(f'  {X_tr.shape[1]} features ({len(te_cols)} target-encoded interactions)', flush=True)

        te = TargetEncoder(categories='auto', target_type='binary', smooth='auto', random_state=42)
        te.fit(X_tr[te_cols], y_tr)
        X_tr[te_cols] = te.transform(X_tr[te_cols])
        X_va[te_cols] = te.transform(X_va[te_cols])
        X_te[te_cols] = te.transform(X_te[te_cols])
        for _df in (X_tr, X_va, X_te):
            _df['te_mean'] = _df[te_cols].mean(axis=1)
            _df['te_std'] = _df[te_cols].std(axis=1)
            _df['te_min'] = _df[te_cols].min(axis=1)
            _df['te_max'] = _df[te_cols].max(axis=1)

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        val_pred = model.predict_proba(X_va)[:, 1]
        te_pred = model.predict_proba(X_te)[:, 1]
        best_iter = model.best_iteration if hasattr(model, 'best_iteration') else XGB_PARAMS['n_estimators']
        print(f'  fold {fold}: logloss {log_loss(y_va, val_pred):.6f}, best_iter {best_iter}', flush=True)
        del X_tr, X_va, X_te, model, te
        gc.collect()
        return val_pred, te_pred

    run_folds(FOLDER, FOLDER, fit_fold, y=y_all)


if __name__ == '__main__':
    main()
