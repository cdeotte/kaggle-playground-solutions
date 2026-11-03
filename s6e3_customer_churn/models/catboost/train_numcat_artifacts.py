#!/usr/bin/env python
"""GPU CatBoost on numeric transforms, digit / frequency artifacts and native categories of the raw, rounded, binned
and crossed values.

Recipes (pair name):
  catboost_numcat_digits_orig_priors_drift   features built over train + test, plus original-data lookups: value
                                             counts in the original data and their ratio to the train + test counts,
                                             original churn rate of every value, TotalCharges percentile among the
                                             original churners / non-churners, train-vs-original drift
                                             log1p(p_train / p_original) of every value, original means / stds of the
                                             numerics per gender, Partner, Dependents, PhoneService (242 numeric
                                             columns)
  catboost_numcat_digits_orig_rows           features built over train + test + original without the original-data
                                             lookups (146 numeric columns); the 7,043 original rows are appended to
                                             the training rows of every fold

Numerics are tenure, MonthlyCharges, TotalCharges and SeniorCitizen; every other column, id included, is treated as a
categorical. Blank TotalCharges are set to 0. Features:
  - every numeric, its log1p, sqrt, square, 1 / (x + 1) and percentile rank; 9 charge relations (MonthlyCharges /
    TotalCharges ratios, TotalCharges - tenure x MonthlyCharges and its absolute value, TotalCharges / (tenure + 1),
    TotalCharges - MonthlyCharges, tenure x MonthlyCharges, tenure^2 x MonthlyCharges, ...)
  - the count of every value of every column
  - first digit, last integer digit, integer part mod 12 / mod 100, number of digits, fractional part, round flag,
    fractional part x 100 / x 1000, 4 integer digits and 4 decimal digits of every numeric
  - Yes / No flags of the 11 Yes / No service columns, the number of Yes and No answers and their difference
  - 65 CatBoost-native categoricals: the raw categoricals; every numeric as an exact / rounded (0 and 1 decimals) /
    integer / 50-quantile bin / even floor string; 10 bigrams and a trigram of InternetService, OnlineSecurity,
    TechSupport, Contract, PaymentMethod; tenure and MonthlyCharges (1 decimal) crossed with id, gender and Partner;
    the last two integer digits and the two decimal digits of every numeric
CatBoost (GPU): depth 8, lr 0.03, l2 3, random_strength 0.5, Bayesian bagging_temperature 0.7, border_count 254,
one_hot_max_size 4, max_ctr_complexity 2, up to 5000 iterations with early stopping (300) on the validation fold
(use_best_model=True). Test = mean of the fold models.

Usage: python train_numcat_artifacts.py <pair>"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402
from models.catboost.lib.cv import run_folds  # noqa: E402

warnings.filterwarnings('ignore')

NUMERICS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
TOP_CATS = ['Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'tenure']
YES_NO_VALUES = {'Yes', 'No', 'No internet service', 'No phone service'}

RECIPES = {
    'catboost_numcat_digits_orig_priors_drift': dict(orig_rows=False, test_folds=25, test_seed=777),
    'catboost_numcat_digits_orig_rows': dict(orig_rows=True, test_folds=25, test_seed=42),
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


def load_data():
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = labels(train[TARGET]).astype(int)
    orig[TARGET] = labels(orig[TARGET]).astype(int)
    for df in (train, test, orig):
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    return train, test, orig


def drift_map(reference, values):
    """log1p(p_values / p_reference) of every value seen in either frame."""
    p_ref = reference.value_counts(normalize=True)
    p_val = values.value_counts(normalize=True)
    return {v: np.log1p(p_val.get(v, 0) / p_ref.get(v, 1e-6)) for v in set(list(p_ref.index) + list(p_val.index))}


def build_features(train, test, orig, orig_rows):
    all_cols = [c for c in train.columns if c != TARGET]
    num_cols = [c for c in NUMERICS if c in all_cols]
    cat_cols = [c for c in all_cols if c not in num_cols]
    lookups = not orig_rows

    parts = [train.drop(columns=[TARGET]), test]
    if orig_rows:
        parts.append(orig.drop(columns=[TARGET]))
    combined = pd.concat(parts, axis=0, ignore_index=True)
    n_train, n_test = len(train), len(test)
    orig_churn_maps = {c: orig.groupby(c)[TARGET].mean().to_dict() for c in orig.columns if c != TARGET}

    # numeric transforms and charge relations
    feat = pd.DataFrame(index=combined.index)
    for c in num_cols:
        vals = combined[c].values.astype(np.float64)
        feat[c] = vals
        feat[f'{c}_log1p'] = np.log1p(np.abs(vals))
        feat[f'{c}_sqrt'] = np.sqrt(np.abs(vals))
        feat[f'{c}_sq'] = vals ** 2
        feat[f'{c}_inv'] = 1.0 / (vals + 1.0)
        feat[f'{c}_rank'] = pd.Series(vals).rank(pct=True).values
    mc = combined['MonthlyCharges'].values.astype(np.float64)
    tc = combined['TotalCharges'].values.astype(np.float64)
    ten = combined['tenure'].values.astype(np.float64)
    feat['charges_ratio'] = mc / (tc + 1.0)
    feat['charges_ratio_inv'] = tc / (mc + 1.0)
    feat['charges_deviation'] = tc - ten * mc
    feat['avg_monthly'] = tc / (ten + 1.0)
    feat['mc_x_tenure'] = mc * ten
    feat['charges_diff'] = tc - mc
    feat['tenure_sq_x_mc'] = (ten ** 2) * mc
    feat['charges_deviation_abs'] = np.abs(tc - ten * mc)
    feat['avg_monthly_ratio'] = feat['avg_monthly'] / (mc + 1.0)

    # value counts of the numerics
    for c in num_cols:
        vc = combined[c].value_counts()
        feat[f'{c}_freq'] = combined[c].map(vc).values.astype(np.float32)
        if lookups:
            vc_orig = orig[c].value_counts()
            feat[f'{c}_freq_orig'] = combined[c].map(vc_orig).fillna(0).values.astype(np.float32)
            feat[f'{c}_freq_ratio'] = feat[f'{c}_freq'] / (feat[f'{c}_freq_orig'] + 1.0)
            feat[f'{c}_orig_churn_prob'] = combined[c].map(orig_churn_maps[c]).fillna(0.5).values.astype(np.float32)

    # digits of the numerics
    for c in num_cols:
        vals = combined[c].values
        s = pd.Series(vals).astype(str)
        feat[f'{c}_first_digit'] = s.str.replace(r'[^0-9]', '', regex=True).str[0].fillna('0').astype(int).values
        int_part = np.floor(np.abs(vals)).astype(np.int64)
        feat[f'{c}_last_digit'] = (int_part % 10).astype(np.int8)
        feat[f'{c}_mod12'] = (int_part % 12).astype(np.int8)
        feat[f'{c}_mod100'] = (int_part % 100).astype(np.int16)
        feat[f'{c}_ndigits'] = np.floor(np.log10(np.abs(vals) + 1) + 1).astype(np.int8)
        frac = np.abs(vals) - np.floor(np.abs(vals))
        feat[f'{c}_frac'] = frac.astype(np.float32)
        feat[f'{c}_is_round'] = (frac < 0.005).astype(np.int8)
        feat[f'{c}_frac_x100'] = (frac * 100).astype(np.int16)
        feat[f'{c}_frac_x1000'] = (frac * 1000).astype(np.int16)
        for d in range(4):
            feat[f'{c}_intdigit_{d}'] = ((int_part // (10 ** d)) % 10).astype(np.int8)
        dec_shifted = (frac * 10000).astype(np.int64)
        for d in range(4):
            feat[f'{c}_decdigit_{d}'] = ((dec_shifted // (10 ** (3 - d))) % 10).astype(np.int8)

    # TotalCharges percentile among the original churners / non-churners
    if lookups:
        tc_churn = np.sort(orig.loc[orig[TARGET] == 1, 'TotalCharges'].values)
        tc_nochurn = np.sort(orig.loc[orig[TARGET] == 0, 'TotalCharges'].values)
        tc_vals = combined['TotalCharges'].values
        feat['tc_pctile_vs_churn'] = np.searchsorted(tc_churn, tc_vals) / (len(tc_churn) + 1)
        feat['tc_pctile_vs_nochurn'] = np.searchsorted(tc_nochurn, tc_vals) / (len(tc_nochurn) + 1)
        feat['tc_pctile_gap'] = feat['tc_pctile_vs_churn'] - feat['tc_pctile_vs_nochurn']

    # value counts of the categoricals
    for c in cat_cols:
        vc = combined[c].value_counts()
        feat[f'{c}_freq'] = combined[c].map(vc).values.astype(np.float32)
        if lookups and c in orig.columns:
            feat[f'{c}_orig_churn_prob'] = combined[c].map(orig_churn_maps[c]).fillna(0.5).values.astype(np.float32)
            vc_orig = orig[c].value_counts()
            feat[f'{c}_freq_orig'] = combined[c].map(vc_orig).fillna(0).values.astype(np.float32)

    # service answers
    yes_no_cols = [c for c in cat_cols if set(combined[c].dropna().unique()) <= YES_NO_VALUES]
    for c in yes_no_cols:
        feat[f'{c}_is_yes'] = (combined[c] == 'Yes').astype(np.int8)
        feat[f'{c}_is_no'] = (combined[c] == 'No').astype(np.int8)
    feat['n_yes_services'] = feat[[f'{c}_is_yes' for c in yes_no_cols]].sum(axis=1).astype(np.int8)
    feat['n_no_services'] = feat[[f'{c}_is_no' for c in yes_no_cols]].sum(axis=1).astype(np.int8)
    feat['yes_minus_no'] = feat['n_yes_services'] - feat['n_no_services']

    # CatBoost-native categoricals
    cat_native = pd.DataFrame(index=combined.index)
    for c in cat_cols:
        cat_native[f'RAW_{c}'] = combined[c].astype(str).fillna('__NA__')
    for c in num_cols:
        cat_native[f'NUMCAT_{c}_exact'] = combined[c].astype(str)
        cat_native[f'NUMCAT_{c}_round0'] = np.round(combined[c], 0).astype(str)
        cat_native[f'NUMCAT_{c}_round1'] = np.round(combined[c], 1).astype(str)
        cat_native[f'NUMCAT_{c}_int'] = np.floor(combined[c].values).astype(int).astype(str)
        try:
            cat_native[f'NUMCAT_{c}_qbin50'] = pd.qcut(combined[c], q=50, duplicates='drop').astype(str)
        except Exception:
            cat_native[f'NUMCAT_{c}_qbin50'] = pd.cut(combined[c], bins=50).astype(str)
        cat_native[f'NUMCAT_{c}_div2'] = (np.floor(combined[c].values / 2) * 2).astype(int).astype(str)
    top_cats = [c for c in cat_cols if c in TOP_CATS]
    for i, c1 in enumerate(top_cats):
        for c2 in top_cats[i + 1:]:
            cat_native[f'BI_{c1}_{c2}'] = combined[c1].astype(str) + '_' + combined[c2].astype(str)
    cat_native[f"TRI_{'_'.join(top_cats[:3])}"] = (
        combined[top_cats[0]].astype(str) + '_'
        + combined[top_cats[1]].astype(str) + '_'
        + combined[top_cats[2]].astype(str)
    )
    for nc in num_cols[:2]:
        nc_str = np.round(combined[nc], 1).astype(str)
        for cc in cat_cols[:3]:
            cat_native[f'CROSS_{nc}_{cc}'] = nc_str + '_' + combined[cc].astype(str)
    for c in num_cols:
        int_part = np.floor(np.abs(combined[c].values)).astype(np.int64)
        d0 = (int_part % 10).astype(str)
        d1 = ((int_part // 10) % 10).astype(str)
        cat_native[f'DIGITPAIR_{c}_01'] = d1 + d0
        frac = np.abs(combined[c].values) - np.floor(np.abs(combined[c].values))
        cat_native[f'DIGITPAIR_{c}_dec01'] = (frac * 100).astype(np.int64).astype(str)

    # train-vs-original drift and original group statistics
    if lookups:
        train_only = combined.iloc[:n_train]
        for c in cat_cols + num_cols:
            if c in orig.columns:
                feat[f'{c}_drift'] = combined[c].map(drift_map(orig[c], train_only[c])).fillna(0).astype(np.float32)
        for nc in num_cols:
            for cc in cat_cols[:5]:
                if cc not in orig.columns:
                    continue
                grp = orig.groupby(cc)[nc].mean()
                feat[f'GRPMEAN_{nc}_by_{cc}'] = combined[cc].map(grp).fillna(0).astype(np.float32)
                grp_std = orig.groupby(cc)[nc].std().fillna(0)
                feat[f'GRPSTD_{nc}_by_{cc}'] = combined[cc].map(grp_std).fillna(0).astype(np.float32)

    print(f'numeric features: {feat.shape[1]}; CatBoost-native categoricals: {cat_native.shape[1]}', flush=True)
    full = pd.concat([feat, cat_native], axis=1)
    return dict(
        train=full.iloc[:n_train].reset_index(drop=True),
        test=full.iloc[n_train:n_train + n_test].reset_index(drop=True),
        orig=full.iloc[n_train + n_test:].reset_index(drop=True),
        features=feat.columns.tolist() + cat_native.columns.tolist(),
        cat_features=cat_native.columns.tolist(),
    )


def make_fit_fold(data, y, y_orig, orig_rows):
    train_all, test_all, orig_all = data['train'], data['test'], data['orig']
    features = data['features']
    cat_idx = [features.index(c) for c in data['cat_features']]

    def fit_fold(fold, tr_idx, va_idx, seed):
        X_tr = train_all.loc[tr_idx, features]
        y_tr = y[tr_idx]
        if orig_rows:
            X_tr = pd.concat([X_tr, orig_all[features]], axis=0)
            y_tr = np.concatenate([y_tr, y_orig])
        X_tr = X_tr.reset_index(drop=True)
        X_val = train_all.loc[va_idx, features].reset_index(drop=True)
        X_te = test_all[features].reset_index(drop=True)
        train_pool = Pool(data=X_tr, label=y_tr, cat_features=cat_idx, feature_names=features)
        val_pool = Pool(data=X_val, label=y[va_idx], cat_features=cat_idx, feature_names=features)
        test_pool = Pool(data=X_te, cat_features=cat_idx, feature_names=features)
        model = CatBoostClassifier(**cb_params(seed))
        model.fit(train_pool, eval_set=val_pool)
        return model.predict_proba(val_pool)[:, 1], model.predict_proba(test_pool)[:, 1]

    return fit_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=list(RECIPES))
    pair = ap.parse_args().pair
    cfg = RECIPES[pair]

    train, test, orig = load_data()
    y, y_orig = train[TARGET].values, orig[TARGET].values
    data = build_features(train, test, orig, cfg['orig_rows'])
    fit_fold = make_fit_fold(data, y, y_orig, cfg['orig_rows'])

    oof, _ = run_folds(fit_fold, y, len(test), name=pair)
    _, test_pred = run_folds(fit_fold, y, len(test), n_folds=cfg['test_folds'], seed=cfg['test_seed'], name=pair)
    print(f'{pair}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(HERE, pair, oof, test_pred)


if __name__ == '__main__':
    main()
