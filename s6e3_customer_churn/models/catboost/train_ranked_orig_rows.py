#!/usr/bin/env python
"""GPU CatBoost (depth 6, Bernoulli) on a compact frame with an exact numeric-triple category, the original rows
appended to every fold and a per-fold importance ranking of the features (pair catboost_triple_cross_ranked_orig_rows).

Numerics are tenure, MonthlyCharges, TotalCharges and SeniorCitizen; every other column, id included, is treated as a
categorical. Blank TotalCharges are set to 0. Features are built over train + test + original (73 columns):
  - every numeric and its log1p; MonthlyCharges / (TotalCharges + 1), TotalCharges - tenure x MonthlyCharges and its
    absolute value, TotalCharges / (tenure + 1), and a 3-level flag of that absolute deviation (< 5, < 50, >= 50)
  - the count of every value of every column; integer part mod 100 of every numeric
  - CatBoost-native categoricals (36): the deviation flag; the raw categoricals; every numeric as an exact / integer /
    even floor string; the exact tenure_MonthlyCharges_TotalCharges triple; 6 bigrams of Contract, InternetService,
    PaymentMethod, OnlineSecurity
Per fold, the 7,043 original rows are appended to the training rows; a 300-iteration depth-4 CatBoost (seed 42) fit
on them ranks the features by importance and the top 150 are kept in that order (all 73 pass).
CatBoost (GPU): depth 6, lr 0.03, l2 3, random_strength 0.5, Bernoulli subsample 0.8, border_count 254,
one_hot_max_size 4, max_ctr_complexity 1, up to 5000 iterations with early stopping (300) on the validation fold
(use_best_model=True), seed 42. Test = mean of the 5 fold models.

Usage: python train_ranked_orig_rows.py"""
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

PAIR = 'catboost_triple_cross_ranked_orig_rows'
SEED = 42
N_TOP_FEATURES = 150
NUMERICS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
TOP_CATS = ['Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity']

CB_PARAMS = dict(
    iterations=5000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_strength=0.5,
    bootstrap_type='Bernoulli',
    subsample=0.8,
    border_count=254,
    grow_policy='SymmetricTree',
    eval_metric='AUC',
    loss_function='Logloss',
    task_type='GPU',
    random_seed=SEED,
    verbose=500,
    early_stopping_rounds=300,
    use_best_model=True,
    one_hot_max_size=4,
    max_ctr_complexity=1,
)
SELECTION_PARAMS = dict(iterations=300, depth=4, task_type='GPU', verbose=0, random_seed=SEED)


def load_data():
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = labels(train[TARGET]).astype(int)
    orig[TARGET] = labels(orig[TARGET]).astype(int)
    for df in (train, test, orig):
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    return train, test, orig


def build_features(train, test, orig):
    all_cols = [c for c in train.columns if c != TARGET]
    num_cols = [c for c in NUMERICS if c in all_cols]
    cat_cols = [c for c in all_cols if c not in num_cols]
    combined = pd.concat([train.drop(columns=[TARGET]), test, orig.drop(columns=[TARGET])], axis=0, ignore_index=True)
    n_train, n_test = len(train), len(test)

    feat = pd.DataFrame(index=combined.index)
    for c in num_cols:
        vals = combined[c].values.astype(np.float64)
        feat[c] = vals
        feat[f'{c}_log1p'] = np.log1p(np.abs(vals))
    mc = combined['MonthlyCharges'].values.astype(np.float64)
    tc = combined['TotalCharges'].values.astype(np.float64)
    ten = combined['tenure'].values.astype(np.float64)
    feat['charges_ratio'] = mc / (tc + 1.0)
    feat['charges_deviation'] = tc - ten * mc
    feat['charges_deviation_abs'] = np.abs(tc - ten * mc)
    feat['avg_monthly'] = tc / (ten + 1.0)
    feat['math_broken_flag'] = pd.cut(
        feat['charges_deviation_abs'], bins=[-np.inf, 5.0, 50.0, np.inf], labels=[0, 1, 2], right=False,
    ).astype(np.int8)

    for c in num_cols:
        vc = combined[c].value_counts()
        feat[f'{c}_freq'] = combined[c].map(vc).values.astype(np.float32)
        int_part = np.floor(np.abs(combined[c].values)).astype(np.int64)
        feat[f'{c}_mod100'] = (int_part % 100).astype(np.int16)
    for c in cat_cols:
        feat[f'{c}_freq'] = combined[c].map(combined[c].value_counts()).values.astype(np.float32)

    cat_native = pd.DataFrame(index=combined.index)
    cat_native['math_broken_flag_str'] = feat['math_broken_flag'].astype(str)
    for c in cat_cols:
        cat_native[f'RAW_{c}'] = combined[c].astype(str).fillna('__NA__')
    for c in num_cols:
        cat_native[f'NUMCAT_{c}_exact'] = combined[c].astype(str)
        cat_native[f'NUMCAT_{c}_int'] = np.floor(combined[c].values).astype(int).astype(str)
        cat_native[f'NUMCAT_{c}_div2'] = (np.floor(combined[c].values / 2) * 2).astype(int).astype(str)
    cat_native['CROSS_exact_all'] = (
        combined['tenure'].astype(str) + '_' + combined['MonthlyCharges'].astype(str) + '_'
        + combined['TotalCharges'].astype(str)
    )
    top_cats = [c for c in TOP_CATS if c in cat_cols]
    for i, c1 in enumerate(top_cats):
        for c2 in top_cats[i + 1:]:
            cat_native[f'BI_{c1}_{c2}'] = combined[c1].astype(str) + '_' + combined[c2].astype(str)

    full = pd.concat([feat, cat_native], axis=1)
    print(f'features: {full.shape[1]} ({cat_native.shape[1]} categorical)', flush=True)
    return dict(
        train=full.iloc[:n_train].reset_index(drop=True),
        test=full.iloc[n_train:n_train + n_test].reset_index(drop=True),
        orig=full.iloc[n_train + n_test:].reset_index(drop=True),
        features=full.columns.tolist(),
        cat_features=cat_native.columns.tolist(),
    )


def main():
    train, test, orig = load_data()
    y, y_orig = train[TARGET].values, orig[TARGET].values
    data = build_features(train, test, orig)
    train_all, test_all, orig_all = data['train'], data['test'], data['orig']
    features, cat_features = data['features'], data['cat_features']

    def fit_fold(fold, tr_idx, va_idx, seed):
        X_tr = pd.concat([train_all.loc[tr_idx, features].reset_index(drop=True), orig_all[features]],
                         axis=0).reset_index(drop=True)
        y_tr = np.concatenate([y[tr_idx], y_orig])
        X_val = train_all.loc[va_idx, features].reset_index(drop=True)
        X_te = test_all[features].reset_index(drop=True)

        sel_pool = Pool(X_tr, y_tr, cat_features=[features.index(c) for c in cat_features], feature_names=features)
        sel_model = CatBoostClassifier(**SELECTION_PARAMS)
        sel_model.fit(sel_pool)
        imp = pd.DataFrame({'feature': features, 'importance': sel_model.get_feature_importance()})
        top_feats = imp.sort_values('importance', ascending=False).head(N_TOP_FEATURES)['feature'].tolist()
        top_cats = [c for c in cat_features if c in top_feats]
        cat_idx = [top_feats.index(c) for c in top_cats]
        print(f'fold {fold}: {len(top_feats)} features kept', flush=True)

        train_pool = Pool(X_tr[top_feats], y_tr, cat_features=cat_idx, feature_names=top_feats)
        val_pool = Pool(X_val[top_feats], y[va_idx], cat_features=cat_idx, feature_names=top_feats)
        test_pool = Pool(X_te[top_feats], cat_features=cat_idx, feature_names=top_feats)
        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(train_pool, eval_set=val_pool)
        return model.predict_proba(val_pool)[:, 1], model.predict_proba(test_pool)[:, 1]

    oof, test_pred = run_folds(fit_fold, y, len(test), name=PAIR)
    print(f'{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
