#!/usr/bin/env python
"""Explainable Boosting Machine (GA2M: shape functions + 25 pairwise interactions) on raw + derived columns and
4 nested target encodings, with a logistic calibration.

Features: the 9 raw numerics, slack = daily - (social + gaming + work), utilization = sum / daily (0 if daily is 0 or
missing), component sum, repair flag (sum > daily and utilization in [0.85, 0.95)), missing flags of the 9 numerics and
of slack / utilization / sum, one-hot categoricals (with a NaN level); NaN -> 0. Plus 4 target encodings of exact '%.2f'
tokens (daily, social, weekend, slack after NaN -> 0; m 20, prior = outer-train rate): outer-train rows from a 3-fold
inner cross-fit (StratifiedKFold seed 777 + fold), held-out and test rows from all outer-train rows.
Model: interpret's ExplainableBoostingClassifier(interactions 25, max_bins 128, max_interaction_bins 32, outer_bags 4,
inner_bags 0, learning_rate 0.03, validation_size 0.15, early_stopping_rounds 50, max_rounds 3000, min_samples_leaf 10,
max_leaves 3, random_state 42, n_jobs 1), fit per outer fold on 50,000 outer-train rows drawn with NumPy's global RNG
(np.random.choice without a seed, so reruns differ slightly).
Calibration: the EBM scores are written fold by fold (held-out rows, then all outer-train rows with that EBM's
predictions), so after the loop folds 0-3 hold the fold-4 EBM's predictions and fold 4 its held-out predictions. Per
fold, LogisticRegression(lbfgs, max_iter 1000) on these scores of the outer-train rows maps the held-out scores and the
fold's test scores to probabilities. Test = mean of the 5 calibrated fold predictions.
Usage: python train_ebm.py"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds as fold_ids, labels, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

NAME = 'ebm_ga2m_te'
CORE_NUMERIC = ['daily_screen_time_hours', 'social_media_hours', 'weekend_screen_time',
                'age', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day',
                'gaming_hours', 'work_study_hours']
CATEGORICAL_COLS = ['gender', 'stress_level', 'academic_work_impact']
TE_FEATURES = ['daily_screen_time_hours', 'social_media_hours', 'weekend_screen_time', 'slack']
SAMPLE_SIZE = 50000


def add_derived(df):
    df = df.copy()
    df['slack'] = df['daily_screen_time_hours'] - (df['social_media_hours'] + df['gaming_hours'] + df['work_study_hours'])
    df['utilization'] = np.where(
        df['daily_screen_time_hours'] > 0,
        (df['social_media_hours'] + df['gaming_hours'] + df['work_study_hours']) / df['daily_screen_time_hours'],
        0
    )
    df['comp_sum'] = df['social_media_hours'] + df['gaming_hours'] + df['work_study_hours']
    df['repair_flag'] = (
        (df['comp_sum'] > df['daily_screen_time_hours']) &
        (df['utilization'] >= 0.85) & (df['utilization'] < 0.95)
    ).astype(float)
    return df


def build_frames(train, test):
    train_core = add_derived(train[CORE_NUMERIC + CATEGORICAL_COLS].copy())
    test_core = add_derived(test[CORE_NUMERIC + CATEGORICAL_COLS].copy())
    for c in CORE_NUMERIC + ['slack', 'utilization', 'comp_sum']:
        train_core[f'{c}_missing'] = train_core[c].isna().astype(float)
        test_core[f'{c}_missing'] = test_core[c].isna().astype(float)
    for c in CATEGORICAL_COLS:
        dummies_tr = pd.get_dummies(train_core[c], prefix=c, dummy_na=True)
        dummies_te = pd.get_dummies(test_core[c], prefix=c, dummy_na=True)
        dummies_tr, dummies_te = dummies_tr.align(dummies_te, join='left', axis=1, fill_value=0)
        train_core = pd.concat([train_core.drop(c, axis=1), dummies_tr], axis=1)
        test_core = pd.concat([test_core.drop(c, axis=1), dummies_te], axis=1)
    train_core, test_core = train_core.align(test_core, join='left', axis=1, fill_value=0)
    return train_core.fillna(0), test_core.fillna(0)


def tokenize_numeric(s):
    return s.map(lambda v: f"{v:.2f}" if pd.notna(v) else "__NA__")


def compute_te_map(tokens, y_fit, prior, smooth=20.0):
    df = pd.DataFrame({"token": tokens, "label": y_fit})
    g = df.groupby("token")["label"].agg(["sum", "count"])
    return (g["sum"] + smooth * prior) / (g["count"] + smooth)


def main():
    from interpret.glassbox import ExplainableBoostingClassifier

    train, test = read_train(), read_test()
    y = labels(train[TARGET]).astype(np.int64)
    folds = fold_ids(y)

    print('Building the feature set...', flush=True)
    train_core, test_core = build_frames(train, test)
    print(f'Features: {train_core.shape[1]}', flush=True)

    train_tokens_dict, test_tokens_dict = {}, {}
    for c in TE_FEATURES:
        train_tokens_dict[c] = tokenize_numeric(train[c] if c in train.columns else train_core[c])
        test_tokens_dict[c] = tokenize_numeric(test[c] if c in test.columns else test_core[c])

    n_folds, n_train, n_test = 5, len(train), len(test)
    oof_scores = np.zeros(n_train)
    test_scores = np.zeros((n_folds, n_test))
    for f in range(n_folds):
        print(f'=== Fold {f} ===', flush=True)
        trn_idx = np.where(folds != f)[0]
        val_idx = np.where(folds == f)[0]
        y_trn = y[trn_idx]
        prior_1 = y_trn.mean()
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=777 + f)

        X_train_fold = train_core.iloc[trn_idx].copy()
        X_val_fold = train_core.iloc[val_idx].copy()
        for te_col in TE_FEATURES:
            oof_te = np.zeros(len(trn_idx))
            for i_fit, i_app in inner_skf.split(np.zeros(len(trn_idx)), y_trn):
                te_map = compute_te_map(train_tokens_dict[te_col].iloc[trn_idx[i_fit]], y_trn[i_fit], prior_1)
                oof_te[i_app] = train_tokens_dict[te_col].iloc[trn_idx[i_app]].map(te_map).fillna(prior_1).values
            full_te_map = compute_te_map(train_tokens_dict[te_col].iloc[trn_idx], y_trn, prior_1)
            X_train_fold[f'te_{te_col}'] = oof_te
            X_val_fold[f'te_{te_col}'] = train_tokens_dict[te_col].iloc[val_idx].map(full_te_map).fillna(prior_1).values
        X_train_fold = X_train_fold.fillna(0)
        X_val_fold = X_val_fold.fillna(0)

        # test encodings from all outer-train rows
        for te_col in TE_FEATURES:
            full_te_map = compute_te_map(train_tokens_dict[te_col].iloc[trn_idx], y_trn, prior_1)
            test_core[f'te_{te_col}'] = test_tokens_dict[te_col].map(full_te_map).fillna(prior_1).values
        X_test_fold = test_core.fillna(0)
        print(f'  Feature count: {X_train_fold.shape[1]}', flush=True)

        ebm = ExplainableBoostingClassifier(
            interactions=25,
            n_jobs=1,
            random_state=42,
            max_bins=128,
            max_interaction_bins=32,
            outer_bags=4,
            inner_bags=0,
            learning_rate=0.03,
            validation_size=0.15,
            early_stopping_rounds=50,
            max_rounds=3000,
            min_samples_leaf=10,
            max_leaves=3
        )
        # a random subsample of the outer-train rows for speed
        sample_size = min(SAMPLE_SIZE, len(X_train_fold))
        sample_idx = np.random.choice(len(X_train_fold), sample_size, replace=False)
        ebm.fit(X_train_fold.iloc[sample_idx], y_trn[sample_idx])

        oof_scores[val_idx] = ebm.predict_proba(X_val_fold)[:, 1]
        test_scores[f] = ebm.predict_proba(X_test_fold)[:, 1]
        oof_scores[trn_idx] = ebm.predict_proba(X_train_fold)[:, 1]
        print(f'  fold {f} held-out AUC: {roc_auc_score(y[val_idx], oof_scores[val_idx]):.6f}', flush=True)

    # logistic calibration per fold
    oof_probs = np.zeros(n_train)
    test_probs = np.zeros((n_folds, n_test))
    for f in range(n_folds):
        trn_idx = np.where(folds != f)[0]
        val_idx = np.where(folds == f)[0]
        cal = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
        cal.fit(oof_scores[trn_idx].reshape(-1, 1), y[trn_idx])
        oof_probs[val_idx] = cal.predict_proba(oof_scores[val_idx].reshape(-1, 1))[:, 1]
        test_probs[f] = cal.predict_proba(test_scores[f].reshape(-1, 1))[:, 1]

    save_pair(HERE, NAME, oof_probs, test_probs.mean(axis=0))
    print(f'{NAME}: OOF AUC {roc_auc_score(y, oof_probs):.6f} | folds '
          + ' '.join(f'{roc_auc_score(y[folds == k], oof_probs[folds == k]):.6f}' for k in range(n_folds)))


if __name__ == '__main__':
    main()
