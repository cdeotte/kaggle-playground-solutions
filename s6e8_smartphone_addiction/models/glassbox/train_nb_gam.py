#!/usr/bin/env python
"""Additive model: exact-token naive Bayes + spline GAM, combined by a degree-2 polynomial logistic regression with
original-data evidence.

NB (sharp part): the 12 columns as exact value tokens ('%.2f' for numerics, raw strings, NaN -> '__NA__'; vocabulary over
train + test); per column and class P(token | y) = (count + 20 prior_y) / (n_y + 20), prior_y = class share of the
fitting rows; score = sum of the 12 log-likelihood ratios + prior log-odds.
GAM (smooth part): 15 numerics (the 9 raw + slack, utilization, sleep + daily, weekend - daily, notifications x app
opens, component sum; NaN -> 0) as cubic B-splines (SplineTransformer, 15 knots), plus missing flags, one-hot categoricals
(with a NaN level), a repair flag (sum > daily and utilization in [0.85, 0.95)) and 4 nested target encodings of exact
tokens (daily, social, weekend, slack; m 20); LogisticRegression(C 0.5, lbfgs, max_iter 500).
Original-data evidence: sum over daily / social / weekend of the smoothed per-value log-odds ratio in the original
7,500 rows (m 20; unseen values 0).
Per outer fold both parts are fit on the outer-train rows; their outer-train scores come from a 3-fold inner cross-fit
(StratifiedKFold seed 777 + fold, the same splits for the NB, the encodings and the GAM). The score arrays are written
fold by fold (held-out rows, then the outer-train rows with that pass's inner scores), so after the loop folds 0-3 hold
the inner scores of the fold-4 pass and fold 4 its held-out scores.
Combiner: per fold, LogisticRegression(C 1, lbfgs, max_iter 1000) on the degree-2 polynomial features of
[NB score, GAM probability, evidence] of the outer-train rows, applied to the held-out rows and to the fold's test
scores. Test = mean of the 5 fold predictions.
Usage: python train_nb_gam.py"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds as fold_ids, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

NAME = 'spline_gam_nb_poly2'
SMOOTH = 20.0  # NB smoothing
C_GAM = 0.5
NUMERIC_COLS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours',
                'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time']
CATEGORICAL_COLS = ['gender', 'stress_level', 'academic_work_impact']
TE_FEATURES = ['daily_screen_time_hours', 'social_media_hours', 'weekend_screen_time', 'slack']
SPLINE_FEATURES = ['daily_screen_time_hours', 'social_media_hours', 'weekend_screen_time', 'age', 'sleep_hours',
                   'notifications_per_day', 'app_opens_per_day', 'gaming_hours', 'work_study_hours', 'slack',
                   'utilization', 'sleep_plus_daily', 'weekend_minus_daily', 'notif_x_app', 'comp_sum']
EVIDENCE_COLS = ['daily_screen_time_hours', 'social_media_hours', 'weekend_screen_time']


def tokenize_series(s, is_numeric=True):
    if is_numeric:
        return s.map(lambda v: f"{v:.2f}" if pd.notna(v) else "__NA__")
    return s.map(lambda v: str(v) if pd.notna(v) else "__NA__")


def token_codes(train, test, feature_cols):
    """Integer codes of the exact value tokens (vocabulary over train + test) and the vocabulary sizes."""
    train_codes = np.zeros((len(train), len(feature_cols)), dtype=np.int32)
    test_codes = np.zeros((len(test), len(feature_cols)), dtype=np.int32)
    vocab_sizes = []
    for j, c in enumerate(feature_cols):
        is_num = c in NUMERIC_COLS
        all_tokens = pd.concat([tokenize_series(train[c], is_num), tokenize_series(test[c], is_num)], ignore_index=True)
        codes, uniques = pd.factorize(all_tokens)
        train_codes[:, j] = codes[:len(train)]
        test_codes[:, j] = codes[len(train):]
        vocab_sizes.append(len(uniques))
    return train_codes, test_codes, vocab_sizes


def build_nb_maps(codes, y_fit, prior_1, prior_0, vocab_sizes):
    log_p1, log_p0 = [], []
    mask1 = y_fit == 1
    mask0 = y_fit == 0
    n1 = mask1.sum()
    n0 = mask0.sum()
    for col_idx, vocab_size in enumerate(vocab_sizes):
        col_codes = codes[:, col_idx]
        counts1 = np.bincount(col_codes[mask1], minlength=vocab_size).astype(np.float64)
        counts0 = np.bincount(col_codes[mask0], minlength=vocab_size).astype(np.float64)
        p1 = (counts1 + SMOOTH * prior_1) / (n1 + SMOOTH)
        p0 = (counts0 + SMOOTH * prior_0) / (n0 + SMOOTH)
        log_p1.append(np.log(p1 + 1e-15))
        log_p0.append(np.log(p0 + 1e-15))
    return log_p1, log_p0


def score_nb(codes, log_p1, log_p0, log_prior_odds):
    log_p1_total = np.zeros(codes.shape[0])
    log_p0_total = np.zeros(codes.shape[0])
    for col_idx in range(codes.shape[1]):
        col_codes = codes[:, col_idx]
        log_p1_total += log_p1[col_idx][col_codes]
        log_p0_total += log_p0[col_idx][col_codes]
    return log_p1_total - log_p0_total + log_prior_odds


def build_gam_features(df):
    feats = pd.DataFrame(index=df.index)
    key_numeric = ['daily_screen_time_hours', 'social_media_hours', 'weekend_screen_time',
                   'age', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day']
    for c in key_numeric:
        feats[c] = df[c]
        feats[f'{c}_missing'] = df[c].isna().astype(float)
    for c in ['gaming_hours', 'work_study_hours']:
        feats[c] = df[c]
        feats[f'{c}_missing'] = df[c].isna().astype(float)
    for c in CATEGORICAL_COLS:
        dummies = pd.get_dummies(df[c], prefix=c, dummy_na=True)
        feats = pd.concat([feats, dummies], axis=1)
    feats['slack'] = df['daily_screen_time_hours'] - (df['social_media_hours'] + df['gaming_hours'] + df['work_study_hours'])
    feats['slack_missing'] = feats['slack'].isna().astype(float)
    feats['utilization'] = np.where(
        df['daily_screen_time_hours'] > 0,
        (df['social_media_hours'] + df['gaming_hours'] + df['work_study_hours']) / df['daily_screen_time_hours'],
        0
    )
    feats['utilization_missing'] = feats['utilization'].isna().astype(float)
    feats['sleep_plus_daily'] = df['sleep_hours'] + df['daily_screen_time_hours']
    feats['sleep_plus_daily_missing'] = feats['sleep_plus_daily'].isna().astype(float)
    feats['weekend_minus_daily'] = df['weekend_screen_time'] - df['daily_screen_time_hours']
    feats['weekend_minus_daily_missing'] = feats['weekend_minus_daily'].isna().astype(float)
    feats['notif_x_app'] = df['notifications_per_day'] * df['app_opens_per_day']
    feats['notif_x_app_missing'] = feats['notif_x_app'].isna().astype(float)
    feats['comp_sum'] = df['social_media_hours'] + df['gaming_hours'] + df['work_study_hours']
    feats['comp_sum_missing'] = feats['comp_sum'].isna().astype(float)
    feats['repair_flag'] = (
        (feats['comp_sum'] > df['daily_screen_time_hours']) &
        (feats['utilization'] >= 0.85) & (feats['utilization'] < 0.95)
    ).astype(float)
    return feats


def tokenize_numeric(s):
    return s.map(lambda v: f"{v:.2f}" if pd.notna(v) else "__NA__")


def compute_te_map(tokens, y_fit, prior, smooth=20.0):
    df = pd.DataFrame({"token": tokens, "label": y_fit})
    g = df.groupby("token")["label"].agg(["sum", "count"])
    return (g["sum"] + smooth * prior) / (g["count"] + smooth)


def original_evidence(train, test):
    """Sum over EVIDENCE_COLS of the smoothed per-value log-odds ratio in the original data (fixed external evidence)."""
    original = read_original()
    ev_tr = np.zeros((len(train), len(EVIDENCE_COLS)), dtype=np.float32)
    ev_te = np.zeros((len(test), len(EVIDENCE_COLS)), dtype=np.float32)
    for i, c in enumerate(EVIDENCE_COLS):
        orig_tokens = original[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "__NA__")
        df = pd.DataFrame({"token": orig_tokens, "label": original[TARGET]})
        rates = df.groupby("token")["label"].mean()
        prior = original[TARGET].mean()
        smooth = 20.0
        log_odds = np.log((rates + smooth * prior) / (1 - rates + smooth * (1 - prior)) / (prior / (1 - prior)))
        ev_tr[:, i] = tokenize_numeric(train[c]).map(log_odds).fillna(0).values
        ev_te[:, i] = tokenize_numeric(test[c]).map(log_odds).fillna(0).values
    return ev_tr.sum(axis=1), ev_te.sum(axis=1)


def main():
    train, test = read_train(), read_test()
    y = labels(train[TARGET]).astype(np.int64)
    folds = fold_ids(y)
    feature_cols = [c for c in train.columns if c not in ['id', TARGET]]

    print('Building the naive Bayes tokens...', flush=True)
    train_codes, test_codes, vocab_sizes = token_codes(train, test, feature_cols)

    print('Building the GAM features...', flush=True)
    train_feats = build_gam_features(train)
    test_feats = build_gam_features(test)
    train_feats, test_feats = train_feats.align(test_feats, join='left', axis=1, fill_value=0)
    train_feats = train_feats.fillna(0)
    test_feats = test_feats.fillna(0)

    train_tokens_dict, test_tokens_dict = {}, {}
    for c in TE_FEATURES:
        if c in train.columns:
            train_tokens_dict[c] = tokenize_numeric(train[c])
            test_tokens_dict[c] = tokenize_numeric(test[c])
        else:
            train_tokens_dict[c] = tokenize_numeric(train_feats[c])
            test_tokens_dict[c] = tokenize_numeric(test_feats[c])

    ev_tr, ev_te = original_evidence(train, test)
    print(f'Evidence standalone AUC: {roc_auc_score(y, ev_tr):.6f}', flush=True)

    n_folds, n_train, n_test = 5, len(train), len(test)
    oof_scores_nb = np.zeros(n_train)
    oof_scores_gam = np.zeros(n_train)
    test_scores_nb = np.zeros((n_folds, n_test))
    test_scores_gam = np.zeros((n_folds, n_test))

    for f in range(n_folds):
        print(f'=== Fold {f} ===', flush=True)
        trn_idx = np.where(folds != f)[0]
        val_idx = np.where(folds == f)[0]
        y_trn = y[trn_idx]
        prior_1 = y_trn.mean()
        prior_0 = 1 - prior_1
        log_prior_odds = math.log(prior_1 / prior_0)
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=777 + f)

        # NB part
        full_nb_p1, full_nb_p0 = build_nb_maps(train_codes[trn_idx], y_trn, prior_1, prior_0, vocab_sizes)
        oof_fold_nb = np.zeros(len(trn_idx))
        for i_fit, i_app in inner_skf.split(np.zeros(len(trn_idx)), y_trn):
            inner_p1, inner_p0 = build_nb_maps(train_codes[trn_idx[i_fit]], y_trn[i_fit], prior_1, prior_0,
                                               vocab_sizes)
            oof_fold_nb[i_app] = score_nb(train_codes[trn_idx[i_app]], inner_p1, inner_p0, log_prior_odds)

        # GAM part
        X_train_fold = train_feats.iloc[trn_idx].copy()
        X_val_fold = train_feats.iloc[val_idx].copy()
        X_test_fold = test_feats.copy()
        for te_col in TE_FEATURES:
            oof_te = np.zeros(len(trn_idx))
            for i_fit, i_app in inner_skf.split(np.zeros(len(trn_idx)), y_trn):
                te_map = compute_te_map(train_tokens_dict[te_col].iloc[trn_idx[i_fit]], y_trn[i_fit], prior_1)
                oof_te[i_app] = train_tokens_dict[te_col].iloc[trn_idx[i_app]].map(te_map).fillna(prior_1).values
            full_te_map = compute_te_map(train_tokens_dict[te_col].iloc[trn_idx], y_trn, prior_1)
            X_train_fold[f'te_{te_col}'] = oof_te
            X_val_fold[f'te_{te_col}'] = train_tokens_dict[te_col].iloc[val_idx].map(full_te_map).fillna(prior_1).values
            X_test_fold[f'te_{te_col}'] = test_tokens_dict[te_col].map(full_te_map).fillna(prior_1).values
        X_train_fold = X_train_fold.fillna(0)
        X_val_fold = X_val_fold.fillna(0)
        X_test_fold = X_test_fold.fillna(0)

        numeric_features = [c for c in SPLINE_FEATURES if c in X_train_fold.columns]
        other_features = [c for c in X_train_fold.columns if c not in numeric_features]
        spline = SplineTransformer(n_knots=15, degree=3, include_bias=False)
        X_train_gam = np.hstack([spline.fit_transform(X_train_fold[numeric_features]), X_train_fold[other_features].values])
        X_val_gam = np.hstack([spline.transform(X_val_fold[numeric_features]), X_val_fold[other_features].values])
        X_test_gam = np.hstack([spline.transform(X_test_fold[numeric_features]), X_test_fold[other_features].values])

        oof_fold_gam = np.zeros(len(trn_idx))
        for i_fit, i_app in inner_skf.split(np.zeros(len(trn_idx)), y_trn):
            inner_lr = LogisticRegression(solver='lbfgs', max_iter=500, random_state=42, C=C_GAM)
            inner_lr.fit(X_train_gam[i_fit], y_trn[i_fit])
            oof_fold_gam[i_app] = inner_lr.predict_proba(X_train_gam[i_app])[:, 1]
        lr_gam = LogisticRegression(solver='lbfgs', max_iter=500, random_state=42, C=C_GAM)
        lr_gam.fit(X_train_gam, y_trn)

        oof_scores_nb[val_idx] = score_nb(train_codes[val_idx], full_nb_p1, full_nb_p0, log_prior_odds)
        oof_scores_gam[val_idx] = lr_gam.predict_proba(X_val_gam)[:, 1]
        test_scores_nb[f] = score_nb(test_codes, full_nb_p1, full_nb_p0, log_prior_odds)
        test_scores_gam[f] = lr_gam.predict_proba(X_test_gam)[:, 1]
        oof_scores_nb[trn_idx] = oof_fold_nb
        oof_scores_gam[trn_idx] = oof_fold_gam
        print(f'  NB AUC: {roc_auc_score(y, oof_scores_nb):.6f}  GAM AUC: {roc_auc_score(y, oof_scores_gam):.6f}',
              flush=True)

    # combiner: degree-2 polynomial logistic regression on [NB score, GAM probability, evidence]
    oof_probs = np.zeros(n_train)
    test_probs = np.zeros((n_folds, n_test))
    for f in range(n_folds):
        trn_idx = np.where(folds != f)[0]
        val_idx = np.where(folds == f)[0]
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_trn = poly.fit_transform(np.column_stack([oof_scores_nb[trn_idx], oof_scores_gam[trn_idx], ev_tr[trn_idx]]))
        cal = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42, C=1.0)
        cal.fit(X_trn, y[trn_idx])
        X_val = poly.transform(np.column_stack([oof_scores_nb[val_idx], oof_scores_gam[val_idx], ev_tr[val_idx]]))
        oof_probs[val_idx] = cal.predict_proba(X_val)[:, 1]
        X_test = poly.transform(np.column_stack([test_scores_nb[f], test_scores_gam[f], ev_te]))
        test_probs[f] = cal.predict_proba(X_test)[:, 1]

    save_pair(HERE, NAME, oof_probs, test_probs.mean(axis=0))
    print(f'{NAME}: OOF AUC {roc_auc_score(y, oof_probs):.6f} | folds '
          + ' '.join(f'{roc_auc_score(y[folds == k], oof_probs[folds == k]):.6f}' for k in range(n_folds)))


if __name__ == '__main__':
    main()
