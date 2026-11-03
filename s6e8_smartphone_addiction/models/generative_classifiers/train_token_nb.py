#!/usr/bin/env python
"""Generative naive Bayes over exact value tokens, with a logistic calibration.

Tokens: the 12 columns as exact value tokens ('%.2f' for numerics, raw strings for categoricals, NaN -> '__NA__');
vocabulary over train + test.
Model: per column and class, P(token | y) = (count + 20 prior_y) / (n_y + 20) with prior_y = class share of the
outer-train rows; score = sum of the 12 log-likelihood ratios + prior log-odds. Per outer fold the held-out and test rows
are scored with the tables of all outer-train rows, the outer-train rows with a 3-fold inner cross-fit
(StratifiedKFold seed 777 + fold). The score array is written fold by fold (held-out rows, then the outer-train rows with
that pass's inner scores), so after the loop folds 0-3 hold the inner scores of the fold-4 pass and fold 4 its held-out
scores.
Calibration: per fold, LogisticRegression(lbfgs, max_iter 1000) on the scores of the outer-train rows maps the held-out
scores and the fold's test scores to probabilities. Test = mean of the 5 calibrated fold predictions.
Usage: python train_token_nb.py"""
import math
import sys
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

NAME = 'naive_bayes_exact_tokens'
SMOOTH = 20.0
N_INNER = 3
NUMERIC_COLS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours',
                'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time']


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
        print(f'  {c}: {len(uniques)} unique tokens')
    return train_codes, test_codes, vocab_sizes


def build_maps(codes, y_fit, prior_1, prior_0, vocab_sizes, smooth):
    log_p1, log_p0 = [], []
    mask1 = y_fit == 1
    mask0 = y_fit == 0
    n1 = mask1.sum()
    n0 = mask0.sum()
    for col_idx, vocab_size in enumerate(vocab_sizes):
        col_codes = codes[:, col_idx]
        counts1 = np.bincount(col_codes[mask1], minlength=vocab_size).astype(np.float64)
        counts0 = np.bincount(col_codes[mask0], minlength=vocab_size).astype(np.float64)
        p1 = (counts1 + smooth * prior_1) / (n1 + smooth)
        p0 = (counts0 + smooth * prior_0) / (n0 + smooth)
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


def main():
    train, test = read_train(), read_test()
    y = labels(train[TARGET]).astype(np.int64)
    folds = fold_ids(y)
    feature_cols = [c for c in train.columns if c not in ['id', TARGET]]
    train_codes, test_codes, vocab_sizes = token_codes(train, test, feature_cols)

    n_folds, n_train, n_test = 5, len(train), len(test)
    oof_scores = np.zeros(n_train)
    test_scores = np.zeros((n_folds, n_test))
    for f in range(n_folds):
        trn_idx = np.where(folds != f)[0]
        val_idx = np.where(folds == f)[0]
        y_trn = y[trn_idx]
        prior_1 = y_trn.mean()
        prior_0 = 1 - prior_1
        log_prior_odds = math.log(prior_1 / prior_0)

        inner_skf = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=777 + f)
        oof_fold = np.zeros(len(trn_idx))
        full_log_p1, full_log_p0 = build_maps(train_codes[trn_idx], y_trn, prior_1, prior_0, vocab_sizes, SMOOTH)
        for i_fit, i_app in inner_skf.split(np.zeros(len(trn_idx)), y_trn):
            inner_log_p1, inner_log_p0 = build_maps(train_codes[trn_idx[i_fit]], y_trn[i_fit], prior_1, prior_0,
                                                    vocab_sizes, SMOOTH)
            oof_fold[i_app] = score_nb(train_codes[trn_idx[i_app]], inner_log_p1, inner_log_p0, log_prior_odds)

        oof_scores[val_idx] = score_nb(train_codes[val_idx], full_log_p1, full_log_p0, log_prior_odds)
        test_scores[f] = score_nb(test_codes, full_log_p1, full_log_p0, log_prior_odds)
        oof_scores[trn_idx] = oof_fold
        print(f'fold {f}: held-out AUC of the raw score '
              f'{roc_auc_score(y[val_idx], oof_scores[val_idx]):.6f}', flush=True)

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
