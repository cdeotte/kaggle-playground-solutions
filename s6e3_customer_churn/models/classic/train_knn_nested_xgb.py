#!/usr/bin/env python
"""XGBoost on original-data KNN, nested train-data KNN and logistic-regression meta features (two presets).

Base columns (19): the 16 categoricals (pandas categories shared by train / test / original) and the 3 numerics
(float32; the 11 blank TotalCharges of the original rows filled with the original median).
KNN (lib/knn_schemes.py; GPU brute force, numerics in fp16): 30 random weight schemes and 30 smart schemes (built from
the importances of an XGBoost fitted on the original rows).
  - Original-data KNN (k = knn_k; reference = the 7,043 original rows, numerics standardized on them) of the train and
    test rows under all 60 schemes.
  - 8 random + 8 smart schemes are kept, with blend weights:
      top_schemes preset: the 8 + 8 prior columns with the largest mean XGBoost gain, weighted by that gain
        (select_knn_schemes.py; in our run random schemes 14, 22, 20, 16, 17, 18, 2, 7 and smart schemes 24, 4, 14, 11,
        21, 7, 27, 26);
      random_schemes preset: per family one scheme from each of 8 random buckets of scheme indices (index mod 10),
        seed 42 + 1000 knn_k + 10 train_knn_k, Dirichlet(0.7) blend weights.
  - Weighted means of the kept original-KNN priors (random, smart, average of both).
Per outer fold:
  - train-data KNN with the 16 kept schemes (k = train_knn_k; reference = the outer-training rows, numerics
    standardized and category churn rates computed on them): directly for the held-out and test rows; for the
    outer-training rows out-of-fold over 5 inner folds (StratifiedKFold seed 42 + fold number 1-5);
  - logistic regressions (lbfgs, C 0.2, standardized inputs) on the logits of the original-KNN priors and of the
    train-KNN priors (random / smart / both, averaged): fitted on the outer-training rows (train-KNN: their inner
    out-of-fold values), they predict the held-out and test rows and, in sample, the outer-training rows;
  - 10 stack columns: the two LR averages and their mean, the weighted original-KNN and train-KNN means and their
    blends;
  - XGBoost (GPU) on the 29 columns: native categoricals (max_cat_to_onehot 8), lr 0.018, depth 4, min_child_weight 8,
    subsample 0.82, colsample_bytree 0.32, alpha 1, lambda 8, up to 6,000 rounds with early stopping 250 (AUC) on the
    fold's validation rows, seed 42.
Test = mean of the 5 fold models. The original rows are never training rows.

Presets:
  xgb_nested_knn_k1_k80_top_schemes      knn_k 1, train_knn_k 80, top schemes (run select_knn_schemes.py first)
  xgb_nested_knn_k10_k40_random_schemes  knn_k 10, train_knn_k 40, random schemes
Usage: python train_knn_nested_xgb.py <preset> [--folds 0,1,2,3,4]
(the original-data KNN features are rebuilt in every run; finished folds are kept in parts/ and skipped)"""
import argparse
import gc
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import TARGET, cache_dir, folds, model_dir, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib.knn_schemes import (CATEGORICAL_COLS, N_RANDOM_SCHEMES, N_SMART_SCHEMES, NUMERIC_COLS, SEED,  # noqa: E402
                             base_matrix, build_random_weight_schemes, build_smart_weight_schemes,
                             build_target_encoding_maps, cat_codes, fit_orig_importance_model, knn_random_chunked,
                             knn_random_full, knn_smart_chunked, knn_smart_full, load_frames, log,
                             make_target_rate_matrix, prefixed_colnames, seed_everything, standardize_from_reference,
                             standardize_ref_and_queries)

warnings.filterwarnings('ignore')

FOLDER = 'classic'

PRESETS = {
    'xgb_nested_knn_k1_k80_top_schemes': dict(knn_k=1, train_knn_k=80, selection='top'),
    'xgb_nested_knn_k10_k40_random_schemes': dict(knn_k=10, train_knn_k=40, selection='random'),
}

N_SPLITS = 5
INNER_SPLITS = 5
EPS = 1e-6
N_SELECT_RANDOM = 8
N_SELECT_SMART = 8

GPU_BATCH_SIZE_ORIG = 4096
BATCH_SIZE_OUTER = 192
BATCH_SIZE_INNER = 256
REF_CHUNK_SIZE = 24576

LR_PARAMS = dict(
    C=0.20,
    solver='lbfgs',
    max_iter=1000,
    random_state=SEED,
)

XGB_PARAMS = dict(
    n_estimators=6000,
    learning_rate=0.018,
    max_depth=4,
    min_child_weight=8,
    subsample=0.82,
    colsample_bytree=0.32,
    reg_alpha=1.0,
    reg_lambda=8.0,
    gamma=0.0,
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    device='cuda',
    enable_categorical=True,
    max_cat_to_onehot=8,
    random_state=SEED,
    early_stopping_rounds=250,
    verbosity=0,
)

STACK_COLS = ['meta_blend_lr', 'meta_tr_blend_lr', 'knn_rand_wmean', 'knn_smart_wmean', 'knn_blend_mean',
              'trknn_rand_wmean', 'trknn_smart_wmean', 'trknn_blend_mean', 'knn_full_blend_mean', 'meta_full_blend_lr']


def safe_logit(x, eps=1e-6):
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def weighted_avg_block(x, weights):
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    w = w / np.sum(w)
    return (x * w[None, :]).sum(axis=1).astype(np.float32)


def fit_lr(X_tr, y_tr, *queries):
    """Standardize + logistic regression fitted on (X_tr, y_tr); P(churn) of every query matrix."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_tr_s, y_tr)
    return [lr.predict_proba(scaler.transform(q))[:, 1].astype(np.float32) for q in queries]


def colname_to_idx(cols):
    return [int(c.split('_')[-1]) for c in cols]


def dirichlet_weights(rng, n, alpha=1.0):
    w = rng.dirichlet(alpha=np.full(n, alpha, dtype=np.float32)).astype(np.float32)
    return w / w.sum()


def choose_diverse_indices(rng, n_total, n_select, n_buckets=10):
    """One index from each of n_select randomly ordered buckets (index mod n_buckets), topped up at random."""
    if n_select > n_total:
        raise ValueError(f'n_select={n_select} > n_total={n_total}')
    buckets = [[] for _ in range(n_buckets)]
    for i in range(n_total):
        buckets[i % n_buckets].append(i)
    chosen = []
    for b in rng.permutation(n_buckets).tolist():
        if len(chosen) >= n_select:
            break
        if buckets[b]:
            chosen.append(rng.choice(buckets[b]))
    if len(chosen) < n_select:
        remaining = np.setdiff1d(np.arange(n_total), np.array(chosen, dtype=int), assume_unique=False)
        chosen.extend(rng.choice(remaining, size=(n_select - len(chosen)), replace=False).tolist())
    chosen = sorted(set(chosen))
    while len(chosen) < n_select:
        remaining = np.setdiff1d(np.arange(n_total), np.array(chosen, dtype=int), assume_unique=False)
        chosen.append(int(rng.choice(remaining)))
        chosen = sorted(set(chosen))
    return chosen


def select_schemes(cfg):
    """(random prior columns, smart prior columns, random blend weights, smart blend weights)."""
    if cfg['selection'] == 'top':
        d = cache_dir(FOLDER)
        if not (d / 'knn_scheme_selection.json').exists():
            raise SystemExit('knn_scheme_selection.json not found in cache/: run select_knn_schemes.py first')
        with open(d / 'knn_scheme_selection.json') as f:
            selected_info = json.load(f)
        selected_random_cols = selected_info['selected_random_cols']
        selected_smart_cols = selected_info['selected_smart_cols']
        imp_random = pd.read_csv(d / 'knn_scheme_importance_random.csv').set_index('feature')
        imp_smart = pd.read_csv(d / 'knn_scheme_importance_smart.csv').set_index('feature')
        rand_w = imp_random.loc[selected_random_cols, 'mean_importance'].values.astype(np.float32)
        smart_w = imp_smart.loc[selected_smart_cols, 'mean_importance'].values.astype(np.float32)
        return selected_random_cols, selected_smart_cols, rand_w / rand_w.sum(), smart_w / smart_w.sum()

    run_seed = int(SEED + 1000 * int(cfg['knn_k']) + 10 * int(cfg['train_knn_k']))
    rng = np.random.default_rng(run_seed)
    selected_random_idx = choose_diverse_indices(rng, N_RANDOM_SCHEMES, N_SELECT_RANDOM, n_buckets=10)
    selected_smart_idx = choose_diverse_indices(rng, N_SMART_SCHEMES, N_SELECT_SMART, n_buckets=10)
    rand_w = dirichlet_weights(rng, N_SELECT_RANDOM, alpha=0.7)
    smart_w = dirichlet_weights(rng, N_SELECT_SMART, alpha=0.7)
    return (prefixed_colnames('knnprior', selected_random_idx), prefixed_colnames('knnsmart', selected_smart_idx),
            rand_w, smart_w)


# ------------------------------------------------------------------ global (original-data) part
def build_global(cfg):
    knn_k = cfg['knn_k']
    train, test, orig = load_frames()
    g = dict(train=train, test=test, y=train[TARGET].values.astype(np.int8))
    orig_cat_codes = cat_codes(orig)
    g['train_cat_codes'], g['test_cat_codes'] = cat_codes(train), cat_codes(test)
    orig_num = orig[NUMERIC_COLS].values.astype(np.float32)
    g['train_num'] = train[NUMERIC_COLS].values.astype(np.float32)
    g['test_num'] = test[NUMERIC_COLS].values.astype(np.float32)
    orig_num_scaled, num_mu, num_sd = standardize_from_reference(orig_num, orig_num)
    train_num_scaled = ((g['train_num'] - num_mu) / num_sd).astype(np.float32)
    test_num_scaled = ((g['test_num'] - num_mu) / num_sd).astype(np.float32)
    orig_y = orig[TARGET].values.astype(np.float32)
    train_target_mean = float(train[TARGET].mean())
    orig_target_mean = float(orig[TARGET].mean())

    log('original-data KNN, random schemes')
    all_random_schemes = build_random_weight_schemes(N_RANDOM_SCHEMES, n_cat=len(CATEGORICAL_COLS),
                                                     n_num=len(NUMERIC_COLS), seed=SEED)
    all_random_names = prefixed_colnames('knnprior', range(N_RANDOM_SCHEMES))
    knn_random_train = knn_random_full(orig_cat_codes, orig_num_scaled, orig_y, g['train_cat_codes'], train_num_scaled,
                                       all_random_schemes, all_random_names, knn_k, batch_size=GPU_BATCH_SIZE_ORIG)
    knn_random_test = knn_random_full(orig_cat_codes, orig_num_scaled, orig_y, g['test_cat_codes'], test_num_scaled,
                                      all_random_schemes, all_random_names, knn_k, batch_size=GPU_BATCH_SIZE_ORIG)

    log('original-data KNN, smart schemes')
    te_maps_orig, te_global_mean_orig = build_target_encoding_maps(orig)
    orig_cat_te = make_target_rate_matrix(orig, te_maps_orig, te_global_mean_orig)
    train_cat_te = make_target_rate_matrix(train, te_maps_orig, te_global_mean_orig)
    test_cat_te = make_target_rate_matrix(test, te_maps_orig, te_global_mean_orig)
    orig_importance = fit_orig_importance_model(orig)
    all_smart_schemes = build_smart_weight_schemes(orig_importance[CATEGORICAL_COLS].values.astype(np.float32),
                                                   orig_importance[NUMERIC_COLS].values.astype(np.float32),
                                                   N_SMART_SCHEMES, seed=SEED)
    all_smart_names = prefixed_colnames('knnsmart', range(N_SMART_SCHEMES))
    knn_smart_train = knn_smart_full(orig_cat_te, orig_num_scaled, orig_y, train_cat_te, train_num_scaled,
                                     train_target_mean, orig_target_mean, all_smart_schemes, all_smart_names, knn_k,
                                     batch_size=GPU_BATCH_SIZE_ORIG)
    knn_smart_test = knn_smart_full(orig_cat_te, orig_num_scaled, orig_y, test_cat_te, test_num_scaled,
                                    train_target_mean, orig_target_mean, all_smart_schemes, all_smart_names, knn_k,
                                    batch_size=GPU_BATCH_SIZE_ORIG)

    selected_random_cols, selected_smart_cols, g['rand_w'], g['smart_w'] = select_schemes(cfg)
    selected_random_idx = colname_to_idx(selected_random_cols)
    selected_smart_idx = colname_to_idx(selected_smart_cols)
    g['selected_random_schemes'] = [all_random_schemes[i] for i in selected_random_idx]
    g['selected_smart_schemes'] = [all_smart_schemes[i] for i in selected_smart_idx]
    g['train_random_names'] = prefixed_colnames('trknnprior', selected_random_idx)
    g['train_smart_names'] = prefixed_colnames('trknnsmart', selected_smart_idx)
    log(f'kept random schemes {selected_random_idx} (weights {np.round(g["rand_w"], 4).tolist()}), '
        f'smart schemes {selected_smart_idx} (weights {np.round(g["smart_w"], 4).tolist()})')

    R_tr = knn_random_train[selected_random_cols].values
    R_te = knn_random_test[selected_random_cols].values
    S_tr = knn_smart_train[selected_smart_cols].values
    S_te = knn_smart_test[selected_smart_cols].values
    g['Rlog_tr'] = safe_logit(R_tr, EPS).astype(np.float32)
    g['Rlog_te'] = safe_logit(R_te, EPS).astype(np.float32)
    g['Slog_tr'] = safe_logit(S_tr, EPS).astype(np.float32)
    g['Slog_te'] = safe_logit(S_te, EPS).astype(np.float32)
    g['Alog_tr'] = np.concatenate([g['Rlog_tr'], g['Slog_tr']], axis=1)
    g['Alog_te'] = np.concatenate([g['Rlog_te'], g['Slog_te']], axis=1)

    g['knn_rand_wmean_tr'] = weighted_avg_block(R_tr, g['rand_w'])
    g['knn_rand_wmean_te'] = weighted_avg_block(R_te, g['rand_w'])
    g['knn_smart_wmean_tr'] = weighted_avg_block(S_tr, g['smart_w'])
    g['knn_smart_wmean_te'] = weighted_avg_block(S_te, g['smart_w'])
    g['knn_blend_mean_tr'] = (0.5 * g['knn_rand_wmean_tr'] + 0.5 * g['knn_smart_wmean_tr']).astype(np.float32)
    g['knn_blend_mean_te'] = (0.5 * g['knn_rand_wmean_te'] + 0.5 * g['knn_smart_wmean_te']).astype(np.float32)

    g['X_train_base'], g['X_test_base'] = base_matrix(train), base_matrix(test)
    return g


# ------------------------------------------------------------------ one outer fold
def run_fold(g, cfg, k, tr_idx, va_idx):
    fold = k + 1
    train_knn_k = cfg['train_knn_k']
    train, test, y = g['train'], g['test'], g['y']
    y_tr_outer, y_va_outer = y[tr_idx], y[va_idx]
    y_tr_outer_f = y_tr_outer.astype(np.float32)
    sel_r, sel_s = g['selected_random_schemes'], g['selected_smart_schemes']
    names_r, names_s = g['train_random_names'], g['train_smart_names']

    # LR on the original-KNN logits: held-out, test and (in sample) outer-training rows
    p_va_rand, p_te_rand, p_tr_rand = fit_lr(g['Rlog_tr'][tr_idx], y_tr_outer, g['Rlog_tr'][va_idx], g['Rlog_te'],
                                             g['Rlog_tr'][tr_idx])
    p_va_smart, p_te_smart, p_tr_smart = fit_lr(g['Slog_tr'][tr_idx], y_tr_outer, g['Slog_tr'][va_idx], g['Slog_te'],
                                                g['Slog_tr'][tr_idx])
    p_va_all, p_te_all, p_tr_all = fit_lr(g['Alog_tr'][tr_idx], y_tr_outer, g['Alog_tr'][va_idx], g['Alog_te'],
                                          g['Alog_tr'][tr_idx])

    # train-data KNN of the held-out and test rows (reference = outer-training rows)
    outer_train_df = train.iloc[tr_idx].copy().reset_index(drop=True)
    outer_valid_df = train.iloc[va_idx].copy().reset_index(drop=True)
    outer_train_cat, outer_valid_cat = g['train_cat_codes'][tr_idx], g['train_cat_codes'][va_idx]
    outer_train_num, outer_valid_num = g['train_num'][tr_idx], g['train_num'][va_idx]
    outer_train_num_s, outer_valid_num_s, outer_test_num_s = standardize_ref_and_queries(
        outer_train_num, outer_valid_num, g['test_num'])

    trR_va = knn_random_chunked(outer_train_cat, outer_train_num_s, y_tr_outer_f, outer_valid_cat, outer_valid_num_s,
                                sel_r, names_r, train_knn_k, batch_size=BATCH_SIZE_OUTER, ref_chunk_size=REF_CHUNK_SIZE,
                                prefix=f'train-KNN random, valid, fold {k}')
    trR_te = knn_random_chunked(outer_train_cat, outer_train_num_s, y_tr_outer_f, g['test_cat_codes'],
                                outer_test_num_s, sel_r, names_r, train_knn_k, batch_size=BATCH_SIZE_OUTER,
                                ref_chunk_size=REF_CHUNK_SIZE, prefix=f'train-KNN random, test, fold {k}')
    te_maps_outer, te_global_outer = build_target_encoding_maps(outer_train_df)
    outer_train_cat_te = make_target_rate_matrix(outer_train_df, te_maps_outer, te_global_outer)
    outer_valid_cat_te = make_target_rate_matrix(outer_valid_df, te_maps_outer, te_global_outer)
    outer_test_cat_te = make_target_rate_matrix(test, te_maps_outer, te_global_outer)
    tr_mean = float(y_tr_outer.mean())
    trS_va = knn_smart_chunked(outer_train_cat_te, outer_train_num_s, y_tr_outer_f, outer_valid_cat_te,
                               outer_valid_num_s, tr_mean, tr_mean, sel_s, names_s, train_knn_k,
                               batch_size=BATCH_SIZE_OUTER, ref_chunk_size=REF_CHUNK_SIZE,
                               prefix=f'train-KNN smart, valid, fold {k}')
    trS_te = knn_smart_chunked(outer_train_cat_te, outer_train_num_s, y_tr_outer_f, outer_test_cat_te,
                               outer_test_num_s, tr_mean, tr_mean, sel_s, names_s, train_knn_k,
                               batch_size=BATCH_SIZE_OUTER, ref_chunk_size=REF_CHUNK_SIZE,
                               prefix=f'train-KNN smart, test, fold {k}')

    # train-data KNN of the outer-training rows: out-of-fold over 5 inner folds
    inner_R_oof = np.zeros((len(tr_idx), len(names_r)), dtype=np.float32)
    inner_S_oof = np.zeros((len(tr_idx), len(names_s)), dtype=np.float32)
    inner_skf = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + fold)
    for inner_fold, (itr_rel, iva_rel) in enumerate(inner_skf.split(np.zeros(len(tr_idx)), y_tr_outer), 1):
        log(f'  inner fold {inner_fold}/{INNER_SPLITS} of outer fold {k}')
        inner_ref_df = outer_train_df.iloc[itr_rel].copy().reset_index(drop=True)
        inner_val_df = outer_train_df.iloc[iva_rel].copy().reset_index(drop=True)
        ref_y = y_tr_outer_f[itr_rel]
        ref_num_s, val_num_s = standardize_ref_and_queries(outer_train_num[itr_rel], outer_train_num[iva_rel])
        inner_R = knn_random_chunked(outer_train_cat[itr_rel], ref_num_s, ref_y, outer_train_cat[iva_rel], val_num_s,
                                     sel_r, names_r, train_knn_k, batch_size=BATCH_SIZE_INNER,
                                     ref_chunk_size=REF_CHUNK_SIZE, prefix='  train-KNN random, inner')
        te_maps_inner, te_global_inner = build_target_encoding_maps(inner_ref_df)
        ref_cat_te = make_target_rate_matrix(inner_ref_df, te_maps_inner, te_global_inner)
        val_cat_te = make_target_rate_matrix(inner_val_df, te_maps_inner, te_global_inner)
        inner_S = knn_smart_chunked(ref_cat_te, ref_num_s, ref_y, val_cat_te, val_num_s, float(ref_y.mean()),
                                    float(ref_y.mean()), sel_s, names_s, train_knn_k, batch_size=BATCH_SIZE_INNER,
                                    ref_chunk_size=REF_CHUNK_SIZE, prefix='  train-KNN smart, inner')
        inner_R_oof[iva_rel] = inner_R[names_r].values.astype(np.float32)
        inner_S_oof[iva_rel] = inner_S[names_s].values.astype(np.float32)
        del inner_R, inner_S, inner_ref_df, inner_val_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # LR on the train-KNN logits (fitted on the inner out-of-fold values of the outer-training rows)
    trR_va_log = safe_logit(trR_va[names_r].values, EPS).astype(np.float32)
    trR_te_log = safe_logit(trR_te[names_r].values, EPS).astype(np.float32)
    inner_R_log = safe_logit(inner_R_oof, EPS).astype(np.float32)
    trS_va_log = safe_logit(trS_va[names_s].values, EPS).astype(np.float32)
    trS_te_log = safe_logit(trS_te[names_s].values, EPS).astype(np.float32)
    inner_S_log = safe_logit(inner_S_oof, EPS).astype(np.float32)
    inner_A_log = np.concatenate([inner_R_log, inner_S_log], axis=1).astype(np.float32)
    trA_va_log = np.concatenate([trR_va_log, trS_va_log], axis=1).astype(np.float32)
    trA_te_log = np.concatenate([trR_te_log, trS_te_log], axis=1).astype(np.float32)
    p_va_tr_rand, p_te_tr_rand, p_tr_tr_rand = fit_lr(inner_R_log, y_tr_outer, trR_va_log, trR_te_log, inner_R_log)
    p_va_tr_smart, p_te_tr_smart, p_tr_tr_smart = fit_lr(inner_S_log, y_tr_outer, trS_va_log, trS_te_log, inner_S_log)
    p_va_tr_all, p_te_tr_all, p_tr_tr_all = fit_lr(inner_A_log, y_tr_outer, trA_va_log, trA_te_log, inner_A_log)

    # the 10 stack columns of the outer-training, held-out and test rows
    rand_w, smart_w = g['rand_w'], g['smart_w']
    meta_blend_tr = ((p_tr_rand + p_tr_smart + p_tr_all) / 3.0).astype(np.float32)
    meta_tr_blend_tr = ((p_tr_tr_rand + p_tr_tr_smart + p_tr_tr_all) / 3.0).astype(np.float32)
    trknn_rand_tr = weighted_avg_block(inner_R_oof, rand_w)
    trknn_smart_tr = weighted_avg_block(inner_S_oof, smart_w)
    trknn_blend_tr = (0.5 * trknn_rand_tr + 0.5 * trknn_smart_tr).astype(np.float32)
    stack_tr = {
        'meta_blend_lr': meta_blend_tr,
        'meta_tr_blend_lr': meta_tr_blend_tr,
        'knn_rand_wmean': g['knn_rand_wmean_tr'][tr_idx],
        'knn_smart_wmean': g['knn_smart_wmean_tr'][tr_idx],
        'knn_blend_mean': g['knn_blend_mean_tr'][tr_idx],
        'trknn_rand_wmean': trknn_rand_tr,
        'trknn_smart_wmean': trknn_smart_tr,
        'trknn_blend_mean': trknn_blend_tr,
        'knn_full_blend_mean': 0.5 * g['knn_blend_mean_tr'][tr_idx] + 0.5 * trknn_blend_tr,
        'meta_full_blend_lr': 0.5 * meta_blend_tr + 0.5 * meta_tr_blend_tr,
    }
    trknn_rand_va = weighted_avg_block(trR_va[names_r].values, rand_w)
    trknn_smart_va = weighted_avg_block(trS_va[names_s].values, smart_w)
    trknn_blend_va = (0.5 * trknn_rand_va + 0.5 * trknn_smart_va).astype(np.float32)
    stack_va = {
        'meta_blend_lr': ((p_va_rand + p_va_smart + p_va_all) / 3.0).astype(np.float32),
        'meta_tr_blend_lr': ((p_va_tr_rand + p_va_tr_smart + p_va_tr_all) / 3.0).astype(np.float32),
        'knn_rand_wmean': g['knn_rand_wmean_tr'][va_idx],
        'knn_smart_wmean': g['knn_smart_wmean_tr'][va_idx],
        'knn_blend_mean': g['knn_blend_mean_tr'][va_idx],
        'trknn_rand_wmean': trknn_rand_va,
        'trknn_smart_wmean': trknn_smart_va,
        'trknn_blend_mean': trknn_blend_va,
        'knn_full_blend_mean': 0.5 * g['knn_blend_mean_tr'][va_idx] + 0.5 * trknn_blend_va,
        'meta_full_blend_lr': np.mean(np.stack([p_va_rand, p_va_smart, p_va_all,
                                                p_va_tr_rand, p_va_tr_smart, p_va_tr_all], axis=1), axis=1),
    }
    meta_blend_te = ((p_te_rand + p_te_smart + p_te_all) / 3.0).astype(np.float32)
    meta_tr_blend_te = ((p_te_tr_rand + p_te_tr_smart + p_te_tr_all) / 3.0).astype(np.float32)
    trknn_rand_te = weighted_avg_block(trR_te[names_r].values, rand_w)
    trknn_smart_te = weighted_avg_block(trS_te[names_s].values, smart_w)
    trknn_blend_te = (0.5 * trknn_rand_te + 0.5 * trknn_smart_te).astype(np.float32)
    stack_te = {
        'meta_blend_lr': meta_blend_te,
        'meta_tr_blend_lr': meta_tr_blend_te,
        'knn_rand_wmean': g['knn_rand_wmean_te'],
        'knn_smart_wmean': g['knn_smart_wmean_te'],
        'knn_blend_mean': g['knn_blend_mean_te'],
        'trknn_rand_wmean': trknn_rand_te,
        'trknn_smart_wmean': trknn_smart_te,
        'trknn_blend_mean': trknn_blend_te,
        'knn_full_blend_mean': 0.5 * g['knn_blend_mean_te'] + 0.5 * trknn_blend_te,
        'meta_full_blend_lr': 0.5 * meta_blend_te + 0.5 * meta_tr_blend_te,
    }

    # final XGBoost on the 19 base + 10 stack columns
    X_tr_fold = g['X_train_base'].iloc[tr_idx].reset_index(drop=True).copy()
    X_va_fold = g['X_train_base'].iloc[va_idx].reset_index(drop=True).copy()
    X_te_fold = g['X_test_base'].copy()
    for c in STACK_COLS:
        X_tr_fold[c] = np.asarray(stack_tr[c], dtype=np.float32)
        X_va_fold[c] = np.asarray(stack_va[c], dtype=np.float32)
        X_te_fold[c] = np.asarray(stack_te[c], dtype=np.float32)

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr_fold, y_tr_outer, eval_set=[(X_va_fold, y_va_outer)], verbose=200)
    p_va = model.predict_proba(X_va_fold)[:, 1].astype(np.float32)
    p_te = model.predict_proba(X_te_fold)[:, 1].astype(np.float32)
    log(f'fold {k}: AUC {roc_auc_score(y_va_outer, p_va):.6f}')
    return p_va, p_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preset', choices=sorted(PRESETS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    pair, cfg = args.preset, PRESETS[args.preset]
    out = model_dir(FOLDER)
    seed_everything(SEED)

    y = target()
    fold_id = folds(y)
    todo = [k for k in fold_list(args.folds) if not have_part(out, pair, k)]
    if todo:
        g = build_global(cfg)
        for k in todo:
            tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
            log(f'{pair}: outer fold {k}')
            p_va, p_te = run_fold(g, cfg, k, tr_idx, va_idx)
            save_part(out, pair, k, p_va, p_te)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    assemble(out, pair, fold_id, y)


if __name__ == '__main__':
    main()
