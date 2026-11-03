#!/usr/bin/env python
"""XGBoost on original-data KNN features and logistic-regression meta features (two presets: k = 10 and k = 1).

Pairs (stack A): xgb_original_knn_k10_lr_meta_a (10 neighbours), xgb_original_knn_k1_lr_meta_a (1 neighbour).
Columns: the 8 categoricals as integer codes (factorized jointly over train + test + original) and the 11 numerics
(missing values -> original medians); for the distances the numerics are standardized on the original rows.
Original-data KNN (the 10,000 original rows are the only neighbours; brute force on the GPU, numerics in fp16):
  - 30 "random" weight schemes (8 generators: uniform ranges, categorical- or numeric-heavy, Dirichlet, jittered,
    two dominant columns, ...); distance = weighted categorical mismatches + weighted squared numeric differences;
    features: the class shares among the k nearest original rows and their mean distance;
  - 30 "smart" schemes from the feature importances of an XGBoost fitted on the original rows (600 trees, lr 0.05,
    depth 4), varied by 8 templates (sqrt, square, categorical- / numeric-heavy, flat mix, top-2 / top-4 boosts) with
    15% jitter; each categorical is replaced by its 3 original class rates (weighted absolute differences);
    features: inverse-distance weighted class shares, rescaled by the train / original class-prior ratio, and the
    mean distance.
8 random and 8 smart schemes are drawn (one per bucket of scheme indices, seed 7147) with Dirichlet(0.7) weights:
weighted means of their class shares and distances (random, smart and the average of both: 12 columns).
LR meta features (per outer fold, fitted on the fold's training rows, predicting its held-out rows; test = mean over
the 5 folds): multinomial logistic regression (lbfgs, C 0.2, standardized inputs) on the logits of the selected random
schemes, of the smart schemes, of both, and the mean of the three (12 columns).
Final model per fold: XGBoost (GPU) on the 43 columns, multi:softprob, lr 0.018, depth 4, min_child_weight 8,
subsample 0.82, colsample_bytree 0.32, alpha 1, lambda 8, up to 6,000 rounds with early stopping 250 (mlogloss) on the
fold's validation rows, seed 42. Test = mean of the 5 fold models. The original rows are never training rows.
Usage: python train_knn_lr_xgb.py <preset> [--folds 0,1,2,3,4]
(the KNN and LR meta features are rebuilt in every run; finished XGBoost folds are kept in parts/ and skipped)"""
import argparse
import gc
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, NUMERICS, TARGET, TARGET_MAP, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

FOLDER = 'classic'
PRESETS = {  # pair -> number of neighbours
    'xgb_original_knn_k10_lr_meta_a': 10,
    'xgb_original_knn_k1_lr_meta_a': 1,
}

SEED = 42
N_SPLITS = 5
DIST_EPS = 1e-6
N_RANDOM_SCHEMES = 30
N_SMART_SCHEMES = 30
N_SELECT_RANDOM = 8
N_SELECT_SMART = 8
SELECTION_SEED = 7147

USE_FP16_NUM = True
PIN_MEMORY = True
APPLY_PRIOR_CALIBRATION = True
GPU_BATCH_SIZE_ORIG = 4096
NUM_CLASSES = 3

CATEGORICAL_COLS = list(CATEGORICALS)
NUMERIC_COLS = list(NUMERICS)
ALL_FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

LR_PARAMS = dict(
    C=0.20,
    solver='lbfgs',
    max_iter=1000,
    random_state=SEED,
    multi_class='multinomial',
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
    objective='multi:softprob',
    num_class=NUM_CLASSES,
    eval_metric='mlogloss',
    tree_method='hist',
    device='cuda',
    random_state=SEED,
    early_stopping_rounds=250,
    verbosity=0,
)
ORIG_XGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=2,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
    gamma=0.0,
    objective='multi:softprob',
    num_class=NUM_CLASSES,
    eval_metric='mlogloss',
    tree_method='hist',
    device='cuda',
    random_state=SEED,
    verbosity=0,
)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_logit(x, eps=1e-6):
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def standardize_ref_and_queries(x_ref, *queries):
    mu = np.nanmean(x_ref, axis=0)
    sd = np.nanstd(x_ref, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    out = [((x_ref - mu) / sd).astype(np.float32)]
    for q in queries:
        out.append(((q - mu) / sd).astype(np.float32))
    return out


def build_lr_feature(x_tr, y_tr, x_va, x_te):
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_va_s = scaler.transform(x_va)
    x_te_s = scaler.transform(x_te)
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(x_tr_s, y_tr)
    return lr.predict_proba(x_va_s).astype(np.float32), lr.predict_proba(x_te_s).astype(np.float32)


def prefixed_colnames(prefix, indices):
    return [f'{prefix}_{i:03d}' for i in indices]


def scheme_output_cols(base_names):
    return [f'{base}_{s}' for base in base_names for s in ('c0', 'c1', 'c2', 'dist')]


def scheme_prob_cols(base_names):
    return [f'{base}_{s}' for base in base_names for s in ('c0', 'c1', 'c2')]


def expand_cat_weights(cat_w):
    return np.repeat(np.asarray(cat_w, dtype=np.float32), NUM_CLASSES)


def one_hot_mean(nn_y):
    return F.one_hot(nn_y, num_classes=NUM_CLASSES).float().mean(dim=1)


def weighted_one_hot(nn_y, weights):
    hot = F.one_hot(nn_y, num_classes=NUM_CLASSES).float()
    numer = (hot * weights.unsqueeze(-1)).sum(dim=1)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return numer / denom


def build_random_weight_schemes(n_schemes, n_cat, n_num, seed=42):
    rng = np.random.default_rng(seed)
    schemes = []
    for k in range(n_schemes):
        mode = k % 8
        if mode == 0:
            cat_w = rng.uniform(0.05, 1.00, n_cat)
            num_w = rng.uniform(0.05, 1.00, n_num)
        elif mode == 1:
            cat_w = rng.uniform(0.30, 1.20, n_cat)
            num_w = rng.uniform(0.01, 0.30, n_num)
        elif mode == 2:
            cat_w = rng.uniform(0.01, 0.30, n_cat)
            num_w = rng.uniform(0.30, 1.20, n_num)
        elif mode == 3:
            z = rng.dirichlet(np.ones(n_cat + n_num))
            z = z * (n_cat + n_num)
            cat_w = z[:n_cat]
            num_w = z[n_cat:]
        elif mode == 4:
            cat_w = np.clip(1.0 + rng.normal(0, 0.25, n_cat), 0.05, 2.0)
            num_w = np.clip(1.0 + rng.normal(0, 0.25, n_num), 0.05, 2.0)
        elif mode == 5:
            cat_w = np.full(n_cat, 0.05, dtype=np.float32)
            num_w = np.full(n_num, 0.05, dtype=np.float32)
            js = rng.choice(n_cat + n_num, size=2, replace=False)
            for j in js:
                if j < n_cat:
                    cat_w[j] = rng.uniform(1.0, 2.5)
                else:
                    num_w[j - n_cat] = rng.uniform(1.0, 2.5)
        elif mode == 6:
            cat_w = rng.uniform(0.05, 0.80, n_cat)
            num_w = rng.uniform(0.20, 1.20, n_num)
        else:
            cat_w = rng.uniform(0.10, 1.50, n_cat)
            num_w = rng.uniform(0.01, 0.20, n_num)
        z = np.concatenate([cat_w, num_w]).astype(np.float32)
        z = z / np.mean(z)
        schemes.append((z[:n_cat].astype(np.float32), z[n_cat:].astype(np.float32)))
    return schemes


def build_smart_weight_schemes(base_cat_w, base_num_w, n_schemes, seed=42):
    rng = np.random.default_rng(seed)
    base_cat_w = np.asarray(base_cat_w, dtype=np.float32)
    base_num_w = np.asarray(base_num_w, dtype=np.float32)
    z = np.concatenate([base_cat_w, base_num_w])
    z = z / np.mean(z)
    base_cat_w = z[:len(base_cat_w)].astype(np.float32)
    base_num_w = z[len(base_cat_w):].astype(np.float32)
    cat_idx_sorted = np.argsort(-base_cat_w)
    num_idx_sorted = np.argsort(-base_num_w)
    modes = ['base', 'sqrt', 'square', 'cat_heavy', 'num_heavy', 'flat_mix', 'top2_boost', 'top4_boost']
    schemes = []
    while len(schemes) < n_schemes:
        mode = modes[len(schemes) % len(modes)]
        cat_w = base_cat_w.copy()
        num_w = base_num_w.copy()
        if mode == 'sqrt':
            cat_w = np.sqrt(cat_w)
            num_w = np.sqrt(num_w)
        elif mode == 'square':
            cat_w = cat_w ** 2
            num_w = num_w ** 2
        elif mode == 'cat_heavy':
            cat_w *= 1.50
            num_w *= 0.60
        elif mode == 'num_heavy':
            cat_w *= 0.60
            num_w *= 1.60
        elif mode == 'flat_mix':
            cat_w = 0.50 * cat_w + 0.50
            num_w = 0.50 * num_w + 0.50
        elif mode == 'top2_boost':
            cat_w[cat_idx_sorted[:2]] *= 2.20
            num_w[num_idx_sorted[:2]] *= 1.40
        elif mode == 'top4_boost':
            cat_w[cat_idx_sorted[:4]] *= 1.80
            num_w[num_idx_sorted[:4]] *= 1.30
        cat_w *= np.clip(1.0 + rng.normal(0, 0.15, len(cat_w)), 0.60, 1.50)
        num_w *= np.clip(1.0 + rng.normal(0, 0.15, len(num_w)), 0.60, 1.50)
        cat_w = np.clip(cat_w, 1e-3, None)
        num_w = np.clip(num_w, 1e-3, None)
        z = np.concatenate([cat_w, num_w])
        z = z / np.mean(z)
        schemes.append((z[:len(base_cat_w)].astype(np.float32), z[len(base_cat_w):].astype(np.float32), mode))
    return schemes


def fit_orig_importance_model(orig_df):
    xo = orig_df[ALL_FEATURE_COLS].copy()
    yo = orig_df[TARGET].values.astype(np.int8)
    for c in CATEGORICAL_COLS:
        xo[c] = pd.to_numeric(xo[c], errors='coerce').fillna(-1).astype(np.int32)
    for c in NUMERIC_COLS:
        xo[c] = pd.to_numeric(xo[c], errors='coerce').fillna(0).astype(np.float32)
    model = XGBClassifier(**ORIG_XGB_PARAMS)
    model.fit(xo, yo, verbose=False)
    imp = pd.Series(model.feature_importances_, index=ALL_FEATURE_COLS).fillna(0.0)
    if imp.sum() <= 0:
        imp[:] = 1.0
    return imp / imp.sum()


def build_target_encoding_maps(df_ref):
    maps = {}
    global_mean = np.bincount(df_ref[TARGET].values, minlength=NUM_CLASSES).astype(np.float32)
    global_mean /= global_mean.sum()
    for c in CATEGORICAL_COLS:
        maps[c] = pd.crosstab(df_ref[c], df_ref[TARGET], normalize='index').reindex(columns=range(NUM_CLASSES), fill_value=0.0)
    return maps, global_mean.astype(np.float32)


def make_target_rate_matrix(df, te_maps, global_mean):
    mats = []
    idx = pd.Index(df.index)
    for c in CATEGORICAL_COLS:
        vals = te_maps[c].reindex(df[c]).copy()
        vals.index = idx
        vals = vals.fillna(pd.Series(global_mean, index=range(NUM_CLASSES)))
        mats.append(vals.values.astype(np.float32))
    return np.concatenate(mats, axis=1).astype(np.float32)


def weighted_family_features(block, base_names, weights, prefix, index):
    out = pd.DataFrame(index=index)
    w = np.asarray(weights, dtype=np.float32)
    w = w / max(w.sum(), 1e-12)
    for cls in range(NUM_CLASSES):
        cols = [f'{b}_c{cls}' for b in base_names]
        out[f'{prefix}_c{cls}'] = (block[cols].values.astype(np.float32) * w[None, :]).sum(axis=1).astype(np.float32)
    dist_cols = [f'{b}_dist' for b in base_names]
    out[f'{prefix}_dist'] = (block[dist_cols].values.astype(np.float32) * w[None, :]).sum(axis=1).astype(np.float32)
    return out


def blend_feature_frames(df_a, df_b, prefix):
    out = pd.DataFrame(index=df_a.index)
    suffix_to_col_b = {c.rsplit('_', 1)[-1]: c for c in df_b.columns}
    for c in df_a.columns:
        suffix = c.rsplit('_', 1)[-1]
        out[f'{prefix}_{suffix}'] = (0.5 * df_a[c].values + 0.5 * df_b[suffix_to_col_b[suffix]].values).astype(np.float32)
    return out


def assign_frame(dst, idx, frame):
    for c in frame.columns:
        dst.loc[idx, c] = frame[c].values.astype(np.float32)


def prob_df(pred, prefix, index):
    return pd.DataFrame({f'{prefix}_c{cls}': pred[:, cls].astype(np.float32) for cls in range(NUM_CLASSES)}, index=index)


def choose_diverse_indices(rng, n_total, n_select, n_buckets=10):
    buckets = [[] for _ in range(n_buckets)]
    for i in range(n_total):
        buckets[i % n_buckets].append(i)
    chosen = []
    for b in rng.permutation(n_buckets).tolist():
        if len(chosen) >= n_select:
            break
        if buckets[b]:
            chosen.append(int(rng.choice(buckets[b])))
    while len(chosen) < n_select:
        remain = np.setdiff1d(np.arange(n_total), np.array(chosen, dtype=int), assume_unique=False)
        chosen.append(int(rng.choice(remain)))
        chosen = sorted(set(chosen))
    return sorted(chosen)


@torch.no_grad()
def build_knn_random_gpu_full(ref_cat_codes, ref_num_scaled, ref_y, query_cat_codes, query_num_scaled, schemes,
                              base_names, n_neighbors, batch_size=4096):
    """Per scheme: class shares of the k nearest reference rows and their mean distance."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_dtype = torch.float16 if USE_FP16_NUM else torch.float32
    ref_cat_t = torch.as_tensor(ref_cat_codes, dtype=torch.int16, device=device)
    ref_num_t = torch.as_tensor(ref_num_scaled, dtype=num_dtype, device=device)
    ref_y_t = torch.as_tensor(ref_y, dtype=torch.int64, device=device)
    query_cat_cpu = torch.as_tensor(query_cat_codes, dtype=torch.int16)
    query_num_cpu = torch.as_tensor(query_num_scaled, dtype=num_dtype)
    if PIN_MEMORY and device.type == 'cuda':
        query_cat_cpu, query_num_cpu = query_cat_cpu.pin_memory(), query_num_cpu.pin_memory()
    n_query = len(query_cat_codes)
    out = np.empty((n_query, len(base_names) * 4), dtype=np.float32)
    for s_idx, (cat_w_np, num_w_np) in enumerate(schemes):
        cat_w_t = torch.as_tensor(cat_w_np, dtype=torch.float32, device=device)
        num_w_t = torch.as_tensor(num_w_np, dtype=num_dtype, device=device)
        probs_scheme = np.empty((n_query, 3), dtype=np.float32)
        dists_scheme = np.empty(n_query, dtype=np.float32)
        for lo in range(0, n_query, batch_size):
            hi = min(lo + batch_size, n_query)
            q_cat = query_cat_cpu[lo:hi].to(device, non_blocking=True)
            q_num = query_num_cpu[lo:hi].to(device, non_blocking=True)
            mismatch = q_cat[:, None, :] != ref_cat_t[None, :, :]
            cat_dist = (mismatch.float() * cat_w_t.view(1, 1, -1)).sum(dim=2)
            num_diff = q_num[:, None, :] - ref_num_t[None, :, :]
            num_dist = ((num_diff * num_diff) * num_w_t.view(1, 1, -1)).sum(dim=2).float()
            dist = cat_dist + num_dist
            top_dist, nn_idx = torch.topk(dist, k=n_neighbors, dim=1, largest=False, sorted=False)
            probs_scheme[lo:hi] = one_hot_mean(ref_y_t[nn_idx]).cpu().numpy().astype(np.float32)
            dists_scheme[lo:hi] = top_dist.mean(dim=1).cpu().numpy().astype(np.float32)
        out[:, s_idx * 4:(s_idx + 1) * 4] = np.column_stack([probs_scheme, dists_scheme]).astype(np.float32)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return pd.DataFrame(out, columns=scheme_output_cols(base_names))


@torch.no_grad()
def build_knn_smart_gpu_full(ref_cat_te, ref_num_scaled, ref_y, query_cat_te, query_num_scaled, train_prior, ref_prior,
                             schemes, base_names, n_neighbors, batch_size=4096):
    """Per scheme: inverse-distance weighted class shares (prior-ratio calibrated) and the mean distance."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_dtype = torch.float16 if USE_FP16_NUM else torch.float32
    ref_cat_te_t = torch.as_tensor(ref_cat_te, dtype=torch.float32, device=device)
    ref_num_t = torch.as_tensor(ref_num_scaled, dtype=num_dtype, device=device)
    ref_y_t = torch.as_tensor(ref_y, dtype=torch.int64, device=device)
    ratio_vec = torch.as_tensor(train_prior / np.clip(ref_prior, 1e-8, None), dtype=torch.float32, device=device)
    query_cat_cpu = torch.as_tensor(query_cat_te, dtype=torch.float32)
    query_num_cpu = torch.as_tensor(query_num_scaled, dtype=num_dtype)
    if PIN_MEMORY and device.type == 'cuda':
        query_cat_cpu, query_num_cpu = query_cat_cpu.pin_memory(), query_num_cpu.pin_memory()
    n_query = len(query_cat_te)
    out = np.empty((n_query, len(base_names) * 4), dtype=np.float32)
    for s_idx, (cat_w_np, num_w_np, _mode) in enumerate(schemes):
        cat_w_t = torch.as_tensor(expand_cat_weights(cat_w_np), dtype=torch.float32, device=device)
        num_w_t = torch.as_tensor(num_w_np, dtype=num_dtype, device=device)
        probs_scheme = np.empty((n_query, 3), dtype=np.float32)
        dists_scheme = np.empty(n_query, dtype=np.float32)
        for lo in range(0, n_query, batch_size):
            hi = min(lo + batch_size, n_query)
            q_cat = query_cat_cpu[lo:hi].to(device, non_blocking=True)
            q_num = query_num_cpu[lo:hi].to(device, non_blocking=True)
            cat_diff = torch.abs(q_cat[:, None, :] - ref_cat_te_t[None, :, :])
            cat_dist = (cat_diff * cat_w_t.view(1, 1, -1)).sum(dim=2)
            num_diff = q_num[:, None, :] - ref_num_t[None, :, :]
            num_dist = ((num_diff * num_diff) * num_w_t.view(1, 1, -1)).sum(dim=2).float()
            dist = cat_dist + num_dist
            top_dist, nn_idx = torch.topk(dist, k=n_neighbors, dim=1, largest=False, sorted=False)
            p = weighted_one_hot(ref_y_t[nn_idx], 1.0 / (top_dist + DIST_EPS))
            if APPLY_PRIOR_CALIBRATION:
                p = p * ratio_vec.view(1, -1)
                p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
            probs_scheme[lo:hi] = p.cpu().numpy().astype(np.float32)
            dists_scheme[lo:hi] = top_dist.mean(dim=1).cpu().numpy().astype(np.float32)
        out[:, s_idx * 4:(s_idx + 1) * 4] = np.column_stack([probs_scheme, dists_scheme]).astype(np.float32)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return pd.DataFrame(out, columns=scheme_output_cols(base_names))


def build_features(knn_k):
    """43-column matrices for train and test (the LR meta block holds out-of-fold values for the train rows)."""
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = train[TARGET].map(TARGET_MAP).astype(np.int8)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype(np.int8)

    num_fill = orig[NUMERIC_COLS].median()
    for df in [train, test, orig]:
        for c in CATEGORICAL_COLS:
            df[c] = df[c].astype(str).fillna('missing')
        for c in NUMERIC_COLS:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(num_fill[c]).astype(np.float32)
    for c in CATEGORICAL_COLS:
        codes, _ = pd.factorize(pd.concat([train[c], test[c], orig[c]], ignore_index=True), sort=False)
        train[c] = codes[:len(train)].astype(np.int16)
        test[c] = codes[len(train):len(train) + len(test)].astype(np.int16)
        orig[c] = codes[len(train) + len(test):].astype(np.int16)

    train_cat_codes = train[CATEGORICAL_COLS].values.astype(np.int16)
    test_cat_codes = test[CATEGORICAL_COLS].values.astype(np.int16)
    orig_cat_codes = orig[CATEGORICAL_COLS].values.astype(np.int16)
    orig_num_scaled, train_num_scaled, test_num_scaled = standardize_ref_and_queries(
        orig[NUMERIC_COLS].values.astype(np.float32), train[NUMERIC_COLS].values.astype(np.float32),
        test[NUMERIC_COLS].values.astype(np.float32))

    y = train[TARGET].values.astype(np.int8)
    orig_y = orig[TARGET].values.astype(np.int8)
    train_prior = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    train_prior /= train_prior.sum()
    orig_prior = np.bincount(orig_y, minlength=NUM_CLASSES).astype(np.float32)
    orig_prior /= orig_prior.sum()

    print('original-data KNN: random schemes', flush=True)
    all_random_schemes = build_random_weight_schemes(N_RANDOM_SCHEMES, len(CATEGORICAL_COLS), len(NUMERIC_COLS), seed=SEED)
    all_random_bases = prefixed_colnames('knnprior', list(range(N_RANDOM_SCHEMES)))
    knn_random_train = build_knn_random_gpu_full(orig_cat_codes, orig_num_scaled, orig_y, train_cat_codes, train_num_scaled,
                                                 all_random_schemes, all_random_bases, knn_k, batch_size=GPU_BATCH_SIZE_ORIG)
    knn_random_test = build_knn_random_gpu_full(orig_cat_codes, orig_num_scaled, orig_y, test_cat_codes, test_num_scaled,
                                                all_random_schemes, all_random_bases, knn_k, batch_size=GPU_BATCH_SIZE_ORIG)

    print('original-data KNN: smart schemes', flush=True)
    te_maps_orig, te_global_orig = build_target_encoding_maps(orig)
    orig_cat_te = make_target_rate_matrix(orig, te_maps_orig, te_global_orig)
    train_cat_te = make_target_rate_matrix(train, te_maps_orig, te_global_orig)
    test_cat_te = make_target_rate_matrix(test, te_maps_orig, te_global_orig)
    orig_importance = fit_orig_importance_model(orig)
    all_smart_schemes = build_smart_weight_schemes(orig_importance[CATEGORICAL_COLS].values.astype(np.float32),
                                                   orig_importance[NUMERIC_COLS].values.astype(np.float32),
                                                   N_SMART_SCHEMES, seed=SEED)
    all_smart_bases = prefixed_colnames('knnsmart', list(range(N_SMART_SCHEMES)))
    knn_smart_train = build_knn_smart_gpu_full(orig_cat_te, orig_num_scaled, orig_y, train_cat_te, train_num_scaled,
                                               train_prior, orig_prior, all_smart_schemes, all_smart_bases, knn_k,
                                               batch_size=GPU_BATCH_SIZE_ORIG)
    knn_smart_test = build_knn_smart_gpu_full(orig_cat_te, orig_num_scaled, orig_y, test_cat_te, test_num_scaled,
                                              train_prior, orig_prior, all_smart_schemes, all_smart_bases, knn_k,
                                              batch_size=GPU_BATCH_SIZE_ORIG)

    rng = np.random.default_rng(SELECTION_SEED)
    selected_random_idx = choose_diverse_indices(rng, N_RANDOM_SCHEMES, N_SELECT_RANDOM, n_buckets=10)
    selected_smart_idx = choose_diverse_indices(rng, N_SMART_SCHEMES, N_SELECT_SMART, n_buckets=8)
    selected_random_bases = [all_random_bases[i] for i in selected_random_idx]
    selected_smart_bases = [all_smart_bases[i] for i in selected_smart_idx]
    rand_w = rng.dirichlet(np.full(len(selected_random_bases), 0.7, dtype=np.float32)).astype(np.float32)
    smart_w = rng.dirichlet(np.full(len(selected_smart_bases), 0.7, dtype=np.float32)).astype(np.float32)
    print(f'selected random schemes {selected_random_idx}, smart schemes {selected_smart_idx}', flush=True)

    r_tr = knn_random_train[scheme_output_cols(selected_random_bases)].copy()
    r_te = knn_random_test[scheme_output_cols(selected_random_bases)].copy()
    s_tr = knn_smart_train[scheme_output_cols(selected_smart_bases)].copy()
    s_te = knn_smart_test[scheme_output_cols(selected_smart_bases)].copy()
    rlog_tr = pd.DataFrame(safe_logit(r_tr[scheme_prob_cols(selected_random_bases)].values).astype(np.float32), index=train.index)
    rlog_te = pd.DataFrame(safe_logit(r_te[scheme_prob_cols(selected_random_bases)].values).astype(np.float32), index=test.index)
    slog_tr = pd.DataFrame(safe_logit(s_tr[scheme_prob_cols(selected_smart_bases)].values).astype(np.float32), index=train.index)
    slog_te = pd.DataFrame(safe_logit(s_te[scheme_prob_cols(selected_smart_bases)].values).astype(np.float32), index=test.index)
    alog_tr = pd.concat([rlog_tr, slog_tr], axis=1)
    alog_te = pd.concat([rlog_te, slog_te], axis=1)

    fixed_rand_train = weighted_family_features(r_tr, selected_random_bases, rand_w, 'knn_rand_wmean', train.index)
    fixed_rand_test = weighted_family_features(r_te, selected_random_bases, rand_w, 'knn_rand_wmean', test.index)
    fixed_smart_train = weighted_family_features(s_tr, selected_smart_bases, smart_w, 'knn_smart_wmean', train.index)
    fixed_smart_test = weighted_family_features(s_te, selected_smart_bases, smart_w, 'knn_smart_wmean', test.index)
    fixed_train = pd.concat([fixed_rand_train, fixed_smart_train,
                             blend_feature_frames(fixed_rand_train, fixed_smart_train, 'knn_blend')], axis=1)
    fixed_test = pd.concat([fixed_rand_test, fixed_smart_test,
                            blend_feature_frames(fixed_rand_test, fixed_smart_test, 'knn_blend')], axis=1)

    x_train_base = train[ALL_FEATURE_COLS].copy()
    x_test_base = test[ALL_FEATURE_COLS].copy()
    for c in ALL_FEATURE_COLS:
        x_train_base[c] = pd.to_numeric(x_train_base[c], errors='coerce').fillna(0).astype(np.float32)
        x_test_base[c] = pd.to_numeric(x_test_base[c], errors='coerce').fillna(0).astype(np.float32)

    meta_cols = [f'{prefix}_c{cls}' for prefix in ['meta_rand_lr', 'meta_smart_lr', 'meta_all_lr', 'meta_blend_lr']
                 for cls in range(NUM_CLASSES)]
    stack_train = pd.DataFrame(0.0, index=train.index, columns=meta_cols + list(fixed_train.columns), dtype=np.float32)
    stack_test = pd.DataFrame(0.0, index=test.index, columns=meta_cols + list(fixed_test.columns), dtype=np.float32)
    for c in fixed_train.columns:
        stack_train[c] = fixed_train[c].values.astype(np.float32)
        stack_test[c] = fixed_test[c].values.astype(np.float32)

    print('LR meta features (all 5 folds)', flush=True)
    fold_id = folds(y)
    for k in range(N_SPLITS):
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        y_tr = y[tr_idx]
        p_va_rand, p_te_rand = build_lr_feature(rlog_tr.iloc[tr_idx].values, y_tr, rlog_tr.iloc[va_idx].values, rlog_te.values)
        p_va_smart, p_te_smart = build_lr_feature(slog_tr.iloc[tr_idx].values, y_tr, slog_tr.iloc[va_idx].values, slog_te.values)
        p_va_all, p_te_all = build_lr_feature(alog_tr.iloc[tr_idx].values, y_tr, alog_tr.iloc[va_idx].values, alog_te.values)
        va_frames = [
            prob_df(p_va_rand, 'meta_rand_lr', va_idx),
            prob_df(p_va_smart, 'meta_smart_lr', va_idx),
            prob_df(p_va_all, 'meta_all_lr', va_idx),
            prob_df((p_va_rand + p_va_smart + p_va_all) / 3.0, 'meta_blend_lr', va_idx),
        ]
        te_frames = [
            prob_df(p_te_rand, 'meta_rand_lr', test.index),
            prob_df(p_te_smart, 'meta_smart_lr', test.index),
            prob_df(p_te_all, 'meta_all_lr', test.index),
            prob_df((p_te_rand + p_te_smart + p_te_all) / 3.0, 'meta_blend_lr', test.index),
        ]
        for frame in va_frames:
            assign_frame(stack_train, va_idx, frame)
        for frame in te_frames:
            stack_test[frame.columns] += frame.values.astype(np.float32) / N_SPLITS
        gc.collect()

    final_train = pd.concat([x_train_base.reset_index(drop=True), stack_train.reset_index(drop=True)], axis=1)
    final_test = pd.concat([x_test_base.reset_index(drop=True), stack_test.reset_index(drop=True)], axis=1)
    return final_train, final_test, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preset', choices=sorted(PRESETS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    pair = args.preset
    out = model_dir(FOLDER)
    seed_everything(SEED)

    final_train, final_test, y = build_features(PRESETS[pair])
    fold_id = folds(y)
    for k in fold_list(args.folds):
        if have_part(out, pair, k):
            print(f'{pair} fold {k}: done', flush=True)
            continue
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        x_tr, x_va = final_train.iloc[tr_idx].copy(), final_train.iloc[va_idx].copy()
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(x_tr, y[tr_idx], eval_set=[(x_va, y[va_idx])], verbose=200)
        p_va = model.predict_proba(x_va).astype(np.float32)
        p_te = model.predict_proba(final_test).astype(np.float32)
        save_part(out, pair, k, p_va, p_te)
        print(f'{pair} fold {k}: balanced accuracy {balanced_accuracy_score(y[va_idx], p_va.argmax(axis=1)):.6f}', flush=True)
        del model, x_tr, x_va
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assemble(out, pair, fold_id, y)


if __name__ == '__main__':
    main()
