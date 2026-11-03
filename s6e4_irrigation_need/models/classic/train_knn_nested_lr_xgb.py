#!/usr/bin/env python
"""XGBoost on original-data KNN, nested train-data KNN and logistic-regression meta features
(pair xgb_nested_knn_lr_meta_b, stack B).

Base columns (22): the 8 categoricals (pandas categories shared by train / test / original), the 11 numerics (float32)
and a threshold score 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10]
- 2 [stage Harvest] - 2 [stage Sowing] - [mulching used], with the flags score <= 0 and score > 3.
KNN (GPU, brute force, numerics in fp16): distance = weighted categorical mismatches + weighted squared standardized
numeric differences ("random" schemes), or weighted absolute differences of the categoricals' per-class target rates +
the same numeric part ("smart" schemes).
  - 30 random weight schemes (10 generators) and 30 smart schemes (feature importances of an XGBoost fitted on the
    original rows, 600 trees, lr 0.05, depth 4; 10 templates with 15% jitter).
  - Original-data KNN (k = 10; neighbours = the 10,000 original rows, numerics standardized on them): random schemes ->
    class shares of the neighbours; smart schemes -> inverse-distance weighted class shares rescaled by the
    train / original class-prior ratio.
  - 8 random + 8 smart schemes are drawn (one per bucket of scheme indices, seed 10442) with Dirichlet(0.7) weights;
    weighted means of their class shares (random, smart, average of both).
Per outer fold:
  - train-data KNN with the 16 selected schemes (k = 40; neighbours = the outer-training rows, numerics standardized
    and smart rates computed on them): directly for the held-out and test rows; for the outer-training rows
    out-of-fold over 5 inner folds (StratifiedKFold seed 42 + fold number 1-5);
  - multinomial logistic regressions (lbfgs, C 0.2, standardized inputs) on the logits of the original-KNN and of the
    train-KNN class shares (random / smart / both, averaged): fitted on the outer-training rows (train-KNN: their
    inner out-of-fold values), they predict the held-out and test rows and, in sample, the outer-training rows;
  - 30 stack columns (x 3 classes: the two LR averages and their mean, the weighted original-KNN and train-KNN means
    and their blends);
  - XGBoost (GPU) on the 52 columns: native categoricals (max_cat_to_onehot 8), multi:softprob, lr 0.018, depth 4,
    min_child_weight 8, subsample 0.82, colsample_bytree 0.32, alpha 1, lambda 8, balanced sample weights, up to 6,000
    rounds with early stopping 250 (mlogloss) on the fold's validation rows, seed 42.
Test = mean of the 5 fold models. The original rows are never training rows.
Usage: python train_knn_nested_lr_xgb.py [--folds 0,1,2,3,4]
(the original-data KNN features are rebuilt in every run; finished folds are kept in parts/ and skipped)"""
import argparse
import gc
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, NUMERICS, TARGET, TARGET_MAP, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'xgb_nested_knn_lr_meta_b'
FOLDER = 'classic'

SEED = 42
N_SPLITS = 5
INNER_SPLITS = 5
N_CLASSES = 3
EPS = 1e-6

CATEGORICAL_COLS = list(CATEGORICALS)
NUMERIC_COLS = list(NUMERICS)
FORMULA_COLS = ['formula_score', 'is_low', 'is_high']
ALL_FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS + FORMULA_COLS

KNN_K = 10
TRAIN_KNN_K = 40
N_RANDOM_SCHEMES = 30
N_SMART_SCHEMES = 30
N_SELECT_RANDOM = 8
N_SELECT_SMART = 8

USE_FP16_NUM = True
PIN_MEMORY = True
DIST_EPS = 1e-6
APPLY_PRIOR_CALIBRATION = True

GPU_BATCH_SIZE_ORIG = 4096
BATCH_SIZE_OUTER = 192
BATCH_SIZE_INNER = 256
REF_CHUNK_SIZE = 24576

RUN_SEED = int(SEED + 1000 * KNN_K + 10 * TRAIN_KNN_K)  # scheme selection and blend weights

LR_PARAMS = dict(
    C=0.20,
    solver='lbfgs',
    multi_class='multinomial',
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
    objective='multi:softprob',
    num_class=N_CLASSES,
    eval_metric='mlogloss',
    tree_method='hist',
    device='cuda',
    enable_categorical=True,
    max_cat_to_onehot=8,
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
    objective='multi:softmax',
    num_class=N_CLASSES,
    tree_method='hist',
    device='cuda',
    enable_categorical=True,
    random_state=SEED,
    verbosity=0,
)

# the 10 stack groups (x 3 classes) of the final model
STACK_GROUPS = ['meta_blend_lr', 'meta_tr_blend_lr', 'knn_rand_wmean', 'knn_smart_wmean', 'knn_blend_mean',
                'trknn_rand_wmean', 'trknn_smart_wmean', 'trknn_blend_mean', 'knn_full_blend_mean', 'meta_full_blend_lr']


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def safe_logit(x, eps=1e-6):
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def standardize_from_reference(x_ref, x):
    mu = np.nanmean(x_ref, axis=0)
    sd = np.nanstd(x_ref, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def standardize_ref_and_queries(x_ref, *queries):
    x_ref_scaled, mu, sd = standardize_from_reference(x_ref, x_ref)
    return [x_ref_scaled] + [((q - mu) / sd).astype(np.float32) for q in queries]


def fit_lr(X_tr, y_tr, *queries):
    """Standardize + multinomial LR fitted on X_tr; (N, 3) probabilities for every query matrix."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_tr_s, y_tr)
    return [lr.predict_proba(scaler.transform(q)).astype(np.float32) for q in queries]


# ------------------------------------------------------------------ per-class target rates (smart schemes)
def build_target_encoding_maps(df):
    """maps[col][k] = {value: mean(y == k)}."""
    maps = {}
    global_means = [(df[TARGET].values == k).mean() for k in range(N_CLASSES)]
    for c in CATEGORICAL_COLS:
        maps[c] = {}
        for k in range(N_CLASSES):
            y_k = (df[TARGET].values == k).astype(float)
            maps[c][k] = pd.Series(y_k, index=df.index).groupby(df[c]).mean().to_dict()
    return maps, global_means


def make_target_rate_matrix(df, te_maps, global_means):
    """(N, 8 * 3) float32: [cat0_k0, cat0_k1, cat0_k2, cat1_k0, ...]."""
    mats = []
    for c in CATEGORICAL_COLS:
        for k in range(N_CLASSES):
            mats.append(df[c].astype(str).map(te_maps[c][k]).fillna(global_means[k]).astype(np.float32).values)
    return np.stack(mats, axis=1).astype(np.float32)


# ------------------------------------------------------------------ weight schemes
def build_random_weight_schemes(n_schemes, n_cat, n_num, seed=42):
    rng = np.random.default_rng(seed)
    schemes = []
    for k in range(n_schemes):
        mode = k % 10
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
            cat_w = rng.uniform(0.0, 1.0, n_cat) * (rng.random(n_cat) < 0.35)
            num_w = rng.uniform(0.0, 1.0, n_num) * (rng.random(n_num) < 0.67)
            cat_w += 1e-3
            num_w += 1e-3
        elif mode == 4:
            z = rng.dirichlet(np.ones(n_cat + n_num))
            z = z * (n_cat + n_num)
            cat_w = z[:n_cat]
            num_w = z[n_cat:]
        elif mode == 5:
            cat_w = np.full(n_cat, 0.05, dtype=np.float32)
            num_w = np.full(n_num, 0.05, dtype=np.float32)
            j = rng.integers(0, n_cat + n_num)
            if j < n_cat:
                cat_w[j] = rng.uniform(1.5, 3.0)
            else:
                num_w[j - n_cat] = rng.uniform(1.5, 3.0)
        elif mode == 6:
            cat_w = np.full(n_cat, 0.05, dtype=np.float32)
            num_w = np.full(n_num, 0.05, dtype=np.float32)
            js = rng.choice(n_cat + n_num, size=2, replace=False)
            for j in js:
                if j < n_cat:
                    cat_w[j] = rng.uniform(1.0, 2.5)
                else:
                    num_w[j - n_cat] = rng.uniform(1.0, 2.5)
        elif mode == 7:
            cat_w = np.clip(1.0 + rng.normal(0, 0.25, n_cat), 0.05, 2.0)
            num_w = np.clip(1.0 + rng.normal(0, 0.25, n_num), 0.05, 2.0)
        elif mode == 8:
            cat_w = rng.uniform(0.05, 0.80, n_cat)
            num_w = rng.uniform(0.20, 1.20, n_num)
        else:
            cat_w = rng.uniform(0.10, 1.50, n_cat)
            num_w = rng.uniform(0.01, 0.20, n_num)
        cat_w = np.asarray(cat_w, dtype=np.float32)
        num_w = np.asarray(num_w, dtype=np.float32)
        z = np.concatenate([cat_w, num_w])
        z = z / np.mean(z)
        schemes.append((z[:n_cat].astype(np.float32), z[n_cat:].astype(np.float32)))
    return schemes


def fit_orig_importance_model(orig_df):
    """XGBoost on the original rows; normalized feature importances."""
    Xo = orig_df[CATEGORICAL_COLS + NUMERIC_COLS].copy()
    yo = orig_df[TARGET].values
    for c in CATEGORICAL_COLS:
        Xo[c] = Xo[c].astype('category')
    for c in NUMERIC_COLS:
        Xo[c] = Xo[c].astype(np.float32)
    model = XGBClassifier(**ORIG_XGB_PARAMS)
    model.fit(Xo, yo, verbose=False)
    imp = pd.Series(model.feature_importances_, index=CATEGORICAL_COLS + NUMERIC_COLS).fillna(0.0)
    if imp.sum() <= 0:
        imp[:] = 1.0
    return imp / imp.sum()


def build_smart_weight_schemes(base_cat_w, base_num_w, n_schemes, seed=42):
    """base_cat_w: (8 * 3,) weights of the per-class rate columns; base_num_w: (11,)."""
    rng = np.random.default_rng(seed)
    base_cat_w = np.asarray(base_cat_w, dtype=np.float32)
    base_num_w = np.asarray(base_num_w, dtype=np.float32)
    z = np.concatenate([base_cat_w, base_num_w])
    z = z / np.mean(z)
    base_cat_w = z[:len(base_cat_w)].astype(np.float32)
    base_num_w = z[len(base_cat_w):].astype(np.float32)

    template_modes = ['base', 'sqrt', 'square', 'cat_heavy', 'num_heavy',
                      'flat_mix', 'top_num1_boost', 'top_num2_boost', 'top2_boost', 'top4_boost']
    cat_idx_sorted = np.argsort(-base_cat_w)
    num_idx_sorted = np.argsort(-base_num_w)
    schemes = []
    while len(schemes) < n_schemes:
        mode = template_modes[len(schemes) % len(template_modes)]
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
        elif mode == 'top_num1_boost':
            num_w[num_idx_sorted[0]] *= 2.20
        elif mode == 'top_num2_boost':
            num_w[num_idx_sorted[0]] *= 1.80
            num_w[num_idx_sorted[1]] *= 1.80
        elif mode == 'top2_boost':
            cat_w[cat_idx_sorted[:2]] *= 2.20
            if len(num_idx_sorted) > 0:
                num_w[num_idx_sorted[0]] *= 1.50
        elif mode == 'top4_boost':
            cat_w[cat_idx_sorted[:4]] *= 1.80
            num_w[num_idx_sorted[:2]] *= 1.30
        cat_w *= np.clip(1.0 + rng.normal(0, 0.15, len(cat_w)), 0.60, 1.50)
        num_w *= np.clip(1.0 + rng.normal(0, 0.15, len(num_w)), 0.60, 1.50)
        cat_w = np.clip(cat_w, 1e-3, None)
        num_w = np.clip(num_w, 1e-3, None)
        z = np.concatenate([cat_w, num_w])
        z = z / np.mean(z)
        schemes.append((z[:len(base_cat_w)].astype(np.float32), z[len(base_cat_w):].astype(np.float32), mode))
    return schemes


def prefixed_colnames_3cls(prefix, indices):
    return [f'{prefix}_{i:03d}_k{k}' for i in indices for k in range(N_CLASSES)]


def weighted_avg_block_3cls(df, scheme_bases, weights):
    """Weighted mean of the class shares of the given schemes; (N, 3)."""
    x = np.zeros((len(df), N_CLASSES), dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    w = w / w.sum()
    for i, base in enumerate(scheme_bases):
        for k in range(N_CLASSES):
            x[:, k] += df[f'{base}_k{k}'].values.astype(np.float32) * w[i]
    return x


def dirichlet_weights(rng, n, alpha=1.0):
    w = rng.dirichlet(alpha=np.full(n, alpha, dtype=np.float32)).astype(np.float32)
    return w / w.sum()


def choose_diverse_indices(rng, n_total, n_select, n_buckets=10):
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
        remaining = np.setdiff1d(np.arange(n_total), np.array(chosen, dtype=int))
        chosen.extend(rng.choice(remaining, size=(n_select - len(chosen)), replace=False).tolist())
    return sorted(set(chosen))


# ------------------------------------------------------------------ GPU KNN, small reference set (original rows)
@torch.no_grad()
def build_knn_random_gpu_full(ref_cat_codes, ref_num_scaled, ref_y_onehot, query_cat_codes, query_num_scaled,
                              schemes, col_names, n_neighbors, batch_size=4096):
    """Per scheme: mean one-hot label of the k nearest reference rows."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_dtype = torch.float16 if USE_FP16_NUM else torch.float32
    ref_cat_t = torch.as_tensor(ref_cat_codes, dtype=torch.int16, device=device)
    ref_num_t = torch.as_tensor(ref_num_scaled, dtype=num_dtype, device=device)
    ref_y_t = torch.as_tensor(ref_y_onehot, dtype=torch.float32, device=device)
    n_query = query_cat_codes.shape[0]
    out = np.empty((n_query, len(schemes) * N_CLASSES), dtype=np.float32)
    query_cat_cpu = torch.as_tensor(query_cat_codes, dtype=torch.int16)
    query_num_cpu = torch.as_tensor(query_num_scaled, dtype=num_dtype)
    if PIN_MEMORY and device.type == 'cuda':
        query_cat_cpu, query_num_cpu = query_cat_cpu.pin_memory(), query_num_cpu.pin_memory()

    for s_idx, (cat_w_np, num_w_np) in enumerate(schemes):
        cat_w_t = torch.as_tensor(cat_w_np, dtype=torch.float32, device=device)
        num_w_t = torch.as_tensor(num_w_np, dtype=num_dtype, device=device)
        preds_scheme = np.empty((n_query, N_CLASSES), dtype=np.float32)
        for lo in range(0, n_query, batch_size):
            hi = min(lo + batch_size, n_query)
            q_cat = query_cat_cpu[lo:hi].to(device, non_blocking=True)
            q_num = query_num_cpu[lo:hi].to(device, non_blocking=True)
            mismatch = (q_cat[:, None, :] != ref_cat_t[None, :, :])
            cat_dist = (mismatch.float() * cat_w_t.view(1, 1, -1)).sum(dim=2)
            num_diff = q_num[:, None, :] - ref_num_t[None, :, :]
            num_dist = ((num_diff * num_diff) * num_w_t.view(1, 1, -1)).sum(dim=2).float()
            dist = cat_dist + num_dist
            nn_idx = torch.topk(dist, k=n_neighbors, dim=1, largest=False, sorted=False).indices
            preds_scheme[lo:hi] = ref_y_t[nn_idx].mean(dim=1).detach().cpu().numpy()
        out[:, s_idx * N_CLASSES:(s_idx + 1) * N_CLASSES] = preds_scheme
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return pd.DataFrame(out, columns=col_names)


@torch.no_grad()
def build_knn_smart_gpu_full(ref_cat_te, ref_num_scaled, ref_y_onehot, query_cat_te, query_num_scaled,
                             train_class_freq, ref_class_freq, schemes, col_names, n_neighbors, batch_size=4096):
    """Per scheme: inverse-distance weighted class shares, prior-ratio calibrated and renormalized."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_dtype = torch.float16 if USE_FP16_NUM else torch.float32
    ref_cat_te_t = torch.as_tensor(ref_cat_te, dtype=torch.float32, device=device)
    ref_num_t = torch.as_tensor(ref_num_scaled, dtype=num_dtype, device=device)
    ref_y_t = torch.as_tensor(ref_y_onehot, dtype=torch.float32, device=device)
    ratio = np.array([train_class_freq[k] / max(ref_class_freq[k], 1e-8) for k in range(N_CLASSES)], dtype=np.float32)
    ratio_t = torch.as_tensor(ratio, dtype=torch.float32, device=device)
    n_query = query_cat_te.shape[0]
    out = np.empty((n_query, len(schemes) * N_CLASSES), dtype=np.float32)
    query_cat_cpu = torch.as_tensor(query_cat_te, dtype=torch.float32)
    query_num_cpu = torch.as_tensor(query_num_scaled, dtype=num_dtype)
    if PIN_MEMORY and device.type == 'cuda':
        query_cat_cpu, query_num_cpu = query_cat_cpu.pin_memory(), query_num_cpu.pin_memory()

    for s_idx, (cat_w_np, num_w_np, _mode) in enumerate(schemes):
        cat_w_t = torch.as_tensor(cat_w_np, dtype=torch.float32, device=device)
        num_w_t = torch.as_tensor(num_w_np, dtype=num_dtype, device=device)
        preds_scheme = np.empty((n_query, N_CLASSES), dtype=np.float32)
        for lo in range(0, n_query, batch_size):
            hi = min(lo + batch_size, n_query)
            q_cat_te = query_cat_cpu[lo:hi].to(device, non_blocking=True)
            q_num = query_num_cpu[lo:hi].to(device, non_blocking=True)
            cat_diff = torch.abs(q_cat_te[:, None, :] - ref_cat_te_t[None, :, :])
            cat_dist = (cat_diff * cat_w_t.view(1, 1, -1)).sum(dim=2)
            num_diff = q_num[:, None, :] - ref_num_t[None, :, :]
            num_dist = ((num_diff * num_diff) * num_w_t.view(1, 1, -1)).sum(dim=2).float()
            dist = cat_dist + num_dist
            top_dist, nn_idx = torch.topk(dist, k=n_neighbors, dim=1, largest=False, sorted=False)
            nn_y = ref_y_t[nn_idx]
            w = 1.0 / (top_dist + DIST_EPS)
            p = (w[:, :, None] * nn_y).sum(dim=1) / w.sum(dim=1, keepdim=True)
            if APPLY_PRIOR_CALIBRATION:
                p = torch.clamp(p * ratio_t[None, :], 0.0, 1.0)
            p = p / (p.sum(dim=1, keepdim=True) + 1e-8)
            preds_scheme[lo:hi] = p.detach().cpu().numpy()
        out[:, s_idx * N_CLASSES:(s_idx + 1) * N_CLASSES] = preds_scheme
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return pd.DataFrame(out, columns=col_names)


# ------------------------------------------------------------------ GPU KNN, large reference set (training rows)
@torch.no_grad()
def build_knn_random_gpu_chunked(ref_cat_codes, ref_num_scaled, ref_y_onehot, query_cat_codes, query_num_scaled,
                                 schemes, col_names, n_neighbors, batch_size=192, ref_chunk_size=24576, prefix=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_dtype = torch.float16 if USE_FP16_NUM else torch.float32
    n_query, n_ref, n_cat = query_cat_codes.shape[0], ref_cat_codes.shape[0], ref_cat_codes.shape[1]
    log(f'{prefix}: n_query={n_query:,} n_ref={n_ref:,} k={n_neighbors}')

    ref_cat_cpu = torch.as_tensor(ref_cat_codes, dtype=torch.int16)
    ref_num_cpu = torch.as_tensor(ref_num_scaled, dtype=num_dtype)
    ref_y_cpu = torch.as_tensor(ref_y_onehot, dtype=torch.float32)
    query_cat_cpu = torch.as_tensor(query_cat_codes, dtype=torch.int16)
    query_num_cpu = torch.as_tensor(query_num_scaled, dtype=num_dtype)
    if PIN_MEMORY and device.type == 'cuda':
        ref_cat_cpu, ref_num_cpu, ref_y_cpu = ref_cat_cpu.pin_memory(), ref_num_cpu.pin_memory(), ref_y_cpu.pin_memory()
        query_cat_cpu, query_num_cpu = query_cat_cpu.pin_memory(), query_num_cpu.pin_memory()

    out = np.empty((n_query, len(schemes) * N_CLASSES), dtype=np.float32)
    for s_idx, (cat_w_np, num_w_np) in enumerate(schemes):
        cat_w_t = torch.as_tensor(cat_w_np, dtype=torch.float32, device=device)
        num_w_t = torch.as_tensor(num_w_np, dtype=num_dtype, device=device)
        preds_scheme = np.empty((n_query, N_CLASSES), dtype=np.float32)
        for q_lo in range(0, n_query, batch_size):
            q_hi = min(q_lo + batch_size, n_query)
            q_cat = query_cat_cpu[q_lo:q_hi].to(device, non_blocking=True)
            q_num = query_num_cpu[q_lo:q_hi].to(device, non_blocking=True)
            bsz = q_cat.shape[0]
            best_dist = torch.full((bsz, n_neighbors), float('inf'), dtype=torch.float32, device=device)
            best_y = torch.zeros((bsz, n_neighbors, N_CLASSES), dtype=torch.float32, device=device)
            for r_lo in range(0, n_ref, ref_chunk_size):
                r_hi = min(r_lo + ref_chunk_size, n_ref)
                ref_cat_t = ref_cat_cpu[r_lo:r_hi].to(device, non_blocking=True)
                ref_num_t = ref_num_cpu[r_lo:r_hi].to(device, non_blocking=True)
                ref_y_t = ref_y_cpu[r_lo:r_hi].to(device, non_blocking=True)
                cat_dist = torch.zeros((bsz, r_hi - r_lo), dtype=torch.float32, device=device)
                for j in range(n_cat):
                    cat_dist += (q_cat[:, None, j] != ref_cat_t[None, :, j]).float() * cat_w_t[j]
                num_diff = q_num[:, None, :] - ref_num_t[None, :, :]
                num_dist = ((num_diff * num_diff) * num_w_t.view(1, 1, -1)).sum(dim=2).float()
                dist_chunk = cat_dist + num_dist
                k_chunk = min(n_neighbors, dist_chunk.shape[1])
                chunk_dist, chunk_idx_local = torch.topk(dist_chunk, k=k_chunk, dim=1, largest=False, sorted=False)
                merged_dist = torch.cat([best_dist, chunk_dist], dim=1)
                merged_y = torch.cat([best_y, ref_y_t[chunk_idx_local]], dim=1)
                keep_idx = torch.topk(merged_dist, k=n_neighbors, dim=1, largest=False, sorted=False).indices
                best_dist = torch.gather(merged_dist, 1, keep_idx)
                best_y = torch.gather(merged_y, 1, keep_idx.unsqueeze(-1).expand(-1, -1, N_CLASSES))
            preds_scheme[q_lo:q_hi] = best_y.mean(dim=1).detach().cpu().numpy()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        out[:, s_idx * N_CLASSES:(s_idx + 1) * N_CLASSES] = preds_scheme
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return pd.DataFrame(out, columns=col_names)


@torch.no_grad()
def build_knn_smart_gpu_chunked(ref_cat_te, ref_num_scaled, ref_y_onehot, query_cat_te, query_num_scaled,
                                train_class_freq, ref_class_freq, schemes, col_names, n_neighbors,
                                batch_size=192, ref_chunk_size=24576, prefix=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_dtype = torch.float16 if USE_FP16_NUM else torch.float32
    n_query, n_ref, n_cat_te = query_cat_te.shape[0], ref_cat_te.shape[0], ref_cat_te.shape[1]
    log(f'{prefix}: n_query={n_query:,} n_ref={n_ref:,} k={n_neighbors}')
    ratio = np.array([train_class_freq[k] / max(ref_class_freq[k], 1e-8) for k in range(N_CLASSES)], dtype=np.float32)
    ratio_t = torch.as_tensor(ratio, dtype=torch.float32, device=device)

    ref_cat_cpu = torch.as_tensor(ref_cat_te, dtype=torch.float32)
    ref_num_cpu = torch.as_tensor(ref_num_scaled, dtype=num_dtype)
    ref_y_cpu = torch.as_tensor(ref_y_onehot, dtype=torch.float32)
    query_cat_cpu = torch.as_tensor(query_cat_te, dtype=torch.float32)
    query_num_cpu = torch.as_tensor(query_num_scaled, dtype=num_dtype)
    if PIN_MEMORY and device.type == 'cuda':
        ref_cat_cpu, ref_num_cpu, ref_y_cpu = ref_cat_cpu.pin_memory(), ref_num_cpu.pin_memory(), ref_y_cpu.pin_memory()
        query_cat_cpu, query_num_cpu = query_cat_cpu.pin_memory(), query_num_cpu.pin_memory()

    out = np.empty((n_query, len(schemes) * N_CLASSES), dtype=np.float32)
    for s_idx, (cat_w_np, num_w_np, _mode) in enumerate(schemes):
        cat_w_t = torch.as_tensor(cat_w_np, dtype=torch.float32, device=device)
        num_w_t = torch.as_tensor(num_w_np, dtype=num_dtype, device=device)
        preds_scheme = np.empty((n_query, N_CLASSES), dtype=np.float32)
        for q_lo in range(0, n_query, batch_size):
            q_hi = min(q_lo + batch_size, n_query)
            q_cat = query_cat_cpu[q_lo:q_hi].to(device, non_blocking=True)
            q_num = query_num_cpu[q_lo:q_hi].to(device, non_blocking=True)
            bsz = q_cat.shape[0]
            best_dist = torch.full((bsz, n_neighbors), float('inf'), dtype=torch.float32, device=device)
            best_y = torch.zeros((bsz, n_neighbors, N_CLASSES), dtype=torch.float32, device=device)
            for r_lo in range(0, n_ref, ref_chunk_size):
                r_hi = min(r_lo + ref_chunk_size, n_ref)
                ref_cat_t = ref_cat_cpu[r_lo:r_hi].to(device, non_blocking=True)
                ref_num_t = ref_num_cpu[r_lo:r_hi].to(device, non_blocking=True)
                ref_y_t = ref_y_cpu[r_lo:r_hi].to(device, non_blocking=True)
                cat_dist = torch.zeros((bsz, r_hi - r_lo), dtype=torch.float32, device=device)
                for j in range(n_cat_te):
                    cat_dist += torch.abs(q_cat[:, None, j] - ref_cat_t[None, :, j]) * cat_w_t[j]
                num_diff = q_num[:, None, :] - ref_num_t[None, :, :]
                num_dist = ((num_diff * num_diff) * num_w_t.view(1, 1, -1)).sum(dim=2).float()
                dist_chunk = cat_dist + num_dist
                k_chunk = min(n_neighbors, dist_chunk.shape[1])
                chunk_dist, chunk_idx_local = torch.topk(dist_chunk, k=k_chunk, dim=1, largest=False, sorted=False)
                merged_dist = torch.cat([best_dist, chunk_dist], dim=1)
                merged_y = torch.cat([best_y, ref_y_t[chunk_idx_local]], dim=1)
                keep_idx = torch.topk(merged_dist, k=n_neighbors, dim=1, largest=False, sorted=False).indices
                best_dist = torch.gather(merged_dist, 1, keep_idx)
                best_y = torch.gather(merged_y, 1, keep_idx.unsqueeze(-1).expand(-1, -1, N_CLASSES))
            w = 1.0 / (best_dist + DIST_EPS)
            p = (w[:, :, None] * best_y).sum(dim=1) / w.sum(dim=1, keepdim=True)
            if APPLY_PRIOR_CALIBRATION:
                p = torch.clamp(p * ratio_t[None, :], 0.0, 1.0)
            p = p / (p.sum(dim=1, keepdim=True) + 1e-8)
            preds_scheme[q_lo:q_hi] = p.detach().cpu().numpy()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        out[:, s_idx * N_CLASSES:(s_idx + 1) * N_CLASSES] = preds_scheme
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return pd.DataFrame(out, columns=col_names)


# ------------------------------------------------------------------ global (original-data) part
def build_global():
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = train[TARGET].map(TARGET_MAP).astype(np.int8)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype(np.int8)
    y = train[TARGET].values.astype(np.int8)
    y_orig = orig[TARGET].values.astype(np.int8)

    for df in (train, test, orig):
        SM = df['Soil_Moisture'].astype(float)
        RM = df['Rainfall_mm'].astype(float)
        TC = df['Temperature_C'].astype(float)
        WS = df['Wind_Speed_kmh'].astype(float)
        CGS = df['Crop_Growth_Stage'].astype(str)
        MU = df['Mulching_Used'].astype(str)
        score = ((SM < 25) * 2 + (RM < 300) * 2 + (TC > 30) + (WS > 10)
                 - (CGS == 'Harvest') * 2 - (CGS == 'Sowing') * 2 - (MU == 'Yes'))
        df['formula_score'] = score.astype(np.float32)
        df['is_low'] = (score <= 0).astype(np.float32)
        df['is_high'] = (score > 3).astype(np.float32)
    for df in (train, test, orig):
        for c in NUMERIC_COLS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float32)
    for c in CATEGORICAL_COLS:
        all_vals = pd.Index(pd.concat([train[c], test[c], orig[c]], axis=0).astype(str).unique())
        for df in (train, test, orig):
            df[c] = pd.Categorical(df[c].astype(str), categories=all_vals)

    g = dict(train=train, test=test, y=y)
    g['train_cat_codes'] = np.stack([train[c].cat.codes.values for c in CATEGORICAL_COLS], axis=1).astype(np.int16)
    g['test_cat_codes'] = np.stack([test[c].cat.codes.values for c in CATEGORICAL_COLS], axis=1).astype(np.int16)
    orig_cat_codes = np.stack([orig[c].cat.codes.values for c in CATEGORICAL_COLS], axis=1).astype(np.int16)
    g['train_num'] = train[NUMERIC_COLS].values.astype(np.float32)
    g['test_num'] = test[NUMERIC_COLS].values.astype(np.float32)
    orig_num = orig[NUMERIC_COLS].values.astype(np.float32)
    orig_num_scaled, num_mu, num_sd = standardize_from_reference(orig_num, orig_num)
    train_num_scaled = ((g['train_num'] - num_mu) / num_sd).astype(np.float32)
    test_num_scaled = ((g['test_num'] - num_mu) / num_sd).astype(np.float32)
    orig_y_onehot = np.stack([(y_orig == k).astype(np.float32) for k in range(N_CLASSES)], axis=1)
    train_class_freq = np.array([(y == k).mean() for k in range(N_CLASSES)], dtype=np.float32)
    orig_class_freq = np.array([(y_orig == k).mean() for k in range(N_CLASSES)], dtype=np.float32)

    log('original-data KNN: random schemes')
    all_random_schemes = build_random_weight_schemes(N_RANDOM_SCHEMES, n_cat=len(CATEGORICAL_COLS),
                                                     n_num=len(NUMERIC_COLS), seed=SEED)
    all_random_names = prefixed_colnames_3cls('knnprior', list(range(N_RANDOM_SCHEMES)))
    knn_random_train = build_knn_random_gpu_full(orig_cat_codes, orig_num_scaled, orig_y_onehot,
                                                 g['train_cat_codes'], train_num_scaled, all_random_schemes,
                                                 all_random_names, KNN_K, batch_size=GPU_BATCH_SIZE_ORIG)
    knn_random_test = build_knn_random_gpu_full(orig_cat_codes, orig_num_scaled, orig_y_onehot,
                                                g['test_cat_codes'], test_num_scaled, all_random_schemes,
                                                all_random_names, KNN_K, batch_size=GPU_BATCH_SIZE_ORIG)

    log('original-data KNN: smart schemes')
    te_maps_orig, te_global_means_orig = build_target_encoding_maps(orig)
    orig_cat_te = make_target_rate_matrix(orig, te_maps_orig, te_global_means_orig)
    train_cat_te = make_target_rate_matrix(train, te_maps_orig, te_global_means_orig)
    test_cat_te = make_target_rate_matrix(test, te_maps_orig, te_global_means_orig)
    orig_importance = fit_orig_importance_model(orig)
    base_cat_w_exp = np.repeat(orig_importance[CATEGORICAL_COLS].values.astype(np.float32), N_CLASSES) / N_CLASSES
    base_num_w = orig_importance[NUMERIC_COLS].values.astype(np.float32)
    all_smart_schemes = build_smart_weight_schemes(base_cat_w_exp, base_num_w, N_SMART_SCHEMES, seed=SEED)
    all_smart_names = prefixed_colnames_3cls('knnsmart', list(range(N_SMART_SCHEMES)))
    knn_smart_train = build_knn_smart_gpu_full(orig_cat_te, orig_num_scaled, orig_y_onehot, train_cat_te,
                                               train_num_scaled, train_class_freq, orig_class_freq, all_smart_schemes,
                                               all_smart_names, KNN_K, batch_size=GPU_BATCH_SIZE_ORIG)
    knn_smart_test = build_knn_smart_gpu_full(orig_cat_te, orig_num_scaled, orig_y_onehot, test_cat_te,
                                              test_num_scaled, train_class_freq, orig_class_freq, all_smart_schemes,
                                              all_smart_names, KNN_K, batch_size=GPU_BATCH_SIZE_ORIG)

    rng = np.random.default_rng(RUN_SEED)
    selected_random_idx = choose_diverse_indices(rng, N_RANDOM_SCHEMES, N_SELECT_RANDOM, n_buckets=10)
    selected_smart_idx = choose_diverse_indices(rng, N_SMART_SCHEMES, N_SELECT_SMART, n_buckets=10)
    selected_random_cols = prefixed_colnames_3cls('knnprior', selected_random_idx)
    selected_smart_cols = prefixed_colnames_3cls('knnsmart', selected_smart_idx)
    selected_random_bases = [f'knnprior_{i:03d}' for i in selected_random_idx]
    selected_smart_bases = [f'knnsmart_{i:03d}' for i in selected_smart_idx]
    g['rand_w'] = dirichlet_weights(rng, len(selected_random_idx), alpha=0.7)
    g['smart_w'] = dirichlet_weights(rng, len(selected_smart_idx), alpha=0.7)
    g['selected_random_schemes'] = [all_random_schemes[i] for i in selected_random_idx]
    g['selected_smart_schemes'] = [all_smart_schemes[i] for i in selected_smart_idx]
    g['train_random_names'] = prefixed_colnames_3cls('trknnprior', selected_random_idx)
    g['train_smart_names'] = prefixed_colnames_3cls('trknnsmart', selected_smart_idx)
    g['train_random_bases'] = [f'trknnprior_{i:03d}' for i in selected_random_idx]
    g['train_smart_bases'] = [f'trknnsmart_{i:03d}' for i in selected_smart_idx]
    log(f'selected random schemes {[int(i) for i in selected_random_idx]}, smart {[int(i) for i in selected_smart_idx]}')

    R_tr = knn_random_train[selected_random_cols]
    R_te = knn_random_test[selected_random_cols]
    S_tr = knn_smart_train[selected_smart_cols]
    S_te = knn_smart_test[selected_smart_cols]
    g['Rlog_tr'] = safe_logit(R_tr.values, EPS).astype(np.float32)
    g['Rlog_te'] = safe_logit(R_te.values, EPS).astype(np.float32)
    g['Slog_tr'] = safe_logit(S_tr.values, EPS).astype(np.float32)
    g['Slog_te'] = safe_logit(S_te.values, EPS).astype(np.float32)
    g['Alog_tr'] = np.concatenate([g['Rlog_tr'], g['Slog_tr']], axis=1)
    g['Alog_te'] = np.concatenate([g['Rlog_te'], g['Slog_te']], axis=1)

    g['rand_wmean_tr'] = weighted_avg_block_3cls(R_tr, selected_random_bases, g['rand_w'])
    g['rand_wmean_te'] = weighted_avg_block_3cls(R_te, selected_random_bases, g['rand_w'])
    g['smart_wmean_tr'] = weighted_avg_block_3cls(S_tr, selected_smart_bases, g['smart_w'])
    g['smart_wmean_te'] = weighted_avg_block_3cls(S_te, selected_smart_bases, g['smart_w'])
    g['blend_wmean_tr'] = 0.5 * g['rand_wmean_tr'] + 0.5 * g['smart_wmean_tr']
    g['blend_wmean_te'] = 0.5 * g['rand_wmean_te'] + 0.5 * g['smart_wmean_te']

    X_train_base = train[ALL_FEATURE_COLS].copy()
    X_test_base = test[ALL_FEATURE_COLS].copy()
    for c in NUMERIC_COLS + FORMULA_COLS:
        X_train_base[c] = X_train_base[c].astype(np.float32)
        X_test_base[c] = X_test_base[c].astype(np.float32)
    for c in CATEGORICAL_COLS:
        X_train_base[c] = X_train_base[c].astype('category')
        X_test_base[c] = X_test_base[c].astype('category')
    g['X_train_base'], g['X_test_base'] = X_train_base, X_test_base
    return g


# ------------------------------------------------------------------ one outer fold
def run_fold(g, k, tr_idx, va_idx):
    fold = k + 1  # 1-based fold number (inner split seed)
    train, test, y = g['train'], g['test'], g['y']
    y_tr_outer, y_va_outer = y[tr_idx], y[va_idx]
    y_tr_onehot = np.stack([(y_tr_outer == c).astype(np.float32) for c in range(N_CLASSES)], axis=1)
    tr_class_freq = np.array([(y_tr_outer == c).mean() for c in range(N_CLASSES)], dtype=np.float32)
    sel_r, sel_s = g['selected_random_schemes'], g['selected_smart_schemes']
    names_r, names_s = g['train_random_names'], g['train_smart_names']
    bases_r, bases_s = g['train_random_bases'], g['train_smart_bases']

    # LR on the original-KNN logits: held-out, test and (in sample) outer-training rows
    p_va_rand, p_te_rand, p_tr_rand = fit_lr(g['Rlog_tr'][tr_idx], y_tr_outer, g['Rlog_tr'][va_idx], g['Rlog_te'], g['Rlog_tr'][tr_idx])
    p_va_smart, p_te_smart, p_tr_smart = fit_lr(g['Slog_tr'][tr_idx], y_tr_outer, g['Slog_tr'][va_idx], g['Slog_te'], g['Slog_tr'][tr_idx])
    p_va_all, p_te_all, p_tr_all = fit_lr(g['Alog_tr'][tr_idx], y_tr_outer, g['Alog_tr'][va_idx], g['Alog_te'], g['Alog_tr'][tr_idx])

    # train-data KNN for the held-out and test rows (neighbours = outer-training rows)
    outer_train_df = train.iloc[tr_idx].copy().reset_index(drop=True)
    outer_valid_df = train.iloc[va_idx].copy().reset_index(drop=True)
    outer_train_cat, outer_valid_cat = g['train_cat_codes'][tr_idx], g['train_cat_codes'][va_idx]
    outer_train_num, outer_valid_num = g['train_num'][tr_idx], g['train_num'][va_idx]
    outer_train_num_s, outer_valid_num_s, outer_test_num_s = standardize_ref_and_queries(
        outer_train_num, outer_valid_num, g['test_num'])

    trR_va = build_knn_random_gpu_chunked(outer_train_cat, outer_train_num_s, y_tr_onehot, outer_valid_cat,
                                          outer_valid_num_s, sel_r, names_r, TRAIN_KNN_K, batch_size=BATCH_SIZE_OUTER,
                                          ref_chunk_size=REF_CHUNK_SIZE, prefix=f'train-KNN random, valid, fold {k}')
    trR_te = build_knn_random_gpu_chunked(outer_train_cat, outer_train_num_s, y_tr_onehot, g['test_cat_codes'],
                                          outer_test_num_s, sel_r, names_r, TRAIN_KNN_K, batch_size=BATCH_SIZE_OUTER,
                                          ref_chunk_size=REF_CHUNK_SIZE, prefix=f'train-KNN random, test, fold {k}')
    te_maps_outer, te_global_outer = build_target_encoding_maps(outer_train_df)
    outer_train_cat_te = make_target_rate_matrix(outer_train_df, te_maps_outer, te_global_outer)
    outer_valid_cat_te = make_target_rate_matrix(outer_valid_df, te_maps_outer, te_global_outer)
    outer_test_cat_te = make_target_rate_matrix(test, te_maps_outer, te_global_outer)
    trS_va = build_knn_smart_gpu_chunked(outer_train_cat_te, outer_train_num_s, y_tr_onehot, outer_valid_cat_te,
                                         outer_valid_num_s, tr_class_freq, tr_class_freq, sel_s, names_s, TRAIN_KNN_K,
                                         batch_size=BATCH_SIZE_OUTER, ref_chunk_size=REF_CHUNK_SIZE,
                                         prefix=f'train-KNN smart, valid, fold {k}')
    trS_te = build_knn_smart_gpu_chunked(outer_train_cat_te, outer_train_num_s, y_tr_onehot, outer_test_cat_te,
                                         outer_test_num_s, tr_class_freq, tr_class_freq, sel_s, names_s, TRAIN_KNN_K,
                                         batch_size=BATCH_SIZE_OUTER, ref_chunk_size=REF_CHUNK_SIZE,
                                         prefix=f'train-KNN smart, test, fold {k}')
    trR_va_wmean = weighted_avg_block_3cls(trR_va, bases_r, g['rand_w'])
    trS_va_wmean = weighted_avg_block_3cls(trS_va, bases_s, g['smart_w'])
    trR_te_wmean = weighted_avg_block_3cls(trR_te, bases_r, g['rand_w'])
    trS_te_wmean = weighted_avg_block_3cls(trS_te, bases_s, g['smart_w'])

    # train-data KNN for the outer-training rows: out-of-fold over 5 inner folds
    inner_R_oof = np.zeros((len(tr_idx), len(names_r)), dtype=np.float32)
    inner_S_oof = np.zeros((len(tr_idx), len(names_s)), dtype=np.float32)
    inner_skf = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + fold)
    for inner_fold, (itr_rel, iva_rel) in enumerate(inner_skf.split(np.zeros(len(tr_idx)), y_tr_outer), 1):
        log(f'  inner fold {inner_fold}/{INNER_SPLITS} of outer fold {k}')
        inner_ref_df = outer_train_df.iloc[itr_rel].copy().reset_index(drop=True)
        inner_val_df = outer_train_df.iloc[iva_rel].copy().reset_index(drop=True)
        ref_yi = y_tr_outer[itr_rel]
        ref_y_oh = np.stack([(ref_yi == c).astype(np.float32) for c in range(N_CLASSES)], axis=1)
        ref_cf = np.array([(ref_yi == c).mean() for c in range(N_CLASSES)], dtype=np.float32)
        ref_num_s, val_num_s = standardize_ref_and_queries(outer_train_num[itr_rel], outer_train_num[iva_rel])
        inner_R = build_knn_random_gpu_chunked(outer_train_cat[itr_rel], ref_num_s, ref_y_oh, outer_train_cat[iva_rel],
                                               val_num_s, sel_r, names_r, TRAIN_KNN_K, batch_size=BATCH_SIZE_INNER,
                                               ref_chunk_size=REF_CHUNK_SIZE, prefix='  train-KNN random, inner')
        te_maps_inner, te_global_inner = build_target_encoding_maps(inner_ref_df)
        ref_cat_te_i = make_target_rate_matrix(inner_ref_df, te_maps_inner, te_global_inner)
        val_cat_te_i = make_target_rate_matrix(inner_val_df, te_maps_inner, te_global_inner)
        inner_S = build_knn_smart_gpu_chunked(ref_cat_te_i, ref_num_s, ref_y_oh, val_cat_te_i, val_num_s, ref_cf, ref_cf,
                                              sel_s, names_s, TRAIN_KNN_K, batch_size=BATCH_SIZE_INNER,
                                              ref_chunk_size=REF_CHUNK_SIZE, prefix='  train-KNN smart, inner')
        inner_R_oof[iva_rel] = inner_R[names_r].values.astype(np.float32)
        inner_S_oof[iva_rel] = inner_S[names_s].values.astype(np.float32)
        del inner_R, inner_S, inner_ref_df, inner_val_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    inner_R_oof_df = pd.DataFrame(inner_R_oof, columns=names_r)
    inner_S_oof_df = pd.DataFrame(inner_S_oof, columns=names_s)

    # LR on the train-KNN logits (fitted on the inner out-of-fold values of the outer-training rows)
    trR_va_log = safe_logit(trR_va[names_r].values, EPS).astype(np.float32)
    trR_te_log = safe_logit(trR_te[names_r].values, EPS).astype(np.float32)
    inner_R_log = safe_logit(inner_R_oof_df.values, EPS).astype(np.float32)
    trS_va_log = safe_logit(trS_va[names_s].values, EPS).astype(np.float32)
    trS_te_log = safe_logit(trS_te[names_s].values, EPS).astype(np.float32)
    inner_S_log = safe_logit(inner_S_oof_df.values, EPS).astype(np.float32)
    inner_A_log = np.concatenate([inner_R_log, inner_S_log], axis=1).astype(np.float32)
    trA_va_log = np.concatenate([trR_va_log, trS_va_log], axis=1).astype(np.float32)
    trA_te_log = np.concatenate([trR_te_log, trS_te_log], axis=1).astype(np.float32)
    p_va_tr_rand, p_te_tr_rand, p_tr_tr_rand = fit_lr(inner_R_log, y_tr_outer, trR_va_log, trR_te_log, inner_R_log)
    p_va_tr_smart, p_te_tr_smart, p_tr_tr_smart = fit_lr(inner_S_log, y_tr_outer, trS_va_log, trS_te_log, inner_S_log)
    p_va_tr_all, p_te_tr_all, p_tr_tr_all = fit_lr(inner_A_log, y_tr_outer, trA_va_log, trA_te_log, inner_A_log)

    # the 10 stack groups for the outer-training, held-out and test rows
    trknn_rand_tr = weighted_avg_block_3cls(inner_R_oof_df, bases_r, g['rand_w'])
    trknn_smart_tr = weighted_avg_block_3cls(inner_S_oof_df, bases_s, g['smart_w'])
    trknn_blend_tr = 0.5 * trknn_rand_tr + 0.5 * trknn_smart_tr
    meta_blend_tr = (p_tr_rand + p_tr_smart + p_tr_all) / 3.0
    meta_tr_blend_tr = (p_tr_tr_rand + p_tr_tr_smart + p_tr_tr_all) / 3.0
    stack_tr = {
        'meta_blend_lr': meta_blend_tr,
        'meta_tr_blend_lr': meta_tr_blend_tr,
        'knn_rand_wmean': g['rand_wmean_tr'][tr_idx],
        'knn_smart_wmean': g['smart_wmean_tr'][tr_idx],
        'knn_blend_mean': g['blend_wmean_tr'][tr_idx],
        'trknn_rand_wmean': trknn_rand_tr,
        'trknn_smart_wmean': trknn_smart_tr,
        'trknn_blend_mean': trknn_blend_tr,
        'knn_full_blend_mean': 0.5 * g['blend_wmean_tr'][tr_idx] + 0.5 * trknn_blend_tr,
        'meta_full_blend_lr': 0.5 * meta_blend_tr + 0.5 * meta_tr_blend_tr,
    }
    meta_blend_va = (p_va_rand + p_va_smart + p_va_all) / 3.0
    meta_tr_blend_va = (p_va_tr_rand + p_va_tr_smart + p_va_tr_all) / 3.0
    trknn_blend_va = 0.5 * trR_va_wmean + 0.5 * trS_va_wmean
    stack_va = {
        'meta_blend_lr': meta_blend_va,
        'meta_tr_blend_lr': meta_tr_blend_va,
        'knn_rand_wmean': g['rand_wmean_tr'][va_idx],
        'knn_smart_wmean': g['smart_wmean_tr'][va_idx],
        'knn_blend_mean': g['blend_wmean_tr'][va_idx],
        'trknn_rand_wmean': trR_va_wmean,
        'trknn_smart_wmean': trS_va_wmean,
        'trknn_blend_mean': trknn_blend_va,
        'knn_full_blend_mean': 0.5 * g['blend_wmean_tr'][va_idx] + 0.5 * trknn_blend_va,
        'meta_full_blend_lr': 0.5 * meta_blend_va + 0.5 * meta_tr_blend_va,
    }
    meta_blend_te = (p_te_rand + p_te_smart + p_te_all) / 3.0
    meta_tr_blend_te = (p_te_tr_rand + p_te_tr_smart + p_te_tr_all) / 3.0
    trknn_blend_te = 0.5 * trR_te_wmean + 0.5 * trS_te_wmean
    stack_te = {
        'meta_blend_lr': meta_blend_te,
        'meta_tr_blend_lr': meta_tr_blend_te,
        'knn_rand_wmean': g['rand_wmean_te'],
        'knn_smart_wmean': g['smart_wmean_te'],
        'knn_blend_mean': g['blend_wmean_te'],
        'trknn_rand_wmean': trR_te_wmean,
        'trknn_smart_wmean': trS_te_wmean,
        'trknn_blend_mean': trknn_blend_te,
        'knn_full_blend_mean': 0.5 * g['blend_wmean_te'] + 0.5 * trknn_blend_te,
        'meta_full_blend_lr': 0.5 * meta_blend_te + 0.5 * meta_tr_blend_te,
    }

    # base columns + stack columns (class-major order: for each class the 10 groups)
    X_tr_fold = g['X_train_base'].iloc[tr_idx].reset_index(drop=True).copy()
    X_va_fold = g['X_train_base'].iloc[va_idx].reset_index(drop=True).copy()
    X_te_fold = g['X_test_base'].copy()
    for c in range(N_CLASSES):
        for grp in STACK_GROUPS:
            X_tr_fold[f'{grp}_k{c}'] = np.asarray(stack_tr[grp][:, c], dtype=np.float32)
            X_va_fold[f'{grp}_k{c}'] = np.asarray(stack_va[grp][:, c], dtype=np.float32)
            X_te_fold[f'{grp}_k{c}'] = np.asarray(stack_te[grp][:, c], dtype=np.float32)

    sw = compute_sample_weight('balanced', y_tr_outer).astype(np.float32)
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr_fold, y_tr_outer, sample_weight=sw, eval_set=[(X_va_fold, y_va_outer)], verbose=500)
    p_va = model.predict_proba(X_va_fold).astype(np.float32)
    p_te = model.predict_proba(X_te_fold).astype(np.float32)
    log(f'{PAIR} fold {k}: balanced accuracy {balanced_accuracy_score(y_va_outer, p_va.argmax(1)):.6f}, '
        f'best iteration {model.best_iteration}')
    return p_va, p_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    out = model_dir(FOLDER)
    seed_everything(SEED)

    g = build_global()
    y = g['y']
    fold_id = folds(y)
    for k in fold_list(args.folds):
        if have_part(out, PAIR, k):
            log(f'{PAIR} fold {k}: done')
            continue
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        p_va, p_te = run_fold(g, k, tr_idx, va_idx)
        save_part(out, PAIR, k, p_va, p_te)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assemble(out, PAIR, fold_id, y)


if __name__ == '__main__':
    main()
