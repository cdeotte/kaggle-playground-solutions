"""Weighted-distance KNN on the GPU, shared by select_knn_schemes.py and train_knn_nested_xgb.py.

Rows are compared on the 16 categoricals and the 3 numerics (standardized on the reference rows):
  random scheme  d = sum_j cat_w[j] * [category j differs] + sum_k num_w[k] * (numeric k difference)^2
                 prior = mean label of the k nearest reference rows
  smart scheme   d = sum_j cat_w[j] * |churn rate of the query's category j - churn rate of the reference's category j|
                     + sum_k num_w[k] * (numeric k difference)^2   (churn rates per category of the reference rows)
                 prior = inverse-distance weighted mean label of the k nearest reference rows (weights 1 / (d + 1e-6)),
                         times (train churn rate / reference churn rate), clipped to [0, 1] (the factor is 1 when the
                         reference rows are train rows)
30 random schemes come from 10 generators (scheme index mod 10); 30 smart schemes from 10 templates around the feature
importances of an XGBoost fitted on the original rows, each with a 15% multiplicative jitter. All weights are rescaled
to mean 1. Brute force: the full distance matrix of a query batch against all reference rows (small reference sets) or
against chunks of 24,576 reference rows with a running top-k (large reference sets)."""
import random
import time

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from common.data import CATEGORICALS, NUMERICS, TARGET, read_original, read_test, read_train

SEED = 42
CATEGORICAL_COLS = list(CATEGORICALS)
NUMERIC_COLS = list(NUMERICS)
ALL_FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS
N_RANDOM_SCHEMES = 30
N_SMART_SCHEMES = 30

PIN_MEMORY = True
DIST_EPS = 1e-6
APPLY_PRIOR_CALIBRATION = True

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
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    device='cuda',
    enable_categorical=True,
    max_cat_to_onehot=8,
    random_state=SEED,
    verbosity=0,
)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_frames():
    """train / test / original with the target as int8 (train, original), the 3 numerics as float32 (blanks filled with
    the original medians) and the 16 categoricals as pandas categories with levels shared by the three frames. The
    original keeps all 7,043 rows."""
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = train[TARGET].map({'Yes': 1, 'No': 0}).astype(np.int8)
    orig[TARGET] = orig[TARGET].map({'Yes': 1, 'No': 0}).astype(np.int8)

    for df in (train, test, orig):
        for c in NUMERIC_COLS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float32)
    num_fill = orig[NUMERIC_COLS].median()
    for df in (train, test, orig):
        for c in NUMERIC_COLS:
            df[c] = df[c].fillna(num_fill[c]).astype(np.float32)

    for c in CATEGORICAL_COLS:
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)
        orig[c] = orig[c].astype(str)
        all_vals = pd.Index(pd.concat([train[c], test[c], orig[c]], axis=0).astype(str).unique())
        train[c] = pd.Categorical(train[c], categories=all_vals)
        test[c] = pd.Categorical(test[c], categories=all_vals)
        orig[c] = pd.Categorical(orig[c], categories=all_vals)
    return train, test, orig


def cat_codes(df):
    return np.stack([df[c].cat.codes.values for c in CATEGORICAL_COLS], axis=1).astype(np.int16)


def base_matrix(df):
    """The 19 raw columns for XGBoost: categoricals as pandas categories, numerics as float32."""
    X = df[ALL_FEATURE_COLS].copy()
    for c in NUMERIC_COLS:
        X[c] = X[c].astype(np.float32)
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype('category')
    return X


def standardize_from_reference(x_ref, x):
    mu = np.nanmean(x_ref, axis=0)
    sd = np.nanstd(x_ref, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def standardize_ref_and_queries(x_ref, *queries):
    x_ref_scaled, mu, sd = standardize_from_reference(x_ref, x_ref)
    return [x_ref_scaled] + [((q - mu) / sd).astype(np.float32) for q in queries]


def prefixed_colnames(prefix, indices):
    return [f'{prefix}_{i:03d}' for i in indices]


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
            num_w *= np.array([1.4, 1.2, 1.2], dtype=np.float32)
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
    """XGBoost on the original rows (19 raw columns); importances normalized to sum 1."""
    Xo = orig_df[ALL_FEATURE_COLS].copy()
    yo = orig_df[TARGET].values.astype(np.int8)
    for c in CATEGORICAL_COLS:
        Xo[c] = Xo[c].astype('category')
    for c in NUMERIC_COLS:
        Xo[c] = Xo[c].astype(np.float32)
    model = XGBClassifier(**ORIG_XGB_PARAMS)
    model.fit(Xo, yo, verbose=False)
    imp = pd.Series(model.feature_importances_, index=ALL_FEATURE_COLS).fillna(0.0)
    if imp.sum() <= 0:
        imp[:] = 1.0
    return imp / imp.sum()


def build_target_encoding_maps(df_ref):
    """Churn rate of every level of every categorical in df_ref (NaN for levels absent from df_ref)."""
    maps = {}
    global_mean = float(df_ref[TARGET].mean())
    for c in CATEGORICAL_COLS:
        maps[c] = df_ref.groupby(c, observed=False)[TARGET].mean().to_dict()
    return maps, global_mean


def make_target_rate_matrix(df, te_maps, global_mean):
    mats = []
    for c in CATEGORICAL_COLS:
        mats.append(df[c].astype(str).map(te_maps[c]).fillna(global_mean).astype(np.float32).values)
    return np.stack(mats, axis=1).astype(np.float32)


def build_smart_weight_schemes(base_cat_w, base_num_w, n_schemes, seed=42):
    rng = np.random.default_rng(seed)
    base_cat_w = np.asarray(base_cat_w, dtype=np.float32)
    base_num_w = np.asarray(base_num_w, dtype=np.float32)
    z = np.concatenate([base_cat_w, base_num_w])
    z = z / np.mean(z)
    base_cat_w = z[:len(base_cat_w)].astype(np.float32)
    base_num_w = z[len(base_cat_w):].astype(np.float32)

    template_modes = ['base', 'sqrt', 'square', 'cat_heavy', 'num_heavy',
                      'flat_mix', 'tenure_boost', 'charges_boost', 'top2_boost', 'top4_boost']
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
        elif mode == 'tenure_boost':
            num_w[0] *= 2.20
        elif mode == 'charges_boost':
            num_w[1] *= 1.80
            num_w[2] *= 1.80
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


# ------------------------------------------------------------------ GPU KNN, small reference set (original rows)
@torch.no_grad()
def knn_random_full(ref_cat_codes, ref_num_scaled, ref_y, query_cat_codes, query_num_scaled, schemes, col_names,
                    n_neighbors, batch_size=4096, fp16=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_dtype = torch.float16 if fp16 else torch.float32
    ref_cat_t = torch.as_tensor(ref_cat_codes, dtype=torch.int16, device=device)
    ref_num_t = torch.as_tensor(ref_num_scaled, dtype=num_dtype, device=device)
    ref_y_t = torch.as_tensor(ref_y, dtype=torch.float32, device=device)
    n_query = query_cat_codes.shape[0]
    assert n_neighbors <= ref_cat_codes.shape[0]
    out = np.empty((n_query, len(schemes)), dtype=np.float32)

    query_cat_cpu = torch.as_tensor(query_cat_codes, dtype=torch.int16)
    query_num_cpu = torch.as_tensor(query_num_scaled, dtype=num_dtype)
    if PIN_MEMORY and device.type == 'cuda':
        query_cat_cpu, query_num_cpu = query_cat_cpu.pin_memory(), query_num_cpu.pin_memory()

    for s_idx, (cat_w_np, num_w_np) in enumerate(schemes):
        cat_w_t = torch.as_tensor(cat_w_np, dtype=torch.float32, device=device)
        num_w_t = torch.as_tensor(num_w_np, dtype=num_dtype, device=device)
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
            out[lo:hi, s_idx] = ref_y_t[nn_idx].mean(dim=1).detach().cpu().numpy().astype(np.float32)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return pd.DataFrame(out, columns=col_names)


@torch.no_grad()
def knn_smart_full(ref_cat_te, ref_num_scaled, ref_y, query_cat_te, query_num_scaled, train_target_mean,
                   ref_target_mean, schemes, col_names, n_neighbors, batch_size=4096, fp16=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_dtype = torch.float16 if fp16 else torch.float32
    ref_cat_te_t = torch.as_tensor(ref_cat_te, dtype=torch.float32, device=device)
    ref_num_t = torch.as_tensor(ref_num_scaled, dtype=num_dtype, device=device)
    ref_y_t = torch.as_tensor(ref_y, dtype=torch.float32, device=device)
    n_query = query_cat_te.shape[0]
    assert n_neighbors <= ref_cat_te.shape[0]
    ratio = float(train_target_mean / max(ref_target_mean, 1e-8))
    out = np.empty((n_query, len(schemes)), dtype=np.float32)

    query_cat_cpu = torch.as_tensor(query_cat_te, dtype=torch.float32)
    query_num_cpu = torch.as_tensor(query_num_scaled, dtype=num_dtype)
    if PIN_MEMORY and device.type == 'cuda':
        query_cat_cpu, query_num_cpu = query_cat_cpu.pin_memory(), query_num_cpu.pin_memory()

    for s_idx, (cat_w_np, num_w_np, _mode) in enumerate(schemes):
        cat_w_t = torch.as_tensor(cat_w_np, dtype=torch.float32, device=device)
        num_w_t = torch.as_tensor(num_w_np, dtype=num_dtype, device=device)
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
            p = ((w * nn_y).sum(dim=1) / w.sum(dim=1)).float()
            if APPLY_PRIOR_CALIBRATION:
                p = torch.clamp(p * ratio, 0.0, 1.0)
            out[lo:hi, s_idx] = p.detach().cpu().numpy().astype(np.float32)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return pd.DataFrame(out, columns=col_names)


# ------------------------------------------------------------------ GPU KNN, large reference set (training rows)
@torch.no_grad()
def knn_random_chunked(ref_cat_codes, ref_num_scaled, ref_y, query_cat_codes, query_num_scaled, schemes, col_names,
                       n_neighbors, batch_size=192, ref_chunk_size=24576, fp16=True, prefix=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_dtype = torch.float16 if fp16 else torch.float32
    n_query, n_ref, n_cat = query_cat_codes.shape[0], ref_cat_codes.shape[0], ref_cat_codes.shape[1]
    assert n_neighbors <= n_ref
    log(f'{prefix}: n_query={n_query:,} n_ref={n_ref:,} k={n_neighbors}')

    ref_cat_cpu = torch.as_tensor(ref_cat_codes, dtype=torch.int16)
    ref_num_cpu = torch.as_tensor(ref_num_scaled, dtype=num_dtype)
    ref_y_cpu = torch.as_tensor(ref_y, dtype=torch.float32)
    query_cat_cpu = torch.as_tensor(query_cat_codes, dtype=torch.int16)
    query_num_cpu = torch.as_tensor(query_num_scaled, dtype=num_dtype)
    if PIN_MEMORY and device.type == 'cuda':
        ref_cat_cpu, ref_num_cpu, ref_y_cpu = ref_cat_cpu.pin_memory(), ref_num_cpu.pin_memory(), ref_y_cpu.pin_memory()
        query_cat_cpu, query_num_cpu = query_cat_cpu.pin_memory(), query_num_cpu.pin_memory()

    out = np.empty((n_query, len(schemes)), dtype=np.float32)
    for s_idx, (cat_w_np, num_w_np) in enumerate(schemes):
        cat_w_t = torch.as_tensor(cat_w_np, dtype=torch.float32, device=device)
        num_w_t = torch.as_tensor(num_w_np, dtype=num_dtype, device=device)
        for q_lo in range(0, n_query, batch_size):
            q_hi = min(q_lo + batch_size, n_query)
            q_cat = query_cat_cpu[q_lo:q_hi].to(device, non_blocking=True)
            q_num = query_num_cpu[q_lo:q_hi].to(device, non_blocking=True)
            bsz = q_cat.shape[0]
            best_dist = torch.full((bsz, n_neighbors), float('inf'), dtype=torch.float32, device=device)
            best_y = torch.zeros((bsz, n_neighbors), dtype=torch.float32, device=device)
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
                best_y = torch.gather(merged_y, 1, keep_idx)
            out[q_lo:q_hi, s_idx] = best_y.mean(dim=1).detach().cpu().numpy().astype(np.float32)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    return pd.DataFrame(out, columns=col_names)


@torch.no_grad()
def knn_smart_chunked(ref_cat_te, ref_num_scaled, ref_y, query_cat_te, query_num_scaled, train_target_mean,
                      ref_target_mean, schemes, col_names, n_neighbors, batch_size=192, ref_chunk_size=24576,
                      fp16=True, prefix=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_dtype = torch.float16 if fp16 else torch.float32
    n_query, n_ref, n_cat = query_cat_te.shape[0], ref_cat_te.shape[0], ref_cat_te.shape[1]
    assert n_neighbors <= n_ref
    log(f'{prefix}: n_query={n_query:,} n_ref={n_ref:,} k={n_neighbors}')

    ref_cat_cpu = torch.as_tensor(ref_cat_te, dtype=torch.float32)
    ref_num_cpu = torch.as_tensor(ref_num_scaled, dtype=num_dtype)
    ref_y_cpu = torch.as_tensor(ref_y, dtype=torch.float32)
    query_cat_cpu = torch.as_tensor(query_cat_te, dtype=torch.float32)
    query_num_cpu = torch.as_tensor(query_num_scaled, dtype=num_dtype)
    if PIN_MEMORY and device.type == 'cuda':
        ref_cat_cpu, ref_num_cpu, ref_y_cpu = ref_cat_cpu.pin_memory(), ref_num_cpu.pin_memory(), ref_y_cpu.pin_memory()
        query_cat_cpu, query_num_cpu = query_cat_cpu.pin_memory(), query_num_cpu.pin_memory()

    ratio = float(train_target_mean / max(ref_target_mean, 1e-8))
    out = np.empty((n_query, len(schemes)), dtype=np.float32)
    for s_idx, (cat_w_np, num_w_np, _mode) in enumerate(schemes):
        cat_w_t = torch.as_tensor(cat_w_np, dtype=torch.float32, device=device)
        num_w_t = torch.as_tensor(num_w_np, dtype=num_dtype, device=device)
        for q_lo in range(0, n_query, batch_size):
            q_hi = min(q_lo + batch_size, n_query)
            q_cat = query_cat_cpu[q_lo:q_hi].to(device, non_blocking=True)
            q_num = query_num_cpu[q_lo:q_hi].to(device, non_blocking=True)
            bsz = q_cat.shape[0]
            best_dist = torch.full((bsz, n_neighbors), float('inf'), dtype=torch.float32, device=device)
            best_y = torch.zeros((bsz, n_neighbors), dtype=torch.float32, device=device)
            for r_lo in range(0, n_ref, ref_chunk_size):
                r_hi = min(r_lo + ref_chunk_size, n_ref)
                ref_cat_t = ref_cat_cpu[r_lo:r_hi].to(device, non_blocking=True)
                ref_num_t = ref_num_cpu[r_lo:r_hi].to(device, non_blocking=True)
                ref_y_t = ref_y_cpu[r_lo:r_hi].to(device, non_blocking=True)
                cat_dist = torch.zeros((bsz, r_hi - r_lo), dtype=torch.float32, device=device)
                for j in range(n_cat):
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
                best_y = torch.gather(merged_y, 1, keep_idx)
            w = 1.0 / (best_dist + DIST_EPS)
            p = ((w * best_y).sum(dim=1) / w.sum(dim=1)).float()
            if APPLY_PRIOR_CALIBRATION:
                p = torch.clamp(p * ratio, 0.0, 1.0)
            out[q_lo:q_hi, s_idx] = p.detach().cpu().numpy().astype(np.float32)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    return pd.DataFrame(out, columns=col_names)
