#!/usr/bin/env python
"""GPU XGBoost on numeric-value tokens with nested target encodings, original-data statistics, a denoising
autoencoder trained on the original rows, nearest original rows and a real-vs-synthetic discriminator.

Label-free columns, built once on the train, test and original rows (TotalCharges blank -> NaN):
- 5 charge relations: tenure x MonthlyCharges, MonthlyCharges / (tenure + 1), TotalCharges / (tenure + 1),
  TotalCharges - MonthlyCharges, TotalCharges / (MonthlyCharges + 1); a missing flag per raw numeric;
- 6 categorical pairs (Contract x InternetService, Contract x PaymentMethod, InternetService x TechSupport,
  InternetService x OnlineSecurity, InternetService x StreamingTV, PaperlessBilling x PaymentMethod);
- exact value tokens ('%.6f') of tenure, MonthlyCharges and TotalCharges and their 3 token pairs;
- 2 row fingerprints: the 3 tokens + Contract, InternetService, PaymentMethod ('core'), plus TechSupport,
  OnlineSecurity, PaperlessBilling ('wide').
That gives 29 string columns and 13 numeric columns (SeniorCitizen counts as numeric).
Per fold (all fitted on the fold's training rows):
- base matrix (42): ordinal codes of the string columns (unknown -> -1) and standardized numerics (missing -> median);
- for 20 keys (Contract, InternetService, PaymentMethod, TechSupport, OnlineSecurity, PaperlessBilling, the 6 pairs,
  the 3 tokens, the 3 token pairs, the 2 fingerprints): target encoding with smoothing 10 (out-of-fold over 5 inner
  StratifiedKFold folds, seed 42 + fold number 1-5, for the training rows; fitted on all training rows for the
  held-out, test and original rows), count and frequency in the training rows; original-data churn mean, count,
  frequency and a seen flag (80 columns); whether the core / wide fingerprint occurs in the original data;
- DAE on the original rows only: input = ordinal codes divided by their training maximum + standardized numerics (42);
  42 -> 128 (BatchNorm, ReLU, dropout 0.05) -> 128 (ReLU) -> 16 -> mirrored decoder; Gaussian noise 0.02 and 3%
  masking, SmoothL1 (beta 0.5), AdamW lr 1e-3, weight decay 1e-5, batch 1024, 30 epochs. Features: mean squared /
  mean absolute / max absolute / L2 reconstruction error, mean absolute error of the categorical and the numeric part,
  max absolute numeric error, and the 16 latent coordinates;
- nearest original rows (exact chunked GPU search) in the DAE input space and in the latent space: distance to the
  nearest, mean of the 3 and 5 nearest, latent density 1 / (mean 5-NN distance), the churn label of the nearest
  latent neighbour and how many of its string columns match the row (count and share);
- discriminator: XGBoost (depth 5, lr 0.05, up to 2000 trees, early stopping 200 on the first 50,000 of its own
  shuffled training rows) trained on all the columns above to tell the original rows (x6, label 1) from the fold's
  training rows (label 0); p(real), its logit and |p - 0.5|;
- 7 products: p(real) x the TE of the MonthlyCharges token / tenure x MonthlyCharges token pair / core fingerprint /
  mean original-seen flag of the tokens / nearest original churn label, TE(tenure token) x TE(MonthlyCharges token),
  DAE squared error x mean token TE.
Final model: XGBoost binary:logistic on the 227 columns (depth 5, lr 0.025, up to 4000 trees, min_child_weight 10,
subsample / colsample_bytree 0.88, lambda 3, GPU hist) with early stopping 200 on the held-out AUC, trained on the
fold's training rows only. Python / numpy / torch are seeded once (2026) before the folds, so the pair is reproduced
only when the 5 folds run in order in one process; both XGBoost models use seed 42. Test = mean of the 5 fold models.
Usage: python train_dae_knn_discriminator_xgb.py [--folds 0,1,2,3,4]"""
import argparse
import gc
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'dae_knn_discriminator_token_te_xgb'
MODEL_DIR = model_dir('neural_nets')
GLOBAL_SEED = 2026
SEED = 42
INNER_FOLDS = 5
TE_SMOOTH = 10.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# DAE
DAE_LATENT_DIM = 16
DAE_HIDDEN_DIM = 128
DAE_BATCH_SIZE = 1024
DAE_EPOCHS = 30
DAE_LR = 1e-3
DAE_WEIGHT_DECAY = 1e-5
DAE_NOISE_STD = 0.02
DAE_MASK_PROB = 0.03
DAE_DROPOUT = 0.05
DAE_PRED_BATCH_SIZE = 32768

# GPU KNN chunks
KNN_QUERY_CHUNK_BASE = 65536
KNN_REF_CHUNK_BASE = 8192
KNN_QUERY_CHUNK_LATENT = 131072
KNN_REF_CHUNK_LATENT = 8192

# discriminator: copies of the original rows on the "real" side
DISC_REAL_MULTIPLIER = 6
EARLY_STOPPING_ROUNDS = 200
XGB_N_JOBS = 0

DISC_PARAMS = dict(
    objective='binary:logistic', eval_metric='auc', max_depth=5, learning_rate=0.05, n_estimators=2000,
    min_child_weight=8, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.0, reg_lambda=2.0, gamma=0.0,
    random_state=SEED, enable_categorical=False, n_jobs=XGB_N_JOBS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    tree_method='hist', device='cuda',
)
FINAL_PARAMS = dict(
    objective='binary:logistic', eval_metric='auc', max_depth=5, learning_rate=0.025, n_estimators=4000,
    min_child_weight=10, subsample=0.88, colsample_bytree=0.88, reg_alpha=0.0, reg_lambda=3.0, gamma=0.0,
    random_state=SEED, enable_categorical=False, n_jobs=XGB_N_JOBS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    tree_method='hist', device='cuda',
)

CAT_PAIR_DEFS = [
    ('Contract', 'InternetService'),
    ('Contract', 'PaymentMethod'),
    ('InternetService', 'TechSupport'),
    ('InternetService', 'OnlineSecurity'),
    ('InternetService', 'StreamingTV'),
    ('PaperlessBilling', 'PaymentMethod'),
]


def seed_everything(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def sigmoid_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def map_target(series):
    if series.dtype == object:
        return series.map({'Yes': 1, 'No': 0}).astype('int8')
    return series.astype('int8')


def clean_total_charges(df):
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df


def infer_feature_types(train_df, test_df, orig_df):
    """String columns -> categorical, the rest numeric (pandas object dtype for strings)."""
    ignore = {TARGET, ID, 'customerID'}
    feature_cols = [c for c in train_df.columns if c not in ignore]
    cat_cols, num_cols = [], []
    for c in feature_cols:
        if str(train_df[c].dtype) in ['object', 'category', 'bool']:
            cat_cols.append(c)
        else:
            num_cols.append(c)
    common = sorted(set(feature_cols) & set(test_df.columns) & set(orig_df.columns))
    cat_cols = [c for c in cat_cols if c in common]
    num_cols = [c for c in num_cols if c in common]
    return common, cat_cols, num_cols


def add_basic_features(df, numeric_cols):
    out = df.copy()
    if {'tenure', 'MonthlyCharges'}.issubset(out.columns):
        out['tenure_x_monthly'] = out['tenure'].astype('float32') * out['MonthlyCharges'].astype('float32')
        out['monthly_per_tenure'] = out['MonthlyCharges'].astype('float32') / (out['tenure'].astype('float32') + 1.0)
    if {'TotalCharges', 'tenure'}.issubset(out.columns):
        out['total_per_tenure'] = out['TotalCharges'].astype('float32') / (out['tenure'].astype('float32') + 1.0)
    if {'TotalCharges', 'MonthlyCharges'}.issubset(out.columns):
        out['charge_gap'] = out['TotalCharges'].astype('float32') - out['MonthlyCharges'].astype('float32')
        out['total_div_monthly'] = (out['TotalCharges'].astype('float32')
                                    / (out['MonthlyCharges'].astype('float32') + 1.0))
    for c in [c for c in numeric_cols if c in out.columns]:
        out[f'{c}_isna'] = out[c].isna().astype('int8')
    return out


def add_pair_columns(df, pair_defs):
    out = df.copy()
    for a, b in pair_defs:
        if a in out.columns and b in out.columns:
            out[f'{a}__{b}'] = out[a].astype(str).fillna('NaN') + '|' + out[b].astype(str).fillna('NaN')
    return out


def add_exact_numeric_token_cols(df, numeric_cols):
    out = df.copy()
    for c in numeric_cols:
        if c not in out.columns:
            continue
        out[f'TOKEN__{c}'] = out[c].map(lambda x: '__nan__' if pd.isna(x) else f'{float(x):.6f}')
    for a, b in [('tenure', 'MonthlyCharges'), ('tenure', 'TotalCharges'), ('MonthlyCharges', 'TotalCharges')]:
        ca, cb = f'TOKEN__{a}', f'TOKEN__{b}'
        if ca in out.columns and cb in out.columns:
            out[f'TOKENPAIR__{a}__{b}'] = out[ca].astype(str) + '|' + out[cb].astype(str)
    return out


def add_row_fingerprint_cols(df):
    out = df.copy()
    pieces1 = ['TOKEN__tenure', 'TOKEN__MonthlyCharges', 'TOKEN__TotalCharges', 'Contract', 'InternetService',
               'PaymentMethod']
    pieces2 = pieces1 + ['TechSupport', 'OnlineSecurity', 'PaperlessBilling']
    existing1 = [c for c in pieces1 if c in out.columns]
    existing2 = [c for c in pieces2 if c in out.columns]
    if existing1:
        out['ROWHASH__core'] = out[existing1].astype(str).fillna('NaN').agg('|'.join, axis=1)
    if existing2:
        out['ROWHASH__wide'] = out[existing2].astype(str).fillna('NaN').agg('|'.join, axis=1)
    return out


class FoldPreprocessor:
    """transform(): ordinal codes + standardized numerics (XGBoost / discriminator input).
    transform_for_dae(): the same with the ordinal codes divided by their training maximum (DAE / KNN input)."""

    def __init__(self, cat_cols, num_cols):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.ord_enc = None
        self.scaler = None
        self.feature_names_ = None
        self.cat_max_ = None

    def fit(self, df_fit):
        cat_df = df_fit[self.cat_cols].copy()
        for c in self.cat_cols:
            cat_df[c] = cat_df[c].astype(str).fillna('NaN')
        self.ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1,
                                      dtype=np.float32)
        self.ord_enc.fit(cat_df)

        num_df = df_fit[self.num_cols].copy().astype(np.float32)
        num_df = num_df.fillna(num_df.median(numeric_only=True))
        self.scaler = StandardScaler()
        self.scaler.fit(num_df)

        cat_train = self.ord_enc.transform(cat_df).astype(np.float32)
        self.cat_max_ = np.maximum(cat_train.max(axis=0), 1.0).astype(np.float32)
        self.feature_names_ = [f'CAT__{c}' for c in self.cat_cols] + [f'NUM__{c}' for c in self.num_cols]
        return self

    def _transform_cat(self, df):
        cat_df = df[self.cat_cols].copy()
        for c in self.cat_cols:
            cat_df[c] = cat_df[c].astype(str).fillna('NaN')
        return self.ord_enc.transform(cat_df).astype(np.float32)

    def _transform_num(self, df):
        num_df = df[self.num_cols].copy().astype(np.float32)
        num_df = num_df.fillna(num_df.median(numeric_only=True))
        return self.scaler.transform(num_df).astype(np.float32)

    def transform(self, df):
        return np.concatenate([self._transform_cat(df), self._transform_num(df)], axis=1).astype(np.float32)

    def transform_for_dae(self, df):
        X_cat = np.clip(self._transform_cat(df), -1.0, None) / self.cat_max_[None, :]
        return np.concatenate([X_cat, self._transform_num(df)], axis=1).astype(np.float32)


def safe_name(col):
    return (col.replace('TOKEN__', 'tok_').replace('TOKENPAIR__', 'tokpair_').replace('ROWHASH__', 'rowhash_')
            .replace('__', '_X_'))


def _smoothed_mapping(key_series, y, global_mean, smooth):
    tmp = pd.DataFrame({'key': key_series.astype(str).fillna('NaN'), 'y': y})
    agg = tmp.groupby('key')['y'].agg(['count', 'mean'])
    count = agg['count']
    mean = agg['mean']
    smooth_mean = (count * mean + smooth * global_mean) / (count + smooth)
    return smooth_mean.to_dict(), count.to_dict()


def nested_target_encode(tr_df, va_df, te_df, y_tr, te_cols, n_inner_folds, seed, smooth=10.0):
    """TE / count / frequency of every key: inner out-of-fold TE for the training rows; mappings fitted on all
    training rows for va_df and te_df."""
    global_mean = float(np.mean(y_tr))
    inner_skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)

    tr_te = pd.DataFrame(index=tr_df.index)
    va_te = pd.DataFrame(index=va_df.index)
    te_te = pd.DataFrame(index=te_df.index)
    tr_freq = pd.DataFrame(index=tr_df.index)
    va_freq = pd.DataFrame(index=va_df.index)
    te_freq = pd.DataFrame(index=te_df.index)

    for col in te_cols:
        tr_key = tr_df[col].astype(str).fillna('NaN')
        va_key = va_df[col].astype(str).fillna('NaN')
        te_key = te_df[col].astype(str).fillna('NaN')

        oof_vals = np.full(len(tr_df), global_mean, dtype=np.float32)
        for i_tr, i_va in inner_skf.split(tr_df, y_tr):
            mp, _ = _smoothed_mapping(tr_key.iloc[i_tr], y_tr[i_tr], global_mean, smooth)
            oof_vals[i_va] = tr_key.iloc[i_va].map(mp).fillna(global_mean).values.astype(np.float32)

        full_mp, full_cnt = _smoothed_mapping(tr_key, y_tr, global_mean, smooth)
        n = len(tr_df)
        freq_map = {k: v / n for k, v in full_cnt.items()}
        safe = safe_name(col)

        tr_te[f'TE__{safe}'] = oof_vals
        va_te[f'TE__{safe}'] = va_key.map(full_mp).fillna(global_mean).values.astype(np.float32)
        te_te[f'TE__{safe}'] = te_key.map(full_mp).fillna(global_mean).values.astype(np.float32)

        tr_freq[f'CNT__{safe}'] = tr_key.map(full_cnt).fillna(0).values.astype(np.float32)
        va_freq[f'CNT__{safe}'] = va_key.map(full_cnt).fillna(0).values.astype(np.float32)
        te_freq[f'CNT__{safe}'] = te_key.map(full_cnt).fillna(0).values.astype(np.float32)

        tr_freq[f'FREQ__{safe}'] = tr_key.map(freq_map).fillna(0).values.astype(np.float32)
        va_freq[f'FREQ__{safe}'] = va_key.map(freq_map).fillna(0).values.astype(np.float32)
        te_freq[f'FREQ__{safe}'] = te_key.map(freq_map).fillna(0).values.astype(np.float32)

    return tr_te, va_te, te_te, tr_freq, va_freq, te_freq


def orig_only_stats(orig_df, apply_df, cols):
    """Original-data churn mean, count, frequency and seen flag of every key."""
    out = pd.DataFrame(index=apply_df.index)
    y = orig_df[TARGET].values.astype(np.float32)
    global_mean = float(np.mean(y))
    for col in cols:
        key_orig = orig_df[col].astype(str).fillna('NaN')
        key_apply = apply_df[col].astype(str).fillna('NaN')
        agg = pd.DataFrame({'key': key_orig, 'y': y}).groupby('key')['y'].agg(['count', 'mean'])
        cnt = agg['count'].to_dict()
        mean = agg['mean'].to_dict()
        n = len(orig_df)
        freq = {k: v / n for k, v in cnt.items()}
        safe = safe_name(col)
        out[f'ORIG_MEAN__{safe}'] = key_apply.map(mean).fillna(global_mean).values.astype(np.float32)
        out[f'ORIG_CNT__{safe}'] = key_apply.map(cnt).fillna(0).values.astype(np.float32)
        out[f'ORIG_FREQ__{safe}'] = key_apply.map(freq).fillna(0).values.astype(np.float32)
        out[f'ORIG_SEEN__{safe}'] = key_apply.isin(set(cnt.keys())).astype(np.float32).values
    return out


class ArrayDataset(Dataset):
    def __init__(self, X):
        self.X = X.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx])


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, dropout=0.05):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_dae(X_real):
    model = DenoisingAutoEncoder(in_dim=X_real.shape[1], hidden_dim=DAE_HIDDEN_DIM, latent_dim=DAE_LATENT_DIM,
                                 dropout=DAE_DROPOUT).to(DEVICE)
    dl = DataLoader(ArrayDataset(X_real), batch_size=DAE_BATCH_SIZE, shuffle=True, num_workers=0,
                    pin_memory=(DEVICE == 'cuda'), drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=DAE_LR, weight_decay=DAE_WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    model.train()
    for epoch in range(DAE_EPOCHS):
        losses = []
        for xb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            noise = torch.randn_like(xb) * DAE_NOISE_STD
            mask = (torch.rand_like(xb) < DAE_MASK_PROB).float()
            x_noisy = (xb + noise) * (1.0 - mask)
            opt.zero_grad(set_to_none=True)
            x_hat, _ = model(x_noisy)
            loss = loss_fn(x_hat, xb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == DAE_EPOCHS:
            print(f'    DAE epoch {epoch + 1:02d}/{DAE_EPOCHS}  loss={np.mean(losses):.6f}', flush=True)
    return model


@torch.no_grad()
def dae_predict(model, X):
    model.eval()
    all_recon, all_z = [], []
    for start in range(0, len(X), DAE_PRED_BATCH_SIZE):
        end = min(start + DAE_PRED_BATCH_SIZE, len(X))
        xb = torch.from_numpy(X[start:end].astype(np.float32)).to(DEVICE, non_blocking=True)
        x_hat, z = model(xb)
        all_recon.append(x_hat.detach().cpu().numpy())
        all_z.append(z.detach().cpu().numpy())
    return np.concatenate(all_recon, 0), np.concatenate(all_z, 0)


def make_dae_features(X, X_hat, Z, n_cat):
    resid = X - X_hat
    abs_resid = np.abs(resid)
    feats = {
        'dae_recon_mse': np.mean(resid ** 2, axis=1).astype(np.float32),
        'dae_recon_mae': np.mean(abs_resid, axis=1).astype(np.float32),
        'dae_recon_maxabs': np.max(abs_resid, axis=1).astype(np.float32),
        'dae_recon_l2': np.sqrt(np.sum(resid ** 2, axis=1)).astype(np.float32),
        'dae_cat_abs_mean': abs_resid[:, :n_cat].mean(axis=1).astype(np.float32),
        'dae_num_abs_mean': abs_resid[:, n_cat:].mean(axis=1).astype(np.float32),
        'dae_num_abs_max': abs_resid[:, n_cat:].max(axis=1).astype(np.float32),
    }
    for j in range(Z.shape[1]):
        feats[f'dae_z{j}'] = Z[:, j].astype(np.float32)
    return pd.DataFrame(feats)


@torch.no_grad()
def gpu_chunked_knn_stats(queries, refs, ks=(1, 3, 5), query_chunk=65536, ref_chunk=8192):
    """Exact Euclidean k-nearest references of every query row, in chunks on the GPU."""
    ks = tuple(sorted(set(ks)))
    max_k = max(ks)
    n_query = queries.shape[0]
    dists_out = {k: np.empty((n_query, k), dtype=np.float32) for k in ks}
    idx1_out = np.empty(n_query, dtype=np.int32)

    refs_t_full = torch.from_numpy(refs.astype(np.float32)).to(DEVICE, non_blocking=True)
    ref_sq_full = (refs_t_full * refs_t_full).sum(dim=1)
    for qs in range(0, n_query, query_chunk):
        qe = min(qs + query_chunk, n_query)
        q_t = torch.from_numpy(queries[qs:qe].astype(np.float32)).to(DEVICE, non_blocking=True)
        q_sq = (q_t * q_t).sum(dim=1, keepdim=True)
        best_d = torch.full((q_t.shape[0], max_k), float('inf'), device=DEVICE, dtype=torch.float32)
        best_i = torch.full((q_t.shape[0], max_k), -1, device=DEVICE, dtype=torch.int64)
        for rs in range(0, refs_t_full.shape[0], ref_chunk):
            re = min(rs + ref_chunk, refs_t_full.shape[0])
            r_t = refs_t_full[rs:re]
            r_sq = ref_sq_full[rs:re].unsqueeze(0)
            d2 = torch.clamp(q_sq + r_sq - 2.0 * (q_t @ r_t.T), min=0.0)
            cand_d = torch.cat([best_d, d2], dim=1)
            cand_i_new = torch.arange(rs, re, device=DEVICE, dtype=torch.int64).unsqueeze(0).expand(q_t.shape[0], -1)
            cand_i = torch.cat([best_i, cand_i_new], dim=1)
            vals, inds = torch.topk(cand_d, k=max_k, dim=1, largest=False, sorted=True)
            best_d = vals
            best_i = torch.gather(cand_i, 1, inds)
        best_d_cpu = torch.sqrt(best_d).cpu().numpy()
        best_i_cpu = best_i[:, 0].cpu().numpy().astype(np.int32)
        for k in ks:
            dists_out[k][qs:qe] = best_d_cpu[:, :k]
        idx1_out[qs:qe] = best_i_cpu

    del refs_t_full, ref_sq_full
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    return dists_out, idx1_out


def make_nn_features_gpu(query_base, query_latent, ref_base, ref_latent, query_cat_raw, ref_cat_raw, orig_target,
                         prefix):
    base_knn, _ = gpu_chunked_knn_stats(query_base, ref_base, ks=(1, 3, 5), query_chunk=KNN_QUERY_CHUNK_BASE,
                                        ref_chunk=KNN_REF_CHUNK_BASE)
    latent_knn, idx1_lat = gpu_chunked_knn_stats(query_latent, ref_latent, ks=(1, 3, 5),
                                                 query_chunk=KNN_QUERY_CHUNK_LATENT, ref_chunk=KNN_REF_CHUNK_LATENT)
    out = pd.DataFrame({
        f'{prefix}_base_d1': base_knn[1][:, 0].astype(np.float32),
        f'{prefix}_base_d3_mean': base_knn[3].mean(axis=1).astype(np.float32),
        f'{prefix}_base_d5_mean': base_knn[5].mean(axis=1).astype(np.float32),
        f'{prefix}_latent_d1': latent_knn[1][:, 0].astype(np.float32),
        f'{prefix}_latent_d3_mean': latent_knn[3].mean(axis=1).astype(np.float32),
        f'{prefix}_latent_d5_mean': latent_knn[5].mean(axis=1).astype(np.float32),
        f'{prefix}_latent_density': (1.0 / (latent_knn[5].mean(axis=1) + 1e-6)).astype(np.float32),
        f'{prefix}_orig_target_nn1': orig_target[idx1_lat].astype(np.float32),
    })
    qcat = query_cat_raw.astype(str).fillna('NaN').reset_index(drop=True)
    rcat = ref_cat_raw.astype(str).fillna('NaN').reset_index(drop=True)
    nearest_real_cat = rcat.iloc[idx1_lat].reset_index(drop=True)
    matches = (qcat.values == nearest_real_cat.values).astype(np.float32)
    out[f'{prefix}_cat_match_count'] = matches.sum(axis=1).astype(np.float32)
    out[f'{prefix}_cat_match_frac'] = matches.mean(axis=1).astype(np.float32)
    return out


def build_discriminator_train(real_df, synth_df, real_multiplier=6):
    real_big = pd.concat([real_df.copy() for _ in range(real_multiplier)], axis=0, ignore_index=True)
    X = pd.concat([real_big, synth_df.reset_index(drop=True)], axis=0, ignore_index=True)
    y = np.concatenate([np.ones(len(real_big), dtype=np.int8), np.zeros(len(synth_df), dtype=np.int8)], axis=0)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def make_disc_features(model, X, prefix='disc'):
    p = model.predict_proba(X)[:, 1].astype(np.float32)
    return pd.DataFrame({
        f'{prefix}_p_real': p,
        f'{prefix}_logit_real': sigmoid_logit(p).astype(np.float32),
        f'{prefix}_margin': np.abs(p - 0.5).astype(np.float32),
    })


def first_matching_col(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if cols:
        return df[cols[0]].values.astype(np.float32)
    return np.zeros(len(df), dtype=np.float32)


def interaction_features(te_df, orig_stats, nn_df, dae_df, disc_df):
    tok_monthly = first_matching_col(te_df, 'TE__tok_MonthlyCharges')
    tok_pair_tm = first_matching_col(te_df, 'TE__tokpair_tenure_X_MonthlyCharges')
    rowhash_core = first_matching_col(te_df, 'TE__rowhash_core')
    tok_tenure = first_matching_col(te_df, 'TE__tok_tenure')
    orig_seen_tok = orig_stats.filter(like='ORIG_SEEN__tok_').mean(axis=1).values.astype(np.float32)
    p_real = disc_df['disc_p_real'].values
    return pd.DataFrame({
        'tok_monthly_x_disc': tok_monthly * p_real,
        'tok_pair_tm_x_disc': tok_pair_tm * p_real,
        'rowhash_core_x_disc': rowhash_core * p_real,
        'tok_tenure_x_monthly': tok_tenure * tok_monthly,
        'orig_seen_tok_x_disc': orig_seen_tok * p_real,
        'nn_target_x_disc': nn_df['nnreal_orig_target_nn1'].values * p_real,
        'dae_x_tok_mean': dae_df['dae_recon_mse'].values * te_df.filter(like='TE__tok_').mean(axis=1).values.astype(
            np.float32),
    })


def load_frames():
    """Train / test / original rows with all label-free columns, and the column lists."""
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = map_target(train[TARGET])
    orig[TARGET] = map_target(orig[TARGET])
    train = train.drop(columns=[ID])
    test = test.drop(columns=[ID])

    train = clean_total_charges(train)
    test = clean_total_charges(test)
    orig = clean_total_charges(orig)

    _, _, num_cols0 = infer_feature_types(train, test, orig)
    train = add_basic_features(train, num_cols0)
    test = add_basic_features(test, num_cols0)
    orig = add_basic_features(orig, num_cols0)

    train = add_pair_columns(train, CAT_PAIR_DEFS)
    test = add_pair_columns(test, CAT_PAIR_DEFS)
    orig = add_pair_columns(orig, CAT_PAIR_DEFS)

    raw_numeric_for_tokens = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges'] if c in train.columns]
    train = add_exact_numeric_token_cols(train, raw_numeric_for_tokens)
    test = add_exact_numeric_token_cols(test, raw_numeric_for_tokens)
    orig = add_exact_numeric_token_cols(orig, raw_numeric_for_tokens)

    train = add_row_fingerprint_cols(train)
    test = add_row_fingerprint_cols(test)
    orig = add_row_fingerprint_cols(orig)

    feature_cols, cat_cols, num_cols = infer_feature_types(train, test, orig)

    te_cat_cols = [c for c in ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'OnlineSecurity',
                               'PaperlessBilling'] if c in train.columns]
    te_cat_pair_cols = [f'{a}__{b}' for a, b in CAT_PAIR_DEFS if f'{a}__{b}' in train.columns]
    te_num_token_cols = [f'TOKEN__{c}' for c in raw_numeric_for_tokens if f'TOKEN__{c}' in train.columns]
    te_num_pair_cols = [c for c in ['TOKENPAIR__tenure__MonthlyCharges', 'TOKENPAIR__tenure__TotalCharges',
                                    'TOKENPAIR__MonthlyCharges__TotalCharges'] if c in train.columns]
    te_rowhash_cols = [c for c in ['ROWHASH__core', 'ROWHASH__wide'] if c in train.columns]
    te_cols = te_cat_cols + te_cat_pair_cols + te_num_token_cols + te_num_pair_cols + te_rowhash_cols

    print(f'train {train.shape} | test {test.shape} | original {orig.shape}', flush=True)
    print(f'features {len(feature_cols)}: {len(cat_cols)} categorical, {len(num_cols)} numeric; '
          f'{len(te_cols)} encoded keys', flush=True)
    return train, test, orig, feature_cols, cat_cols, num_cols, te_cols


def run_fold(k, tr_idx, va_idx, train, test, orig, feature_cols, cat_cols, num_cols, te_cols):
    fold = k + 1
    tr_df = train.iloc[tr_idx].reset_index(drop=True)
    va_df = train.iloc[va_idx].reset_index(drop=True)
    te_df = test.reset_index(drop=True)
    orig_df = orig.reset_index(drop=True)

    y_tr = tr_df[TARGET].values.astype(np.int8)
    y_va = va_df[TARGET].values.astype(np.int8)
    y_orig = orig_df[TARGET].values.astype(np.int8)

    tr_X_df = tr_df[feature_cols].copy()
    va_X_df = va_df[feature_cols].copy()
    te_X_df = te_df[feature_cols].copy()
    orig_X_df = orig_df[feature_cols].copy()

    # nested target encodings, counts, frequencies; original-data statistics
    print('  target encodings / counts ...', flush=True)
    te_tr, te_va, te_te, fq_tr, fq_va, fq_te = nested_target_encode(tr_df, va_df, te_df, y_tr, te_cols, INNER_FOLDS,
                                                                    seed=SEED + fold, smooth=TE_SMOOTH)
    orig_tr = orig_only_stats(orig_df, tr_df, te_cols)
    orig_va = orig_only_stats(orig_df, va_df, te_cols)
    orig_te = orig_only_stats(orig_df, te_df, te_cols)
    orig_orig = orig_only_stats(orig_df, orig_df, te_cols)
    # encodings fitted on all training rows, applied to the original rows (discriminator only)
    _, te_orig, _, _, fq_orig, _ = nested_target_encode(tr_df, orig_df, orig_df.iloc[:1].copy(), y_tr, te_cols,
                                                        INNER_FOLDS, seed=SEED + fold, smooth=TE_SMOOTH)

    # does the row's fingerprint occur in the original data
    exact_flags = {name: pd.DataFrame(index=df.index) for name, df in
                   [('tr', tr_df), ('va', va_df), ('te', te_df), ('orig', orig_df)]}
    for col, flag in [('ROWHASH__core', 'orig_core_exact'), ('ROWHASH__wide', 'orig_wide_exact')]:
        if col in tr_df.columns:
            orig_set = set(orig_df[col].astype(str).fillna('NaN'))
            for name, df in [('tr', tr_df), ('va', va_df), ('te', te_df), ('orig', orig_df)]:
                exact_flags[name][flag] = df[col].astype(str).fillna('NaN').isin(orig_set).astype(np.float32)

    # base and DAE matrices
    prep = FoldPreprocessor(cat_cols=cat_cols, num_cols=num_cols)
    prep.fit(tr_X_df)
    X_tr_base = pd.DataFrame(prep.transform(tr_X_df), columns=prep.feature_names_)
    X_va_base = pd.DataFrame(prep.transform(va_X_df), columns=prep.feature_names_)
    X_te_base = pd.DataFrame(prep.transform(te_X_df), columns=prep.feature_names_)
    X_orig_base = pd.DataFrame(prep.transform(orig_X_df), columns=prep.feature_names_)
    X_tr_dae = prep.transform_for_dae(tr_X_df)
    X_va_dae = prep.transform_for_dae(va_X_df)
    X_te_dae = prep.transform_for_dae(te_X_df)
    X_orig_dae = prep.transform_for_dae(orig_X_df)

    # DAE on the original rows
    print('  DAE on the original rows ...', flush=True)
    dae_model = train_dae(X_orig_dae)
    Xhat_tr, Z_tr = dae_predict(dae_model, X_tr_dae)
    Xhat_va, Z_va = dae_predict(dae_model, X_va_dae)
    Xhat_te, Z_te = dae_predict(dae_model, X_te_dae)
    Xhat_orig, Z_orig = dae_predict(dae_model, X_orig_dae)
    dae_tr = make_dae_features(X_tr_dae, Xhat_tr, Z_tr, len(cat_cols))
    dae_va = make_dae_features(X_va_dae, Xhat_va, Z_va, len(cat_cols))
    dae_te = make_dae_features(X_te_dae, Xhat_te, Z_te, len(cat_cols))
    dae_orig = make_dae_features(X_orig_dae, Xhat_orig, Z_orig, len(cat_cols))

    # nearest original rows
    print('  nearest original rows ...', flush=True)
    Z_orig32 = Z_orig.astype(np.float32)
    nn_tr = make_nn_features_gpu(X_tr_dae, Z_tr.astype(np.float32), X_orig_dae, Z_orig32, tr_df[cat_cols],
                                 orig_df[cat_cols], y_orig, 'nnreal')
    nn_va = make_nn_features_gpu(X_va_dae, Z_va.astype(np.float32), X_orig_dae, Z_orig32, va_df[cat_cols],
                                 orig_df[cat_cols], y_orig, 'nnreal')
    nn_te = make_nn_features_gpu(X_te_dae, Z_te.astype(np.float32), X_orig_dae, Z_orig32, te_df[cat_cols],
                                 orig_df[cat_cols], y_orig, 'nnreal')
    nn_orig = make_nn_features_gpu(X_orig_dae, Z_orig32, X_orig_dae, Z_orig32, orig_df[cat_cols], orig_df[cat_cols],
                                   y_orig, 'nnreal')

    # real (original x6) vs synthetic (fold training rows) discriminator
    print('  real-vs-synthetic discriminator ...', flush=True)
    disc_real = pd.concat([X_orig_base, te_orig.reset_index(drop=True), fq_orig.reset_index(drop=True),
                           orig_orig.reset_index(drop=True), exact_flags['orig'].reset_index(drop=True),
                           dae_orig.reset_index(drop=True), nn_orig.reset_index(drop=True)], axis=1)
    disc_synth = pd.concat([X_tr_base, te_tr.reset_index(drop=True), fq_tr.reset_index(drop=True),
                            orig_tr.reset_index(drop=True), exact_flags['tr'].reset_index(drop=True),
                            dae_tr.reset_index(drop=True), nn_tr.reset_index(drop=True)], axis=1)
    disc_train_X, disc_train_y = build_discriminator_train(disc_real, disc_synth, DISC_REAL_MULTIPLIER)
    disc_model = xgb.XGBClassifier(**DISC_PARAMS)
    n_eval = min(50000, len(disc_train_X))
    disc_model.fit(disc_train_X, disc_train_y, eval_set=[(disc_train_X.iloc[:n_eval], disc_train_y[:n_eval])],
                   verbose=False)

    disc_tr = make_disc_features(disc_model, pd.concat([X_tr_base, te_tr, fq_tr, orig_tr, exact_flags['tr'], dae_tr,
                                                        nn_tr], axis=1))
    disc_va = make_disc_features(disc_model, pd.concat([X_va_base, te_va, fq_va, orig_va, exact_flags['va'], dae_va,
                                                        nn_va], axis=1))
    disc_te = make_disc_features(disc_model, pd.concat([X_te_base, te_te, fq_te, orig_te, exact_flags['te'], dae_te,
                                                        nn_te], axis=1))

    inter_tr = interaction_features(te_tr, orig_tr, nn_tr, dae_tr, disc_tr)
    inter_va = interaction_features(te_va, orig_va, nn_va, dae_va, disc_va)
    inter_te = interaction_features(te_te, orig_te, nn_te, dae_te, disc_te)

    X_tr_final = pd.concat([X_tr_base, te_tr, fq_tr, orig_tr, exact_flags['tr'], dae_tr, nn_tr, disc_tr, inter_tr],
                           axis=1)
    X_va_final = pd.concat([X_va_base, te_va, fq_va, orig_va, exact_flags['va'], dae_va, nn_va, disc_va, inter_va],
                           axis=1)
    X_te_final = pd.concat([X_te_base, te_te, fq_te, orig_te, exact_flags['te'], dae_te, nn_te, disc_te, inter_te],
                           axis=1)
    print(f'  final features: train {X_tr_final.shape} | valid {X_va_final.shape} | test {X_te_final.shape}',
          flush=True)

    model = xgb.XGBClassifier(**FINAL_PARAMS)
    model.fit(X_tr_final, y_tr, eval_set=[(X_va_final, y_va)], verbose=100)
    # with early stopping, predict_proba uses the best iteration
    va_pred = model.predict_proba(X_va_final)[:, 1].astype(np.float32)
    te_pred = model.predict_proba(X_te_final)[:, 1].astype(np.float32)
    print(f'  Fold {fold} AUC: {roc_auc_score(y_va, va_pred):.6f} | best iteration '
          f'{getattr(model, "best_iteration", None)}', flush=True)
    return va_pred, te_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(GLOBAL_SEED)
    train, test, orig, feature_cols, cat_cols, num_cols, te_cols = load_frames()
    y = train[TARGET].values
    fold_ids = folds(y)
    for k in fold_list(args.folds):
        print('\n' + '-' * 80, flush=True)
        print(f'FOLD {k + 1}/5', flush=True)
        print('-' * 80, flush=True)
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        va_pred, te_pred = run_fold(k, tr_idx, va_idx, train, test, orig, feature_cols, cat_cols, num_cols, te_cols)
        save_part(MODEL_DIR, PAIR, k, va_pred, te_pred)
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
