#!/usr/bin/env python
"""Denoising-autoencoder, nearest-original and real-vs-synthetic discriminator features for a GPU XGBoost.

Label-free features, built once on the train, test and original rows:
- raw columns: every numeric column as float32 (median-filled); Driver, Compound, Race, Year, PitStop, LapNumber, Stint,
  TyreLife, Position, Position_Change (and any other column with < 100 distinct train + test values) as string
  categoricals;
- 10 ratios / interactions (tyre life / lap, lap / stint, tyre life / pit stops, degradation / tyre life,
  delta / |degradation|, lap time / tyre life, lap time x progress, degradation x progress, position + |position change|,
  late-stint-old-tyre flag) and missing flags;
- 6 categorical pair keys (Driver|Race, Driver|Compound, Race|Compound, Year|Race, Stint|Compound, PitStop|Stint);
- exact value tokens ('%.6f') of the numeric columns and 4 token pairs;
- lap time, lap time delta, cumulative degradation, race progress, tyre life and lap number snapped to the nearest value
  that occurs in the original data (snapped value, its '%.3f' token) plus their 2-decimal rounded tokens;
- two row fingerprints (snapped lap time / delta / degradation + Driver, Race, Compound; the same + Year, Stint, PitStop,
  rounded race progress and tyre life).

Per outer fold (fold = 1..5, seeds 42 + fold):
- base matrix: string columns ordinal-encoded, numerics standardized (fit on the outer-train rows);
- nested target encodings (5 inner folds, smoothing 10) and inner-fold frequencies of every categorical, pair, token and
  fingerprint key (held-out, test and original rows get the maps of all outer-train rows); original-only target mean,
  count and seen flag of the same keys; flags for fingerprints that occur in the original data;
- denoising autoencoder (128 hidden, 16 latent, dropout 0.05; inputs perturbed by Gaussian noise 0.02 and 3% masking;
  SmoothL1 loss; AdamW lr 1e-3, weight decay 1e-5, batch 2048, 25 epochs) trained on all original rows: reconstruction
  errors and the 16 latent coordinates;
- 5 nearest original rows (Euclidean) in the base and the latent space: mean distance to the nearest 1 / 3 / 5, latent
  density, original target of the nearest 1 / 3 / 5 and categorical agreement with the nearest one;
- real-vs-synthetic discriminator (XGBoost depth 5, lr 0.05, up to 2000 trees, early stopping 200): original rows x 4
  versus the outer-train rows; p(real), its logit, |p - 0.5| and their products with the first 4 latent coordinates.
Final model: XGBoost (GPU hist, depth 5, lr 0.025, up to 4000 trees, min_child_weight 10, subsample / colsample 0.88,
lambda 3, early stopping 200 on the held-out fold's AUC) on the outer-train rows (original rows are not training rows).
Python / numpy / torch are seeded once (2026) before the folds. Predictions clipped to [1e-7, 1 - 1e-7]; test = mean of
the 5 fold models.
Usage: python train_dae_xgb.py [--folds 0,1,2,3,4]"""
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'dae_knn_discriminator_xgb'
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

GLOBAL_SEED = 2026  # python / numpy / torch, set once before the fold loop
SEED = 42           # inner target-encoding folds and XGBoost seeds: SEED + fold (fold = 1..5)
INNER_FOLDS = 5
TE_SMOOTH = 10.0
DAE_LATENT_DIM = 16
DAE_HIDDEN_DIM = 128
DAE_DROPOUT = 0.05
DAE_BATCH_SIZE = 2048
DAE_EPOCHS = 25
DAE_LR = 1e-3
DAE_WEIGHT_DECAY = 1e-5
DAE_NOISE_STD = 0.02
DAE_MASK_PROB = 0.03
DAE_PRED_BATCH_SIZE = 32768
KNN_K = (1, 3, 5)
DISC_REAL_MULTIPLIER = 4
EARLY_STOPPING_ROUNDS = 200

DISC_PARAMS = dict(
    objective='binary:logistic', eval_metric='auc', max_depth=5, learning_rate=0.05,
    n_estimators=2000, min_child_weight=8, subsample=0.85, colsample_bytree=0.85,
    reg_alpha=0.0, reg_lambda=2.0, gamma=0.0, random_state=SEED,
    enable_categorical=False, n_jobs=0, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    tree_method='hist', device='cuda',
)
FINAL_PARAMS = dict(
    objective='binary:logistic', eval_metric='auc', max_depth=5, learning_rate=0.025,
    n_estimators=4000, min_child_weight=10, subsample=0.88, colsample_bytree=0.88,
    reg_alpha=0.0, reg_lambda=3.0, gamma=0.0, random_state=SEED,
    enable_categorical=False, n_jobs=0, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    tree_method='hist', device='cuda',
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def safe_name(c):
    return ''.join(ch if ch.isalnum() else '_' for ch in str(c)).strip('_')


def infer_groups(train, test, feature_cols):
    cats, nums, numeric_like = [], [], []
    both = pd.concat([train[feature_cols], test[feature_cols]], ignore_index=True)
    for c in feature_cols:
        if pd.api.types.is_object_dtype(both[c]) or str(both[c].dtype).startswith('string'):
            cats.append(c)
        else:
            numeric_like.append(c)
            if both[c].nunique(dropna=False) < 100:
                cats.append(c)
            else:
                nums.append(c)
    first = [c for c in ['Driver', 'Compound', 'Race', 'Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position',
                         'Position_Change'] if c in feature_cols]
    out = []
    for c in first + cats:
        if c not in out:
            out.append(c)
    return out, nums, numeric_like


def sigmoid_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def add_basic_features(df, numeric_cols):
    out = df.copy()

    def num(c):
        if c in out.columns:
            return pd.to_numeric(out[c], errors='coerce').fillna(0).astype('float32')
        return pd.Series(0, index=out.index, dtype='float32')

    lap, tyre, stint, pit = num('LapNumber'), num('TyreLife'), num('Stint'), num('PitStop')
    pos, pchg = num('Position'), num('Position_Change')
    lapt, delta, degr, prog = num('LapTime (s)'), num('LapTime_Delta'), num('Cumulative_Degradation'), num('RaceProgress')
    out['tyre_per_lap'] = tyre / (lap + 1)
    out['lap_per_stint'] = lap / (stint + 1)
    out['tyre_after_pit'] = tyre / (pit + 1)
    out['degr_per_tyre'] = degr / (tyre + 1)
    out['delta_per_degr'] = delta / (np.abs(degr) + 1)
    out['laptime_per_tyre'] = lapt / (tyre + 1)
    out['laptime_x_progress'] = lapt * prog
    out['degr_x_progress'] = degr * prog
    out['position_pressure'] = pos + pchg.abs()
    out['late_stint_old_tyre'] = ((prog > 0.55).astype('int8') * (tyre > 12).astype('int8')).astype('int8')
    for c in numeric_cols:
        if c in out.columns:
            out[f'{safe_name(c)}_isna'] = out[c].isna().astype('int8')
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
        p = safe_name(c)
        vals = pd.to_numeric(out[c], errors='coerce')
        out[f'TOKEN__{p}'] = vals.map(lambda x: '__nan__' if pd.isna(x) else f'{float(x):.6f}')
    pair_defs = [('LapTime (s)', 'LapTime_Delta'), ('LapTime (s)', 'Cumulative_Degradation'), ('TyreLife', 'LapNumber'),
                 ('RaceProgress', 'Cumulative_Degradation')]
    for a, b in pair_defs:
        ca, cb = f'TOKEN__{safe_name(a)}', f'TOKEN__{safe_name(b)}'
        if ca in out.columns and cb in out.columns:
            out[f'TOKENPAIR__{safe_name(a)}__{safe_name(b)}'] = out[ca].astype(str) + '|' + out[cb].astype(str)
    return out


def build_snap_ref(orig_series):
    ref = pd.to_numeric(orig_series, errors='coerce').dropna().astype(np.float32).unique()
    return np.sort(ref)


def snap_to_ref(values, ref):
    """Nearest value of `ref` (sorted distinct original values); NaN -> 0."""
    arr = pd.to_numeric(values, errors='coerce').astype(np.float32).values
    if len(ref) == 0:
        return np.nan_to_num(arr, nan=0.0).astype(np.float32)
    out = arr.copy()
    mask = ~np.isnan(arr)
    idx = np.where(mask)[0]
    vals = arr[idx]
    pos = np.clip(np.searchsorted(ref, vals), 0, len(ref) - 1)
    left = np.clip(pos - 1, 0, len(ref) - 1)
    right = pos
    choose_right = np.abs(vals - ref[right]) <= np.abs(vals - ref[left])
    out[idx] = np.where(choose_right, ref[right], ref[left]).astype(np.float32)
    out[~mask] = np.nan
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def add_snapped_numeric_token_cols(train_df, test_df, orig_df, numeric_cols):
    """In place: SNAP__ (nearest original value), TOKEN_SNAP__ ('%.3f' of it) and TOKEN_ROUND__ (2 decimals)."""
    snap_cols = []
    for c in numeric_cols:
        if c not in train_df.columns or c not in test_df.columns or c not in orig_df.columns:
            continue
        p = safe_name(c)
        ref = build_snap_ref(orig_df[c])
        for df in [train_df, test_df, orig_df]:
            df[f'SNAP__{p}'] = snap_to_ref(df[c], ref)
            df[f'TOKEN_SNAP__{p}'] = pd.Series(df[f'SNAP__{p}'], index=df.index).map(
                lambda x: '__nan__' if pd.isna(x) else f'{float(x):.3f}')
            df[f'TOKEN_ROUND__{p}'] = pd.Series(pd.to_numeric(df[c], errors='coerce').round(2), index=df.index).map(
                lambda x: '__nan__' if pd.isna(x) else f'{float(x):.2f}')
        snap_cols.extend([f'SNAP__{p}', f'TOKEN_SNAP__{p}', f'TOKEN_ROUND__{p}'])
    return snap_cols


def add_row_fingerprint_cols(df):
    out = df.copy()
    pieces1 = ['TOKEN_SNAP__LapTime_s', 'TOKEN_SNAP__LapTime_Delta', 'TOKEN_SNAP__Cumulative_Degradation', 'Driver', 'Race',
               'Compound']
    pieces2 = pieces1 + ['Year', 'Stint', 'PitStop', 'TOKEN_ROUND__RaceProgress', 'TOKEN_ROUND__TyreLife']
    keep1 = [c for c in pieces1 if c in out.columns]
    keep2 = [c for c in pieces2 if c in out.columns]
    out['ROWHASH__core'] = out[keep1].astype(str).fillna('NaN').agg('|'.join, axis=1) if keep1 else 'NA'
    out['ROWHASH__wide'] = out[keep2].astype(str).fillna('NaN').agg('|'.join, axis=1) if keep2 else 'NA'
    return out


def orig_only_stats(orig_df, query_df, te_cols, target_col):
    """Original-data target mean / count / seen flag of each key."""
    out = pd.DataFrame(index=query_df.index)
    prior = float(orig_df[target_col].mean())
    for c in te_cols:
        if c not in orig_df.columns or c not in query_df.columns:
            continue
        tmp = pd.DataFrame({'key': orig_df[c].astype(str).fillna('NaN'), 'y': orig_df[target_col].values.astype(np.float32)})
        stat = tmp.groupby('key')['y'].agg(['mean', 'count'])
        key = query_df[c].astype(str).fillna('NaN')
        out[f'orig_cnt__{safe_name(c)}'] = key.map(stat['count']).fillna(0).astype(np.float32)
        out[f'orig_seen__{safe_name(c)}'] = (out[f'orig_cnt__{safe_name(c)}'] > 0).astype(np.float32)
        out[f'orig_p1__{safe_name(c)}'] = key.map(stat['mean']).fillna(prior).astype(np.float32)
    return out


def nested_target_encode(tr_df, va_df, te_df, y_tr, te_cols, n_inner_folds, seed, smooth):
    """Smoothed target mean and frequency of each key: inner-fold out-of-fold values for the training rows, maps of all
    training rows for the other two frames."""
    tr_out, va_out, te_out = pd.DataFrame(index=tr_df.index), pd.DataFrame(index=va_df.index), pd.DataFrame(index=te_df.index)
    fq_tr, fq_va, fq_te = pd.DataFrame(index=tr_df.index), pd.DataFrame(index=va_df.index), pd.DataFrame(index=te_df.index)
    inner = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
    prior = float(np.mean(y_tr))
    for c in te_cols:
        if c not in tr_df.columns:
            continue
        oof_prob = np.zeros(len(tr_df), dtype=np.float32)
        oof_freq = np.zeros(len(tr_df), dtype=np.float32)
        for tr2_idx, va2_idx in inner.split(tr_df, y_tr):
            key_tr = tr_df.iloc[tr2_idx][c].astype(str).fillna('NaN')
            key_va = tr_df.iloc[va2_idx][c].astype(str).fillna('NaN')
            stats = pd.DataFrame({'key': key_tr.values, 'y': y_tr[tr2_idx].astype(np.float32)}).groupby('key')['y'].agg(['mean', 'count'])
            enc = (stats['mean'] * stats['count'] + prior * smooth) / (stats['count'] + smooth)
            oof_prob[va2_idx] = key_va.map(enc).fillna(prior).astype(np.float32).values
            oof_freq[va2_idx] = key_va.map(stats['count']).fillna(0).astype(np.float32).values
        name = safe_name(c)
        tr_out[f'te_p1__{name}'] = oof_prob
        fq_tr[f'fq__{name}'] = oof_freq
        key_all = tr_df[c].astype(str).fillna('NaN')
        stats_f = pd.DataFrame({'key': key_all.values, 'y': y_tr.astype(np.float32)}).groupby('key')['y'].agg(['mean', 'count'])
        enc_f = (stats_f['mean'] * stats_f['count'] + prior * smooth) / (stats_f['count'] + smooth)
        for src_df, out_df, fq_df in [(va_df, va_out, fq_va), (te_df, te_out, fq_te)]:
            key = src_df[c].astype(str).fillna('NaN')
            out_df[f'te_p1__{name}'] = key.map(enc_f).fillna(prior).astype(np.float32).values
            fq_df[f'fq__{name}'] = key.map(stats_f['count']).fillna(0).astype(np.float32).values
    return tr_out, va_out, te_out, fq_tr, fq_va, fq_te


class FoldPreprocessor:
    """Ordinal codes of the string columns + standardized numerics; the DAE input rescales the codes to (0, 1)."""

    def __init__(self, cat_cols, num_cols):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.ord = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.num_scaler = StandardScaler()
        self.dae_cat_max = None
        self.feature_names_ = None

    def fit(self, df):
        self.ord.fit(df[self.cat_cols].astype(str).fillna('NaN'))
        num = df[self.num_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32)
        num = num.fillna(num.median())
        self.num_scaler.fit(num)
        cat_enc = self.ord.transform(df[self.cat_cols].astype(str).fillna('NaN')) if self.cat_cols else np.zeros((len(df), 0))
        self.dae_cat_max = np.maximum(cat_enc.max(axis=0), 1.0) if self.cat_cols else np.array([])
        self.feature_names_ = [f'cat__{c}' for c in self.cat_cols] + [f'num__{c}' for c in self.num_cols]

    def _num(self, df):
        num = df[self.num_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32)
        if not self.num_cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        return self.num_scaler.transform(num.fillna(num.median())).astype(np.float32)

    def _cat(self, df):
        if not self.cat_cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        return self.ord.transform(df[self.cat_cols].astype(str).fillna('NaN')).astype(np.float32)

    def transform(self, df):
        return np.concatenate([self._cat(df), self._num(df)], axis=1).astype(np.float32)

    def transform_for_dae(self, df):
        cat = self._cat(df)
        if self.cat_cols:
            cat = (cat + 1.0) / (self.dae_cat_max + 2.0)
        return np.concatenate([cat, self._num(df)], axis=1).astype(np.float32)


class NumpyDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class DAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, dropout):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, in_dim))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_dae(X):
    dl = DataLoader(NumpyDataset(X), batch_size=DAE_BATCH_SIZE, shuffle=True, drop_last=True)
    model = DAE(X.shape[1], DAE_HIDDEN_DIM, DAE_LATENT_DIM, DAE_DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=DAE_LR, weight_decay=DAE_WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss()
    for _ in range(DAE_EPOCHS):
        model.train()
        for xb in dl:
            xb = xb.to(DEVICE)
            noise = torch.randn_like(xb) * DAE_NOISE_STD
            mask = (torch.rand_like(xb) < DAE_MASK_PROB).float()
            x_in = xb * (1.0 - mask) + noise
            opt.zero_grad()
            x_hat, _ = model(x_in)
            loss = loss_fn(x_hat, xb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def dae_predict(model, X):
    model.eval()
    dl = DataLoader(NumpyDataset(X), batch_size=DAE_PRED_BATCH_SIZE, shuffle=False)
    all_recon, all_z = [], []
    for xb in dl:
        x_hat, z = model(xb.to(DEVICE))
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
        'dae_cat_abs_mean': abs_resid[:, :n_cat].mean(axis=1).astype(np.float32) if n_cat else np.zeros(len(X), dtype=np.float32),
        'dae_num_abs_mean': abs_resid[:, n_cat:].mean(axis=1).astype(np.float32) if X.shape[1] > n_cat else np.zeros(len(X), dtype=np.float32),
    }
    for j in range(Z.shape[1]):
        feats[f'dae_z{j}'] = Z[:, j].astype(np.float32)
    return pd.DataFrame(feats)


def make_nn_features(query_base, query_latent, ref_base, ref_latent, query_cat_raw, ref_cat_raw, orig_target, prefix):
    """Distances to the nearest original rows in the base and the DAE latent space; in the latent space also the
    original target of the neighbours and the categorical agreement with the nearest one."""
    out = pd.DataFrame(index=np.arange(query_base.shape[0]))
    for space_name, Q, R in [('base', query_base, ref_base), ('latent', query_latent, ref_latent)]:
        knn = NearestNeighbors(n_neighbors=max(KNN_K), metric='euclidean', n_jobs=32)
        knn.fit(R)
        d, idx = knn.kneighbors(Q)
        out[f'{prefix}_{space_name}_d1'] = d[:, 0].astype(np.float32)
        out[f'{prefix}_{space_name}_d3_mean'] = d[:, :3].mean(axis=1).astype(np.float32)
        out[f'{prefix}_{space_name}_d5_mean'] = d[:, :5].mean(axis=1).astype(np.float32)
        if space_name == 'latent':
            out[f'{prefix}_latent_density'] = (1.0 / (d[:, :5].mean(axis=1) + 1e-6)).astype(np.float32)
            neigh1 = orig_target[idx[:, 0]].astype(np.int8)
            out[f'{prefix}_orig_target_nn1'] = neigh1.astype(np.float32)
            out[f'{prefix}_orig_target_top3'] = orig_target[idx[:, :3]].mean(axis=1).astype(np.float32)
            out[f'{prefix}_orig_target_top5'] = orig_target[idx[:, :5]].mean(axis=1).astype(np.float32)
            qcat = query_cat_raw.astype(str).fillna('NaN').reset_index(drop=True)
            rcat = ref_cat_raw.astype(str).fillna('NaN').reset_index(drop=True)
            nearest_real_cat = rcat.iloc[idx[:, 0]].reset_index(drop=True)
            matches = (qcat.values == nearest_real_cat.values).astype(np.float32)
            out[f'{prefix}_cat_match_count'] = matches.sum(axis=1).astype(np.float32)
            out[f'{prefix}_cat_match_frac'] = matches.mean(axis=1).astype(np.float32)
    return out


def build_discriminator_train(real_df, synth_df, real_multiplier=4):
    real_big = pd.concat([real_df.copy() for _ in range(real_multiplier)], axis=0, ignore_index=True)
    real_y = np.ones(len(real_big), dtype=np.int8)
    synth_y = np.zeros(len(synth_df), dtype=np.int8)
    X = pd.concat([real_big, synth_df.reset_index(drop=True)], axis=0, ignore_index=True)
    y = np.concatenate([real_y, synth_y])
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def make_disc_features(model, X, prefix='disc'):
    p = model.predict_proba(X)[:, 1].astype(np.float32)
    return pd.DataFrame({f'{prefix}_p_real': p, f'{prefix}_logit_real': sigmoid_logit(p).astype(np.float32),
                         f'{prefix}_margin': np.abs(p - 0.5).astype(np.float32)})


def build_frames():
    """Label-free feature frames of the train, test and original rows and the column groups."""
    train, test, orig = read_train(), read_test(), read_original()
    y = train[TARGET].astype(np.int8).to_numpy()
    train[TARGET] = y
    orig[TARGET] = orig[TARGET].astype(np.int8).to_numpy()
    feature_cols0 = [c for c in train.columns if c not in [ID, TARGET]]
    base_cats, _, numeric_like = infer_groups(train, test, feature_cols0)
    for df in [train, test, orig]:
        for c in numeric_like:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float32)
                df[c] = df[c].fillna(df[c].median())
        for c in base_cats:
            if c in df.columns:
                df[c] = df[c].astype(str).fillna('NaN')

    train, test, orig = (add_basic_features(df, numeric_like) for df in (train, test, orig))
    cat_pair_defs = [('Driver', 'Race'), ('Driver', 'Compound'), ('Race', 'Compound'), ('Year', 'Race'), ('Stint', 'Compound'),
                     ('PitStop', 'Stint')]
    train, test, orig = (add_pair_columns(df, cat_pair_defs) for df in (train, test, orig))
    train, test, orig = (add_exact_numeric_token_cols(df, numeric_like) for df in (train, test, orig))
    snap_priority = [c for c in ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'TyreLife', 'LapNumber']
                     if c in train.columns]
    add_snapped_numeric_token_cols(train, test, orig, snap_priority)
    train, test, orig = (add_row_fingerprint_cols(df) for df in (train, test, orig))

    common = [c for c in train.columns if c in test.columns and c in orig.columns]
    feature_cols = sorted(set([c for c in common if c not in [TARGET, ID]]))
    cat_cols = [c for c in feature_cols if train[c].dtype == 'object' or str(train[c].dtype) in ['object', 'category', 'bool']]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    te_cat_cols = [c for c in base_cats if c in train.columns]
    te_cat_pair_cols = [f'{a}__{b}' for a, b in cat_pair_defs if f'{a}__{b}' in train.columns]
    te_num_token_cols = [c for c in train.columns if c.startswith('TOKEN__') or c.startswith('TOKEN_SNAP__')
                         or c.startswith('TOKEN_ROUND__') or c.startswith('TOKENPAIR__')]
    te_rowhash_cols = [c for c in ['ROWHASH__core', 'ROWHASH__wide'] if c in train.columns]
    te_cols = te_cat_cols + te_cat_pair_cols + te_num_token_cols + te_rowhash_cols
    print(f'train {train.shape} | test {test.shape} | original {orig.shape} | categorical {len(cat_cols)} | '
          f'numeric {len(num_cols)} | encoded keys {len(te_cols)}', flush=True)
    return train, test, orig, y, feature_cols, cat_cols, num_cols, te_cols


def run_fold(fold, tr_idx, va_idx, train, test, orig, feature_cols, cat_cols, num_cols, te_cols):
    """Outer fold `fold` (1..5): fold-wise encodings, DAE, nearest-original and discriminator blocks, final XGBoost."""
    tr_df = train.iloc[tr_idx].reset_index(drop=True)
    va_df = train.iloc[va_idx].reset_index(drop=True)
    te_df = test.reset_index(drop=True)
    orig_df = orig.reset_index(drop=True)
    y_tr = tr_df[TARGET].values.astype(np.int8)
    y_va = va_df[TARGET].values.astype(np.int8)
    y_orig = orig_df[TARGET].values.astype(np.int8)
    tr_X_df, va_X_df, te_X_df, orig_X_df = (d[feature_cols].copy() for d in (tr_df, va_df, te_df, orig_df))

    print('target / frequency / original encodings', flush=True)
    te_tr, te_va, te_te, fq_tr, fq_va, fq_te = nested_target_encode(tr_df, va_df, te_df, y_tr, te_cols, INNER_FOLDS,
                                                                    SEED + fold, TE_SMOOTH)
    _, te_orig, _, _, fq_orig, _ = nested_target_encode(tr_df, orig_df, orig_df.iloc[:1].copy(), y_tr, te_cols, INNER_FOLDS,
                                                        SEED + fold, TE_SMOOTH)
    orig_tr, orig_va, orig_te, orig_orig = (orig_only_stats(orig_df, q, te_cols, TARGET) for q in (tr_df, va_df, te_df, orig_df))
    exact_flags = []
    for qdf in [tr_df, va_df, te_df, orig_df]:
        ef = pd.DataFrame(index=qdf.index)
        for c in ['ROWHASH__core', 'ROWHASH__wide']:
            if c in qdf.columns:
                orig_set = set(orig_df[c].astype(str).fillna('NaN'))
                ef[f'orig_exact__{c}'] = qdf[c].astype(str).fillna('NaN').isin(orig_set).astype(np.float32)
        exact_flags.append(ef)
    exact_flags_tr, exact_flags_va, exact_flags_te, exact_flags_orig = exact_flags

    prep = FoldPreprocessor(cat_cols=cat_cols, num_cols=num_cols)
    prep.fit(tr_X_df)
    X_tr_base_np, X_va_base_np, X_te_base_np, X_orig_base_np = (prep.transform(d) for d in (tr_X_df, va_X_df, te_X_df, orig_X_df))
    X_tr_dae_np, X_va_dae_np, X_te_dae_np, X_orig_dae_np = (prep.transform_for_dae(d) for d in (tr_X_df, va_X_df, te_X_df, orig_X_df))
    X_tr_base, X_va_base, X_te_base, X_orig_base = (pd.DataFrame(a, columns=prep.feature_names_)
                                                    for a in (X_tr_base_np, X_va_base_np, X_te_base_np, X_orig_base_np))

    print('DAE on the original rows', flush=True)
    dae_model = train_dae(X_orig_dae_np)
    Xhat_tr, Z_tr = dae_predict(dae_model, X_tr_dae_np)
    Xhat_va, Z_va = dae_predict(dae_model, X_va_dae_np)
    Xhat_te, Z_te = dae_predict(dae_model, X_te_dae_np)
    Xhat_orig, Z_orig = dae_predict(dae_model, X_orig_dae_np)
    n_cat = len(cat_cols)
    dae_tr = make_dae_features(X_tr_dae_np, Xhat_tr, Z_tr, n_cat)
    dae_va = make_dae_features(X_va_dae_np, Xhat_va, Z_va, n_cat)
    dae_te = make_dae_features(X_te_dae_np, Xhat_te, Z_te, n_cat)
    dae_orig = make_dae_features(X_orig_dae_np, Xhat_orig, Z_orig, n_cat)

    print('nearest-original features', flush=True)
    nn_tr = make_nn_features(X_tr_base_np, Z_tr, X_orig_base_np, Z_orig, tr_df[cat_cols], orig_df[cat_cols], y_orig, 'nn')
    nn_va = make_nn_features(X_va_base_np, Z_va, X_orig_base_np, Z_orig, va_df[cat_cols], orig_df[cat_cols], y_orig, 'nn')
    nn_te = make_nn_features(X_te_base_np, Z_te, X_orig_base_np, Z_orig, te_df[cat_cols], orig_df[cat_cols], y_orig, 'nn')
    nn_orig = make_nn_features(X_orig_base_np, Z_orig, X_orig_base_np, Z_orig, orig_df[cat_cols], orig_df[cat_cols], y_orig, 'nn')

    print('real-vs-synthetic discriminator', flush=True)
    disc_real = pd.concat([X_orig_base, dae_orig, nn_orig, te_orig, fq_orig, orig_orig, exact_flags_orig], axis=1).fillna(0)
    disc_synth = pd.concat([X_tr_base, dae_tr, nn_tr, te_tr, fq_tr, orig_tr, exact_flags_tr], axis=1).fillna(0)
    disc_train_X, disc_train_y = build_discriminator_train(disc_real, disc_synth, DISC_REAL_MULTIPLIER)
    disc_model = xgb.XGBClassifier(**{**DISC_PARAMS, 'random_state': SEED + fold})
    disc_model.fit(disc_train_X, disc_train_y, eval_set=[(disc_synth, np.zeros(len(disc_synth), dtype=np.int8))], verbose=False)
    disc_tr = make_disc_features(disc_model, disc_synth, prefix='disc')
    disc_va = make_disc_features(disc_model, pd.concat([X_va_base, dae_va, nn_va, te_va, fq_va, orig_va, exact_flags_va], axis=1).fillna(0),
                                 prefix='disc')
    disc_te = make_disc_features(disc_model, pd.concat([X_te_base, dae_te, nn_te, te_te, fq_te, orig_te, exact_flags_te], axis=1).fillna(0),
                                 prefix='disc')
    inter_tr, inter_va, inter_te = pd.DataFrame(index=tr_df.index), pd.DataFrame(index=va_df.index), pd.DataFrame(index=te_df.index)
    for base_col in ['disc_p_real', 'disc_logit_real', 'disc_margin']:
        if base_col in disc_tr.columns:
            for zc in [c for c in dae_tr.columns if c.startswith('dae_z')][:4]:
                inter_tr[f'{base_col}__{zc}'] = disc_tr[base_col].values * dae_tr[zc].values
                inter_va[f'{base_col}__{zc}'] = disc_va[base_col].values * dae_va[zc].values
                inter_te[f'{base_col}__{zc}'] = disc_te[base_col].values * dae_te[zc].values

    X_tr_final = pd.concat([X_tr_base, dae_tr, nn_tr, te_tr, fq_tr, orig_tr, exact_flags_tr, disc_tr, inter_tr], axis=1).fillna(0)
    X_va_final = pd.concat([X_va_base, dae_va, nn_va, te_va, fq_va, orig_va, exact_flags_va, disc_va, inter_va], axis=1).fillna(0)
    X_te_final = pd.concat([X_te_base, dae_te, nn_te, te_te, fq_te, orig_te, exact_flags_te, disc_te, inter_te], axis=1).fillna(0)
    cols_union = sorted(set(X_tr_final.columns) | set(X_va_final.columns) | set(X_te_final.columns))
    X_tr_final = X_tr_final.reindex(columns=cols_union, fill_value=0).astype(np.float32)
    X_va_final = X_va_final.reindex(columns=cols_union, fill_value=0).astype(np.float32)
    X_te_final = X_te_final.reindex(columns=cols_union, fill_value=0).astype(np.float32)
    print(f'final feature count: {X_tr_final.shape[1]}', flush=True)
    model = xgb.XGBClassifier(**{**FINAL_PARAMS, 'random_state': SEED + fold})
    model.fit(X_tr_final, y_tr, eval_set=[(X_va_final, y_va)], verbose=500)
    va_pred = model.predict_proba(X_va_final)[:, 1].astype(np.float32)
    te_pred = model.predict_proba(X_te_final)[:, 1].astype(np.float32)
    return va_pred, te_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(GLOBAL_SEED)
    train, test, orig, y, feature_cols, cat_cols, num_cols, te_cols = build_frames()
    fold_ids = folds(y)
    for k in fold_list(args.folds):
        fold = k + 1
        print(f'fold {fold}/5', flush=True)
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        va_pred, te_pred = run_fold(fold, tr_idx, va_idx, train, test, orig, feature_cols, cat_cols, num_cols, te_cols)
        print(f'fold {fold} AUC = {roc_auc_score(y[va_idx], va_pred):.9f}', flush=True)
        save_part(MODEL_DIR, PAIR, k, np.clip(va_pred, 1e-7, 1 - 1e-7), np.clip(te_pred, 1e-7, 1 - 1e-7))
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
