#!/usr/bin/env python
"""TabTransformer with arithmetic / digit / bin / numeric-as-category tokens, the original rows as training rows,
target encodings of all categorical pairs (logit, logit^2, logit^3) and ridge side predictions.

Features, built once on train + test + original:
- the 11 numerics and 8 arithmetic features (Rainfall / Previous_Irrigation and its inverse, total water, water per
  area, Soil_Moisture - Humidity, Temperature x Sunlight, Temperature x Wind_Speed, Soil_pH x Electrical_Conductivity);
- 51 categorical tokens: the 8 categoricals; integer and decimal digits of Rainfall_mm, Previous_Irrigation_mm,
  Soil_Moisture, Humidity, Temperature_C and Soil_pH (22 digit columns); quantile bins (KBinsDiscretizer on all rows)
  of Rainfall and Previous_Irrigation (160 and 80 bins), Soil_Moisture and Humidity (120) and Temperature (80); the
  combos Crop x Season x Irrigation_Type, Soil x Crop x Region and Water_Source x Irrigation_Type x Region; every
  numeric rounded to a 0.1 step (0.5 for Rainfall and Previous_Irrigation) as a category.
Tokens are indexed with one sorted vocabulary per column over train + test + original (0 = unseen).
Per fold:
- training rows = the fold's training rows + all original rows;
- numerics standardized on the training rows;
- target encoding of all 1,275 pairs of the 51 tokens: per-class frequencies of the pair key (no smoothing), out-of-fold
  over 5 inner folds (StratifiedKFold seed 42 + fold, folds numbered 1-5) for the training rows, fitted on all training
  rows for the held-out and test rows; each probability p becomes logit(p), logit(p)^2, logit(p)^3 (p clipped to
  [1e-5, 1 - 1e-5]); standardized (11,475 columns);
- a ridge regression (alpha 10) of the one-hot target on [numerics, TE columns]; its 3 outputs, clipped to [0, 1]
  (held-out / test rows also renormalized to sum 1), are appended to the numerics.
Model: 16-d token embeddings -> 2 post-norm Transformer blocks (4 heads, FFN 64, GELU, dropout 0.2), flattened; the TE
columns go through BatchNorm -> Linear(256) -> ReLU -> dropout; [tokens, TE branch, numerics + ridge outputs] ->
MLP 128-64 (BatchNorm, ReLU, dropout 0.2) -> 3 logits.
Training: cross-entropy, Adam lr 1e-3, weight decay 1e-5, ReduceLROnPlateau on the held-out balanced accuracy (factor
0.5, patience 3, min lr 1e-5), batch 4096, 10 epochs; the best epoch predicts (softmax). Seed 42 (python / numpy /
torch, deterministic cuDNN) once before the folds. Test = mean of the 5 fold models.
Memory: the TE matrices hold 11,475 float32 columns for every training, held-out and test row.
Usage: python train_tabtransformer.py [--folds 0,1,2,3,4]"""
import argparse
import copy
import gc
import os
import random
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import (CATEGORICALS, ID, NUMERICS, TARGET, TARGET_MAP, folds, model_dir, read_original,  # noqa: E402
                         read_test, read_train)
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'tabtransformer_pair_te_logit3_ridge_a'
MODEL_DIR = model_dir('neural_nets')
N_CLASSES = 3
SEED = 42
TE_INNER_SPLITS = 5
USE_X_ORIG = True
USE_LOGIT3 = True
TE_SMOOTH = 0.0

BATCH_SIZE = 4096
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 10
EMBED_DIM = 16
NUM_HEADS = 4
FF_DIM = 64
NUM_TRANSFORMER_BLOCKS = 2
MLP_HIDDEN = [128, 64]
DROPOUT = 0.2

BASE_CATS = list(CATEGORICALS)
BASE_NUMS = list(NUMERICS)


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision('high')
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_str(frame, cols):
    for c in cols:
        frame[c] = frame[c].astype(str).fillna('__MISSING__')


def add_digit_features(df, col, int_positions=None, dec_positions=None):
    vals = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32).values
    vals_abs = np.abs(vals)
    if int_positions is not None:
        int_part = np.floor(vals_abs).astype(np.int64)
        for p in int_positions:
            digit = ((int_part // (10 ** p)) % 10).astype(np.int16)
            df[f'{col}_int_d{p}_'] = pd.Series(digit, index=df.index).astype('category')
    if dec_positions is not None:
        frac = vals_abs - np.floor(vals_abs)
        for p in dec_positions:
            digit = (np.floor(frac * (10 ** p)).astype(np.int64) % 10).astype(np.int16)
            df[f'{col}_dec_d{p}_'] = pd.Series(digit, index=df.index).astype('category')
    return df


def feature_engineering_fit(df_all, num_cols_base):
    """New numeric columns start with '_', new categorical columns end with '_'."""
    df = df_all.copy()
    eps = 1e-6
    df['_Rainfall_DIV_Previous_Irrigation'] = (df['Rainfall_mm'] / (df['Previous_Irrigation_mm'] + eps)).astype('float32')
    df['_Previous_Irrigation_DIV_Rainfall'] = (df['Previous_Irrigation_mm'] / (df['Rainfall_mm'] + eps)).astype('float32')
    df['_Water_Total'] = (df['Rainfall_mm'] + df['Previous_Irrigation_mm']).astype('float32')
    df['_Water_Per_Area'] = (df['_Water_Total'] / (df['Field_Area_hectare'] + eps)).astype('float32')
    df['_Moisture_Humidity_Gap'] = (df['Soil_Moisture'] - df['Humidity']).astype('float32')
    df['_Temp_x_Sun'] = (df['Temperature_C'] * df['Sunlight_Hours']).astype('float32')
    df['_Temp_x_Wind'] = (df['Temperature_C'] * df['Wind_Speed_kmh']).astype('float32')
    df['_PH_x_EC'] = (df['Soil_pH'] * df['Electrical_Conductivity']).astype('float32')

    digit_cfg = {
        'Rainfall_mm': ([0, 1, 2], [1, 2]),
        'Previous_Irrigation_mm': ([0, 1, 2], [1, 2]),
        'Soil_Moisture': ([0, 1], [1]),
        'Humidity': ([0, 1], [1]),
        'Temperature_C': ([0, 1], [1]),
        'Soil_pH': ([0], [1, 2]),
    }
    for col, (int_pos, dec_pos) in digit_cfg.items():
        df = add_digit_features(df, col=col, int_positions=int_pos, dec_positions=dec_pos)

    bin_config = {
        'Rainfall_mm': [160, 80],
        'Previous_Irrigation_mm': [160, 80],
        'Soil_Moisture': [120],
        'Humidity': [120],
        'Temperature_C': [80],
    }
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
            binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
            df[f'{col}_{n_bins}_bin_'] = pd.Categorical(binned.astype(str))

    important_combos = [
        ('Crop_Type', 'Season', 'Irrigation_Type'),
        ('Soil_Type', 'Crop_Type', 'Region'),
        ('Water_Source', 'Irrigation_Type', 'Region'),
    ]
    for cols in important_combos:
        combo_series = df[cols[0]].astype(str)
        for c in cols[1:]:
            combo_series = combo_series + '_' + df[c].astype(str)
        codes, _ = pd.factorize(combo_series)
        df['_'.join(cols) + '_'] = pd.Categorical(codes.astype('int32').astype(str))

    round_steps = {
        'Soil_pH': 0.1,
        'Soil_Moisture': 0.1,
        'Organic_Carbon': 0.1,
        'Electrical_Conductivity': 0.1,
        'Temperature_C': 0.1,
        'Humidity': 0.1,
        'Rainfall_mm': 0.5,
        'Sunlight_Hours': 0.1,
        'Wind_Speed_kmh': 0.1,
        'Field_Area_hectare': 0.1,
        'Previous_Irrigation_mm': 0.5,
    }
    for col in num_cols_base:
        rounded = ((pd.to_numeric(df[col], errors='coerce') / round_steps[col]).round() * round_steps[col]).astype('float32')
        df[f'{col}_cat_'] = pd.Categorical(rounded.astype(str))

    new_cat_cols = [c for c in df.columns if c.endswith('_')]
    new_num_cols = [c for c in df.columns if c.startswith('_')]
    return df, new_cat_cols, new_num_cols


def build_cat_vocab(train_df, test_df, orig_df, cat_cols):
    vocab = {}
    for c in cat_cols:
        parts = [train_df[c].astype(str).fillna('__MISSING__'), test_df[c].astype(str).fillna('__MISSING__')]
        if orig_df is not None:
            parts.append(orig_df[c].astype(str).fillna('__MISSING__'))
        vocab[c] = sorted(pd.concat(parts, axis=0).unique().tolist())
    return vocab


def encode_cat_matrix(df, cat_cols, cat_vocab):
    mats = []
    for c in cat_cols:
        mapping = {v: i + 1 for i, v in enumerate(cat_vocab[c])}
        mats.append(df[c].astype(str).fillna('__MISSING__').map(mapping).fillna(0).astype(np.int64).to_numpy())
    return np.stack(mats, axis=1) if mats else np.zeros((len(df), 0), dtype=np.int64)


def pair_key(df, c1, c2):
    return df[c1].astype(np.int64).astype(str) + '|' + df[c2].astype(np.int64).astype(str)


def smoothed_multiclass_mapping(x_df, y, pair_cols, smooth):
    y = np.asarray(y, dtype=np.int64)
    prior = np.bincount(y, minlength=N_CLASSES).astype(np.float32)
    prior = prior / prior.sum()
    maps = []
    for c1, c2 in pair_cols:
        key = pair_key(x_df, c1, c2)
        stats = pd.DataFrame({'key': key, 'y': y})
        counts = pd.crosstab(stats['key'], stats['y']).reindex(columns=range(N_CLASSES), fill_value=0)
        total = counts.sum(axis=1).astype(np.float32).values[:, None]
        enc = (counts.values.astype(np.float32) + smooth * prior[None, :]) / (total + smooth)
        maps.append(pd.DataFrame(enc, index=counts.index, columns=range(N_CLASSES)))
    return maps, prior


def transform_pair_mapping(x_df, pair_cols, mappings, prior):
    out = np.zeros((len(x_df), len(pair_cols) * N_CLASSES), dtype=np.float32)
    for j, (c1, c2) in enumerate(pair_cols):
        key = pair_key(x_df, c1, c2)
        mapped = mappings[j].reindex(key).fillna(pd.Series(prior, index=range(N_CLASSES))).to_numpy(dtype=np.float32)
        out[:, j * N_CLASSES:(j + 1) * N_CLASSES] = mapped
    return out


def make_logit3_features(tr_m, va_m, te_m, eps=1e-5):
    out = []
    for m in (tr_m, va_m, te_m):
        m = np.clip(m, eps, 1.0 - eps)
        z = np.log(m / (1.0 - m)).astype(np.float32)
        out.append(np.hstack([z, z ** 2, z ** 3]).astype(np.float32))
    return tuple(out)


def build_pair_te_features(xtr_int, ytr, xva_int, xte_int, pair_cols, inner_splits=5, seed=0, smooth=0.0, use_logit3=True):
    tr_pair = np.zeros((len(xtr_int), len(pair_cols) * N_CLASSES), dtype=np.float32)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    for it, iv in inner_cv.split(xtr_int, ytr):
        mappings, prior = smoothed_multiclass_mapping(xtr_int.iloc[it].reset_index(drop=True), ytr[it], pair_cols, smooth)
        tr_pair[iv] = transform_pair_mapping(xtr_int.iloc[iv].reset_index(drop=True), pair_cols, mappings, prior)
    full_maps, full_prior = smoothed_multiclass_mapping(xtr_int.reset_index(drop=True), ytr, pair_cols, smooth)
    va_pair = transform_pair_mapping(xva_int.reset_index(drop=True), pair_cols, full_maps, full_prior)
    te_pair = transform_pair_mapping(xte_int.reset_index(drop=True), pair_cols, full_maps, full_prior)
    if use_logit3:
        tr_pair, va_pair, te_pair = make_logit3_features(tr_pair, va_pair, te_pair)
    return tr_pair, va_pair, te_pair


class IrrigationDataset(Dataset):
    def __init__(self, x_cat, x_num, x_te, y=None):
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.x_num = torch.as_tensor(x_num, dtype=torch.float32)
        self.x_te = torch.as_tensor(x_te, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x_cat.shape[0]

    def __getitem__(self, idx):
        item = {'x_cat': self.x_cat[idx], 'x_num': self.x_num[idx], 'x_te': self.x_te[idx]}
        if self.y is not None:
            item['y'] = self.y[idx]
        return item


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, embed_dim))
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        return self.ln2(x + self.ff(x))


class TabTransformerTEModel(nn.Module):
    def __init__(self, cat_cardinalities, num_numeric, te_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card + 1, EMBED_DIM, padding_idx=0) for card in cat_cardinalities])
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT) for _ in range(NUM_TRANSFORMER_BLOCKS)])
        te_hidden = min(256, max(32, te_dim // 2))
        self.te_branch = nn.Sequential(nn.BatchNorm1d(te_dim), nn.Linear(te_dim, te_hidden), nn.ReLU(), nn.Dropout(DROPOUT))
        layers = []
        prev = len(cat_cardinalities) * EMBED_DIM + te_hidden + num_numeric
        for h in MLP_HIDDEN:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(DROPOUT)]
            prev = h
        layers.append(nn.Linear(prev, N_CLASSES))
        self.head = nn.Sequential(*layers)

    def forward(self, x_cat, x_num, x_te):
        x = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        for block in self.transformer_blocks:
            x = block(x)
        return self.head(torch.cat([x.flatten(start_dim=1), self.te_branch(x_te), x_num], dim=1))


def run_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_n = 0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch['x_cat'].to(device), batch['x_num'].to(device), batch['x_te'].to(device))
        y = batch['y'].to(device)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * y.shape[0]
        total_n += y.shape[0]
    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    preds = []
    for batch in loader:
        logits = model(batch['x_cat'].to(device), batch['x_num'].to(device), batch['x_te'].to(device))
        preds.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def make_loader(x_cat, x_num, x_te, y, shuffle, device):
    return DataLoader(IrrigationDataset(x_cat, x_num, x_te, y), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0,
                      pin_memory=device.type == 'cuda')


def fit_torch_model(x_cat_tr, x_num_tr, x_te_tr, y_tr, x_cat_va, x_num_va, x_te_va, y_va, cat_cardinalities, device):
    train_loader = make_loader(x_cat_tr, x_num_tr, x_te_tr, y_tr, True, device)
    valid_loader = make_loader(x_cat_va, x_num_va, x_te_va, y_va, False, device)
    model = TabTransformerTEModel(cat_cardinalities=cat_cardinalities, num_numeric=x_num_tr.shape[1],
                                  te_dim=x_te_tr.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-5)

    best_score = -np.inf
    best_state = None
    bad_epochs = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_score = balanced_accuracy_score(y_va, predict_proba(model, valid_loader, device).argmax(axis=1))
        scheduler.step(val_score)
        print(f'    epoch {epoch:03d} | train_loss={train_loss:.6f} | val_bac={val_score:.6f}', flush=True)
        if val_score > best_score:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOPPING_PATIENCE:
                print(f'    early stopping at epoch {epoch}', flush=True)
                break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, flush=True)

    train_df, test_df, orig_df = read_train(), read_test(), read_original()
    train_df[TARGET] = train_df[TARGET].map(TARGET_MAP).astype(np.int64)
    orig_df[TARGET] = orig_df[TARGET].map(TARGET_MAP).astype(np.int64)
    y = train_df[TARGET].to_numpy()
    X = train_df.drop(columns=[ID, TARGET]).copy()
    X_test = test_df.drop(columns=[ID]).copy()
    y_orig = orig_df[TARGET].to_numpy()
    X_orig = orig_df.drop(columns=[TARGET]).copy()
    for frame in [X, X_test, X_orig]:
        for c in BASE_NUMS:
            frame[c] = pd.to_numeric(frame[c], errors='coerce').astype('float32')
        safe_str(frame, BASE_CATS)

    df_all = pd.concat([X, X_test, X_orig], axis=0).reset_index(drop=True)
    df_tf, new_cat_cols, new_num_cols = feature_engineering_fit(df_all, BASE_NUMS)
    n_train, n_test = len(X), len(X_test)
    X = df_tf.iloc[:n_train].reset_index(drop=True)
    X_test = df_tf.iloc[n_train:n_train + n_test].reset_index(drop=True)
    X_orig = df_tf.iloc[n_train + n_test:].reset_index(drop=True)

    cat_cols = BASE_CATS + new_cat_cols
    num_cols = BASE_NUMS + new_num_cols
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna('__MISSING__')
        X_test[c] = X_test[c].astype(str).fillna('__MISSING__')
        X_orig[c] = X_orig[c].astype(str).fillna('__MISSING__')
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0).astype('float32')
        X_test[c] = pd.to_numeric(X_test[c], errors='coerce').fillna(0).astype('float32')
        X_orig[c] = pd.to_numeric(X_orig[c], errors='coerce').fillna(0).astype('float32')

    cat_vocab = build_cat_vocab(X, X_test, X_orig, cat_cols)
    cat_cardinalities = [len(cat_vocab[c]) for c in cat_cols]
    X_cat_all = encode_cat_matrix(X, cat_cols, cat_vocab)
    X_test_cat_all = encode_cat_matrix(X_test, cat_cols, cat_vocab)
    X_orig_cat_all = encode_cat_matrix(X_orig, cat_cols, cat_vocab)
    X_teenc = pd.DataFrame(X_cat_all, columns=cat_cols)
    X_test_teenc = pd.DataFrame(X_test_cat_all, columns=cat_cols)
    X_orig_teenc = pd.DataFrame(X_orig_cat_all, columns=cat_cols)
    pair_cols = list(combinations(cat_cols, 2))
    print(f'{len(cat_cols)} token columns, {len(num_cols)} numeric columns, {len(pair_cols)} TE pairs', flush=True)

    fold_ids = folds(y)
    for k in fold_list(args.folds):
        fold = k + 1
        print('\n' + '=' * 88 + f'\nfold {fold}\n' + '=' * 88, flush=True)
        t, v = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        Xt_raw = X.iloc[t].reset_index(drop=True).copy()
        Xv_raw = X.iloc[v].reset_index(drop=True).copy()
        yt = y[t].copy()
        yv = y[v].copy()
        Xt_cat = X_cat_all[t].copy()
        Xv_cat = X_cat_all[v].copy()
        Xt_int = X_teenc.iloc[t].reset_index(drop=True).copy()
        Xv_int = X_teenc.iloc[v].reset_index(drop=True).copy()

        if USE_X_ORIG:
            Xt_raw = pd.concat([Xt_raw, X_orig], axis=0).reset_index(drop=True)
            Xt_cat = np.vstack([Xt_cat, X_orig_cat_all])
            Xt_int = pd.concat([Xt_int, X_orig_teenc], axis=0).reset_index(drop=True)
            yt = np.concatenate([yt, y_orig], axis=0)

        scaler_num = StandardScaler()
        Xt_num = scaler_num.fit_transform(Xt_raw[num_cols]).astype(np.float32)
        Xv_num = scaler_num.transform(Xv_raw[num_cols]).astype(np.float32)
        Xtest_num = scaler_num.transform(X_test[num_cols]).astype(np.float32)

        tr_te, va_te, te_te = build_pair_te_features(Xt_int, yt, Xv_int, X_test_teenc.copy(), pair_cols=pair_cols,
                                                     inner_splits=TE_INNER_SPLITS, seed=SEED + fold, smooth=TE_SMOOTH,
                                                     use_logit3=USE_LOGIT3)
        scaler_te = StandardScaler()
        tr_te = scaler_te.fit_transform(tr_te).astype(np.float32)
        va_te = scaler_te.transform(va_te).astype(np.float32)
        te_te = scaler_te.transform(te_te).astype(np.float32)

        Xt_ridge = np.hstack([Xt_num, tr_te]).astype(np.float32)
        Xv_ridge = np.hstack([Xv_num, va_te]).astype(np.float32)
        Xtest_ridge = np.hstack([Xtest_num, te_te]).astype(np.float32)
        ridge = Ridge(alpha=10.0)
        ridge.fit(Xt_ridge, np.eye(N_CLASSES, dtype=np.float32)[yt])
        ridge_tr_pred = np.clip(ridge.predict(Xt_ridge), 0, 1).astype(np.float32)
        ridge_va_pred = np.clip(ridge.predict(Xv_ridge), 0, 1).astype(np.float32)
        ridge_te_pred = np.clip(ridge.predict(Xtest_ridge), 0, 1).astype(np.float32)
        del Xt_ridge, Xv_ridge, Xtest_ridge
        ridge_va_pred /= np.clip(ridge_va_pred.sum(axis=1, keepdims=True), 1e-6, None)
        ridge_te_pred /= np.clip(ridge_te_pred.sum(axis=1, keepdims=True), 1e-6, None)
        print(f'fold {fold} ridge bac: {balanced_accuracy_score(yv, ridge_va_pred.argmax(axis=1)):.6f}', flush=True)

        Xt_num_full = np.hstack([Xt_num, ridge_tr_pred]).astype(np.float32)
        Xv_num_full = np.hstack([Xv_num, ridge_va_pred]).astype(np.float32)
        Xtest_num_full = np.hstack([Xtest_num, ridge_te_pred]).astype(np.float32)

        model = fit_torch_model(Xt_cat, Xt_num_full, tr_te, yt, Xv_cat, Xv_num_full, va_te, yv, cat_cardinalities, device)
        fold_oof = predict_proba(model, make_loader(Xv_cat, Xv_num_full, va_te, None, False, device), device)
        fold_test = predict_proba(model, make_loader(X_test_cat_all.copy(), Xtest_num_full, te_te, None, False, device),
                                  device)
        print(f'fold {fold} tabtransformer bac: {balanced_accuracy_score(yv, fold_oof.argmax(axis=1)):.6f}', flush=True)
        save_part(MODEL_DIR, PAIR, k, fold_oof, fold_test)

        del model, tr_te, va_te, te_te, ridge
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
