#!/usr/bin/env python
"""DeepFM on categorical tokens only: original-snapped numerics, original-quantile bins, original expected-value
deciles and categorical crosses.

Token fields (44), built once on train + test (the original rows are a reference only, not training rows):
- the 8 categoricals;
- every numeric snapped to the nearest distinct original-data value, as a token (rounded to 4 decimals);
- every numeric binned by a quantile KBinsDiscretizer (min(100, distinct original values) bins) fitted on the
  original rows;
- for every categorical: floor(5 x original-data mean of the label code (Low 0, Medium 1, High 2)), clipped to 0-10
  (unseen levels -> mean 1.0);
- the crosses Crop_Type x Season, Water_Source x Irrigation_Type, snapped Rainfall_mm x Season, snapped
  Previous_Irrigation_mm x Irrigation_Type, snapped Soil_Moisture x Crop_Growth_Stage and Crop_Type x Season x
  Irrigation_Type.
Each field is factorized on train + test and offset into one shared vocabulary.
Model: DeepFM with 3 outputs: per-token linear weights (3 per token) + bias, the FM pairwise-interaction vector of
16-d token embeddings projected to 3 logits, and an MLP on the concatenated embeddings (128-64-32, BatchNorm, ReLU,
dropout 0.3); the three parts are summed. Xavier-uniform init of both embedding tables.
Training: class-weighted cross-entropy (weights n / n_class of the training rows, normalized to mean 1), AdamW lr 3e-3,
weight decay 1e-4, batch 2048, 20 epochs; the epoch with the best held-out balanced accuracy gives the held-out and test
predictions (softmax). Seed 42 (python / numpy / torch, deterministic cuDNN) once before the folds. Test = mean of the
5 fold models.
Usage: python train_deepfm.py [--folds 0,1,2,3,4]"""
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import (CATEGORICALS, CLASSES, ID, NUMERICS, TARGET, TARGET_MAP, folds, model_dir,  # noqa: E402
                         read_original, read_test, read_train)
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'deepfm_snapped_quantile_tokens_a'
MODEL_DIR = model_dir('neural_nets')
SEED = 42
BATCH_SIZE = 2048
EPOCHS = 20
LR = 3e-3
WEIGHT_DECAY = 1e-4
EMBED_DIM = 16
MLP_HIDDEN = [128, 64, 32]
DROPOUT = 0.3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_CATS = list(CATEGORICALS)
BASE_NUMS = list(NUMERICS)


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_nearest_val(vals, ref_vals):
    ref_vals = np.sort(np.unique(ref_vals[~pd.isna(ref_vals)]))
    if len(ref_vals) == 0:
        return np.zeros_like(vals, dtype=np.float32)
    idx = np.searchsorted(ref_vals, vals)
    idx = np.clip(idx, 1, len(ref_vals) - 1)
    left = ref_vals[idx - 1]
    right = ref_vals[idx]
    return np.where(np.abs(vals - left) <= np.abs(vals - right), left, right)


class TokenDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


class DeepFM(nn.Module):
    def __init__(self, vocab_size, num_fields, embed_dim, mlp_hidden, dropout, num_classes):
        super().__init__()
        self.linear_emb = nn.Embedding(vocab_size, num_classes)
        self.fm_emb = nn.Embedding(vocab_size, embed_dim)
        self.fm_proj = nn.Linear(embed_dim, num_classes)
        self.bias = nn.Parameter(torch.zeros(num_classes))

        mlp_layers = []
        input_dim = num_fields * embed_dim
        for hidden in mlp_hidden:
            mlp_layers.extend([nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden
        mlp_layers.append(nn.Linear(input_dim, num_classes))
        self.mlp = nn.Sequential(*mlp_layers)

        nn.init.xavier_uniform_(self.linear_emb.weight)
        nn.init.xavier_uniform_(self.fm_emb.weight)

    def forward(self, x):
        linear_out = self.linear_emb(x).sum(dim=1) + self.bias
        emb = self.fm_emb(x)
        sum_emb = emb.sum(dim=1)
        fm_vec = 0.5 * (sum_emb.pow(2) - emb.pow(2).sum(dim=1))
        fm_out = self.fm_proj(fm_vec)
        deep_out = self.mlp(emb.reshape(x.size(0), -1))
        return linear_out + fm_out + deep_out


def load_data():
    train = read_train()
    test = read_test()
    orig = read_original()

    y = train[TARGET].map(TARGET_MAP).astype(np.int64).values
    y_orig = orig[TARGET].map(TARGET_MAP).astype(np.int64).values

    train = train.drop(columns=[ID, TARGET], errors='ignore')
    test = test.drop(columns=[ID], errors='ignore')
    orig = orig.drop(columns=[TARGET], errors='ignore')

    for df in [train, test, orig]:
        for col in BASE_CATS:
            df[col] = df[col].astype(str).fillna('MISSING')
        for col in BASE_NUMS:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32').fillna(0)
    return train, test, orig, y, y_orig


def engineer_tokens(train, test, orig, y_orig):
    n_train = len(train)
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    categorical_features = BASE_CATS.copy()

    orig_labeled = orig.copy()
    orig_labeled[TARGET] = y_orig

    print('Engineering token features', flush=True)
    for col in BASE_NUMS:
        snapped = get_nearest_val(combined[col].to_numpy(dtype=np.float32), orig[col].to_numpy(dtype=np.float32))
        token_col = f'{col}_snapped_token'
        combined[token_col] = np.round(snapped, 4).astype(str)
        categorical_features.append(token_col)

    for col in BASE_NUMS:
        fit_vals = orig[[col]].dropna()
        n_bins = int(min(100, max(2, fit_vals[col].nunique())))
        kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        kb.fit(fit_vals)
        binned = kb.transform(combined[[col]].fillna(0)).reshape(-1)
        token_col = f'{col}_quantile_token'
        combined[token_col] = binned.astype(np.int32).astype(str)
        categorical_features.append(token_col)

    for col in BASE_CATS:
        exp_map = orig_labeled.groupby(col)[TARGET].mean()
        exp_val = pd.to_numeric(combined[col].map(exp_map), errors='coerce').fillna(1.0)
        token_col = f'ORIG_exp_decile_{col}'
        combined[token_col] = np.floor(exp_val * 5.0).clip(0, 10).astype(np.int32).astype(str)
        categorical_features.append(token_col)

    cross_pairs = [
        ('Crop_Type', 'Season'),
        ('Water_Source', 'Irrigation_Type'),
        ('Rainfall_mm_snapped_token', 'Season'),
        ('Previous_Irrigation_mm_snapped_token', 'Irrigation_Type'),
        ('Soil_Moisture_snapped_token', 'Crop_Growth_Stage'),
    ]
    for a, b in cross_pairs:
        token_col = f'cross_{a}_{b}'
        combined[token_col] = combined[a].astype(str) + '_X_' + combined[b].astype(str)
        categorical_features.append(token_col)

    triple_col = 'triple_Crop_Season_Irrigation'
    combined[triple_col] = (combined['Crop_Type'].astype(str) + '_X_' + combined['Season'].astype(str) + '_X_'
                            + combined['Irrigation_Type'].astype(str))
    categorical_features.append(triple_col)

    combined = combined[categorical_features].copy()
    offsets = []
    current_offset = 0
    for col in categorical_features:
        combined[col] = pd.factorize(combined[col].astype(str))[0].astype(np.int64)
        offsets.append(current_offset)
        current_offset += int(combined[col].nunique())
    for col, offset in zip(categorical_features, offsets):
        combined[col] = combined[col] + offset

    x_train = combined.iloc[:n_train].to_numpy(dtype=np.int64, copy=True)
    x_test = combined.iloc[n_train:].to_numpy(dtype=np.int64, copy=True)
    return x_train, x_test, categorical_features, current_offset


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    preds = []
    for xb in loader:
        if isinstance(xb, (list, tuple)):
            xb = xb[0]
        logits = model(xb.to(DEVICE))
        preds.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(preds)


def train_one_fold(xtr, ytr, xva, yva, xte, vocab_size):
    tr_loader = DataLoader(TokenDataset(xtr, ytr), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va_loader = DataLoader(TokenDataset(xva, yva), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    te_loader = DataLoader(TokenDataset(xte, None), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DeepFM(vocab_size=vocab_size, num_fields=xtr.shape[1], embed_dim=EMBED_DIM, mlp_hidden=MLP_HIDDEN,
                   dropout=DROPOUT, num_classes=len(CLASSES)).to(DEVICE)
    class_counts = np.bincount(ytr, minlength=len(CLASSES)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_bac = -1.0
    best_va = None
    best_te = None
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        va_pred = predict_proba(model, va_loader)
        va_bac = balanced_accuracy_score(yva, va_pred.argmax(axis=1))
        if va_bac > best_bac:
            best_bac = va_bac
            best_va = va_pred
            best_te = predict_proba(model, te_loader)
        print(f'  epoch {epoch + 1:02d} bac={va_bac:.6f}', flush=True)
    return best_va, best_te, best_bac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(SEED)
    print(f'Using device: {DEVICE}', flush=True)
    train, test, orig, y, y_orig = load_data()
    x_train, x_test, feature_names, vocab_size = engineer_tokens(train, test, orig, y_orig)
    print(f'DeepFM with {len(feature_names)} token fields and vocab size {vocab_size}', flush=True)

    fold_ids = folds(y)
    for k in fold_list(args.folds):
        print(f'Fold {k + 1}', flush=True)
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        va_pred, te_pred, best_bac = train_one_fold(x_train[tr_idx], y[tr_idx], x_train[va_idx], y[va_idx], x_test,
                                                    vocab_size)
        print(f'fold {k + 1} bac: {best_bac:.6f}', flush=True)
        save_part(MODEL_DIR, PAIR, k, va_pred, te_pred)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
