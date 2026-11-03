#!/usr/bin/env python
"""Neural additive model (NAM) with an unseen-profile flag, original-neighbourhood density and original-quantile bins.

Features, built once on train + test (the original rows are a reference only, not training rows):
- the 8 categoricals and the 11 numerics;
- is_impossible_clone: 1 when the row's combination of the 8 categoricals never occurs in the original data;
- distances to the 1st and 20th nearest original rows (numerics standardized with original-data statistics,
  sklearn NearestNeighbors) and their ratio d20 / (d1 + 1e-5);
- water_total = Rainfall_mm + Previous_Irrigation_mm, Soil_Moisture - water_total, water_total / (Field_Area_hectare + 1),
  Temperature_C - Humidity / 10, Electrical_Conductivity x Soil_pH;
- 10-, 50- and 100-bin quantile bins (KBinsDiscretizer fitted on the original rows) of every numeric, as categoricals.
Categoricals (42) are factorized on train + test; the 19 continuous columns are standardized on train + test.
Model: NAM with 3 outputs: one embedding (3 logits) per categorical and one small network per continuous column
(1 -> 32 -> 32 -> 3, LayerNorm, SiLU, dropout 0.15); the logits are the sum of all 61 contributions plus a bias.
Training: KL divergence to 0.01-smoothed one-hot targets, AdamW lr 3e-3, weight decay 1e-5, batch 1024 (last partial
batch dropped), ReduceLROnPlateau on the held-out balanced accuracy (factor 0.5, patience 3, min lr 1e-5), up to 35
epochs with early stopping (patience 8); the best epoch gives the held-out and test predictions (softmax). Seed 42
(python / numpy / torch, deterministic cuDNN) once before the folds. Test = mean of the 5 fold models.
Usage: python train_nam.py [--folds 0,1,2,3,4]"""
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import (CATEGORICALS, CLASSES, ID, NUMERICS, TARGET, TARGET_MAP, folds, model_dir,  # noqa: E402
                         read_original, read_test, read_train)
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'nam_orig_density_multires_bins_a'
MODEL_DIR = model_dir('neural_nets')
SEED = 42
BATCH_SIZE = 1024
EPOCHS = 35
PATIENCE = 8
LR = 3e-3
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 32
DROPOUT = 0.15
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


def smooth_labels(y, smoothing=0.01):
    y = y.astype(np.int64)
    n_classes = len(CLASSES)
    out = np.full((len(y), n_classes), smoothing / max(n_classes - 1, 1), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0 - smoothing
    return out


def is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


class TabDataset(Dataset):
    def __init__(self, cat_data, num_data, targets=None):
        self.cat_data = torch.tensor(cat_data, dtype=torch.long)
        self.num_data = torch.tensor(num_data, dtype=torch.float32)
        self.targets = None if targets is None else torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        if self.targets is None:
            return self.cat_data[idx], self.num_data[idx]
        return self.cat_data[idx], self.num_data[idx], self.targets[idx]


class FeatureNN(nn.Module):
    def __init__(self, hidden_dim, dropout, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(-1))


class NeuralAdditiveModel(nn.Module):
    def __init__(self, cat_cards, num_count, hidden_dim, dropout, out_dim):
        super().__init__()
        self.cat_embs = nn.ModuleList([nn.Embedding(int(card), out_dim) for card in cat_cards])
        self.num_nets = nn.ModuleList([FeatureNN(hidden_dim, dropout, out_dim) for _ in range(num_count)])
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x_cat, x_num):
        cat_outputs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)]
        num_outputs = [net(x_num[:, i]) for i, net in enumerate(self.num_nets)]
        return torch.stack(cat_outputs + num_outputs, dim=1).sum(dim=1) + self.bias


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    out = []
    for batch in loader:
        x_cat, x_num = batch[0], batch[1]
        logits = model(x_cat.to(DEVICE, non_blocking=True), x_num.to(DEVICE, non_blocking=True))
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def build_features():
    train = read_train().drop(columns=[ID])
    test = read_test().drop(columns=[ID])
    orig = read_original()
    y_train = train[TARGET].map(TARGET_MAP).astype(np.int64).values

    train_x = train.drop(columns=[TARGET]).copy()
    test_x = test.copy()
    orig_x = orig.drop(columns=[TARGET]).copy()
    for df in [train_x, test_x, orig_x]:
        for col in BASE_CATS:
            df[col] = df[col].astype(str)
        for col in BASE_NUMS:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
            med = np.nanmedian(df[col].values)
            if np.isnan(med):
                med = 0.0
            df[col] = df[col].fillna(med)

    print('Features: unseen-profile flag, original density, interactions, multi-resolution bins ...', flush=True)
    n_train = len(train_x)
    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)

    orig_profiles = set(orig_x[BASE_CATS].apply(tuple, axis=1))
    comb_profiles = combined[BASE_CATS].apply(tuple, axis=1)
    combined['is_impossible_clone'] = (~comb_profiles.isin(orig_profiles)).astype(int).astype(str)

    scaler = StandardScaler().fit(orig_x[BASE_NUMS])
    orig_scaled = scaler.transform(orig_x[BASE_NUMS]).astype(np.float32)
    comb_scaled = scaler.transform(combined[BASE_NUMS]).astype(np.float32)
    nn_model = NearestNeighbors(n_neighbors=min(20, len(orig_x)), n_jobs=-1)
    nn_model.fit(orig_scaled)
    distances, _ = nn_model.kneighbors(comb_scaled)
    nn_last_idx = distances.shape[1] - 1
    combined['dist_to_orig_nn1'] = distances[:, 0].astype(np.float32)
    combined['dist_to_orig_nn20'] = distances[:, nn_last_idx].astype(np.float32)
    combined['orig_density_score'] = (distances[:, nn_last_idx] / (distances[:, 0] + 1e-5)).astype(np.float32)

    combined['water_total'] = (combined['Rainfall_mm'] + combined['Previous_Irrigation_mm']).astype(np.float32)
    combined['moisture_minus_water'] = (combined['Soil_Moisture'] - combined['water_total']).astype(np.float32)
    combined['water_area_ratio'] = (combined['water_total'] / (combined['Field_Area_hectare'] + 1.0)).astype(np.float32)
    combined['heat_humidity_gap'] = (combined['Temperature_C'] - combined['Humidity'] / 10.0).astype(np.float32)
    combined['ec_ph_interaction'] = (combined['Electrical_Conductivity'] * combined['Soil_pH']).astype(np.float32)

    for col in BASE_NUMS:
        orig_non_null = orig_x[[col]].dropna()
        if len(orig_non_null) == 0:
            continue
        for n_bins in [10, 50, 100]:
            effective_bins = min(n_bins, max(2, len(orig_non_null)))
            kb = KBinsDiscretizer(n_bins=effective_bins, encode='ordinal', strategy='quantile')
            kb.fit(orig_non_null)
            transformed = kb.transform(combined[[col]].fillna(0)).flatten()
            combined[f'{col}_bins_{n_bins}'] = transformed.astype(int).astype(str)

    cat_features = [c for c in combined.columns if is_text(combined[c])]
    num_features = [c for c in combined.columns if c not in cat_features]
    cat_cardinalities = []
    for c in cat_features:
        codes, uniques = pd.factorize(combined[c].astype(str), sort=False)
        combined[c] = codes.astype(np.int64)
        cat_cardinalities.append(len(uniques))
    combined[num_features] = StandardScaler().fit_transform(combined[num_features]).astype(np.float32)

    X_train_cat = combined.iloc[:n_train][cat_features].values.astype(np.int64)
    X_train_num = combined.iloc[:n_train][num_features].values.astype(np.float32)
    X_test_cat = combined.iloc[n_train:][cat_features].values.astype(np.int64)
    X_test_num = combined.iloc[n_train:][num_features].values.astype(np.float32)
    print(f'NAM with {len(num_features)} continuous feature nets and {len(cat_features)} categorical embeddings',
          flush=True)
    return X_train_cat, X_train_num, X_test_cat, X_test_num, y_train, cat_cardinalities, len(num_features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(SEED)
    print(f'Device: {DEVICE}', flush=True)
    X_train_cat, X_train_num, X_test_cat, X_test_num, y_train, cat_cardinalities, n_num = build_features()

    test_loader = DataLoader(TabDataset(X_test_cat, X_test_num), batch_size=BATCH_SIZE, shuffle=False)
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    fold_ids = folds(y_train)
    for k in fold_list(args.folds):
        print(f'\n--- Fold {k + 1} ---', flush=True)
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        y_tr_smooth = smooth_labels(y_train[tr_idx], smoothing=0.01)
        train_loader = DataLoader(TabDataset(X_train_cat[tr_idx], X_train_num[tr_idx], y_tr_smooth),
                                  batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(TabDataset(X_train_cat[va_idx], X_train_num[va_idx], y_train[va_idx]),
                                batch_size=BATCH_SIZE, shuffle=False)

        model = NeuralAdditiveModel(cat_cards=cat_cardinalities, num_count=n_num, hidden_dim=HIDDEN_DIM,
                                    dropout=DROPOUT, out_dim=len(CLASSES)).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                                               min_lr=1e-5)

        best_bac = -1.0
        best_preds = None
        best_test_fold = None
        patience_counter = 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for x_cat, x_num, y_b in train_loader:
                x_cat = x_cat.to(DEVICE, non_blocking=True)
                x_num = x_num.to(DEVICE, non_blocking=True)
                y_b = y_b.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                loss = loss_fn(torch.log_softmax(model(x_cat, x_num), dim=1), y_b)
                loss.backward()
                optimizer.step()

            val_preds_epoch = predict_proba(model, val_loader)
            val_bac = balanced_accuracy_score(y_train[va_idx], val_preds_epoch.argmax(axis=1))
            scheduler.step(val_bac)
            if val_bac > best_bac + 1e-6:
                best_bac = val_bac
                best_preds = val_preds_epoch.copy()
                best_test_fold = predict_proba(model, test_loader)
                patience_counter = 0
            else:
                patience_counter += 1
            print(f'epoch {epoch:02d} | val_bac={val_bac:.6f} | best_bac={best_bac:.6f}', flush=True)
            if patience_counter >= PATIENCE:
                print(f'Early stopping at epoch {epoch}', flush=True)
                break

        print(f'Best fold {k + 1} BAC: {best_bac:.6f}', flush=True)
        save_part(MODEL_DIR, PAIR, k, best_preds, best_test_fold)
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y_train)


if __name__ == '__main__':
    main()
