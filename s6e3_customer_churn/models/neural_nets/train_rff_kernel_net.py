#!/usr/bin/env python
"""Random-Fourier-feature kernel network on categorical embeddings, original-data KMeans distances and RankGauss
numerics.

Cleaning: TotalCharges blank -> NaN -> tenure x MonthlyCharges (train, test and original rows); numerics as float32.
Features, built once on train + test (the original rows only shape features; they are not training rows):
- 20 categoricals: the 16 columns + 4 crosses (Contract x InternetService, PaymentMethod x PaperlessBilling,
  InternetService x OnlineSecurity, Contract x PaymentMethod), factorized on train + test;
- 38 numerics: the 3 raw numerics; the distances to the 32 centroids of a KMeans (k 32, n_init 10, seed 42) fitted on
  the standardized original numerics (scaler fitted on the original rows); a RankGauss (QuantileTransformer, normal
  output, seed 42) of the 3 numerics fitted on the original rows. All 38 are then standardized on train + test.
Model: 8-d embedding per categorical; the 38 numerics go through a frozen random Fourier feature map
cos(x W + b) * sqrt(2 / 1024) with W ~ N(0, 0.5^2) (38 x 1024) and b ~ U(0, 2 pi); [embeddings, RFF] -> dropout 0.3 ->
Linear 128 -> BatchNorm -> ReLU -> dropout 0.3 -> 1 logit.
Training: BCE with logits on labels smoothed by 0.01 (y (1 - 0.01) + 0.005), AdamW lr 2e-3, weight decay 1e-4, cosine
annealing over 35 epochs (eta_min 1e-5), batch 1024 (last partial batch dropped), up to 35 epochs; early stopping after
8 epochs without a gain in held-out AUC; the best epoch predicts the held-out and test rows. Seed 42 (python / numpy /
torch, deterministic cuDNN) once at the start, so the pair is reproduced only when the 5 folds run in order in one
process. Test = mean of the 5 fold models.
Usage: python train_rff_kernel_net.py [--folds 0,1,2,3,4]"""
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
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, ID, NUMERICS, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'rff_kernel_net_orig_kmeans_rankgauss'
MODEL_DIR = model_dir('neural_nets')
SEED = 42
BATCH_SIZE = 1024
EPOCHS = 35
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
LABEL_SMOOTHING = 0.01

RFF_DIM = 1024  # random Fourier features
RFF_SIGMA = 0.5  # RBF bandwidth
EMB_DIM = 8
MLP_HIDDEN = 128
DROPOUT = 0.3
N_KMEANS_CLUSTERS = 32

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


def smooth_labels(y, smoothing=LABEL_SMOOTHING):
    return y * (1.0 - smoothing) + 0.5 * smoothing


def build_features():
    print('Loading train / test / original ...', flush=True)
    train = read_train().drop(columns=[ID])
    test = read_test().drop(columns=[ID])
    orig = read_original()

    train[TARGET] = train[TARGET].map({'No': 0, 'Yes': 1}).astype(np.int8)
    orig[TARGET] = orig[TARGET].map({'No': 0, 'Yes': 1}).astype(np.int8)
    y_train = train[TARGET].values
    train = train.drop(columns=[TARGET])

    for df in (train, test, orig):
        df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        for col in BASE_NUMS:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        miss = df['TotalCharges'].isna()
        if miss.any():
            df.loc[miss, 'TotalCharges'] = df.loc[miss, 'tenure'].fillna(0) * df.loc[miss, 'MonthlyCharges'].fillna(0)

    n_train = len(train)
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # 1. distances to KMeans centroids of the original rows
    scaler = StandardScaler().fit(orig[BASE_NUMS])
    orig_scaled = scaler.transform(orig[BASE_NUMS])
    comb_scaled = scaler.transform(combined[BASE_NUMS])
    kmeans = KMeans(n_clusters=N_KMEANS_CLUSTERS, random_state=SEED, n_init=10)
    kmeans.fit(orig_scaled)
    comb_kmeans_dists = kmeans.transform(comb_scaled)
    kmeans_cols = [f'kmeans_dist_{i}' for i in range(N_KMEANS_CLUSTERS)]
    for i, col in enumerate(kmeans_cols):
        combined[col] = comb_kmeans_dists[:, i].astype('float32')

    # 2. RankGauss fitted on the original rows
    qt = QuantileTransformer(output_distribution='normal', random_state=SEED)
    qt.fit(orig[BASE_NUMS])
    comb_gauss = qt.transform(combined[BASE_NUMS])
    gauss_cols = [f'{col}_gauss' for col in BASE_NUMS]
    for i, col in enumerate(gauss_cols):
        combined[col] = comb_gauss[:, i].astype('float32')

    # 3. categorical crosses
    cross_pairs = [
        ('Contract', 'InternetService'),
        ('PaymentMethod', 'PaperlessBilling'),
        ('InternetService', 'OnlineSecurity'),
        ('Contract', 'PaymentMethod'),
    ]
    for a, b in cross_pairs:
        combined[f'{a}__{b}'] = combined[a].astype(str) + '_' + combined[b].astype(str)

    cat_features = BASE_CATS + [f'{a}__{b}' for a, b in cross_pairs]
    num_features = BASE_NUMS + kmeans_cols + gauss_cols

    cat_cardinalities = []
    for c in cat_features:
        combined[c] = pd.factorize(combined[c].astype(str))[0]
        cat_cardinalities.append(combined[c].nunique())

    combined[num_features] = StandardScaler().fit_transform(combined[num_features]).astype('float32')
    print(f'{len(cat_features)} categorical, {len(num_features)} numeric features', flush=True)

    X_train_cat = combined.iloc[:n_train][cat_features].values.astype(np.int64)
    X_train_num = combined.iloc[:n_train][num_features].values.astype(np.float32)
    X_test_cat = combined.iloc[n_train:][cat_features].values.astype(np.int64)
    X_test_num = combined.iloc[n_train:][num_features].values.astype(np.float32)
    return X_train_cat, X_train_num, X_test_cat, X_test_num, y_train, cat_cardinalities


class TabDataset(Dataset):
    def __init__(self, cat_data, num_data, targets=None):
        self.cat_data = torch.tensor(cat_data, dtype=torch.long)
        self.num_data = torch.tensor(num_data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.cat_data[idx], self.num_data[idx], self.targets[idx]
        return self.cat_data[idx], self.num_data[idx]


class RandomFourierFeatures(nn.Module):
    """Frozen random map of the continuous inputs: cos(x W + b) * sqrt(2 / out_dim)."""

    def __init__(self, in_dim, out_dim, sigma):
        super().__init__()
        self.out_dim = out_dim
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * sigma, requires_grad=False)
        self.b = nn.Parameter(torch.rand(out_dim) * 2 * np.pi, requires_grad=False)

    def forward(self, x):
        return torch.cos(torch.matmul(x, self.W) + self.b) * np.sqrt(2.0 / self.out_dim)


class RFFKernelNetwork(nn.Module):
    def __init__(self, cat_cards, num_count, emb_dim, rff_dim, rff_sigma, mlp_hidden, dropout):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, emb_dim) for card in cat_cards])
        self.rff = RandomFourierFeatures(in_dim=num_count, out_dim=rff_dim, sigma=rff_sigma)
        total_dim = len(cat_cards) * emb_dim + rff_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_dim, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x_cat, x_num):
        x_cat_flat = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        x = torch.cat([x_cat_flat, self.rff(x_num)], dim=1)
        return self.classifier(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(SEED)
    X_train_cat, X_train_num, X_test_cat, X_test_num, y_train, cat_cardinalities = build_features()

    criterion = nn.BCEWithLogitsLoss()
    test_loader = DataLoader(TabDataset(X_test_cat, X_test_num), batch_size=BATCH_SIZE, shuffle=False)

    fold_ids = folds(y_train)
    for k in fold_list(args.folds):
        print(f'\n--- Fold {k + 1} ---', flush=True)
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        y_tr_smooth = smooth_labels(y_train[tr_idx])
        train_loader = DataLoader(TabDataset(X_train_cat[tr_idx], X_train_num[tr_idx], y_tr_smooth),
                                  batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(TabDataset(X_train_cat[va_idx], X_train_num[va_idx], y_train[va_idx]),
                                batch_size=BATCH_SIZE, shuffle=False)

        model = RFFKernelNetwork(cat_cards=cat_cardinalities, num_count=X_train_num.shape[1], emb_dim=EMB_DIM,
                                 rff_dim=RFF_DIM, rff_sigma=RFF_SIGMA, mlp_hidden=MLP_HIDDEN, dropout=DROPOUT).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

        best_auc = 0
        patience_counter = 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for x_cat, x_num, y_b in train_loader:
                x_cat, x_num, y_b = x_cat.to(DEVICE), x_num.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x_cat, x_num), y_b)
                loss.backward()
                optimizer.step()

            model.eval()
            val_preds_epoch = []
            with torch.no_grad():
                for x_cat, x_num, _ in val_loader:
                    val_preds_epoch.extend(torch.sigmoid(model(x_cat.to(DEVICE), x_num.to(DEVICE))).cpu().numpy())
            val_auc = roc_auc_score(y_train[va_idx], val_preds_epoch)
            scheduler.step()

            if val_auc > best_auc:
                best_auc = val_auc
                best_preds = np.array(val_preds_epoch)
                patience_counter = 0
                test_preds_fold = []
                with torch.no_grad():
                    for x_cat, x_num in test_loader:
                        test_preds_fold.extend(torch.sigmoid(model(x_cat.to(DEVICE), x_num.to(DEVICE))).cpu().numpy())
            else:
                patience_counter += 1
            print(f'  epoch {epoch:02d} | val AUC {val_auc:.6f} | best {best_auc:.6f}', flush=True)
            if patience_counter >= PATIENCE:
                print(f'Early stopping at epoch {epoch}', flush=True)
                break

        print(f'Best Fold {k + 1} AUC: {best_auc:.6f}', flush=True)
        save_part(MODEL_DIR, PAIR, k, best_preds, np.array(test_preds_fold))
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y_train)


if __name__ == '__main__':
    main()
