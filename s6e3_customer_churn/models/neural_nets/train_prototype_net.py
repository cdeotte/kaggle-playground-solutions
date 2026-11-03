#!/usr/bin/env python
"""Prototypical network (learned churn / no-churn prototypes in a latent space) with Mahalanobis distances to the
original classes.

Features, built once on train + test (TotalCharges NaN -> tenure x MonthlyCharges):
- categoricals (19, factorized on train + test): the 16 categoricals and Contract x PaymentMethod,
  InternetService x OnlineSecurity, MultipleLines x InternetService;
- numerics (8): tenure, MonthlyCharges, TotalCharges, TotalCharges - tenure x MonthlyCharges, MonthlyCharges /
  (TotalCharges + 1); Mahalanobis distances to the original no-churn and churn rows (class means and pseudo-inverse
  covariances of these 5 columns standardized on the original rows; train + test standardized on train + test) and
  their difference. All 8 go through a QuantileTransformer (normal output, seed 42) fitted on train + test.
Network: 8-d categorical embeddings + numerics -> MLP 256-128-64 (BatchNorm, SiLU, dropout 0.25) -> 32-d latent z;
16 learnable prototypes per class; logit = logsumexp(-|z - p1|^2) - logsumexp(-|z - p0|^2).
Training: BCE on 0.015-smoothed labels, AdamW lr 2e-3, weight decay 1e-4, cosine annealing (T_max 35, min lr 1e-5),
batch 1024 (last partial batch dropped), up to 35 epochs with early stopping on the held-out AUC (patience 8); the best
epoch predicts (sigmoid). Python / numpy / torch are seeded with 42 (deterministic cuDNN) before the folds.
Usage: python train_prototype_net.py"""
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'prototype_net_original_mahalanobis'
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 42
N_FOLDS = 5
BATCH_SIZE = 1024
EPOCHS = 35
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
LATENT_DIM = 32
NUM_PROTO_0 = 16
NUM_PROTO_1 = 16
EMB_DIM = 8
MLP_HIDDEN = [256, 128, 64]
DROPOUT = 0.25

BASE_CATS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
             'Contract', 'PaperlessBilling', 'PaymentMethod']
BASE_NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
CROSS_PAIRS = [('Contract', 'PaymentMethod'), ('InternetService', 'OnlineSecurity'), ('MultipleLines', 'InternetService')]


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def smooth_labels(y, smoothing=0.015):
    return y * (1.0 - smoothing) + 0.5 * smoothing


def compute_mahalanobis(x, mean, inv_cov):
    x_mu = x - mean
    return np.sqrt(np.maximum(np.sum(np.dot(x_mu, inv_cov) * x_mu, axis=1), 0))


def load_data():
    train = read_train().drop(columns=[ID], errors='ignore')
    test = read_test().drop(columns=[ID], errors='ignore')
    orig = read_original()
    train[TARGET] = train[TARGET].map({'No': 0, 'Yes': 1}).astype(np.int8)
    orig[TARGET] = orig[TARGET].map({'No': 0, 'Yes': 1}).astype(np.int8)
    y_train = train[TARGET].values
    train = train.drop(columns=[TARGET])
    y_orig = orig[TARGET].values
    for df in [train, test, orig]:
        df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        for col in BASE_NUMS:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        miss = df['TotalCharges'].isna()
        if miss.any():
            df.loc[miss, 'TotalCharges'] = df.loc[miss, 'tenure'].fillna(0) * df.loc[miss, 'MonthlyCharges'].fillna(0)
    return train, test, orig, y_train, y_orig


def build_features():
    train, test, orig, y_train, y_orig = load_data()
    n_train = len(train)
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    for df in [combined, orig]:
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
    num_base_extended = BASE_NUMS + ['charges_deviation', 'monthly_to_total_ratio']

    orig_num_scaled = StandardScaler().fit_transform(orig[num_base_extended])
    comb_num_scaled = StandardScaler().fit_transform(combined[num_base_extended])
    orig_0, orig_1 = orig_num_scaled[y_orig == 0], orig_num_scaled[y_orig == 1]
    mean_0, cov_0 = np.mean(orig_0, axis=0), np.cov(orig_0, rowvar=False)
    mean_1, cov_1 = np.mean(orig_1, axis=0), np.cov(orig_1, rowvar=False)
    combined['mahalanobis_dist_0'] = compute_mahalanobis(comb_num_scaled, mean_0, np.linalg.pinv(cov_0)).astype(
        'float32')
    combined['mahalanobis_dist_1'] = compute_mahalanobis(comb_num_scaled, mean_1, np.linalg.pinv(cov_1)).astype(
        'float32')
    combined['mahalanobis_diff'] = (combined['mahalanobis_dist_1'] - combined['mahalanobis_dist_0']).astype('float32')
    for a, b in CROSS_PAIRS:
        combined[f'{a}__{b}'] = combined[a].astype(str) + '_' + combined[b].astype(str)

    cat_features = BASE_CATS + [f'{a}__{b}' for a, b in CROSS_PAIRS]
    num_features = num_base_extended + ['mahalanobis_dist_0', 'mahalanobis_dist_1', 'mahalanobis_diff']
    cat_cardinalities = []
    for c in cat_features:
        combined[c] = pd.factorize(combined[c].astype(str))[0]
        cat_cardinalities.append(combined[c].nunique())
    qt = QuantileTransformer(output_distribution='normal', random_state=SEED)
    combined[num_features] = qt.fit_transform(combined[num_features]).astype('float32')
    print(f'{len(num_features)} numeric, {len(cat_features)} categorical features', flush=True)
    return (combined.iloc[:n_train][cat_features].values.astype(np.int64),
            combined.iloc[:n_train][num_features].values.astype(np.float32),
            combined.iloc[n_train:][cat_features].values.astype(np.int64),
            combined.iloc[n_train:][num_features].values.astype(np.float32),
            y_train, cat_cardinalities)


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


class PrototypicalNetwork(nn.Module):
    def __init__(self, cat_cards, num_count, emb_dim, latent_dim, num_proto_0, num_proto_1, mlp_hidden, dropout):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, emb_dim) for card in cat_cards])
        layers, prev_dim = [], len(cat_cards) * emb_dim + num_count
        for hidden in mlp_hidden:
            layers.extend([nn.Linear(prev_dim, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(dropout)])
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)
        self.proto_0 = nn.Parameter(torch.randn(num_proto_0, latent_dim))
        self.proto_1 = nn.Parameter(torch.randn(num_proto_1, latent_dim))

    def forward(self, x_cat, x_num):
        z = self.encoder(torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)] + [x_num], dim=1))
        sim_0 = torch.logsumexp(-torch.cdist(z, self.proto_0, p=2.0) ** 2, dim=1)
        sim_1 = torch.logsumexp(-torch.cdist(z, self.proto_1, p=2.0) ** 2, dim=1)
        return sim_1 - sim_0


@torch.no_grad()
def predict(model, loader):
    preds = []
    for batch in loader:
        preds.extend(torch.sigmoid(model(batch[0].to(DEVICE), batch[1].to(DEVICE))).cpu().numpy())
    return preds


def main():
    seed_everything(SEED)
    X_train_cat, X_train_num, X_test_cat, X_test_num, y_train, cat_cardinalities = build_features()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_train), dtype=np.float32)
    test_preds = np.zeros(len(X_test_cat), dtype=np.float32)
    criterion = nn.BCEWithLogitsLoss()
    test_loader = DataLoader(TabDataset(X_test_cat, X_test_num), batch_size=BATCH_SIZE, shuffle=False)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_cat, y_train), 1):
        train_loader = DataLoader(TabDataset(X_train_cat[tr_idx], X_train_num[tr_idx], smooth_labels(y_train[tr_idx])),
                                  batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(TabDataset(X_train_cat[va_idx], X_train_num[va_idx], y_train[va_idx]),
                                batch_size=BATCH_SIZE, shuffle=False)
        model = PrototypicalNetwork(cat_cards=cat_cardinalities, num_count=X_train_num.shape[1], emb_dim=EMB_DIM,
                                    latent_dim=LATENT_DIM, num_proto_0=NUM_PROTO_0, num_proto_1=NUM_PROTO_1,
                                    mlp_hidden=MLP_HIDDEN, dropout=DROPOUT).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
        best_auc, patience_counter = 0, 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for x_cat, x_num, y_b in train_loader:
                x_cat, x_num, y_b = x_cat.to(DEVICE), x_num.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x_cat, x_num), y_b)
                loss.backward()
                optimizer.step()
            model.eval()
            val_preds_epoch = predict(model, val_loader)
            val_auc = roc_auc_score(y_train[va_idx], val_preds_epoch)
            scheduler.step()
            if val_auc > best_auc:
                best_auc, best_preds, patience_counter = val_auc, np.array(val_preds_epoch), 0
                test_preds_fold = predict(model, test_loader)
            else:
                patience_counter += 1
            if patience_counter >= PATIENCE:
                break
        print(f'Fold {fold} | best AUC: {best_auc:.6f}', flush=True)
        oof[va_idx] = best_preds
        test_preds += np.array(test_preds_fold) / N_FOLDS
    print(f'OOF AUC: {roc_auc_score(y_train, oof):.6f}', flush=True)
    save_pair(MODEL_DIR, PAIR, oof, test_preds)


if __name__ == '__main__':
    main()
