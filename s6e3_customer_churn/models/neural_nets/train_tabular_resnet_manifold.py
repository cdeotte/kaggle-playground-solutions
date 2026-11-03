#!/usr/bin/env python
"""Tabular ResNet on categorical embeddings, cyclic tenure features and an original-data PCA / random-projection
manifold.

Features, built once on train + test (TotalCharges NaN -> tenure x MonthlyCharges):
- the 16 categoricals (factorized on train + test) as 12-d embeddings;
- numerics (31): tenure, MonthlyCharges, TotalCharges; sin / cos of tenure with periods 12 and 24; 12 PCA components and
  12 Gaussian random-projection components (seed 42), both fitted on the original rows (one-hot categoricals + the
  3 numerics standardized on the original rows) and applied to train + test. The numerics are standardized on each
  fold's training rows.
Network: embeddings + numerics -> Linear(256) -> BatchNorm -> ReLU -> 4 pre-activation residual blocks
(BN, ReLU, Linear, BN, ReLU, dropout 0.25, Linear) -> BN -> ReLU -> Linear -> 1 logit.
Training: BCE on 0.02-smoothed labels, AdamW lr 2e-3, weight decay 1e-4, batch 1024 (last partial batch dropped), up to
35 epochs; ReduceLROnPlateau on the held-out AUC (factor 0.5, patience 3, min lr 1e-5); early stopping (patience 8)
and the best epoch predicts (sigmoid). Python / numpy / torch are seeded with 42 (deterministic cuDNN) before the folds.
Usage: python train_tabular_resnet_manifold.py"""
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'tabular_resnet_original_pca_projection'
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 42
N_FOLDS = 5
BATCH_SIZE = 1024
EPOCHS = 35
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
HIDDEN_DIM = 256
RES_BLOCKS = 4
DROPOUT = 0.25
EMB_DIM = 12

BASE_CATS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
             'Contract', 'PaperlessBilling', 'PaymentMethod']
BASE_NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def smooth_labels(y, smoothing=0.02):
    return y * (1.0 - smoothing) + 0.5 * smoothing


def load_data():
    train = read_train().drop(columns=[ID], errors='ignore')
    test = read_test().drop(columns=[ID], errors='ignore')
    orig = read_original()
    train[TARGET] = train[TARGET].map({'No': 0, 'Yes': 1}).astype(np.int8)
    orig[TARGET] = orig[TARGET].map({'No': 0, 'Yes': 1}).astype(np.int8)
    y_train = train[TARGET].values
    train = train.drop(columns=[TARGET])
    orig_features = orig.drop(columns=[TARGET])
    for df in [train, test, orig_features]:
        df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        for col in BASE_NUMS:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        miss = df['TotalCharges'].isna()
        if miss.any():
            df.loc[miss, 'TotalCharges'] = df.loc[miss, 'tenure'].fillna(0) * df.loc[miss, 'MonthlyCharges'].fillna(0)
    return train, test, orig_features, y_train


def build_features():
    train, test, orig_features, y_train = load_data()
    n_train = len(train)
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    for period in (12, 24):
        combined[f'tenure_sin_{period}'] = np.sin(2 * np.pi * combined['tenure'] / float(period)).astype('float32')
        combined[f'tenure_cos_{period}'] = np.cos(2 * np.pi * combined['tenure'] / float(period)).astype('float32')

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    orig_cat_encoded = ohe.fit_transform(orig_features[BASE_CATS].astype(str))
    scaler_orig = StandardScaler().fit(orig_features[BASE_NUMS])
    orig_full_matrix = np.hstack([scaler_orig.transform(orig_features[BASE_NUMS]), orig_cat_encoded])
    pca = PCA(n_components=12, random_state=SEED).fit(orig_full_matrix)
    grp = GaussianRandomProjection(n_components=12, random_state=SEED).fit(orig_full_matrix)
    comb_full_matrix = np.hstack([scaler_orig.transform(combined[BASE_NUMS]),
                                  ohe.transform(combined[BASE_CATS].astype(str))])
    pca_features = pca.transform(comb_full_matrix)
    grp_features = grp.transform(comb_full_matrix)
    for i in range(pca_features.shape[1]):
        combined[f'orig_pca_{i}'] = pca_features[:, i].astype('float32')
        combined[f'orig_grp_{i}'] = grp_features[:, i].astype('float32')

    cat_features = BASE_CATS
    num_features = [c for c in combined.columns if c not in cat_features]
    cat_cardinalities = []
    for c in cat_features:
        combined[c] = pd.factorize(combined[c].astype(str))[0]
        cat_cardinalities.append(combined[c].nunique())
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


class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(dim, dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x):
        out = self.lin1(self.relu(self.norm1(x)))
        out = self.lin2(self.dropout(self.relu(self.norm2(out))))
        return x + out


class TabularResNet(nn.Module):
    def __init__(self, cat_cards, num_count, emb_dim, hidden_dim, res_blocks, dropout):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, emb_dim) for card in cat_cards])
        self.initial_proj = nn.Sequential(nn.Linear(len(cat_cards) * emb_dim + num_count, hidden_dim),
                                          nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(res_blocks)])
        self.head = nn.Sequential(nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x_cat, x_num):
        x = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)] + [x_num], dim=1)
        x = self.initial_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


@torch.no_grad()
def predict(model, loader):
    preds = []
    for batch in loader:
        preds.extend(torch.sigmoid(model(batch[0].to(DEVICE), batch[1].to(DEVICE))).cpu().numpy())
    return preds


def main():
    seed_everything(SEED)
    X_train_cat, X_train_num_raw, X_test_cat, X_test_num_raw, y_train, cat_cardinalities = build_features()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_train), dtype=np.float32)
    test_preds = np.zeros(len(X_test_cat), dtype=np.float32)
    criterion = nn.BCEWithLogitsLoss()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_cat, y_train), 1):
        scaler_cv = StandardScaler()
        X_tr_num = scaler_cv.fit_transform(X_train_num_raw[tr_idx])
        X_va_num = scaler_cv.transform(X_train_num_raw[va_idx])
        X_te_num = scaler_cv.transform(X_test_num_raw)
        train_loader = DataLoader(TabDataset(X_train_cat[tr_idx], X_tr_num, smooth_labels(y_train[tr_idx])),
                                  batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(TabDataset(X_train_cat[va_idx], X_va_num, y_train[va_idx]), batch_size=BATCH_SIZE,
                                shuffle=False)
        test_loader = DataLoader(TabDataset(X_test_cat, X_te_num), batch_size=BATCH_SIZE, shuffle=False)

        model = TabularResNet(cat_cards=cat_cardinalities, num_count=X_tr_num.shape[1], emb_dim=EMB_DIM,
                              hidden_dim=HIDDEN_DIM, res_blocks=RES_BLOCKS, dropout=DROPOUT).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                                               min_lr=1e-5)
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
            scheduler.step(val_auc)
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
