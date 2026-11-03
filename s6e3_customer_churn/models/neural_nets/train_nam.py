#!/usr/bin/env python
"""Neural additive model with original-data profile, neighbourhood-density and quantile-bin features.

Features, built once on train + test (TotalCharges NaN -> tenure x MonthlyCharges):
- categoricals (25, factorized on train + test): the 15 string columns; a flag for rows whose 16-column categorical
  profile never occurs in the original data; 10-, 50- and 100-bin quantile bins of tenure, MonthlyCharges and
  TotalCharges (KBinsDiscretizer fitted on the original rows; NaN -> 0);
- numerics (8): SeniorCitizen, tenure, MonthlyCharges, TotalCharges, TotalCharges - tenure x MonthlyCharges, and, in the
  space of the 3 numerics standardized on the original rows, the distances to the 1st and 20th nearest original rows
  and their ratio d20 / (d1 + 1e-5). All 8 are standardized on train + test.
Network: the logit is a bias plus one scalar embedding per categorical plus, per numeric, an MLP 1 -> 32 -> 32 -> 1
(LayerNorm, SiLU, dropout 0.15).
Training: BCE on 0.01-smoothed labels, AdamW lr 3e-3, weight decay 1e-5, batch 1024 (last partial batch dropped), up to
35 epochs; ReduceLROnPlateau on the held-out AUC (factor 0.5, patience 3, min lr 1e-5); early stopping (patience 8)
and the best epoch predicts (sigmoid). Python / numpy / torch are seeded with 42 (deterministic cuDNN) before the folds.
Usage: python train_nam.py"""
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'nam_original_density_bins'
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 42
N_FOLDS = 5
BATCH_SIZE = 1024
EPOCHS = 35
LR = 3e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 8
HIDDEN_DIM = 32
DROPOUT = 0.15

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


def smooth_labels(y, smoothing=0.01):
    return y * (1.0 - smoothing) + 0.5 * smoothing


def load_data():
    train = read_train().drop(columns=[ID], errors='ignore')
    test = read_test().drop(columns=[ID], errors='ignore')
    orig = read_original()
    train[TARGET] = train[TARGET].map({'No': 0, 'Yes': 1}).astype(np.int8)
    orig[TARGET] = orig[TARGET].map({'No': 0, 'Yes': 1}).astype(np.int8)
    y_train = train[TARGET].values
    train = train.drop(columns=[TARGET])
    for df in [train, test, orig]:
        df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        for col in BASE_NUMS:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        miss = df['TotalCharges'].isna()
        if miss.any():
            df.loc[miss, 'TotalCharges'] = df.loc[miss, 'tenure'].fillna(0) * df.loc[miss, 'MonthlyCharges'].fillna(0)
    return train, test, orig, y_train


def build_features():
    train, test, orig, y_train = load_data()
    n_train = len(train)
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    orig_profiles = set(orig[BASE_CATS].apply(tuple, axis=1))
    comb_profiles = combined[BASE_CATS].apply(tuple, axis=1)
    combined['is_impossible_clone'] = (~comb_profiles.isin(orig_profiles)).astype(int).astype(str)

    scaler = StandardScaler().fit(orig[BASE_NUMS])
    knn = NearestNeighbors(n_neighbors=20, n_jobs=-1).fit(scaler.transform(orig[BASE_NUMS]))
    distances, _ = knn.kneighbors(scaler.transform(combined[BASE_NUMS]))
    combined['dist_to_orig_nn1'] = distances[:, 0].astype('float32')
    combined['dist_to_orig_nn20'] = distances[:, 19].astype('float32')
    combined['orig_density_score'] = (distances[:, 19] / (distances[:, 0] + 1e-5)).astype('float32')
    combined['charges_deviation'] = (combined['TotalCharges'] - combined['tenure'] * combined['MonthlyCharges']).astype(
        'float32')
    for col in BASE_NUMS:
        for n_bins in [10, 50, 100]:
            kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            kb.fit(orig[[col]].dropna())
            combined[f'{col}_bins_{n_bins}'] = kb.transform(combined[[col]].fillna(0)).flatten().astype(int).astype(str)

    cat_features = [c for c in combined.columns if combined[c].dtype == object or str(combined[c].dtype) == 'string']
    num_features = [c for c in combined.columns if c not in cat_features]
    cat_cardinalities = []
    for c in cat_features:
        combined[c] = pd.factorize(combined[c].astype(str))[0]
        cat_cardinalities.append(combined[c].nunique())
    combined[num_features] = StandardScaler().fit_transform(combined[num_features]).astype('float32')
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


class FeatureNN(nn.Module):
    """A small MLP for one continuous feature: (batch,) -> (batch, 1)."""

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(-1))


class NeuralAdditiveModel(nn.Module):
    def __init__(self, cat_cards, num_count, hidden_dim, dropout):
        super().__init__()
        self.cat_embs = nn.ModuleList([nn.Embedding(card, 1) for card in cat_cards])
        self.num_nets = nn.ModuleList([FeatureNN(hidden_dim, dropout) for _ in range(num_count)])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_cat, x_num):
        cat_outputs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)]
        num_outputs = [net(x_num[:, i]) for i, net in enumerate(self.num_nets)]
        return torch.stack(cat_outputs + num_outputs, dim=1).sum(dim=1).squeeze(-1) + self.bias


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
        model = NeuralAdditiveModel(cat_cards=cat_cardinalities, num_count=X_train_num.shape[1], hidden_dim=HIDDEN_DIM,
                                    dropout=DROPOUT).to(DEVICE)
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
