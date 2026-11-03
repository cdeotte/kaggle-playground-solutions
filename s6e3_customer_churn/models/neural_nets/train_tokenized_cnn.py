#!/usr/bin/env python
"""Tokenized 1D-CNN over feature tokens, with original-data GMM archetype probabilities and centroid similarities.

Features, built once on train + test (TotalCharges NaN -> tenure x MonthlyCharges):
- the 16 categoricals (factorized on train + test);
- numerics (24): tenure, MonthlyCharges, TotalCharges, TotalCharges - tenure x MonthlyCharges, MonthlyCharges /
  (TotalCharges + 1), the fractional part of MonthlyCharges; the 16 component probabilities of a Gaussian mixture
  (full covariance, seed 42) fitted on the 3 numerics of the original rows (standardized on the original rows); cosine
  similarity of the standardized numerics to the original churn and no-churn centroids. All 24 are standardized on
  train + test.
Network: every categorical (embedding) and every numeric (Linear(1, 16)) becomes a 16-d token; the 40 tokens form a
sequence for Conv1d(16 -> 32, k 3) -> BN -> ReLU -> MaxPool(2) -> Conv1d(32 -> 64, k 3) -> BN -> ReLU -> global
average pool -> dropout 0.3 -> Linear(32) -> ReLU -> Linear -> 1 logit.
Training: BCE on 0.01-smoothed labels, AdamW lr 1e-3, weight decay 1e-4, cosine annealing (T_max 40, min lr 1e-5),
batch 1024 (last partial batch dropped), up to 40 epochs with early stopping on the held-out AUC (patience 10); the best
epoch predicts (sigmoid). Python / numpy / torch are seeded with 42 (deterministic cuDNN) before the folds.
Usage: python train_tokenized_cnn.py"""
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'tokenized_cnn_original_gmm_archetypes'
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 42
N_FOLDS = 5
BATCH_SIZE = 1024
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
EMBED_DIM = 16
CNN_CHANNELS_1 = 32
CNN_CHANNELS_2 = 64
DROPOUT = 0.3

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
    combined['charges_deviation'] = (combined['TotalCharges'] - combined['tenure'] * combined['MonthlyCharges']).astype(
        'float32')
    combined['monthly_to_total_ratio'] = (combined['MonthlyCharges'] / (combined['TotalCharges'] + 1)).astype('float32')
    combined['MonthlyCharges_fractional'] = (combined['MonthlyCharges'] - np.floor(combined['MonthlyCharges'])).astype(
        'float32')

    scaler_orig = StandardScaler().fit(orig[BASE_NUMS])
    orig_scaled = scaler_orig.transform(orig[BASE_NUMS])
    gmm = GaussianMixture(n_components=16, covariance_type='full', random_state=SEED).fit(orig_scaled)
    comb_scaled = scaler_orig.transform(combined[BASE_NUMS])
    gmm_probs = gmm.predict_proba(comb_scaled)
    for i in range(16):
        combined[f'GMM_Archetype_{i}'] = gmm_probs[:, i].astype('float32')
    orig_churn_centroid = orig_scaled[y_orig == 1].mean(axis=0, keepdims=True)
    orig_nochurn_centroid = orig_scaled[y_orig == 0].mean(axis=0, keepdims=True)
    combined['cosine_sim_churn'] = cosine_similarity(comb_scaled, orig_churn_centroid).flatten().astype('float32')
    combined['cosine_sim_nochurn'] = cosine_similarity(comb_scaled, orig_nochurn_centroid).flatten().astype('float32')

    cat_features = BASE_CATS
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


class TokenizedTabularCNN(nn.Module):
    def __init__(self, cat_cards, num_count, emb_dim, cnn_ch1, cnn_ch2, dropout):
        super().__init__()
        self.cat_embs = nn.ModuleList([nn.Embedding(card, emb_dim) for card in cat_cards])
        self.num_projs = nn.ModuleList([nn.Linear(1, emb_dim) for _ in range(num_count)])
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=emb_dim, out_channels=cnn_ch1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(cnn_ch1), nn.ReLU(), nn.MaxPool1d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=cnn_ch1, out_channels=cnn_ch2, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(cnn_ch2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(cnn_ch2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x_cat, x_num):
        cat_tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)]
        num_tokens = [proj(x_num[:, i:i + 1]) for i, proj in enumerate(self.num_projs)]
        x = torch.stack(cat_tokens + num_tokens, dim=1).permute(0, 2, 1)
        x = self.conv2(self.conv1(x)).squeeze(-1)
        return self.classifier(x).squeeze(-1)


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
        model = TokenizedTabularCNN(cat_cards=cat_cardinalities, num_count=X_train_num.shape[1], emb_dim=EMBED_DIM,
                                    cnn_ch1=CNN_CHANNELS_1, cnn_ch2=CNN_CHANNELS_2, dropout=DROPOUT).to(DEVICE)
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
