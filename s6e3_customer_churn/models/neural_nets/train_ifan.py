#!/usr/bin/env python
"""Instance-wise feature attention network (IFAN) with original-data noise, micro-cluster and kernel-PCA features.

Features, built once on train + test (TotalCharges NaN -> tenure x MonthlyCharges):
- the 16 categoricals (factorized on train + test);
- numerics (13): tenure, MonthlyCharges, TotalCharges; TotalCharges / (MonthlyCharges + 1e-5) and its absolute
  difference from tenure; for MonthlyCharges and TotalCharges the distance to the nearest value of the original data;
  the original churn rate of the row's KMeans cluster (50 clusters, seed 42, n_init 10, fitted on the 3 numerics of the
  original rows standardized on the original rows; unseen -> 0.22); 5 cosine kernel-PCA components (seed 42) fitted on
  the same original matrix. All 13 go through a QuantileTransformer (normal output, seed 42) fitted on train + test.
Network: every categorical (embedding) and every numeric (Linear(1, 16)) becomes a 16-d token; an attention net
(flattened tokens -> 128 -> LayerNorm -> ReLU -> one softmax weight per token) rescales the tokens; the flattened
weighted tokens -> MLP 256-128 (BatchNorm, ReLU, dropout 0.3) -> 1 logit.
Training: BCE on 0.015-smoothed labels, AdamW lr 2e-3, weight decay 1e-4, cosine annealing (T_max 35, min lr 1e-5),
batch 1024 (last partial batch dropped), up to 35 epochs with early stopping on the held-out AUC (patience 8); the best
epoch predicts (sigmoid). Python / numpy / torch are seeded with 42 (deterministic cuDNN) before the folds.
Usage: python train_ifan.py"""
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
from sklearn.decomposition import KernelPCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'ifan_original_noise_kmeans_kpca'
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 42
N_FOLDS = 5
BATCH_SIZE = 1024
EPOCHS = 35
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
EMBED_DIM = 16
MLP_HIDDEN = [256, 128]
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


def smooth_labels(y, smoothing=0.015):
    return y * (1.0 - smoothing) + 0.5 * smoothing


def get_univariate_noise(vals, ref_vals):
    """Absolute distance from each value to the nearest reference (original) value."""
    ref_vals = np.sort(np.unique(ref_vals))
    idx = np.clip(np.searchsorted(ref_vals, vals), 1, len(ref_vals) - 1)
    left, right = ref_vals[idx - 1], ref_vals[idx]
    nearest = np.where(np.abs(vals - left) <= np.abs(vals - right), left, right)
    return np.abs(vals - nearest)


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
    combined['calculated_tenure'] = (combined['TotalCharges'] / (combined['MonthlyCharges'] + 1e-5)).astype('float32')
    combined['tenure_math_error'] = np.abs(combined['calculated_tenure'] - combined['tenure']).astype('float32')
    for col in ['MonthlyCharges', 'TotalCharges']:
        combined[f'{col}_injected_noise'] = get_univariate_noise(combined[col].fillna(0).values,
                                                                 orig[col].dropna().values).astype('float32')

    scaler = StandardScaler().fit(orig[BASE_NUMS])
    orig_scaled = scaler.transform(orig[BASE_NUMS])
    comb_scaled = scaler.transform(combined[BASE_NUMS])
    kmeans = KMeans(n_clusters=50, random_state=SEED, n_init=10)
    orig_clusters = kmeans.fit_predict(orig_scaled)
    comb_clusters = kmeans.predict(comb_scaled)
    cluster_target_map = pd.Series(y_orig).groupby(orig_clusters).mean().to_dict()
    combined['micro_cluster_target_mean'] = pd.Series(comb_clusters).map(cluster_target_map).fillna(0.22).astype(
        'float32')
    kpca = KernelPCA(n_components=5, kernel='cosine', fit_inverse_transform=False, random_state=SEED, n_jobs=-1)
    kpca.fit(orig_scaled)
    comb_kpca = kpca.transform(comb_scaled)
    for i in range(5):
        combined[f'cosine_kpca_{i}'] = comb_kpca[:, i].astype('float32')

    cat_features = BASE_CATS.copy()
    num_features = [c for c in combined.columns if c not in cat_features]
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


class IFAN(nn.Module):
    """Instance-wise feature attention network: a per-row softmax weight for every feature token."""

    def __init__(self, cat_cards, num_count, emb_dim, mlp_hidden, dropout):
        super().__init__()
        self.cat_embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cat_cards])
        self.num_projs = nn.ModuleList([nn.Linear(1, emb_dim) for _ in range(num_count)])
        self.total_features = len(cat_cards) + num_count
        self.attention_net = nn.Sequential(
            nn.Linear(self.total_features * emb_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, self.total_features),
            nn.Softmax(dim=1),
        )
        layers, input_dim = [], self.total_features * emb_dim
        for hidden in mlp_hidden:
            layers.extend([nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden
        layers.append(nn.Linear(input_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)]
        projs = [proj(x_num[:, i:i + 1]) for i, proj in enumerate(self.num_projs)]
        x_stacked = torch.stack(embs + projs, dim=1)
        attention_weights = self.attention_net(x_stacked.view(x_stacked.size(0), -1)).unsqueeze(-1)
        x_masked = x_stacked * attention_weights
        return self.classifier(x_masked.view(x_masked.size(0), -1)).squeeze(-1)


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
        model = IFAN(cat_cards=cat_cardinalities, num_count=X_train_num.shape[1], emb_dim=EMBED_DIM,
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
