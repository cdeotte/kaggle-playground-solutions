#!/usr/bin/env python
"""FT-Transformer (PyTorch, CLS token) on frequency / rank / transform / digit / service features of the numerics,
4 categorical crosses and per-fold nested target encodings of all 20 categoricals.

Features, built once on train + test (the 3 numerics as float32, NaN -> the train median):
- 20 categoricals (factorized on train + test): the 16 categoricals and Contract x InternetService, PaymentMethod x
  Contract, InternetService x OnlineSecurity, Contract x PaperlessBilling;
- numerics: the 3 raw values; their frequency (share of rows) and percentile rank over train + test; log1p and sqrt;
  TotalCharges - tenure x MonthlyCharges, MonthlyCharges / (TotalCharges + 1), TotalCharges / (tenure + 1); count of
  'Yes' over 8 service columns, has-internet flag; for MonthlyCharges and TotalCharges the length, first and last digit
  of the value string without the decimal point and the fractional part.
Per fold: scikit-learn `TargetEncoder` (5 inner shuffled folds, seed 42, smooth 'auto') of the 20 categoricals, fitted on
the fold's training rows (out-of-fold values for them), adds 20 numerics; the 48 numerics are standardized on the fold's
training rows (NaN -> 0).
Network: a CLS token, a 128-d token per categorical (embedding of code + 1, padding index 0) and per numeric
(x w_i + b_i), 3 post-norm Transformer encoder layers (8 heads, FFN 512, dropout 0.2); CLS output -> LayerNorm -> ReLU
-> Linear -> 1 logit.
Training: BCE, AdamW lr 5e-4, weight decay 1e-4, cosine annealing over 40 epochs, batch 2048 (last partial batch
dropped), mixed precision; early stopping on the held-out AUC (patience 6) and the best epoch predicts (sigmoid).
Python / numpy / torch are seeded with 42 (deterministic cuDNN) before the folds.
Usage: python train_ft_transformer_cls.py"""
import gc
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.amp
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, TargetEncoder
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, model_dir, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'ft_transformer_cls_engineered_nested_te'
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5
BATCH_SIZE = 2048
EPOCHS = 40
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 6
D_TOKEN = 128
N_BLOCKS = 3
N_HEADS = 8
DROPOUT = 0.2

CATS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod']
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
SVC = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies']
CROSS_PAIRS = [
    ('Contract', 'InternetService'),
    ('PaymentMethod', 'Contract'),
    ('InternetService', 'OnlineSecurity'),
    ('Contract', 'PaperlessBilling'),
]


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            s = df[c].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
            df[c] = pd.to_numeric(s, errors='coerce').astype(np.float32)
    return df


def build_features():
    train, test = read_train(), read_test()
    train[TARGET] = train[TARGET].map({'No': 0, 'Yes': 1}).astype(np.int8)
    y_train = train[TARGET].values
    train = coerce_numeric(train, NUMS)
    test = coerce_numeric(test, NUMS)
    for c in NUMS:
        med = train[c].median()
        train[c] = train[c].fillna(med)
        test[c] = test[c].fillna(med)

    combined = pd.concat([train.drop(columns=[TARGET, ID], errors='ignore'), test.drop(columns=[ID], errors='ignore')],
                         axis=0, ignore_index=True)
    for col in NUMS:
        freq = combined[col].value_counts(normalize=True)
        combined[f'FREQ_{col}'] = combined[col].map(freq).astype(np.float32)
        combined[f'RANK_{col}'] = combined[col].rank(method='average', pct=True).astype(np.float32)
    for col in NUMS:
        combined[f'LOG1P_{col}'] = np.log1p(combined[col].clip(lower=0)).astype(np.float32)
        combined[f'SQRT_{col}'] = np.sqrt(combined[col].clip(lower=0)).astype(np.float32)
    combined['charges_deviation'] = (combined['TotalCharges'] - combined['tenure'] * combined['MonthlyCharges']).astype(
        np.float32)
    combined['monthly_to_total_ratio'] = (combined['MonthlyCharges'] / (combined['TotalCharges'] + 1)).astype(np.float32)
    combined['avg_monthly_charges'] = (combined['TotalCharges'] / (combined['tenure'] + 1)).astype(np.float32)
    combined['service_count'] = (combined[SVC] == 'Yes').sum(axis=1).astype(np.float32)
    combined['has_internet'] = (combined['InternetService'] != 'No').astype(np.float32)
    for col in ['MonthlyCharges', 'TotalCharges']:
        c_str = combined[col].astype(str).str.replace('.', '', regex=False)
        combined[f'{col}_len'] = c_str.str.len().astype(np.float32)
        combined[f'{col}_first_digit'] = c_str.str[0].replace('-', '0').astype(np.float32)
        combined[f'{col}_last_digit'] = c_str.str[-1].replace('-', '0').astype(np.float32)
        combined[f'{col}_frac'] = (combined[col] % 1).astype(np.float32)

    cross_cats = []
    for a, b in CROSS_PAIRS:
        combined[f'{a}__{b}'] = combined[a].astype(str) + '|' + combined[b].astype(str)
        cross_cats.append(f'{a}__{b}')
    all_cats = CATS + cross_cats
    for c in all_cats:
        combined[c] = pd.factorize(combined[c].astype(str))[0]
    return combined.iloc[:len(train)].copy(), combined.iloc[len(train):].copy(), y_train, all_cats


class TabularDataset(Dataset):
    def __init__(self, x_cat, x_num, y=None):
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.x_cat)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_cat[idx], self.x_num[idx], self.y[idx]
        return self.x_cat[idx], self.x_num[idx]


class FTTransformer(nn.Module):
    def __init__(self, cat_cardinalities, num_features, d_token=128, n_blocks=3, n_heads=8, dropout=0.2):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(card + 1, d_token, padding_idx=0)
                                             for card in cat_cardinalities])
        self.num_weights = nn.Parameter(torch.randn(num_features, d_token))
        self.num_biases = nn.Parameter(torch.randn(num_features, d_token))
        nn.init.xavier_uniform_(self.num_weights)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 1))

    def forward(self, x_cat, x_num):
        cat_tokens = [emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_embeddings)]
        num_tokens = x_num.unsqueeze(-1) * self.num_weights.unsqueeze(0) + self.num_biases.unsqueeze(0)
        cls_tokens = self.cls_token.expand(x_cat.size(0), -1, -1)
        if len(cat_tokens) > 0:
            x_seq = torch.cat([cls_tokens] + cat_tokens + [num_tokens], dim=1)
        else:
            x_seq = torch.cat([cls_tokens, num_tokens], dim=1)
        return self.head(self.transformer(x_seq)[:, 0, :]).squeeze(-1)


@torch.no_grad()
def predict(model, loader):
    preds = []
    for batch in loader:
        with torch.amp.autocast('cuda'):
            preds.extend(torch.sigmoid(model(batch[0].to(DEVICE), batch[1].to(DEVICE))).cpu().numpy())
    return np.array(preds)


def main():
    seed_everything(SEED)
    train_fe, test_fe, y_train, all_cats = build_features()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(train_fe), dtype=np.float32)
    test_preds = np.zeros(len(test_fe), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_fe, y_train), 1):
        X_tr = train_fe.iloc[tr_idx].copy().reset_index(drop=True)
        y_tr = y_train[tr_idx]
        X_va = train_fe.iloc[va_idx].copy().reset_index(drop=True)
        y_va = y_train[va_idx]
        X_te = test_fe.copy().reset_index(drop=True)

        te_mean_cols = [f'TE_{c}' for c in all_cats]
        te_enc = TargetEncoder(cv=INNER_FOLDS, shuffle=True, smooth='auto', target_type='binary', random_state=SEED)
        X_tr[te_mean_cols] = te_enc.fit_transform(X_tr[all_cats].to_numpy(), y_tr)
        X_va[te_mean_cols] = te_enc.transform(X_va[all_cats].to_numpy())
        X_te[te_mean_cols] = te_enc.transform(X_te[all_cats].to_numpy())

        num_cols = [c for c in X_tr.columns if c not in all_cats]
        X_tr_cat = X_tr[all_cats].values + 1
        X_va_cat = X_va[all_cats].values + 1
        X_te_cat = X_te[all_cats].values + 1
        cat_cards = [int(np.max(X_tr_cat[:, i])) + 1 for i in range(len(all_cats))]
        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(X_tr[num_cols].fillna(0).astype(np.float32))
        X_va_num = scaler.transform(X_va[num_cols].fillna(0).astype(np.float32))
        X_te_num = scaler.transform(X_te[num_cols].fillna(0).astype(np.float32))
        if fold == 1:
            print(f'{len(num_cols)} numeric + {len(all_cats)} categorical tokens', flush=True)

        train_loader = DataLoader(TabularDataset(X_tr_cat, X_tr_num, y_tr), batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True)
        val_loader = DataLoader(TabularDataset(X_va_cat, X_va_num, y_va), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TabularDataset(X_te_cat, X_te_num), batch_size=BATCH_SIZE, shuffle=False)

        model = FTTransformer(cat_cardinalities=cat_cards, num_features=len(num_cols), d_token=D_TOKEN,
                              n_blocks=N_BLOCKS, n_heads=N_HEADS, dropout=DROPOUT).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler_amp = torch.amp.GradScaler('cuda')

        best_auc, best_val_preds, fold_test_preds, patience = 0, None, None, 0
        for epoch in range(EPOCHS):
            model.train()
            for x_c, x_n, y_b in train_loader:
                x_c, x_n, y_b = x_c.to(DEVICE), x_n.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(x_c, x_n), y_b)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            scheduler.step()

            model.eval()
            val_preds = predict(model, val_loader)
            val_auc = roc_auc_score(y_va, val_preds)
            print(f'  epoch {epoch + 1:02d} | val_auc {val_auc:.6f}', flush=True)
            if val_auc > best_auc:
                best_auc, best_val_preds, patience = val_auc, val_preds, 0
                fold_test_preds = predict(model, test_loader)
            else:
                patience += 1
            if patience >= PATIENCE:
                break

        oof_preds[va_idx] = best_val_preds
        test_preds += fold_test_preds / N_FOLDS
        print(f'Fold {fold} | Best AUC: {best_auc:.6f}', flush=True)

        del model, train_loader, val_loader, test_loader, X_tr, X_va, X_te
        gc.collect()
        torch.cuda.empty_cache()
    print(f'OOF AUC: {roc_auc_score(y_train, oof_preds):.6f}', flush=True)
    save_pair(MODEL_DIR, PAIR, oof_preds, test_preds)


if __name__ == '__main__':
    main()
