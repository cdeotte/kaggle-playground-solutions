#!/usr/bin/env python
"""Liquid neural network on the raw columns plus service-count, charge-relation and bin features (categoricals fed as
ordinal numbers).

Features (train and test separately; no original rows):
- the 19 columns; MonthlyCharges / TotalCharges are read as strings, stripped and parsed (blank TotalCharges ->
  tenure x MonthlyCharges); string columns stripped;
- _ServiceCount = number of the 9 service columns equal to Yes / Fiber optic / DSL;
- _AvgMonthlyChargeFromTotal = TotalCharges / (tenure + 1), _ChargeGap = TotalCharges - tenure x MonthlyCharges;
- tenure bins (edges 6, 12, 24, 48) and 8-quantile bins of MonthlyCharges (qcut within the frame) as categoricals.
Inputs per fold: the columns that are not constant on the training rows; the 17 categoricals ordinal-encoded on the
training rows (unknown -> 0) and fed as plain numbers; the continuous columns min-max scaled (scaler fitted on the
training values + N(0, 1e-5) noise, noise seed 42).
Model: 3 liquid layers of 256 units, h <- h + (SiLU(W x + U h) - h) / tau with a learnable tau per unit (the next layer
gets the raw layer output), each followed by a residual connection when the shapes match, GroupNorm(4 groups) and
dropout 0.2; linear readout to 1 logit.
Training: BCE with logits, AdamW lr 1e-3, weight decay 1e-4, OneCycle (20% warm-up, stepped per batch), batch 256,
gradient-norm clip 0.1, bf16 autocast, up to 30 epochs with early stopping on the held-out AUC (patience 10); the best
epoch predicts (sigmoid, fp32). Seed 42 once before the folds, so the pair is reproduced only when the 5 folds run in
order in one process. Test = mean of the 5 fold models.
Usage: python train_liquid_nn_ordinal.py [--folds 0,1,2,3,4]"""
import argparse
import gc
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import DATA_DIR, ID, TARGET, folds, model_dir  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'liquid_nn_ordinal_service_bins'
MODEL_DIR = model_dir('neural_nets')
SEED = 42
STRING_DTYPES = {'TotalCharges': 'string', 'MonthlyCharges': 'string'}


class CFG:
    hidden_dim = 256
    output_dim = 1
    num_layers = 3
    num_groups = 4
    dropout_prob = 0.2
    learning_rate = 1e-3
    weight_decay = 1e-4
    warmup_ratio = 0.2
    clip_grad_norm = True
    max_norm = 0.1
    batch_size = 256
    max_epochs = 30
    patience = 10
    noise_std = 1e-5
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clean_totalcharges(df):
    df = df.copy()
    for c in ['TotalCharges', 'MonthlyCharges']:
        df[c] = df[c].astype('string').str.strip()
    df['TotalCharges'] = df['TotalCharges'].replace('', pd.NA).replace(' ', pd.NA)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    return df


def strip_string_cols(df):
    df = df.copy()
    for c in df.select_dtypes(include=['object', 'string']).columns.tolist():
        df[c] = df[c].astype('string').str.strip()
    return df


def feature_engineering(df):
    df = df.copy()
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    yes_map = {'Yes': 1, 'Fiber optic': 1, 'DSL': 1}
    tmp = pd.DataFrame(index=df.index)
    for c in service_cols:
        tmp[c] = df[c].astype(str).map(yes_map).fillna(0).astype('int8')
    df['_ServiceCount'] = tmp.sum(axis=1).astype('float32')

    df['_AvgMonthlyChargeFromTotal'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    df['_ChargeGap'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')

    df['tenure_bin_'] = pd.cut(df['tenure'], bins=[-np.inf, 6, 12, 24, 48, np.inf], labels=False).fillna(-1).astype(
        'int32').astype('category')
    try:
        q = pd.qcut(df['MonthlyCharges'], q=8, labels=False, duplicates='drop')
        df['MonthlyCharges_bin_'] = q.fillna(-1).astype('int32').astype('category')
    except Exception:
        pass
    return df


class LiquidNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.tau = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x, h):
        dx = F.silu(self.W(x) + self.U(h))
        dh = (dx - h) / (self.tau + 1e-6)
        return h + dh


class LiquidNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2, num_groups=4, dropout_prob=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.liquid_layers = nn.ModuleList([LiquidNeuron(input_dim if i == 0 else hidden_dim, hidden_dim)
                                            for i in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.GroupNorm(num_groups, hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_prob)
        self.readout_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        for i in range(self.num_layers):
            residual = x
            h = self.liquid_layers[i](x, h)
            x = h
            if residual.shape == h.shape:
                h = h + residual
            h = self.layer_norms[i](h)
            h = self.dropout(h)
        return self.readout_linear(h)


class LNNClassifier:
    def __init__(self, random_state: int = SEED, device: str = CFG.device):
        self.random_state = random_state
        self.device = torch.device(device)

    def _preprocess_data(self, X: pd.DataFrame, y: Optional[np.ndarray], training: bool):
        if training:
            self._non_constant_columns = X.columns[X.nunique(dropna=False) > 1]
        X = X[self._non_constant_columns].copy()

        categorical_features = [col for col in X.columns if X[col].dtype.name in ['object', 'category', 'string']]
        X_cat = (X[categorical_features].astype(str).to_numpy() if len(categorical_features)
                 else np.zeros((len(X), 0), dtype=np.float32))
        X_cont = (X.drop(columns=categorical_features).to_numpy(dtype=np.float32)
                  if len(categorical_features) < X.shape[1] else np.zeros((len(X), 0), dtype=np.float32))

        if training:
            self._categorical_encoders = [
                OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(X_cat[:, i:i + 1])
                for i in range(X_cat.shape[1])
            ]
        if X_cat.shape[1] > 0:
            X_cat = np.concatenate([self._categorical_encoders[i].transform(X_cat[:, i:i + 1])
                                    for i in range(X_cat.shape[1])], axis=1).astype(np.float32)
            X_cat[X_cat < 0] = 0
        else:
            X_cat = np.zeros((len(X), 0), dtype=np.float32)

        if X_cont.shape[1] > 0:
            if training:
                noise = (np.random.default_rng(self.random_state).normal(0.0, CFG.noise_std, X_cont.shape)
                         .astype(np.float32))
                self._cont_feature_preprocessor = MinMaxScaler().fit(X_cont + noise)
            X_cont = self._cont_feature_preprocessor.transform(X_cont).astype(np.float32)
        else:
            X_cont = np.zeros((len(X), 0), dtype=np.float32)

        X_cat = torch.tensor(X_cat, dtype=torch.float32, device=self.device)
        X_cont = torch.tensor(X_cont, dtype=torch.float32, device=self.device)
        X_cat_cont = torch.cat([X_cat, X_cont], dim=1)
        if y is not None:
            y = torch.tensor(y.astype(np.float32), dtype=torch.float32, device=self.device)
        return X_cat_cont, y

    def fit(self, X: pd.DataFrame, y: np.ndarray, eval_set: Tuple[pd.DataFrame, np.ndarray]):
        X_train, y_train = self._preprocess_data(X, y, training=True)
        X_val, y_val = self._preprocess_data(eval_set[0], eval_set[1], training=False)

        self.model = LiquidNetwork(input_dim=X_train.shape[1], hidden_dim=CFG.hidden_dim, output_dim=CFG.output_dim,
                                   num_layers=CFG.num_layers, num_groups=CFG.num_groups,
                                   dropout_prob=CFG.dropout_prob).to(self.device)
        total_steps = CFG.max_epochs * math.ceil(len(X) / CFG.batch_size)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.learning_rate, total_steps=total_steps,
                                                        pct_start=CFG.warmup_ratio, cycle_momentum=False)
        loss_fn = torch.nn.BCEWithLogitsLoss().to(self.device)
        use_amp = self.device.type == 'cuda'
        best = {'epoch': -1, 'val_auc': -np.inf, 'state': None}
        remaining_patience = CFG.patience

        for epoch in range(CFG.max_epochs):
            self.model.train()
            train_losses = []
            for batch_idx in torch.randperm(len(y_train), device=self.device).split(CFG.batch_size):
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    logits = self.model(X_train[batch_idx]).squeeze(-1).float()
                    loss = loss_fn(logits, y_train[batch_idx])
                loss.backward()
                if CFG.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), CFG.max_norm)
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

            self.model.eval()
            val_losses, val_probs = [], []
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    for start in range(0, len(y_val), CFG.batch_size):
                        logits = self.model(X_val[start:start + CFG.batch_size]).squeeze(-1).float()
                        val_losses.append(loss_fn(logits, y_val[start:start + CFG.batch_size]).item())
                        val_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            val_probs = np.concatenate(val_probs)
            val_auc = roc_auc_score(y_val.detach().cpu().numpy(), val_probs)
            print(f'Epoch {epoch + 1} | Train Loss: {float(np.mean(train_losses)):.6f} | '
                  f'Val Loss: {float(np.mean(val_losses)):.6f} | Val AUC: {val_auc:.6f}', flush=True)
            if val_auc > best['val_auc']:
                best = {'epoch': epoch, 'val_auc': val_auc,
                        'state': {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}}
                remaining_patience = CFG.patience
            else:
                remaining_patience -= 1
            if remaining_patience == 0:
                print('Early stopping triggered.', flush=True)
                break

        self.model.load_state_dict(best['state'])
        print(f'Best epoch: {best["epoch"] + 1} | best val AUC: {best["val_auc"]:.6f}', flush=True)
        return self

    def predict_proba(self, X: pd.DataFrame, batch_size: Optional[int] = 8192) -> np.ndarray:
        X_all, _ = self._preprocess_data(X, y=None, training=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                logits = self.model(X_all[start:start + batch_size]).squeeze(-1).float()
                preds.append(torch.sigmoid(logits).cpu().numpy())
        return np.clip(np.concatenate(preds), 0, 1)


def load_data():
    train = pd.read_csv(DATA_DIR / 'train.csv', dtype=STRING_DTYPES)
    test = pd.read_csv(DATA_DIR / 'test.csv', dtype=STRING_DTYPES)
    if train[TARGET].dtype == 'O' or str(train[TARGET].dtype).startswith('string'):
        train[TARGET] = train[TARGET].astype(str).str.strip().map({'Yes': 1, 'No': 0})
    train[TARGET] = train[TARGET].astype('int64')

    train = clean_totalcharges(train)
    test = clean_totalcharges(test)
    for df in (train, test):
        if df['TotalCharges'].isna().any():
            df['TotalCharges'] = df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'])
    train = strip_string_cols(train)
    test = strip_string_cols(test)

    X = train.drop([ID, TARGET], axis=1).copy()
    y = train[TARGET].copy()
    X_test = test.drop([ID], axis=1).copy()
    X = feature_engineering(X)
    X_test = feature_engineering(X_test)
    print(f'X {X.shape} | X_test {X_test.shape}', flush=True)
    return X, y, X_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(SEED)
    X, y, X_test = load_data()
    clean_memory()

    fold_ids = folds(y.to_numpy())
    for k in fold_list(args.folds):
        print('#' * 25, flush=True)
        print(f'### Fold {k + 1}/5 ...', flush=True)
        print('#' * 25, flush=True)
        train_idx, valid_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        model = LNNClassifier()
        model.fit(X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].to_numpy(),
                  eval_set=(X.iloc[valid_idx].reset_index(drop=True), y.iloc[valid_idx].to_numpy()))
        oof_pred = model.predict_proba(X.iloc[valid_idx].reset_index(drop=True))
        test_pred = model.predict_proba(X_test)
        print(f'\nFold {k + 1} | AUC: {roc_auc_score(y.iloc[valid_idx].to_numpy(), oof_pred):.6f}\n', flush=True)
        save_part(MODEL_DIR, PAIR, k, oof_pred, test_pred)
        del model, oof_pred, test_pred
        clean_memory()
    assemble(MODEL_DIR, PAIR, fold_ids, y.to_numpy())


if __name__ == '__main__':
    main()
