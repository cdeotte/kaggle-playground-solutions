#!/usr/bin/env python
"""Liquid neural network on the raw columns plus water-balance / load features and 8-quantile bins.

Features (no original rows):
- the 8 categoricals (stripped strings) and the 11 numerics (float32, missing -> median of the frame);
- _WaterBalance = Rainfall_mm + Previous_Irrigation_mm - Soil_Moisture, _TempHumidity = Temperature_C x Humidity,
  _IrrigationLoad = Field_Area_hectare x Previous_Irrigation_mm, _MgmtCount = mulch yes + drip / sprinkler +
  non-rain water source + Summer / Kharif season;
- 8-quantile bins of Rainfall_mm and Soil_Moisture as categoricals (qcut within train and within test separately).
Inputs per fold: the columns that are not constant on the training rows; the 10 categoricals ordinal-encoded on the
training rows (unknown -> 0) and fed as plain numbers; the continuous columns min-max scaled (scaler fitted on the
training values + N(0, 1e-5) noise, noise seed 42).
Model: 3 liquid layers of 256 units, h <- h + (SiLU(W x + U h) - h) / tau with a learnable tau per unit (the next layer
gets the raw layer output), each followed by a residual connection when the shapes match, GroupNorm(4 groups) and
dropout 0.2; linear 3-class readout.
Training: cross-entropy, AdamW lr 1e-3, weight decay 1e-4, OneCycle (20% warm-up, stepped per batch), batch 256,
gradient-norm clip 0.1, bf16 autocast, up to 30 epochs with early stopping on the held-out balanced accuracy (patience
10); the best epoch predicts (softmax, fp32). Seed 42 once before the folds. Test = mean of the 5 fold models.
Usage: python train_liquid_nn.py [--folds 0,1,2,3,4]"""
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, model_dir, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'liquid_nn_water_balance_bins_a'
MODEL_DIR = model_dir('neural_nets')
SEED = 42


class CFG:
    hidden_dim = 256
    output_dim = 3
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


def feature_engineering(df):
    df = df.copy()
    df['_WaterBalance'] = (pd.to_numeric(df['Rainfall_mm'], errors='coerce').fillna(0)
                           + pd.to_numeric(df['Previous_Irrigation_mm'], errors='coerce').fillna(0)
                           - pd.to_numeric(df['Soil_Moisture'], errors='coerce').fillna(0)).astype('float32')
    df['_TempHumidity'] = (pd.to_numeric(df['Temperature_C'], errors='coerce').fillna(0)
                           * pd.to_numeric(df['Humidity'], errors='coerce').fillna(0)).astype('float32')
    df['_IrrigationLoad'] = (pd.to_numeric(df['Field_Area_hectare'], errors='coerce').fillna(0)
                             * pd.to_numeric(df['Previous_Irrigation_mm'], errors='coerce').fillna(0)).astype('float32')

    tmp = pd.DataFrame(index=df.index)
    tmp['m'] = df['Mulching_Used'].astype(str).eq('Yes').astype('int8')
    tmp['i'] = df['Irrigation_Type'].astype(str).isin(['Drip', 'Sprinkler']).astype('int8')
    tmp['w'] = (~df['Water_Source'].astype(str).eq('Rainwater')).astype('int8')
    tmp['s'] = df['Season'].astype(str).isin(['Summer', 'Kharif']).astype('int8')
    df['_MgmtCount'] = tmp.sum(axis=1).astype('float32')

    for col, name in [('Rainfall_mm', 'Rainfall_bin_'), ('Soil_Moisture', 'Soil_Moisture_bin_')]:
        try:
            q = pd.qcut(pd.to_numeric(df[col], errors='coerce'), q=8, labels=False, duplicates='drop')
            df[name] = q.fillna(-1).astype('int32').astype('category')
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
    def __init__(self, input_dim, hidden_dim=64, output_dim=3, num_layers=2, num_groups=4, dropout_prob=0.2):
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
                noise = np.random.default_rng(self.random_state).normal(0.0, CFG.noise_std, X_cont.shape).astype(np.float32)
                self._cont_feature_preprocessor = MinMaxScaler().fit(X_cont + noise)
            X_cont = self._cont_feature_preprocessor.transform(X_cont).astype(np.float32)
        else:
            X_cont = np.zeros((len(X), 0), dtype=np.float32)

        X_cat = torch.tensor(X_cat, dtype=torch.float32, device=self.device)
        X_cont = torch.tensor(X_cont, dtype=torch.float32, device=self.device)
        X_cat_cont = torch.cat([X_cat, X_cont], dim=1)
        if y is not None:
            y = torch.tensor(y.astype(np.int64), dtype=torch.long, device=self.device)
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
        loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        use_amp = self.device.type == 'cuda'
        best = {'epoch': -1, 'val_bac': -np.inf, 'state': None}
        remaining_patience = CFG.patience

        for epoch in range(CFG.max_epochs):
            self.model.train()
            train_losses = []
            for batch_idx in torch.randperm(len(y_train), device=self.device).split(CFG.batch_size):
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    logits = self.model(X_train[batch_idx]).float()
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
                        logits = self.model(X_val[start:start + CFG.batch_size]).float()
                        val_losses.append(loss_fn(logits, y_val[start:start + CFG.batch_size]).item())
                        val_probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            val_probs = np.concatenate(val_probs)
            val_bac = balanced_accuracy_score(y_val.detach().cpu().numpy(), val_probs.argmax(axis=1))
            print(f'Epoch {epoch + 1} | Train Loss: {float(np.mean(train_losses)):.6f} | '
                  f'Val Loss: {float(np.mean(val_losses)):.6f} | Val BAC: {val_bac:.6f}', flush=True)
            if val_bac > best['val_bac']:
                best = {'epoch': epoch, 'val_bac': val_bac,
                        'state': {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}}
                remaining_patience = CFG.patience
            else:
                remaining_patience -= 1
            if remaining_patience == 0:
                break

        self.model.load_state_dict(best['state'])
        return self

    def predict_proba(self, X: pd.DataFrame, batch_size: Optional[int] = 8192) -> np.ndarray:
        X_all, _ = self._preprocess_data(X, y=None, training=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                logits = self.model(X_all[start:start + batch_size]).float()
                preds.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(SEED)
    clean_memory()
    train = read_train()
    test = read_test()
    train[TARGET] = train[TARGET].map(TARGET_MAP).astype('int64')
    X = train.drop([ID, TARGET], axis=1).copy()
    y = train[TARGET].copy()
    X_test = test.drop([ID], axis=1).copy()

    for df in (X, X_test):
        obj_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        for c in obj_cols:
            df[c] = df[c].astype('string').str.strip()
        for c in [c for c in df.columns if c not in obj_cols]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            med = float(np.nanmedian(df[c].values)) if np.isfinite(np.nanmedian(df[c].values)) else 0.0
            df[c] = df[c].fillna(med).astype('float32')

    X = feature_engineering(X)
    X_test = feature_engineering(X_test)
    cat_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    print(f'cat_cols: {cat_cols}', flush=True)
    print(f'num_cols: {[c for c in X.columns if c not in cat_cols]}', flush=True)

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
        print(f'Fold {k + 1} | BAC: {balanced_accuracy_score(y.iloc[valid_idx].to_numpy(), oof_pred.argmax(axis=1)):.6f}',
              flush=True)
        save_part(MODEL_DIR, PAIR, k, oof_pred, test_pred)
        del model, oof_pred, test_pred
        clean_memory()
    assemble(MODEL_DIR, PAIR, fold_ids, y.to_numpy())


if __name__ == '__main__':
    main()
