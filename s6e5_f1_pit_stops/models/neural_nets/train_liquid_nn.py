#!/usr/bin/env python
"""Liquid neural network on F1 state features, quantile bins, rounded keys, combo keys and original-data priors.

Features (built once on train + test, with the original rows as reference only; the original rows are not trained on):
- the 14 columns: Driver, Compound, Race as string categoricals, every numeric as float32;
- 6 F1 state features: LapNumber - TyreLife, (|LapTime_Delta| + Cumulative_Degradation) / (TyreLife + 1),
  LapTime / (RaceProgress + 0.01), (Position + |Position_Change|) / (RaceProgress + 0.02), PitStop + Stint + RaceProgress,
  LapTime_Delta / (|LapTime| + 1);
- categoricals: 8-quantile bins (edges from train + test + original) of the high-cardinality numerics (LapTime (s),
  LapTime_Delta, Cumulative_Degradation, RaceProgress) and of TyreLife, LapNumber, Position; the 4 high-cardinality
  numerics rounded to 2 and to 3 decimals; 6 combo keys Driver x Compound, Driver x Race, Race x Year, Compound x TyreLife,
  Stint x TyreLife, Driver x PitStop;
- original-data target mean (unseen keys -> original prior) and log1p count of Driver, Compound, Race, of every numeric
  with < 100 distinct values in train + test, of every bin / round / combo key and of the 4 high-cardinality numerics
  (all 101,371 original rows).
Inputs per fold: the columns that are not constant on the outer-train rows; categoricals ordinal-encoded on the
outer-train rows (unknown -> 0) and fed as plain numbers, continuous columns min-max scaled (scaler fit on the
outer-train values + N(0, 1e-5) noise, noise seed 42 + fold with folds numbered 1-5).
Model: 3 liquid layers of 256 units (h <- h + (SiLU(W x + U h) - h) / tau with a learnable tau per unit; the next layer
gets the raw layer output), each followed by a residual connection when the shapes match, GroupNorm(4 groups) and
dropout 0.2; linear 2-class readout. Cross-entropy, AdamW lr 1e-3, weight decay 1e-4, OneCycle schedule (20% warm-up,
stepped per batch), batch 256, gradient-norm clipping 0.1, bf16 autocast, up to 30 epochs with early stopping on the
held-out fold's AUC (patience 10); the best epoch predicts (softmax probability of class 1). Seed 42 once at the start
(the folds run in sequence in one process). Test = mean of the 5 fold models.
Usage: python train_liquid_nn.py [--folds 0,1,2,3,4]"""
import argparse
import gc
import math
import os
import random
import re
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
from common.data import ID, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'liquid_nn_f1_state_original_priors'
MODEL_DIR = model_dir('neural_nets')
SEED = 42


class CFG:
    hidden_dim = 256
    output_dim = 2
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def safe_name(x):
    return re.sub(r'[^0-9A-Za-z_]+', '_', str(x)).strip('_')


def safe_num(s):
    return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')


def safe_str(s):
    return s.astype('string').fillna('missing').str.strip()


def infer_groups(train, test, orig):
    """cats: string columns + numerics with < 100 distinct values in train + test; nums: the other numerics;
    numeric_like: every numeric column."""
    feature_cols = [c for c in train.columns if c not in [ID, TARGET] and c in test.columns and c in orig.columns]
    cats, nums, numeric_like = [], [], []
    both = pd.concat([train[feature_cols], test[feature_cols]], axis=0, ignore_index=True)
    for c in feature_cols:
        if train[c].dtype == object or test[c].dtype == object or orig[c].dtype == object:
            cats.append(c)
        else:
            numeric_like.append(c)
            if both[c].nunique(dropna=False) < 100:
                cats.append(c)
            else:
                nums.append(c)
    return cats, nums, numeric_like


def add_quantile_bin(train, test, orig, col, q=8):
    name = f'{safe_name(col)}_bin_'
    ref = pd.concat([train[col], test[col], orig[col]], ignore_index=True)
    try:
        _, bins = pd.qcut(safe_num(ref), q=q, labels=False, retbins=True, duplicates='drop')
        for df in (train, test, orig):
            df[name] = pd.cut(safe_num(df[col]), bins=bins, labels=False, include_lowest=True).fillna(-1).astype('int32').astype('category')
    except Exception:
        for df in (train, test, orig):
            df[name] = '0'
    return name


def feature_engineering(train, test, orig, y_orig, cats, nums, numeric_like):
    """Adds the features to train, test and orig in place; returns train and test."""
    for df in (train, test, orig):
        for c in cats:
            df[c] = safe_str(df[c])
        for c in numeric_like:  # low-cardinality numerics are back to float32 here
            df[c] = safe_num(df[c])

    new_cat_cols = []
    high_card = [c for c in ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress'] if c in nums]

    for df in (train, test, orig):
        lap = safe_num(df['LapNumber'])
        tyre = safe_num(df['TyreLife'])
        lt = safe_num(df['LapTime (s)'])
        ld = safe_num(df['LapTime_Delta'])
        deg = safe_num(df['Cumulative_Degradation'])
        prog = safe_num(df['RaceProgress'])
        pos = safe_num(df['Position'])
        pchg = safe_num(df['Position_Change'])
        pit = safe_num(df['PitStop'])
        stint = safe_num(df['Stint'])
        df['_LapTyreBalance'] = (lap - tyre).astype('float32')
        df['_TimeDegradation'] = ((ld.abs() + deg) / (tyre + 1.0)).astype('float32')
        df['_LapTimeProgress'] = (lt / (prog + 0.01)).astype('float32')
        df['_PositionPressure'] = ((pos + pchg.abs()) / (prog + 0.02)).astype('float32')
        df['_PitState'] = (pit + stint + prog).astype('float32')
        df['_DeltaTimeRatio'] = (ld / (lt.abs() + 1.0)).astype('float32')

    for col in high_card + [c for c in ['TyreLife', 'LapNumber', 'Position'] if c in numeric_like]:
        new_cat_cols.append(add_quantile_bin(train, test, orig, col, q=8))
    for col in high_card:
        base = safe_name(col)
        for dec in [2, 3]:
            name = f'{base}_round{dec}_'
            for df in (train, test, orig):
                df[name] = safe_num(df[col]).round(dec).astype(str).astype('category')
            new_cat_cols.append(name)

    for a, b in [('Driver', 'Compound'), ('Driver', 'Race'), ('Race', 'Year'), ('Compound', 'TyreLife'), ('Stint', 'TyreLife'), ('Driver', 'PitStop')]:
        if a in train.columns and b in train.columns:
            name = f'{safe_name(a)}__{safe_name(b)}_'
            for df in (train, test, orig):
                df[name] = (safe_str(df[a]) + '__' + safe_str(df[b])).astype('category')
            new_cat_cols.append(name)

    prior = float(np.mean(y_orig))
    prior_cols = list(dict.fromkeys(cats + new_cat_cols + high_card))
    for col in prior_cols:
        if col not in orig.columns:
            continue
        stats = pd.DataFrame({'key': orig[col].astype(str).values, 'target': y_orig}).groupby('key', dropna=False)['target'].agg(['mean', 'count'])
        mean_name = f'ORIG_mean_{safe_name(col)}'
        count_name = f'ORIG_count_{safe_name(col)}'
        for df in (train, test, orig):
            key = df[col].astype(str)
            df[mean_name] = key.map(stats['mean']).fillna(prior).astype('float32')
            df[count_name] = np.log1p(key.map(stats['count']).fillna(0).astype('float32'))

    return train, test


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
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, num_layers=2, num_groups=4, dropout_prob=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.liquid_layers = nn.ModuleList([LiquidNeuron(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.GroupNorm(num_groups, hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_prob)
        self.readout_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
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
        X_cat = X[categorical_features].astype(str).to_numpy() if categorical_features else np.zeros((len(X), 0), dtype=np.float32)
        X_cont = X.drop(columns=categorical_features).to_numpy(dtype=np.float32) if len(categorical_features) < X.shape[1] else np.zeros((len(X), 0), dtype=np.float32)
        if training:
            self._categorical_features = categorical_features
            self._categorical_encoders = [OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(X_cat[:, i:i + 1]) for i in range(X_cat.shape[1])]
        if X_cat.shape[1] > 0:
            X_cat = np.concatenate([self._categorical_encoders[i].transform(X_cat[:, i:i + 1]) for i in range(X_cat.shape[1])], axis=1).astype(np.float32)
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
        self.model = LiquidNetwork(input_dim=X_train.shape[1], hidden_dim=CFG.hidden_dim, output_dim=CFG.output_dim, num_layers=CFG.num_layers, num_groups=CFG.num_groups, dropout_prob=CFG.dropout_prob).to(self.device)
        total_steps = CFG.max_epochs * math.ceil(len(X) / CFG.batch_size)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.learning_rate, total_steps=total_steps, pct_start=CFG.warmup_ratio, cycle_momentum=False)
        loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        best = {'epoch': -1, 'val_auc': -np.inf, 'state': None}
        remaining_patience = CFG.patience
        use_amp = self.device.type == 'cuda'
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
            val_probs = []
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    for start in range(0, len(y_val), CFG.batch_size):
                        logits = self.model(X_val[start:start + CFG.batch_size]).float()
                        val_probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
            val_probs = np.concatenate(val_probs)
            val_auc = roc_auc_score(y_val.detach().cpu().numpy(), val_probs)
            print(f'Epoch {epoch + 1} | Train Loss: {float(np.mean(train_losses)):.6f} | Val AUC: {val_auc:.6f}', flush=True)
            if val_auc > best['val_auc']:
                best = {'epoch': epoch, 'val_auc': val_auc, 'state': {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}}
                remaining_patience = CFG.patience
            else:
                remaining_patience -= 1
            if remaining_patience == 0:
                break
        self.model.load_state_dict(best['state'])
        self.best_epoch_ = best['epoch'] + 1
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
    run_folds = fold_list(ap.parse_args().folds)

    seed_everything(SEED)
    train, test, orig = read_train(), read_test(), read_original()
    y = train[TARGET].astype('int64').to_numpy()
    y_orig = orig[TARGET].astype('int64').to_numpy()
    cats, nums, numeric_like = infer_groups(train, test, orig)
    X = train.drop([ID, TARGET], axis=1).copy()
    X_test = test.drop([ID], axis=1).copy()
    X_orig = orig.drop([TARGET], axis=1).copy()
    X, X_test = feature_engineering(X, X_test, X_orig, y_orig, cats, nums, numeric_like)
    cat_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    print(f'{X.shape[1]} columns: {len(cat_cols)} categorical, {X.shape[1] - len(cat_cols)} continuous', flush=True)

    fold_ids = folds(y)
    for k in range(5):
        if k not in run_folds:
            continue
        print(f'### Fold {k}', flush=True)
        train_idx, valid_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        model = LNNClassifier(random_state=SEED + k + 1)  # folds numbered 1-5 in the seed
        model.fit(X.iloc[train_idx].reset_index(drop=True), y[train_idx], eval_set=(X.iloc[valid_idx].reset_index(drop=True), y[valid_idx]))
        oof_pred = model.predict_proba(X.iloc[valid_idx].reset_index(drop=True))[:, 1]
        test_fold = model.predict_proba(X_test.reset_index(drop=True))[:, 1]
        print(f'Fold {k} | AUC: {roc_auc_score(y[valid_idx], oof_pred):.6f} | best epoch {model.best_epoch_}', flush=True)
        save_part(MODEL_DIR, PAIR, k, oof_pred, test_fold)
        del model, oof_pred, test_fold
        clean_memory()
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
