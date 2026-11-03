#!/usr/bin/env python
"""Tabular ResNet with categorical embeddings on F1 support features, artifact keys, original-data priors and an
original-data manifold (PCA + Gaussian random projection fit on the original rows).

Features (train + test + original rows; the original rows are never trained on):
- categoricals: Driver, Compound, Race and the numerics with < 100 unique values in train + test; for the high-cardinality
  numerics (LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress) the values rounded to 2 and 3 decimals and a
  32-quantile rank bin (edges from the ranks over train + test + original, applied to each frame's own ranks); 6 combos
  (Driver x Compound, Driver x Race, Race x Year, Compound x TyreLife, Stint x TyreLife, Driver x PitStop). Codes are
  factorized over train + test;
- numerics: the high-cardinality raw numerics; sin / cos of LapNumber, TyreLife, LapTime_Delta and RaceProgress at periods
  8, 12 and 24; 7 F1 support features (lap - tyre life, |delta| + degradation and the same per tyre lap, lap time /
  progress, position pressure, pit + stint + progress, delta / lap time); the original-data target mean (original mean
  for unseen keys) and log count of every categorical and high-cardinality numeric; 12 PCA and 12 Gaussian random
  projection components (seed 42) of the one-hot categoricals + standardized numerics, both fit on the original rows.
  Standardized on the outer-train rows of each fold.
Model: embeddings (dim 12) + numerics -> Linear 256, BatchNorm, ReLU -> 4 residual blocks (BatchNorm, ReLU, Linear,
BatchNorm, ReLU, dropout 0.25, Linear, + identity) -> BatchNorm, ReLU, Linear -> 2 logits. KL-divergence loss on
label-smoothed (0.02) two-class targets, AdamW lr 2e-3, weight decay 1e-4, batch 1024 (last partial batch dropped),
ReduceLROnPlateau on the validation AUC (factor 0.5, patience 3, min lr 1e-5), up to 35 epochs with early stopping
(patience 8) on the held-out fold; the best epoch predicts the held-out and test rows (softmax probability of class 1).
Seeded once (42) before the fold loop, so the folds share one random stream: run all 5 folds in one process to reproduce
the stacked pair. Test = mean of the 5 fold models.
Usage: python train_resnet_manifold.py [--folds 0,1,2,3,4]"""
import argparse
import gc
import os
import random
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'tabular_resnet_original_manifold_priors'
MODEL_DIR = model_dir('neural_nets')
SEED = 42
BATCH_SIZE = 1024
EPOCHS = 35
PATIENCE = 8
LR = 2e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 256
RES_BLOCKS = 4
DROPOUT = 0.25
EMB_DIM = 12
PCA_COMPONENTS = 12
GRP_COMPONENTS = 12
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_everything(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def smooth_labels(y: np.ndarray, smoothing: float = 0.02) -> np.ndarray:
    y = y.astype(np.int64)
    out = np.full((len(y), 2), smoothing, dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0 - smoothing
    return out


def safe_name(x):
    return re.sub(r'[^0-9A-Za-z_]+', '_', str(x)).strip('_')


def safe_num(s):
    return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)


def safe_str(s):
    return s.astype('string').fillna('missing').str.strip()


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown='ignore', dtype=np.float32)


def infer_groups(train, test, orig):
    """Object columns and numerics with < 100 unique values (train + test) are categorical; numeric_like = all numerics."""
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
    return feature_cols, cats, nums, numeric_like


def add_cycle_features(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        vals = safe_num(df[col])
        for period in (8.0, 12.0, 24.0):
            pname = str(int(period))
            df[f'{safe_name(col)}_sin_{pname}'] = np.sin(2.0 * np.pi * vals / period).astype(np.float32)
            df[f'{safe_name(col)}_cos_{pname}'] = np.cos(2.0 * np.pi * vals / period).astype(np.float32)
    return df


def add_support_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    df['lap_tyre_gap'] = (lap - tyre).astype(np.float32)
    df['time_stress'] = (np.abs(ld) + deg).astype(np.float32)
    df['stress_per_tyre'] = ((np.abs(ld) + deg) / (tyre + 1.0)).astype(np.float32)
    df['lap_time_progress'] = (lt / (prog + 0.01)).astype(np.float32)
    df['position_pressure'] = ((pos + np.abs(pchg)) / (prog + 0.02)).astype(np.float32)
    df['pit_stint_progress'] = (pit + stint + prog).astype(np.float32)
    df['delta_time_ratio'] = (ld / (np.abs(lt) + 1.0)).astype(np.float32)
    return df


def build_original_manifold_features(combined_df, orig_features_df, base_cats, base_nums, seed, pca_components, grp_components):
    """PCA and Gaussian random projection of [standardized numerics, one-hot categoricals], fit on the original rows."""
    combined_df = combined_df.copy()
    ohe = make_ohe()
    orig_cat_encoded = ohe.fit_transform(orig_features_df[base_cats].astype(str)) if base_cats else np.zeros((len(orig_features_df), 0), dtype=np.float32)
    scaler = StandardScaler()
    orig_num_scaled = scaler.fit_transform(orig_features_df[base_nums]).astype(np.float32) if base_nums else np.zeros((len(orig_features_df), 0), dtype=np.float32)
    orig_full = np.hstack([orig_num_scaled, orig_cat_encoded]).astype(np.float32)
    n_comp = max(1, min(pca_components, orig_full.shape[1], len(orig_features_df) - 1))
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(orig_full)
    grp_n = max(1, min(grp_components, max(1, orig_full.shape[1])))
    grp = GaussianRandomProjection(n_components=grp_n, random_state=seed)
    grp.fit(orig_full)
    comb_cat = ohe.transform(combined_df[base_cats].astype(str)) if base_cats else np.zeros((len(combined_df), 0), dtype=np.float32)
    comb_num = scaler.transform(combined_df[base_nums]).astype(np.float32) if base_nums else np.zeros((len(combined_df), 0), dtype=np.float32)
    comb_full = np.hstack([comb_num, comb_cat]).astype(np.float32)
    pca_features = pca.transform(comb_full).astype(np.float32)
    grp_features = grp.transform(comb_full).astype(np.float32)
    for i in range(pca_features.shape[1]):
        combined_df[f'orig_pca_{i}'] = pca_features[:, i]
    for i in range(grp_features.shape[1]):
        combined_df[f'orig_grp_{i}'] = grp_features[:, i]
    return combined_df


def add_categorical_artifacts(train, test, orig, high_card):
    """Rounded-value and 32-quantile rank-bin keys of the high-cardinality numerics, plus 6 categorical combos (in place)."""
    new_cats = []
    for col in high_card:
        base = safe_name(col)
        for dec in [2, 3]:
            n = f'cat_{base}_r{dec}'
            for df in (train, test, orig):
                df[n] = safe_num(df[col]).round(dec).astype(str)
            new_cats.append(n)
        qn = f'cat_{base}_q32'
        ref = pd.concat([train[col], test[col], orig[col]], ignore_index=True)
        try:
            _, bins = pd.qcut(safe_num(ref).rank(method='first'), q=32, labels=False, retbins=True, duplicates='drop')
        except Exception:
            bins = None
        for df in (train, test, orig):
            if bins is None:
                df[qn] = '0'
            else:
                df[qn] = pd.cut(safe_num(df[col]).rank(method='first'), bins=bins, labels=False, include_lowest=True).astype('Int16').astype(str)
        new_cats.append(qn)
    for a, b in [('Driver', 'Compound'), ('Driver', 'Race'), ('Race', 'Year'), ('Compound', 'TyreLife'), ('Stint', 'TyreLife'), ('Driver', 'PitStop')]:
        if a in train.columns and b in train.columns:
            n = f'combo_{safe_name(a)}__{safe_name(b)}'
            for df in (train, test, orig):
                df[n] = safe_str(df[a]) + '|' + safe_str(df[b])
            new_cats.append(n)
    return new_cats


def add_original_priors(train, test, orig, y_orig, cols):
    """Original-data target mean and log count of each key (in place)."""
    new_nums = []
    prior = float(np.mean(y_orig))
    for col in cols:
        stats = pd.DataFrame({'key': orig[col].astype(str).values, 'target': y_orig}).groupby('key', dropna=False)['target'].agg(['mean', 'count'])
        m = f'orig_mean_{safe_name(col)}'
        c = f'orig_count_{safe_name(col)}'
        for df in (train, test, orig):
            key = df[col].astype(str)
            df[m] = key.map(stats['mean']).fillna(prior).astype(np.float32)
            df[c] = np.log1p(key.map(stats['count']).fillna(0).astype(np.float32))
        new_nums += [m, c]
    return new_nums


class TabDataset(Dataset):
    def __init__(self, cat_data, num_data, targets=None):
        self.cat_data = torch.tensor(cat_data, dtype=torch.long)
        self.num_data = torch.tensor(num_data, dtype=torch.float32)
        self.targets = None if targets is None else torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        if self.targets is None:
            return self.cat_data[idx], self.num_data[idx]
        return self.cat_data[idx], self.num_data[idx], self.targets[idx]


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
        identity = x
        out = self.norm1(x)
        out = self.relu(out)
        out = self.lin1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.lin2(out)
        return identity + out


class TabularResNet(nn.Module):
    def __init__(self, cat_cards, num_count, emb_dim, hidden_dim, res_blocks, dropout, out_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(int(card), emb_dim) for card in cat_cards])
        total_input_dim = (len(cat_cards) * emb_dim) + num_count
        self.initial_proj = nn.Sequential(nn.Linear(total_input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(res_blocks)])
        self.head = nn.Sequential(nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1) if embs else x_num
        x = self.initial_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def infer_loader(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x_cat, x_num, _ = batch
            else:
                x_cat, x_num = batch
            x_cat = x_cat.to(DEVICE, non_blocking=True)
            x_num = x_num.to(DEVICE, non_blocking=True)
            probs = torch.softmax(model(x_cat, x_num), dim=1)[:, 1].cpu().numpy()
            preds.append(probs)
    return np.concatenate(preds, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    run_folds = fold_list(ap.parse_args().folds)

    seed_everything(SEED)
    train, test, orig = read_train(), read_test(), read_original()
    y_train = train[TARGET].astype(np.int8).values
    y_orig = orig[TARGET].astype(np.int8).values
    _, cats, nums, numeric_like = infer_groups(train, test, orig)
    train_x = train.drop(columns=[ID, TARGET]).copy()
    test_x = test.drop(columns=[ID]).copy()
    orig_x = orig.drop(columns=[TARGET]).copy()
    for df in [train_x, test_x, orig_x]:
        for c in cats:
            df[c] = safe_str(df[c])
        for c in numeric_like:
            df[c] = safe_num(df[c])
    high_card = [c for c in ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress'] if c in nums]
    cycle_cols = [c for c in ['LapNumber', 'TyreLife', 'LapTime_Delta', 'RaceProgress'] if c in train_x.columns]
    train_x = add_cycle_features(train_x, cycle_cols)
    test_x = add_cycle_features(test_x, cycle_cols)
    orig_x = add_cycle_features(orig_x, cycle_cols)
    train_x = add_support_numeric_features(train_x)
    test_x = add_support_numeric_features(test_x)
    orig_x = add_support_numeric_features(orig_x)
    new_cats = add_categorical_artifacts(train_x, test_x, orig_x, high_card)
    add_original_priors(train_x, test_x, orig_x, y_orig, list(dict.fromkeys(cats + new_cats + high_card)))
    cat_features = list(dict.fromkeys(cats + new_cats))
    base_nums = [c for c in numeric_like if c in train_x.columns]
    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)
    combined = build_original_manifold_features(combined, orig_x.copy(), cats, base_nums, SEED, PCA_COMPONENTS, GRP_COMPONENTS)
    n_train = len(train_x)
    num_features = [c for c in combined.columns if c not in cat_features]
    cat_cardinalities = []
    for c in cat_features:
        codes, uniques = pd.factorize(combined[c].astype(str), sort=False)
        combined[c] = codes.astype(np.int64)
        cat_cardinalities.append(max(1, len(uniques)))
    for c in num_features:
        combined[c] = safe_num(combined[c])
    X_train_cat = combined.iloc[:n_train][cat_features].values.astype(np.int64)
    X_train_num_raw = combined.iloc[:n_train][num_features].values.astype(np.float32)
    X_test_cat = combined.iloc[n_train:][cat_features].values.astype(np.int64)
    X_test_num_raw = combined.iloc[n_train:][num_features].values.astype(np.float32)
    print(f'Categorical features: {len(cat_features)}', flush=True)
    print(f'Numeric features: {len(num_features)}', flush=True)

    fold_ids = folds(y_train)
    criterion = nn.KLDivLoss(reduction='batchmean')
    for k in run_folds:
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        print(f'\n--- Fold {k} ---', flush=True)
        y_tr_smooth = smooth_labels(y_train[tr_idx], smoothing=0.02)
        scaler_cv = StandardScaler()
        X_tr_num_scaled = scaler_cv.fit_transform(X_train_num_raw[tr_idx]).astype(np.float32)
        X_va_num_scaled = scaler_cv.transform(X_train_num_raw[va_idx]).astype(np.float32)
        X_te_num_scaled = scaler_cv.transform(X_test_num_raw).astype(np.float32)
        train_ds = TabDataset(X_train_cat[tr_idx], X_tr_num_scaled, y_tr_smooth)
        val_ds = TabDataset(X_train_cat[va_idx], X_va_num_scaled, y_train[va_idx])
        test_ds = TabDataset(X_test_cat, X_te_num_scaled)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        model = TabularResNet(cat_cards=cat_cardinalities, num_count=len(num_features), emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, res_blocks=RES_BLOCKS, dropout=DROPOUT, out_dim=2).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-5)
        best_auc = -1.0
        best_val_preds = None
        best_test_preds = None
        patience_counter = 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for x_cat, x_num, y_b in train_loader:
                x_cat = x_cat.to(DEVICE, non_blocking=True)
                x_num = x_num.to(DEVICE, non_blocking=True)
                y_b = y_b.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                logits = model(x_cat, x_num)
                loss = criterion(torch.log_softmax(logits, dim=1), y_b)
                loss.backward()
                optimizer.step()
            val_preds_epoch = infer_loader(model, val_loader)
            val_auc = roc_auc_score(y_train[va_idx], val_preds_epoch)
            scheduler.step(val_auc)
            if val_auc > best_auc + 1e-6:
                best_auc = val_auc
                best_val_preds = val_preds_epoch.copy()
                best_test_preds = infer_loader(model, test_loader)
                patience_counter = 0
            else:
                patience_counter += 1
            print(f'epoch {epoch:02d} | val_auc={val_auc:.6f} | best_auc={best_auc:.6f}', flush=True)
            if patience_counter >= PATIENCE:
                print(f'Early stopping at epoch {epoch}', flush=True)
                break
        save_part(MODEL_DIR, PAIR, k, best_val_preds, best_test_preds)
        print(f'Best Fold {k} AUC: {roc_auc_score(y_train[va_idx], best_val_preds):.6f}', flush=True)
        del model, optimizer, scheduler, train_loader, val_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assemble(MODEL_DIR, PAIR, fold_ids, y_train)


if __name__ == '__main__':
    main()
