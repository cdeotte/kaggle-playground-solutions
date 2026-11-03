#!/usr/bin/env python
"""Field-aware factorization machine on label-encoded categoricals, original-data value bins and decimal digits.

Fields (22):
- the 9 columns Driver, Compound, Race, Year, PitStop, LapNumber, Stint, Position, Position_Change, label-encoded on
  train + test + original;
- 5 bin fields: the sorted distinct original-data values of TyreLife, LapTime_Delta, Cumulative_Degradation, LapTime (s)
  and RaceProgress split into 100 / 512 / 2048 / 512 / 512 equal-count chunks (an exact original value maps to its chunk,
  any other value to the first chunk whose upper end is >= the value);
- the first decimal digit of LapTime_Delta, Cumulative_Degradation and LapTime (s);
- the 5 continuous columns as numeric fields (median-filled, standardized on each fold's training rows).
Model: FFM with linear terms and k = 16 field-aware latent factors for every pair of the 22 fields (categorical and
numeric); BCE, AdamW lr 2e-3, weight decay 1e-6, batch 4096, up to 30 epochs with early stopping on the held-out fold's
AUC (patience 5), best epoch restored.
All original rows are appended once to the training rows of every fold. Seed 42 (python / numpy / torch, deterministic
cuDNN), set once before the folds. Predictions clipped to [1e-7, 1 - 1e-7]; test = mean of the 5 fold models.
Usage: python train_ffm.py [--folds 0,1,2,3,4]"""
import argparse
import gc
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'ffm_original_bins_decimal_digits'
MODEL_DIR = model_dir('neural_nets')
SEED = 42
ORIG_COPIES = 1  # copies of the original rows appended to every fold's training rows

NUMERIC_COLS = ['TyreLife', 'LapTime_Delta', 'Cumulative_Degradation', 'LapTime (s)', 'RaceProgress']
NUMERIC_BIN_CARDINALITY = {
    'TyreLife': 100,
    'LapTime_Delta': 512,
    'Cumulative_Degradation': 2048,
    'LapTime (s)': 512,
    'RaceProgress': 512,
}
DECIMAL_DIGIT_COLS = ['LapTime_Delta', 'Cumulative_Degradation', 'LapTime (s)']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4096
MAX_EPOCHS = 30
PATIENCE = 5
LR = 2e-3
WEIGHT_DECAY = 1e-6
NUM_WORKERS = 0
PIN_MEMORY = (DEVICE == 'cuda')
FFM_K = 16


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def reduce_mem_float32(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    return df


def first_decimal_digit(s):
    arr = pd.to_numeric(s, errors='coerce').fillna(0).to_numpy(dtype=np.float32, copy=False)
    out = np.floor(np.abs(arr) * 10.0 + 1e-6).astype(np.int32) % 10
    return pd.Series(out, index=s.index, dtype='int32')


def make_apply_bins_fn(value_to_bin, edges, col_name):
    interval_ends = np.array([b for a, b in edges], dtype=np.float32)

    def apply_bins(s):
        arr = pd.to_numeric(s, errors='coerce').to_numpy(dtype=np.float32, copy=False)
        out = np.full(len(arr), -1, dtype=np.int32)
        for i, v in enumerate(arr):
            if not np.isfinite(v):
                out[i] = -1
            else:
                key = float(v)
                if key in value_to_bin:
                    out[i] = value_to_bin[key]
                else:
                    pos = np.searchsorted(interval_ends, v, side='left')
                    out[i] = int(np.clip(pos, 0, len(interval_ends) - 1))
        return pd.Series(out, index=s.index, name=f'{col_name}_bin', dtype='int32')
    return apply_bins


def safe_cat_to_long(df, cat_cols):
    out = df[cat_cols].copy()
    for c in cat_cols:
        x = out[c].astype(np.int64).values
        x = np.where(x < 0, 0, x + 1)  # 0 reserved for missing/unknown
        out[c] = x.astype(np.int64)
    return out


def fit_standardizer(x_num):
    mu = np.nanmean(x_num, axis=0).astype(np.float32)
    sd = np.nanstd(x_num, axis=0).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    return mu, sd


def apply_standardizer(x_num, mu, sd):
    x = x_num.astype(np.float32)
    x = np.where(np.isnan(x), mu, x)
    return ((x - mu) / sd).astype(np.float32)


def preprocess():
    """Label-encoded categoricals, original-value bins and decimal digits for the train, test and original rows."""
    train_df = read_train().drop(columns=[ID])
    test_df = read_test().drop(columns=[ID])
    orig_df = read_original()
    train_df[TARGET] = train_df[TARGET].astype(np.int8)
    orig_df[TARGET] = orig_df[TARGET].astype(np.int8)

    for df in [train_df, test_df, orig_df]:
        reduce_mem_float32(df, NUMERIC_COLS)
        for c in NUMERIC_COLS:
            med = pd.to_numeric(df[c], errors='coerce').median()
            if not np.isfinite(med):
                med = 0.0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(med).astype('float32')

    feature_cols = [c for c in train_df.columns if c != TARGET]
    base_cat_cols = [c for c in feature_cols if c not in NUMERIC_COLS]
    print(f'base categorical columns ({len(base_cat_cols)}): {base_cat_cols}', flush=True)

    for col in base_cat_cols:
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str), orig_df[col].astype(str)], axis=0,
                             ignore_index=True).fillna('__MISSING__')
        le = LabelEncoder()
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str).fillna('__MISSING__')).astype('int32')
        test_df[col] = le.transform(test_df[col].astype(str).fillna('__MISSING__')).astype('int32')
        orig_df[col] = le.transform(orig_df[col].astype(str).fillna('__MISSING__')).astype('int32')

    # bins of the distinct original values (equal-count chunks of the sorted values)
    for col in NUMERIC_COLS:
        desired_card = NUMERIC_BIN_CARDINALITY[col]
        uniq = np.sort(pd.to_numeric(orig_df[col], errors='coerce').dropna().unique())
        if len(uniq) == 0:
            raise ValueError(f'No valid unique values found in original for column {col}')
        actual_card = min(desired_card, len(uniq))
        chunks = [arr for arr in np.array_split(uniq, actual_card) if len(arr)]
        value_to_bin, edges = {}, []
        for bin_id, arr in enumerate(chunks):
            for v in arr:
                value_to_bin[float(v)] = int(bin_id)
            edges.append((float(arr[0]), float(arr[-1])))
        apply_bins = make_apply_bins_fn(value_to_bin, edges, col)
        new_col = f'{col}_bin'
        train_df[new_col] = apply_bins(train_df[col])
        test_df[new_col] = apply_bins(test_df[col])
        orig_df[new_col] = apply_bins(orig_df[col])

    for col in DECIMAL_DIGIT_COLS:
        new_col = f'{col}_d1'
        train_df[new_col] = first_decimal_digit(train_df[col])
        test_df[new_col] = first_decimal_digit(test_df[col])
        orig_df[new_col] = first_decimal_digit(orig_df[col])

    final_feature_cols = [c for c in train_df.columns if c != TARGET]
    final_cat_cols = [c for c in final_feature_cols if c not in NUMERIC_COLS]
    print(f'{len(final_feature_cols)} fields: {len(final_cat_cols)} categorical, {len(NUMERIC_COLS)} numeric', flush=True)
    return train_df, test_df, orig_df, final_feature_cols, final_cat_cols


class FFMDataset(Dataset):
    def __init__(self, X_cat, X_num, y=None):
        self.X_cat = torch.from_numpy(X_cat).long()
        self.X_num = torch.from_numpy(X_num).float()
        self.y = None if y is None else torch.from_numpy(y).float()

    def __len__(self):
        return self.X_cat.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X_cat[idx], self.X_num[idx]
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


class FieldAwareFM(nn.Module):
    """Bias + linear terms + sum over field pairs (i, j) of <v_{i, field j}, v_{j, field i}>; a numeric field's vectors
    are scaled by its value."""

    def __init__(self, cat_cardinalities, n_num, k):
        super().__init__()
        self.n_cat = len(cat_cardinalities)
        self.n_num = n_num
        self.n_fields = self.n_cat + self.n_num
        self.k = k
        self.bias = nn.Parameter(torch.zeros(1))
        self.w_cat = nn.ModuleList([nn.Embedding(card, 1) for card in cat_cardinalities])
        self.w_num = nn.Parameter(torch.zeros(n_num, 1))
        self.v_cat_to_all = nn.ModuleList([nn.Embedding(card, self.n_fields * k) for card in cat_cardinalities])
        self.v_num_to_all = nn.Parameter(torch.randn(n_num, self.n_fields, k) * 0.01)
        for emb in self.w_cat:
            nn.init.zeros_(emb.weight)
        for emb in self.v_cat_to_all:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, x_cat, x_num):
        B = x_cat.size(0)
        out = self.bias.expand(B)
        for i in range(self.n_cat):
            out = out + self.w_cat[i](x_cat[:, i]).squeeze(1)
        out = out + (x_num @ self.w_num).squeeze(1)

        v_cat_all = []
        for i in range(self.n_cat):
            v = self.v_cat_to_all[i](x_cat[:, i]).view(B, self.n_fields, self.k)
            v_cat_all.append(v)
        v_cat_all = torch.stack(v_cat_all, dim=1)
        v_num_all = self.v_num_to_all.unsqueeze(0) * x_num.unsqueeze(2).unsqueeze(3)

        inter = x_num.new_zeros(B)
        for i in range(self.n_cat):
            vi_to = v_cat_all[:, i]
            for j in range(i + 1, self.n_cat):
                vj_to = v_cat_all[:, j]
                inter = inter + (vi_to[:, j, :] * vj_to[:, i, :]).sum(dim=1)
        for i in range(self.n_cat):
            vi_to = v_cat_all[:, i]
            for m in range(self.n_num):
                j_field = self.n_cat + m
                inter = inter + (vi_to[:, j_field, :] * v_num_all[:, m, i, :]).sum(dim=1)
        for a in range(self.n_num):
            fa = self.n_cat + a
            for b in range(a + 1, self.n_num):
                fb = self.n_cat + b
                inter = inter + (v_num_all[:, a, fb, :] * v_num_all[:, b, fa, :]).sum(dim=1)
        return out + inter


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        if len(batch) == 2:
            x_cat, x_num = batch
        else:
            x_cat, x_num, _ = batch
        x_cat = x_cat.to(device, non_blocking=True)
        x_num = x_num.to(device, non_blocking=True)
        logits = model(x_cat, x_num)
        out.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def train_one_fold(Xtr_cat, Xtr_num, ytr, Xva_cat, Xva_num, yva, cat_cardinalities, device):
    ds_tr = FFMDataset(Xtr_cat, Xtr_num, ytr)
    ds_va = FFMDataset(Xva_cat, Xva_num, yva)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    model = FieldAwareFM(cat_cardinalities=cat_cardinalities, n_num=Xtr_num.shape[1], k=FFM_K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    best_auc = -1.0
    best_state = None
    best_epoch = -1
    patience_left = PATIENCE
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for x_cat, x_num, yb in dl_tr:
            x_cat = x_cat.to(device, non_blocking=True)
            x_num = x_num.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x_cat, x_num)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            bs = yb.shape[0]
            total_loss += float(loss.item()) * bs
            n_seen += bs
        p_va = predict_loader(model, dl_va, device)
        auc = roc_auc_score(yva, p_va)
        avg_loss = total_loss / max(1, n_seen)
        if auc > best_auc + 1e-6:
            best_auc = auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
        print(f'  epoch {epoch:03d} | loss {avg_loss:.5f} | val_auc {auc:.6f} | best {best_auc:.6f} @ {best_epoch}', flush=True)
        if patience_left <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(SEED)
    train_df, test_df, orig_df, feature_cols, cat_cols = preprocess()
    num_cols = NUMERIC_COLS
    y = train_df[TARGET].astype(np.int8).values

    X_test_cat_df = safe_cat_to_long(test_df[feature_cols], cat_cols)
    X_test_num_raw = test_df[num_cols].to_numpy(dtype=np.float32, copy=True)
    combined_cats = pd.concat([
        safe_cat_to_long(train_df[feature_cols], cat_cols),
        safe_cat_to_long(test_df[feature_cols], cat_cols),
        safe_cat_to_long(orig_df[feature_cols], cat_cols),
    ], axis=0, ignore_index=True)
    cat_cardinalities = [int(combined_cats[c].max()) + 1 for c in cat_cols]
    orig_X_cat_df = safe_cat_to_long(orig_df[feature_cols], cat_cols)
    orig_X_num_raw = orig_df[num_cols].to_numpy(dtype=np.float32, copy=True)
    orig_y = orig_df[TARGET].astype(np.float32).to_numpy(copy=True)

    fold_ids = folds(y)
    for k in fold_list(args.folds):
        fold = k + 1
        print('=' * 70, flush=True)
        print(f'fold {fold}/5', flush=True)
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        X_tr_df = train_df.iloc[tr_idx].copy()
        X_va_df = train_df.iloc[va_idx].copy()
        y_tr = y[tr_idx].astype(np.float32)
        y_va = y[va_idx].astype(np.float32)

        X_tr_cat_df = safe_cat_to_long(X_tr_df[feature_cols], cat_cols)
        X_va_cat_df = safe_cat_to_long(X_va_df[feature_cols], cat_cols)
        X_tr_num_raw = X_tr_df[num_cols].to_numpy(dtype=np.float32, copy=True)
        X_va_num_raw = X_va_df[num_cols].to_numpy(dtype=np.float32, copy=True)

        # all original rows appended to the training rows
        X_tr_cat_df = pd.concat([X_tr_cat_df] + [orig_X_cat_df] * ORIG_COPIES, axis=0, ignore_index=True)
        X_tr_num_raw = np.concatenate([X_tr_num_raw] + [orig_X_num_raw] * ORIG_COPIES, axis=0)
        y_tr = np.concatenate([y_tr] + [orig_y.astype(np.float32)] * ORIG_COPIES, axis=0)

        mu, sd = fit_standardizer(X_tr_num_raw)
        X_tr_num = apply_standardizer(X_tr_num_raw, mu, sd)
        X_va_num = apply_standardizer(X_va_num_raw, mu, sd)
        X_te_num = apply_standardizer(X_test_num_raw, mu, sd)

        Xtr_cat = X_tr_cat_df.to_numpy(dtype=np.int64, copy=True)
        Xva_cat = X_va_cat_df.to_numpy(dtype=np.int64, copy=True)
        Xte_cat = X_test_cat_df.to_numpy(dtype=np.int64, copy=True)

        model, best_auc, best_epoch = train_one_fold(Xtr_cat, X_tr_num, y_tr, Xva_cat, X_va_num, y_va,
                                                     cat_cardinalities=cat_cardinalities, device=DEVICE)

        dl_va = DataLoader(FFMDataset(Xva_cat, X_va_num, y_va), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                           pin_memory=PIN_MEMORY)
        dl_te = DataLoader(FFMDataset(Xte_cat, X_te_num), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                           pin_memory=PIN_MEMORY)
        p_va = np.clip(predict_loader(model, dl_va, DEVICE), 1e-7, 1 - 1e-7)
        p_te = np.clip(predict_loader(model, dl_te, DEVICE), 1e-7, 1 - 1e-7)
        print(f'fold {fold} AUC = {roc_auc_score(y_va, p_va):.6f} | best_epoch = {best_epoch}', flush=True)
        save_part(MODEL_DIR, PAIR, k, p_va, p_te)

        del X_tr_df, X_va_df, X_tr_cat_df, X_va_cat_df, X_tr_num_raw, X_va_num_raw, X_tr_num, X_va_num, X_te_num
        del Xtr_cat, Xva_cat, Xte_cat, dl_va, dl_te, model, p_va, p_te
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
