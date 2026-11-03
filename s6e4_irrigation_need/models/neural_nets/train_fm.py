#!/usr/bin/env python
"""Field-aware FM / FM on label-encoded categoricals, original-data value bins, first decimal digits and the numerics.

Fields (35), built once:
- the 8 categoricals, label-encoded on train + test + original;
- 11 bin fields: the sorted distinct original-data values of each numeric split into equal-count chunks (96 to 256
  chunks per column, NUMERIC_BIN_CARDINALITY). A value maps to the chunk whose [first, last] value range holds it; a value
  outside every range maps to chunk 0 when it is below the first chunk, otherwise to the last chunk;
- the first decimal digit of Rainfall_mm, Previous_Irrigation_mm, Soil_Moisture, Temperature_C and Humidity;
- the 11 numerics as numeric fields (float32), standardized on each fold's training rows.
Categorical codes are shifted by 1 (0 = missing).

Presets (models with 3 outputs, class order Low, Medium, High):
- ffm: field-aware factorization machine. Per-class bias and linear terms (categorical embeddings and numeric weights
  with 3 outputs) plus the field-aware interaction sum over all pairs of the 35 fields, <v_{i, field j}, v_{j, field i}>
  with k = 16 factors per (feature, other field); numeric fields scale their factors by the value. The scalar
  interaction enters the 3 logits through a learned 1 -> 3 linear map (initialized to ones).
- fm: factorization machine. Per-class bias and linear terms plus the FM pairwise-interaction sum of k = 32 factors over
  the 35 fields (numeric factors scaled by the value); the scalar interaction is added to each of the 3 logits.

Training (both): all original rows are appended once to every fold's training rows. Class-weighted cross-entropy
(weights n / n_class of the training rows, normalized to mean 1), AdamW lr 2e-3, weight decay 1e-6, batch 4096, up to 30
epochs with early stopping on the held-out fold's balanced accuracy (patience 5); the best epoch predicts (softmax).
Seed 42 (python / numpy / torch, deterministic cuDNN) is set once before the folds. Test = mean of the 5 fold models.
Usage: python train_fm.py {ffm,fm} [--folds 0,1,2,3,4]"""
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import (CATEGORICALS, ID, NUMERICS, TARGET, TARGET_MAP, folds, model_dir,  # noqa: E402
                         read_original, read_test, read_train)
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PRESETS = {
    'ffm': dict(pair='ffm_orig_bins_decimal_digits_a', k=16),
    'fm': dict(pair='fm_orig_bins_decimal_digits_a', k=32),
}
MODEL_DIR = model_dir('neural_nets')
SEED = 42

CAT_BASE_COLS = list(CATEGORICALS)
NUMERIC_COLS = list(NUMERICS)
ADD_ORIG_COPIES = 1
NUMERIC_BIN_CARDINALITY = {
    'Soil_pH': 96,
    'Soil_Moisture': 192,
    'Organic_Carbon': 96,
    'Electrical_Conductivity': 96,
    'Temperature_C': 128,
    'Humidity': 192,
    'Rainfall_mm': 256,
    'Sunlight_Hours': 96,
    'Wind_Speed_kmh': 96,
    'Field_Area_hectare': 96,
    'Previous_Irrigation_mm': 256,
}
DECIMAL_DIGIT_COLS = ['Rainfall_mm', 'Previous_Irrigation_mm', 'Soil_Moisture', 'Temperature_C', 'Humidity']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4096
MAX_EPOCHS = 30
PATIENCE = 5
LR = 2e-3
WEIGHT_DECAY = 1e-6
NUM_WORKERS = 0
PIN_MEMORY = DEVICE == 'cuda'


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def reduce_mem_float32(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    return df


def first_decimal_digit(s):
    arr = s.to_numpy(dtype=np.float32, copy=False)
    out = np.full(len(arr), -1, dtype=np.int32)
    for i, v in enumerate(arr):
        if pd.isna(v):
            out[i] = -1
        else:
            out[i] = int(np.floor(np.abs(v) * 10.0 + 1e-6)) % 10
    return pd.Series(out, index=s.index, dtype='int32')


def make_apply_bins_fn(value_to_bin, edges, col_name):
    interval_starts = np.array([a for a, _ in edges], dtype=np.float32)
    interval_ends = np.array([b for _, b in edges], dtype=np.float32)

    def apply_bins(s):
        arr = s.to_numpy(dtype=np.float32, copy=False)
        out = np.full(len(arr), -1, dtype=np.int32)
        for i, v in enumerate(arr):
            if pd.isna(v):
                out[i] = -1
            else:
                key = float(v)
                if key in value_to_bin:
                    out[i] = value_to_bin[key]
                else:
                    placed = False
                    for b in range(len(interval_starts)):
                        if interval_starts[b] <= v <= interval_ends[b]:
                            out[i] = b
                            placed = True
                            break
                    if not placed:
                        out[i] = 0 if v < interval_starts[0] else len(interval_starts) - 1
        return pd.Series(out, index=s.index, name=f'{col_name}_bin', dtype='int32')

    return apply_bins


def preprocess():
    """Label-encoded categoricals, original-value bins and decimal digits for the train, test and original rows."""
    print('Loading train / test / original ...', flush=True)
    train_df = read_train().drop(columns=[ID])
    test_df = read_test().drop(columns=[ID])
    orig_df = read_original()

    train_df[TARGET] = train_df[TARGET].map(TARGET_MAP).astype('int8')
    orig_df[TARGET] = orig_df[TARGET].map(TARGET_MAP).astype('int8')

    train_df = reduce_mem_float32(train_df, NUMERIC_COLS)
    test_df = reduce_mem_float32(test_df, NUMERIC_COLS)
    orig_df = reduce_mem_float32(orig_df, NUMERIC_COLS)

    for col in CAT_BASE_COLS:
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str), orig_df[col].astype(str)], axis=0,
                             ignore_index=True).fillna('__MISSING__')
        le = LabelEncoder()
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str).fillna('__MISSING__')).astype('int32')
        test_df[col] = le.transform(test_df[col].astype(str).fillna('__MISSING__')).astype('int32')
        orig_df[col] = le.transform(orig_df[col].astype(str).fillna('__MISSING__')).astype('int32')

    # equal-count chunks of the distinct original values
    for col in NUMERIC_COLS:
        desired_card = NUMERIC_BIN_CARDINALITY[col]
        uniq = np.sort(orig_df[col].dropna().unique())
        if len(uniq) == 0:
            raise ValueError(f"No valid unique values found in original for column '{col}'")
        actual_card = min(desired_card, len(uniq))
        chunks = np.array_split(uniq, actual_card)
        value_to_bin = {}
        edges = []
        for bin_id, arr in enumerate(chunks):
            for v in arr:
                value_to_bin[float(v)] = int(bin_id)
            edges.append((float(arr[0]), float(arr[-1])))
        apply_bins = make_apply_bins_fn(value_to_bin, edges, col)
        train_df[f'{col}_bin'] = apply_bins(train_df[col])
        test_df[f'{col}_bin'] = apply_bins(test_df[col])
        orig_df[f'{col}_bin'] = apply_bins(orig_df[col])

    for col in DECIMAL_DIGIT_COLS:
        new_col = f'{col}_d1'
        train_df[new_col] = first_decimal_digit(train_df[col])
        test_df[new_col] = first_decimal_digit(test_df[col])
        orig_df[new_col] = first_decimal_digit(orig_df[col])

    final_feature_cols = [c for c in train_df.columns if c != TARGET]
    final_cat_cols = [c for c in final_feature_cols if c not in NUMERIC_COLS]
    print(f'{len(final_feature_cols)} fields: {len(final_cat_cols)} categorical, {len(NUMERIC_COLS)} numeric', flush=True)
    return train_df, test_df, orig_df, final_feature_cols, final_cat_cols


def safe_cat_to_long(df, cat_cols):
    out = df[cat_cols].copy()
    for c in cat_cols:
        x = out[c].astype(np.int64).values
        x = np.where(x < 0, 0, x + 1)
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


class FieldDataset(Dataset):
    def __init__(self, X_cat, X_num, y=None):
        self.X_cat = torch.from_numpy(X_cat).long()
        self.X_num = torch.from_numpy(X_num).float()
        self.y = None if y is None else torch.from_numpy(y).long()

    def __len__(self):
        return self.X_cat.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X_cat[idx], self.X_num[idx]
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


class FieldAwareFM(nn.Module):
    def __init__(self, cat_cardinalities, n_num, k, num_classes):
        super().__init__()
        self.n_cat = len(cat_cardinalities)
        self.n_num = n_num
        self.n_fields = self.n_cat + self.n_num
        self.k = k
        self.num_classes = num_classes

        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.w_cat = nn.ModuleList([nn.Embedding(card, num_classes) for card in cat_cardinalities])
        self.w_num = nn.Parameter(torch.zeros(n_num, num_classes))
        self.v_cat_to_all = nn.ModuleList([nn.Embedding(card, self.n_fields * k) for card in cat_cardinalities])
        self.v_num_to_all = nn.Parameter(torch.randn(n_num, self.n_fields, k) * 0.01)

        for emb in self.w_cat:
            nn.init.zeros_(emb.weight)
        for emb in self.v_cat_to_all:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        self.out_layer = nn.Linear(1, num_classes, bias=False)
        nn.init.ones_(self.out_layer.weight)

    def forward(self, x_cat, x_num):
        B = x_cat.size(0)
        out = self.bias.unsqueeze(0).expand(B, self.num_classes).clone()
        for i in range(self.n_cat):
            out = out + self.w_cat[i](x_cat[:, i])
        out = out + x_num @ self.w_num

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
            for m in range(self.n_num):
                j = self.n_cat + m
                inter = inter + (vi_to[:, j, :] * v_num_all[:, m, i, :]).sum(dim=1)
        for m1 in range(self.n_num):
            i = self.n_cat + m1
            for m2 in range(m1 + 1, self.n_num):
                j = self.n_cat + m2
                inter = inter + (v_num_all[:, m1, j, :] * v_num_all[:, m2, i, :]).sum(dim=1)

        return out + self.out_layer(inter.unsqueeze(1))


class FMModel(nn.Module):
    def __init__(self, cat_cardinalities, n_num, k, num_classes):
        super().__init__()
        self.n_cat = len(cat_cardinalities)
        self.n_num = n_num
        self.k = k
        self.num_classes = num_classes

        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.w_cat = nn.ModuleList([nn.Embedding(card, num_classes) for card in cat_cardinalities])
        self.cat_factor = nn.ModuleList([nn.Embedding(card, k) for card in cat_cardinalities])
        self.w_num = nn.Parameter(torch.zeros(n_num, num_classes))
        self.num_factor = nn.Parameter(torch.randn(n_num, k) * 0.01)

        for emb in self.w_cat:
            nn.init.zeros_(emb.weight)
        for emb in self.cat_factor:
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

    def forward(self, x_cat, x_num):
        batch = x_cat.size(0)
        out = self.bias.unsqueeze(0).expand(batch, self.num_classes).clone()

        cat_vs = []
        for i in range(self.n_cat):
            xi = x_cat[:, i]
            out = out + self.w_cat[i](xi)
            cat_vs.append(self.cat_factor[i](xi))

        out = out + x_num @ self.w_num
        num_vs = x_num.unsqueeze(-1) * self.num_factor.unsqueeze(0)

        all_vs = cat_vs + [num_vs[:, j, :] for j in range(self.n_num)]
        stacked = torch.stack(all_vs, dim=1)
        sum_v = stacked.sum(dim=1)
        inter = 0.5 * ((sum_v * sum_v) - (stacked * stacked).sum(dim=1))
        inter = inter.sum(dim=1, keepdim=True)
        return out + inter


MODELS = {'ffm': FieldAwareFM, 'fm': FMModel}


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
        out.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def train_one_fold(preset, Xtr_cat, Xtr_num, ytr, Xva_cat, Xva_num, yva, cat_cardinalities, device):
    dl_tr = DataLoader(FieldDataset(Xtr_cat, Xtr_num, ytr), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                       pin_memory=PIN_MEMORY)
    dl_va = DataLoader(FieldDataset(Xva_cat, Xva_num, yva), batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = MODELS[preset](cat_cardinalities=cat_cardinalities, n_num=Xtr_num.shape[1], k=PRESETS[preset]['k'],
                           num_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    class_counts = np.bincount(ytr.astype(np.int64), minlength=3).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

    best_bac = -1.0
    best_state = None
    best_epoch = -1
    patience_left = PATIENCE
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for x_cat, x_num, y in dl_tr:
            x_cat = x_cat.to(device, non_blocking=True)
            x_num = x_num.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x_cat, x_num), y)
            loss.backward()
            opt.step()
            bs = y.shape[0]
            total_loss += float(loss.item()) * bs
            n_seen += bs

        p_va = predict_loader(model, dl_va, device)
        bac = balanced_accuracy_score(yva, p_va.argmax(axis=1))
        if bac > best_bac + 1e-6:
            best_bac = bac
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
        print(f'  epoch {epoch:03d} | loss {total_loss / max(1, n_seen):.5f} | val_bac {bac:.6f} | '
              f'best {best_bac:.6f} @ {best_epoch}', flush=True)
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_bac, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preset', choices=sorted(PRESETS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    preset = args.preset
    pair = PRESETS[preset]['pair']

    seed_everything(SEED)
    train_df, test_df, orig_df, features, cat_cols = preprocess()
    num_cols = NUMERIC_COLS
    y = train_df[TARGET].astype(np.int8).values

    X_test_cat_df = safe_cat_to_long(test_df[features], cat_cols)
    X_test_num_raw = test_df[num_cols].to_numpy(dtype=np.float32, copy=True)
    combined_cats = pd.concat([
        safe_cat_to_long(train_df[features], cat_cols),
        safe_cat_to_long(test_df[features], cat_cols),
        safe_cat_to_long(orig_df[features], cat_cols),
    ], axis=0, ignore_index=True)
    cat_cardinalities = [int(combined_cats[c].max()) + 1 for c in cat_cols]

    orig_X_cat_df = safe_cat_to_long(orig_df[features], cat_cols)
    orig_X_num_raw = orig_df[num_cols].to_numpy(dtype=np.float32, copy=True)
    orig_y = orig_df[TARGET].astype(np.int64).to_numpy(copy=True)
    print(f'{preset}: k = {PRESETS[preset]["k"]}, device {DEVICE}, original copies {ADD_ORIG_COPIES}', flush=True)

    fold_ids = folds(y)
    for k in fold_list(args.folds):
        print('=' * 70, flush=True)
        print(f'Fold {k + 1}/5', flush=True)
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        X_tr_df = train_df.iloc[tr_idx].copy()
        X_va_df = train_df.iloc[va_idx].copy()
        y_tr = y[tr_idx].astype(np.int64)
        y_va = y[va_idx].astype(np.int64)

        X_tr_cat_df = safe_cat_to_long(X_tr_df[features], cat_cols)
        X_va_cat_df = safe_cat_to_long(X_va_df[features], cat_cols)
        X_tr_num_raw = X_tr_df[num_cols].to_numpy(dtype=np.float32, copy=True)
        X_va_num_raw = X_va_df[num_cols].to_numpy(dtype=np.float32, copy=True)

        if ADD_ORIG_COPIES > 0:
            X_tr_cat_df = pd.concat([X_tr_cat_df] + [orig_X_cat_df] * ADD_ORIG_COPIES, axis=0, ignore_index=True)
            X_tr_num_raw = np.concatenate([X_tr_num_raw] + [orig_X_num_raw] * ADD_ORIG_COPIES, axis=0)
            y_tr = np.concatenate([y_tr] + [orig_y] * ADD_ORIG_COPIES, axis=0)

        mu, sd = fit_standardizer(X_tr_num_raw)
        X_tr_num = apply_standardizer(X_tr_num_raw, mu, sd)
        X_va_num = apply_standardizer(X_va_num_raw, mu, sd)
        X_te_num = apply_standardizer(X_test_num_raw, mu, sd)

        X_tr_cat = X_tr_cat_df.to_numpy(dtype=np.int64, copy=True)
        X_va_cat = X_va_cat_df.to_numpy(dtype=np.int64, copy=True)
        X_te_cat = X_test_cat_df.to_numpy(dtype=np.int64, copy=True)

        model, best_bac, best_epoch = train_one_fold(preset, X_tr_cat, X_tr_num, y_tr, X_va_cat, X_va_num, y_va,
                                                     cat_cardinalities, DEVICE)

        dl_va = DataLoader(FieldDataset(X_va_cat, X_va_num, y_va), batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        dl_te = DataLoader(FieldDataset(X_te_cat, X_te_num, None), batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        p_va = predict_loader(model, dl_va, DEVICE)
        p_te = predict_loader(model, dl_te, DEVICE)
        print(f'Fold {k + 1} BAC = {balanced_accuracy_score(y_va, p_va.argmax(axis=1)):.6f} | best epoch {best_epoch}',
              flush=True)
        save_part(MODEL_DIR, pair, k, p_va, p_te)

        del X_tr_cat_df, X_va_cat_df, X_tr_num_raw, X_va_num_raw, X_tr_num, X_va_num, X_te_num, X_tr_cat, X_va_cat
        del X_te_cat, model, dl_va, dl_te, p_va, p_te
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, pair, fold_ids, y)


if __name__ == '__main__':
    main()
