#!/usr/bin/env python
"""Field-aware FM / FM on label-encoded categoricals, original-data value bins, first decimal digits and the numerics.

Fields (24), built once:
- the 16 categoricals, label-encoded on train + test + original (as strings);
- 3 bin fields: the sorted distinct original-data values of tenure / MonthlyCharges / TotalCharges split into 100 /
  512 / 2048 equal-count chunks (fewer when there are fewer distinct values). A value maps to the chunk whose
  [first, last] value range holds it; a value outside every range maps to chunk 0 when it is below the first chunk,
  otherwise to the last chunk; missing -> -1;
- the first decimal digit of MonthlyCharges and TotalCharges (floor(|x| * 10) mod 10);
- the 3 numerics as numeric fields (float32), standardized on each fold's training rows (missing -> training mean).
Categorical codes are shifted by 1 (0 = missing). TotalCharges is blank (-> NaN) for 11 original rows.

Presets (one logit each):
- ffm: field-aware factorization machine. Bias + linear terms (1-d categorical embeddings and numeric weights) + the
  field-aware interaction sum over all pairs of the 24 fields, <v_{i, field j}, v_{j, field i}>, k = 16 factors per
  (feature, other field); numeric fields scale their factors by the value. Batch 4096, up to 30 epochs, weight decay
  1e-6.
- fm: factorization machine. Bias + linear terms + the FM pairwise-interaction sum of k = 32 factors over the 24 fields
  (numeric factors scaled by the value). Batch 8192, up to 50 epochs, weight decay 1e-5.

Training (both): all original rows are appended once to every fold's training rows. BCE with logits, AdamW lr 2e-3,
early stopping on the held-out AUC (patience 5); the best epoch predicts (sigmoid). Seed 42 (python / numpy / torch,
deterministic cuDNN) is set once before the folds, so a pair is reproduced only when its 5 folds run in order in one
process. Test = mean of the 5 fold models.
Usage: python train_factorization_machines.py {ffm,fm} [--folds 0,1,2,3,4]"""
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
from common.data import ID, NUMERICS, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PRESETS = {
    'ffm': dict(pair='ffm_orig_bins_decimal_digits', k=16, batch_size=4096, max_epochs=30, weight_decay=1e-6),
    'fm': dict(pair='fm_orig_bins_decimal_digits', k=32, batch_size=8192, max_epochs=50, weight_decay=1e-5),
}
MODEL_DIR = model_dir('neural_nets')
SEED = 42

NUMERIC_COLS = list(NUMERICS)
ADD_ORIG_COPIES = 1
NUMERIC_BIN_CARDINALITY = {
    'tenure': 100,
    'MonthlyCharges': 256 * 2,
    'TotalCharges': 1024 * 2,
}
DECIMAL_DIGIT_COLS = ['MonthlyCharges', 'TotalCharges']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATIENCE = 5
LR = 2e-3
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


def clean_total_charges_inplace(df):
    if 'TotalCharges' in df.columns:
        s = df['TotalCharges'].astype(str).str.strip()
        s = s.replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        df['TotalCharges'] = pd.to_numeric(s, errors='coerce').astype('float32')
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

    for df in (train_df, test_df, orig_df):
        if TARGET in df.columns:
            df[TARGET] = df[TARGET].map({'Yes': 1, 'No': 0}).astype('int8')

    train_df = clean_total_charges_inplace(train_df)
    test_df = clean_total_charges_inplace(test_df)
    orig_df = clean_total_charges_inplace(orig_df)

    train_df = reduce_mem_float32(train_df, NUMERIC_COLS)
    test_df = reduce_mem_float32(test_df, NUMERIC_COLS)
    orig_df = reduce_mem_float32(orig_df, NUMERIC_COLS)

    feature_cols = [c for c in train_df.columns if c != TARGET]
    base_cat_cols = [c for c in feature_cols if c not in NUMERIC_COLS]

    for col in base_cat_cols:
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
    # 0 = missing, other codes shifted by 1
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
        self.y = None if y is None else torch.from_numpy(y).float()

    def __len__(self):
        return self.X_cat.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X_cat[idx], self.X_num[idx]
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


class FieldAwareFM(nn.Module):
    """bias + sum w_cat(x) + sum w_num x_num + sum_{i<j} <v_{i->j}, v_{j->i}> (x_i x_j), x_i = 1 for categoricals."""

    def __init__(self, cat_cardinalities, n_num, k):
        super().__init__()
        self.n_cat = len(cat_cardinalities)
        self.n_num = n_num
        self.n_fields = self.n_cat + self.n_num
        self.k = k

        self.bias = nn.Parameter(torch.zeros(1))
        self.w_cat = nn.ModuleList([nn.Embedding(card, 1) for card in cat_cardinalities])
        self.w_num = nn.Parameter(torch.zeros(n_num, 1))
        # per categorical field: factors toward every field, [card, n_fields * k]
        self.v_cat_to_all = nn.ModuleList([nn.Embedding(card, self.n_fields * k) for card in cat_cardinalities])
        # per numeric field: [n_num, n_fields, k]
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
        v_cat_all = torch.stack(v_cat_all, dim=1)  # [B, n_cat, n_fields, k]
        v_num_all = self.v_num_to_all.unsqueeze(0) * x_num.unsqueeze(2).unsqueeze(3)  # [B, n_num, n_fields, k]

        inter = x_num.new_zeros(B)
        # categorical-categorical pairs
        for i in range(self.n_cat):
            vi_to = v_cat_all[:, i]
            for j in range(i + 1, self.n_cat):
                vj_to = v_cat_all[:, j]
                inter = inter + (vi_to[:, j, :] * vj_to[:, i, :]).sum(dim=1)
        # categorical-numeric pairs
        for i in range(self.n_cat):
            vi_to = v_cat_all[:, i]
            for m in range(self.n_num):
                j_field = self.n_cat + m
                inter = inter + (vi_to[:, j_field, :] * v_num_all[:, m, i, :]).sum(dim=1)
        # numeric-numeric pairs
        for a in range(self.n_num):
            fa = self.n_cat + a
            for b in range(a + 1, self.n_num):
                fb = self.n_cat + b
                inter = inter + (v_num_all[:, a, fb, :] * v_num_all[:, b, fa, :]).sum(dim=1)
        return out + inter


class FactorizationMachine(nn.Module):
    """bias + sum w_i x_i + 0.5 sum_f ((sum_i v_if x_i)^2 - sum_i v_if^2 x_i^2), x_i = 1 for categoricals."""

    def __init__(self, cat_cardinalities, n_num, k):
        super().__init__()
        self.n_cat = len(cat_cardinalities)
        self.n_num = n_num
        self.k = k

        self.w_cat = nn.ModuleList([nn.Embedding(card, 1) for card in cat_cardinalities])
        self.v_cat = nn.ModuleList([nn.Embedding(card, k) for card in cat_cardinalities])
        self.w_num = nn.Parameter(torch.zeros(n_num, 1))
        self.v_num = nn.Parameter(torch.randn(n_num, k) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))

        for emb in self.w_cat:
            nn.init.zeros_(emb.weight)
        for emb in self.v_cat:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, x_cat, x_num):
        B = x_cat.size(0)
        lin = self.bias.expand(B)
        for j in range(self.n_cat):
            lin = lin + self.w_cat[j](x_cat[:, j]).squeeze(1)
        lin = lin + (x_num @ self.w_num).squeeze(1)

        Vx_cat = torch.stack([self.v_cat[j](x_cat[:, j]) for j in range(self.n_cat)], dim=1)  # [B, n_cat, k]
        Vx_num = self.v_num.unsqueeze(0) * x_num.unsqueeze(2)  # [B, n_num, k]
        Vx = torch.cat([Vx_cat, Vx_num], dim=1)
        sum_vx = Vx.sum(dim=1)
        inter = 0.5 * (sum_vx * sum_vx - (Vx * Vx).sum(dim=1)).sum(dim=1)
        return lin + inter


MODELS = {'ffm': FieldAwareFM, 'fm': FactorizationMachine}


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
        out.append(torch.sigmoid(model(x_cat, x_num)).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def train_one_fold(preset, Xtr_cat, Xtr_num, ytr, Xva_cat, Xva_num, yva, cat_cardinalities, device):
    cfg = PRESETS[preset]
    dl_tr = DataLoader(FieldDataset(Xtr_cat, Xtr_num, ytr), batch_size=cfg['batch_size'], shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    dl_va = DataLoader(FieldDataset(Xva_cat, Xva_num, yva), batch_size=cfg['batch_size'], shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = MODELS[preset](cat_cardinalities=cat_cardinalities, n_num=Xtr_num.shape[1], k=cfg['k']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=cfg['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    best_epoch = -1
    patience_left = PATIENCE
    for epoch in range(1, cfg['max_epochs'] + 1):
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

        auc = roc_auc_score(yva, predict_loader(model, dl_va, device))
        if auc > best_auc + 1e-6:
            best_auc = auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
        print(f'  epoch {epoch:03d} | loss {total_loss / max(1, n_seen):.5f} | val_auc {auc:.6f} | '
              f'best {best_auc:.6f} @ {best_epoch}', flush=True)
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preset', choices=sorted(PRESETS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    preset = args.preset
    cfg = PRESETS[preset]
    pair = cfg['pair']

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
    orig_y = orig_df[TARGET].astype(np.float32).to_numpy(copy=True)
    print(f'{preset}: k = {cfg["k"]}, device {DEVICE}, original copies {ADD_ORIG_COPIES}', flush=True)

    fold_ids = folds(y)
    for k in fold_list(args.folds):
        print('=' * 70, flush=True)
        print(f'Fold {k + 1}/5', flush=True)
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        X_tr_df = train_df.iloc[tr_idx].copy()
        X_va_df = train_df.iloc[va_idx].copy()
        y_tr = y[tr_idx].astype(np.float32)
        y_va = y[va_idx].astype(np.float32)

        X_tr_cat_df = safe_cat_to_long(X_tr_df[features], cat_cols)
        X_va_cat_df = safe_cat_to_long(X_va_df[features], cat_cols)
        X_tr_num_raw = X_tr_df[num_cols].to_numpy(dtype=np.float32, copy=True)
        X_va_num_raw = X_va_df[num_cols].to_numpy(dtype=np.float32, copy=True)

        if ADD_ORIG_COPIES > 0:
            X_tr_cat_df = pd.concat([X_tr_cat_df] + [orig_X_cat_df] * ADD_ORIG_COPIES, axis=0, ignore_index=True)
            X_tr_num_raw = np.concatenate([X_tr_num_raw] + [orig_X_num_raw] * ADD_ORIG_COPIES, axis=0)
            y_tr = np.concatenate([y_tr] + [orig_y] * ADD_ORIG_COPIES, axis=0)

        # standardize on the fold's training rows (with the original rows)
        mu, sd = fit_standardizer(X_tr_num_raw)
        X_tr_num = apply_standardizer(X_tr_num_raw, mu, sd)
        X_va_num = apply_standardizer(X_va_num_raw, mu, sd)
        X_te_num = apply_standardizer(X_test_num_raw, mu, sd)

        X_tr_cat = X_tr_cat_df.to_numpy(dtype=np.int64, copy=True)
        X_va_cat = X_va_cat_df.to_numpy(dtype=np.int64, copy=True)
        X_te_cat = X_test_cat_df.to_numpy(dtype=np.int64, copy=True)

        model, best_auc, best_epoch = train_one_fold(preset, X_tr_cat, X_tr_num, y_tr, X_va_cat, X_va_num, y_va,
                                                     cat_cardinalities, DEVICE)

        dl_va = DataLoader(FieldDataset(X_va_cat, X_va_num, y_va), batch_size=cfg['batch_size'], shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        dl_te = DataLoader(FieldDataset(X_te_cat, X_te_num), batch_size=cfg['batch_size'], shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        p_va = predict_loader(model, dl_va, DEVICE)
        p_te = predict_loader(model, dl_te, DEVICE)
        print(f'Fold {k + 1} AUC = {roc_auc_score(y_va, p_va):.6f} | best epoch {best_epoch}', flush=True)
        save_part(MODEL_DIR, pair, k, p_va, p_te)

        del X_tr_df, X_va_df, X_tr_cat_df, X_va_cat_df, X_tr_num_raw, X_va_num_raw, X_tr_num, X_va_num, X_te_num
        del X_tr_cat, X_va_cat, X_te_cat, model, dl_va, dl_te, p_va, p_te
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, pair, fold_ids, y)


if __name__ == '__main__':
    main()
