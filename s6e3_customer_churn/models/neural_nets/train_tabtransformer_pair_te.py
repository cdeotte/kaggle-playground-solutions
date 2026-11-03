#!/usr/bin/env python
"""TabTransformer on categorical tokens, numerics and cuML pairwise target encodings (logit, logit^2, logit^3), with the
original rows appended to every fold's training rows.

Features, built once on train + test + original:
- 15 string categoricals (SeniorCitizen stays numeric);
- 7 numerics: SeniorCitizen, tenure, MonthlyCharges, TotalCharges (NaN -> MonthlyCharges x tenure) and MonthlyCharges /
  TotalCharges, TotalCharges / tenure, TotalCharges / MonthlyCharges (+1e-6 in the denominators);
- 19 extra categoricals: integer digits 0-2 and decimal digits 1-2 of MonthlyCharges, integer digits 0-3 and decimal
  digits 1-2 of TotalCharges, integer digits 0-1 of tenure; 200 / 4000-bin quantile bins (KBinsDiscretizer) of
  MonthlyCharges / TotalCharges; a category code of the exact value of each numeric (TotalCharges rounded to 0
  decimals).
The 34 categoricals get embedding vocabularies over train + test + original (sorted strings, unseen -> 0).
Per fold: the fold's training rows + all 7,043 original rows are the training set. The numerics are standardized on it.
All C(34, 2) = 561 categorical pairs are target-encoded with cuML `TargetEncoder` (smooth 0, mean, 5 inner random
folds, seed 42 + fold number 1-5): out-of-fold encodings for the training rows, the encoding fitted on all training
rows for the held-out and test rows. Each encoding m (clipped to [1e-5, 1 - 1e-5]) gives z = logit(m), z^2 and z^3
(1,683 columns), standardized on the training rows.
Network: each categorical -> 16-d token, 2 post-norm Transformer blocks (4 heads, FFN 64 with GELU, dropout 0.2), tokens
flattened; the encodings go through BatchNorm -> Linear(256) -> ReLU -> dropout; concatenated with the numerics ->
MLP 128-64 (BatchNorm, ReLU, dropout 0.2) -> 1 logit.
Training: BCE on 0.01-smoothed labels, Adam lr 1e-3, weight decay 1e-5, batch 4096, gradient clipping 1, 10 epochs,
ReduceLROnPlateau on the held-out AUC (factor 0.5, patience 5, min lr 1e-5); the best epoch predicts (sigmoid).
Python / numpy / torch are seeded with 42 (deterministic cuDNN) before the folds.
Usage: python train_tabtransformer_pair_te.py"""
import copy
import gc
import os
import random
import sys
import warnings
from itertools import combinations
from pathlib import Path

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from cuml.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'tabtransformer_pair_te_logit3_digits_original_rows'
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 42
N_SPLITS = 5
TE_INNER_SPLITS = 5
BATCH_SIZE = 4096
EPOCHS = 10
USE_X_ORIG = True
USE_LOGIT3 = True
MAX_PAIRS = None
TE_SMOOTH = 0

EMBED_DIM = 16
NUM_HEADS = 4
FF_DIM = 64
NUM_TRANSFORMER_BLOCKS = 2
MLP_HIDDEN = (128, 64)
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-5
LABEL_SMOOTHING = 0.01
PATIENCE_ES = 15
PATIENCE_LR = 5
LR_FACTOR = 0.5
MIN_LR = 1e-5
GRAD_CLIP_NORM = 1.0


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ----------------------------------------------------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------------------------------------------------
def safe_totalcharges(df):
    out = pd.to_numeric(df['TotalCharges'], errors='coerce')
    miss = out.isna()
    if miss.any():
        out.loc[miss] = (pd.to_numeric(df.loc[miss, 'MonthlyCharges'], errors='coerce').fillna(0).astype(np.float32)
                         * pd.to_numeric(df.loc[miss, 'tenure'], errors='coerce').fillna(0).astype(np.float32))
    return out.astype(np.float32)


def add_digit_features(df, col, int_positions=None, dec_positions=None):
    vals_abs = np.abs(pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32).values)
    if int_positions is not None:
        int_part = np.floor(vals_abs).astype(np.int64)
        for p in int_positions:
            digit = ((int_part // (10 ** p)) % 10).astype(np.int16)
            df[f'{col}_int_d{p}_'] = pd.Series(digit, index=df.index).astype('category')
    if dec_positions is not None:
        frac = vals_abs - np.floor(vals_abs)
        for p in dec_positions:
            digit = (np.floor(frac * (10 ** p)).astype(np.int64) % 10).astype(np.int16)
            df[f'{col}_dec_d{p}_'] = pd.Series(digit, index=df.index).astype('category')
    return df


def feature_engineering(df, num_cols_for_cat):
    """New categoricals end with '_', new numerics start with '_'."""
    df = df.copy()
    df['_MonthlyCharges_DIV_TotalCharges'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1e-6)).astype('float32')
    df['_TotalCharges_DIV_tenure'] = (df['TotalCharges'] / (df['tenure'] + 1e-6)).astype('float32')
    df['_TotalCharges_DIV_MonthlyCharges'] = (df['TotalCharges'] / (df['MonthlyCharges'] + 1e-6)).astype('float32')

    df = add_digit_features(df, col='MonthlyCharges', int_positions=[0, 1, 2], dec_positions=[1, 2])
    df = add_digit_features(df, col='TotalCharges', int_positions=[0, 1, 2, 3], dec_positions=[1, 2])
    df = add_digit_features(df, col='tenure', int_positions=[0, 1], dec_positions=None)

    for col, n_bins in [('MonthlyCharges', 200), ('TotalCharges', 4000)]:
        kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
        df[f'{col}_bin_'] = kb.fit_transform(df[[col]]).ravel().astype('int32')
        df[f'{col}_bin_'] = df[f'{col}_bin_'].astype('category')

    for col in num_cols_for_cat:
        round_flag = (col == 'TotalCharges')
        series = df[col].round(0) if round_flag else df[col]
        _, uniques = pd.factorize(series)
        code_map = {cat: i for i, cat in enumerate(uniques)}
        df[f'{col}_cat_'] = series.map(code_map).fillna(-1).astype('int32')
        df[f'{col}_cat_'] = df[f'{col}_cat_'].astype('category')

    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]
    return df, new_cat_cols, new_num_cols


def encode_cat_frame(df, cols, maps):
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = df[c].astype(str).map(maps[c]).fillna(0).astype(np.int64)
    return out


def label_encode_frames(train_df, test_df, orig_df, cols):
    train_out, test_out, orig_out = train_df.copy(), test_df.copy(), orig_df.copy()
    for c in cols:
        all_vals = pd.concat([train_out[c].astype('string').fillna('__MISSING__'),
                              test_out[c].astype('string').fillna('__MISSING__'),
                              orig_out[c].astype('string').fillna('__MISSING__')], axis=0)
        uniq = pd.Index(all_vals.unique())
        mapping = pd.Series(np.arange(len(uniq), dtype=np.int32), index=uniq)
        for df in (train_out, test_out, orig_out):
            df[c] = df[c].astype('string').fillna('__MISSING__').map(mapping).astype(np.int32)
    return train_out, test_out, orig_out


def build_features():
    train_df, test_df, orig_df = read_train(), read_test(), read_original()
    train_df[TARGET] = train_df[TARGET].map({'Yes': 1, 'No': 0}).astype(np.int8)
    orig_df[TARGET] = orig_df[TARGET].map({'Yes': 1, 'No': 0}).astype(np.int8)
    for df in (train_df, test_df, orig_df):
        df['TotalCharges'] = safe_totalcharges(df)
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').astype(np.float32)
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').astype(np.float32)

    X = train_df.drop([ID, TARGET], axis=1).copy()
    y = train_df[TARGET].copy()
    X_test = test_df.drop([ID], axis=1).copy()
    X_orig = orig_df.drop([TARGET], axis=1).copy()
    y_orig = orig_df[TARGET].copy()
    base_cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    base_num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    df_all = pd.concat([X, X_test, X_orig], axis=0).reset_index(drop=True)
    df_all_fe, new_cat_cols, new_num_cols = feature_engineering(df_all, num_cols_for_cat=base_num_cols)
    n_train, n_test = len(X), len(X_test)
    X = df_all_fe.iloc[:n_train].reset_index(drop=True)
    X_test = df_all_fe.iloc[n_train:n_train + n_test].reset_index(drop=True)
    X_orig = df_all_fe.iloc[n_train + n_test:].reset_index(drop=True)
    cat_cols = base_cat_cols + new_cat_cols
    num_cols = base_num_cols + new_num_cols

    for c in base_cat_cols:
        for df in (X, X_test, X_orig):
            df[c] = df[c].astype(str).astype('category')
    X, X_test, X_orig = X.copy(), X_test.copy(), X_orig.copy()
    for c in cat_cols:
        for df in (X, X_test, X_orig):
            df[c] = df[c].astype(str).fillna('__MISSING__')
    for c in num_cols:
        for df in (X, X_test, X_orig):
            df[c] = df[c].astype('float32')

    cat_maps, cat_cardinalities = {}, {}
    for c in cat_cols:
        vals = sorted(pd.concat([X[c], X_test[c], X_orig[c]], axis=0).astype(str).unique().tolist())
        cat_maps[c] = {v: i + 1 for i, v in enumerate(vals)}
        cat_cardinalities[c] = len(cat_maps[c]) + 1
    X_cat = encode_cat_frame(X, cat_cols, cat_maps)
    X_test_cat = encode_cat_frame(X_test, cat_cols, cat_maps)
    X_orig_cat = encode_cat_frame(X_orig, cat_cols, cat_maps)
    X_teenc, X_test_teenc, X_orig_teenc = label_encode_frames(X, X_test, X_orig, cat_cols)
    print(f'{len(cat_cols)} categorical, {len(num_cols)} numeric columns', flush=True)
    return dict(y=y, y_orig=y_orig, cat_cols=cat_cols, num_cols=num_cols, cat_cardinalities=cat_cardinalities,
                X_num=X[num_cols].values.astype(np.float32), X_test_num=X_test[num_cols].values.astype(np.float32),
                X_orig_num=X_orig[num_cols].values.astype(np.float32),
                X_cat=X_cat[cat_cols].values.astype(np.int64), X_test_cat=X_test_cat[cat_cols].values.astype(np.int64),
                X_orig_cat=X_orig_cat[cat_cols].values.astype(np.int64),
                X_teenc=X_teenc, X_test_teenc=X_test_teenc, X_orig_teenc=X_orig_teenc)


def to_cudf(df, cols):
    return cudf.DataFrame({c: cudf.Series(df[c].values).astype('int32') for c in cols})


def make_logit3_features(tr_m, va_m, te_m, eps=1e-5):
    out = []
    for m in (tr_m, va_m, te_m):
        m = np.clip(m, eps, 1.0 - eps)
        z = np.log(m / (1.0 - m)).astype(np.float32)
        out.append(np.hstack([z, z ** 2, z ** 3]).astype(np.float32))
    return tuple(out)


def build_pair_te_features_gpu(Xtr_int, ytr, Xva_int, Xte_int, pair_cols, all_cat_cols, inner_splits=5, seed=0,
                               smooth=0, use_logit3=True):
    n_pair = len(pair_cols)
    tr_pair = np.zeros((len(Xtr_int), n_pair), dtype=np.float32)
    va_pair = np.zeros((len(Xva_int), n_pair), dtype=np.float32)
    te_pair = np.zeros((len(Xte_int), n_pair), dtype=np.float32)
    Xtr_g = to_cudf(Xtr_int, all_cat_cols)
    Xva_g = to_cudf(Xva_int, all_cat_cols)
    Xte_g = to_cudf(Xte_int, all_cat_cols)
    ytr_g = cudf.Series(np.asarray(ytr, dtype=np.int32))
    te = TargetEncoder(n_folds=inner_splits, smooth=smooth, seed=seed, split_method='random', stat='mean',
                       output_type='cupy')
    for j, (f1, f2) in enumerate(pair_cols):
        Xtr_ij = Xtr_g[[f1, f2]]
        tr_pair[:, j] = cp.asnumpy(te.fit_transform(Xtr_ij, ytr_g)).ravel().astype(np.float32)
        te.fit(Xtr_ij, ytr_g)
        va_pair[:, j] = cp.asnumpy(te.transform(Xva_g[[f1, f2]])).ravel().astype(np.float32)
        te_pair[:, j] = cp.asnumpy(te.transform(Xte_g[[f1, f2]])).ravel().astype(np.float32)
        del Xtr_ij
        gc.collect()
    if use_logit3:
        tr_pair, va_pair, te_pair = make_logit3_features(tr_pair, va_pair, te_pair)
    return tr_pair, va_pair, te_pair


# ----------------------------------------------------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------------------------------------------------
class TabDataset(Dataset):
    def __init__(self, X_cat_np, X_num_np, X_te_np, y_np=None):
        self.X_cat = torch.tensor(X_cat_np, dtype=torch.long)
        self.X_num = torch.tensor(X_num_np, dtype=torch.float32)
        self.X_te = torch.tensor(X_te_np, dtype=torch.float32)
        self.y = None if y_np is None else torch.tensor(y_np, dtype=torch.float32)

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        item = {'x_cat': self.X_cat[idx], 'x_num': self.X_num[idx], 'x_te': self.X_te[idx]}
        if self.y is not None:
            item['y'] = self.y[idx]
        return item


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        return self.norm2(x + self.dropout(self.ff(x)))


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.block(x)


class TabTransformerTE(nn.Module):
    def __init__(self, cat_cardinalities, num_numeric, te_dim, embed_dim=16, num_heads=4, ff_dim=64,
                 num_transformer_blocks=2, mlp_hidden=(128, 64), dropout=0.2):
        super().__init__()
        self.cat_cols = list(cat_cardinalities.keys())
        self.num_numeric = num_numeric
        self.embeddings = nn.ModuleList([nn.Embedding(cat_cardinalities[c], embed_dim) for c in self.cat_cols])
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                                                 for _ in range(num_transformer_blocks)])
        self.te_bn = nn.BatchNorm1d(te_dim)
        te_hidden = min(256, max(32, te_dim // 2))
        self.te_proj = nn.Sequential(nn.Linear(te_dim, te_hidden), nn.ReLU(), nn.Dropout(dropout))
        prev = len(self.cat_cols) * embed_dim + te_hidden + num_numeric
        mlp_layers = []
        for h in mlp_hidden:
            mlp_layers.append(MLPBlock(prev, h, dropout))
            prev = h
        self.mlp = nn.Sequential(*mlp_layers)
        self.out = nn.Linear(prev, 1)

    def forward(self, x_cat, x_num, x_te):
        x_cat_tokens = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        for block in self.transformer_blocks:
            x_cat_tokens = block(x_cat_tokens)
        feats = [x_cat_tokens.reshape(x_cat_tokens.size(0), -1), self.te_proj(self.te_bn(x_te))]
        if self.num_numeric > 0:
            feats.append(x_num)
        return self.out(self.mlp(torch.cat(feats, dim=1))).squeeze(1)


def smooth_targets(y, smoothing=0.0):
    if smoothing <= 0:
        return y
    return y * (1.0 - smoothing) + 0.5 * smoothing


def evaluate_model(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['x_cat'].to(device), batch['x_num'].to(device), batch['x_te'].to(device))
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            targets.append(batch['y'].numpy())
    preds, targets = np.concatenate(preds), np.concatenate(targets)
    return roc_auc_score(targets, preds), preds


def predict_model(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['x_cat'].to(device), batch['x_num'].to(device), batch['x_te'].to(device))
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(preds)


def train_one_fold(Xtr_cat, Xtr_num, tr_te, ytr, Xva_cat, Xva_num, va_te, yva, Xte_cat, Xte_num, te_te,
                   cat_cardinalities):
    loader_kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=(DEVICE == 'cuda'), drop_last=False)
    train_loader = DataLoader(TabDataset(Xtr_cat, Xtr_num, tr_te, ytr), shuffle=True, **loader_kw)
    val_loader = DataLoader(TabDataset(Xva_cat, Xva_num, va_te, yva), shuffle=False, **loader_kw)
    test_loader = DataLoader(TabDataset(Xte_cat, Xte_num, te_te, None), shuffle=False, **loader_kw)

    model = TabTransformerTE(cat_cardinalities=cat_cardinalities, num_numeric=Xtr_num.shape[1], te_dim=tr_te.shape[1],
                             embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM,
                             num_transformer_blocks=NUM_TRANSFORMER_BLOCKS, mlp_hidden=MLP_HIDDEN,
                             dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=LR_FACTOR,
                                                           patience=PATIENCE_LR, min_lr=MIN_LR)
    criterion = nn.BCEWithLogitsLoss()

    best_auc, best_state, bad_epochs = -np.inf, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            yb = batch['y'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch['x_cat'].to(DEVICE), batch['x_num'].to(DEVICE), batch['x_te'].to(DEVICE))
            loss = criterion(logits, smooth_targets(yb, LABEL_SMOOTHING))
            loss.backward()
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_losses.append(loss.item())
        val_auc, _ = evaluate_model(model, val_loader, DEVICE)
        scheduler.step(val_auc)
        print(f'Epoch {epoch:02d} | train_loss={float(np.mean(train_losses)):.5f} | val_auc={val_auc:.6f} '
              f'| lr={optimizer.param_groups[0]["lr"]:.6g}', flush=True)
        if val_auc > best_auc:
            best_auc, bad_epochs = val_auc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE_ES:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    _, val_pred = evaluate_model(model, val_loader, DEVICE)
    test_pred = predict_model(model, test_loader, DEVICE)
    del train_loader, val_loader, test_loader
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    return val_pred.astype(np.float32), test_pred.astype(np.float32), best_auc


def main():
    seed_everything(SEED)
    f = build_features()
    y, cat_cols = f['y'], f['cat_cols']
    pair_cols = list(combinations(cat_cols, 2))
    if MAX_PAIRS is not None:
        pair_cols = pair_cols[:MAX_PAIRS]
    print(f'{len(pair_cols)} target-encoded pairs', flush=True)

    outer_kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    test_pred = np.zeros(len(f['X_test_num']), dtype=np.float32)
    for fold, (t_idx, v_idx) in enumerate(outer_kfold.split(np.zeros(len(y)), y), 1):
        print(f'\n======== Fold {fold} ========', flush=True)
        yt = y.iloc[t_idx].values.astype(np.float32)
        yv = y.iloc[v_idx].values.astype(np.float32)
        Xt_cat, Xv_cat = f['X_cat'][t_idx].copy(), f['X_cat'][v_idx].copy()
        Xt_num, Xv_num = f['X_num'][t_idx].copy(), f['X_num'][v_idx].copy()
        Xt_int, Xv_int = f['X_teenc'].iloc[t_idx].copy(), f['X_teenc'].iloc[v_idx].copy()
        if USE_X_ORIG:
            Xt_cat = np.vstack([Xt_cat, f['X_orig_cat']])
            Xt_num = np.vstack([Xt_num, f['X_orig_num']])
            Xt_int = pd.concat([Xt_int, f['X_orig_teenc']], axis=0).reset_index(drop=True)
            yt = np.concatenate([yt, f['y_orig'].values.astype(np.float32)], axis=0)
        X_test_cat_fold = f['X_test_cat'].copy()
        X_test_num_fold = f['X_test_num'].copy()

        scaler_num = StandardScaler()
        Xt_num = scaler_num.fit_transform(Xt_num).astype(np.float32)
        Xv_num = scaler_num.transform(Xv_num).astype(np.float32)
        X_test_num_fold = scaler_num.transform(X_test_num_fold).astype(np.float32)

        tr_te, va_te, te_te = build_pair_te_features_gpu(Xt_int, yt.astype(np.int32), Xv_int, f['X_test_teenc'].copy(),
                                                         pair_cols=pair_cols, all_cat_cols=cat_cols,
                                                         inner_splits=TE_INNER_SPLITS, seed=SEED + fold,
                                                         smooth=TE_SMOOTH, use_logit3=USE_LOGIT3)
        scaler_te = StandardScaler()
        tr_te = scaler_te.fit_transform(tr_te).astype(np.float32)
        va_te = scaler_te.transform(va_te).astype(np.float32)
        te_te = scaler_te.transform(te_te).astype(np.float32)

        fold_val_pred, fold_test_pred, best_auc = train_one_fold(
            Xt_cat, Xt_num, tr_te, yt, Xv_cat, Xv_num, va_te, yv, X_test_cat_fold, X_test_num_fold, te_te,
            f['cat_cardinalities'])
        oof[v_idx] = fold_val_pred
        test_pred += fold_test_pred / N_SPLITS
        print(f'Fold {fold} AUC: {roc_auc_score(yv, oof[v_idx]):.6f} | best epoch AUC: {best_auc:.6f}', flush=True)

        del Xt_cat, Xv_cat, Xt_num, Xv_num, Xt_int, Xv_int, X_test_cat_fold, X_test_num_fold, tr_te, va_te, te_te
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    print(f'OOF AUC: {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(MODEL_DIR, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
