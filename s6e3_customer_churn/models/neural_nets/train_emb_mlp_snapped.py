#!/usr/bin/env python
"""Embedding MLP on the 16 categoricals, fold-local snapped-value categories / rare flags of the 3 numerics and the 3
standardized numerics. Two presets.

Inputs, built inside each fold from its training rows:
- 22 categorical embeddings: the 16 categoricals and, for tenure, MonthlyCharges and TotalCharges, the value snapped to
  the nearest value that occurs at least 25 times in the fold's training rows (frequent values stay as they are; ties go
  to the larger value) plus a rare flag (value not frequent). Vocabularies come from the fold's training rows
  (unseen -> 0);
- the 3 numerics, standardized on the fold's training rows.
Network: embeddings (N(0, 0.02) init, dropout 0.1) + numerics -> MLP 512-256 (ReLU, dropout) -> 1 logit.
Training: BCE on 0.02-smoothed labels, AdamW, batch 8192, up to 80 epochs with early stopping on the held-out AUC
(patience 10); the best epoch predicts (sigmoid).
Presets:
- emb_mlp_snapped_numerics: embedding dim int(sqrt(cardinality)) clipped to 4-128, constant lr 2e-3, weight decay 1e-4,
  MLP dropout 0.25; no seeding.
- emb_mlp_snapped_numerics_layernorm_cosine: embedding dim round(1.8 cardinality^0.25) clipped to 4-64, LayerNorm after
  each hidden layer, MLP dropout 0.3, lr 1e-3 (cosine annealing to 5e-5 with T_max 79 epochs, stepped from the 2nd
  epoch), weight decay 3e-4, gradient clipping 1; numpy / torch seeded with 42.
Usage: python train_emb_mlp_snapped.py {emb_mlp_snapped_numerics,emb_mlp_snapped_numerics_layernorm_cosine}"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, NUMERICS, TARGET, labels, model_dir, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

PRESETS = {
    'emb_mlp_snapped_numerics': dict(
        seed=None, lr=2e-3, weight_decay=1e-4, mlp_dropout=0.25, emb_dim_rule='sqrt', layer_norm=False,
        cosine=False, grad_clip=None, test_folds=50),
    'emb_mlp_snapped_numerics_layernorm_cosine': dict(
        seed=42, lr=1e-3, weight_decay=3e-4, mlp_dropout=0.30, emb_dim_rule='quarter_power', layer_norm=True,
        cosine=True, grad_clip=1.0, test_folds=100),
}
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUMS = list(NUMERICS)
CAT_FEATS = list(CATEGORICALS)
RARE_MIN_COUNT = 25
RANDOM_STATE = 42
EPOCHS = 80
BATCH_SIZE = 8192
PATIENCE = 10
EMB_DROPOUT = 0.10
HIDDEN = (512, 256)
WARMUP_EPOCHS = 1


def make_vocab_maps(train_df: pd.DataFrame, cols):
    """value -> id maps from the fold's training rows; 0 is reserved for unseen values."""
    maps, sizes = {}, {}
    for c in cols:
        uniq = pd.Series(train_df[c].values).astype(str).unique().tolist()
        v2i = {v: i + 1 for i, v in enumerate(uniq)}
        maps[c] = v2i
        sizes[c] = len(v2i) + 1
    return maps, sizes


def encode_with_maps(df: pd.DataFrame, cols, maps):
    X = np.zeros((len(df), len(cols)), dtype=np.int64)
    for j, c in enumerate(cols):
        X[:, j] = pd.Series(df[c].values).astype(str).map(maps[c]).fillna(0).astype(np.int64).values
    return X


def emb_dim_from_card(card: int, rule: str) -> int:
    if rule == 'sqrt':
        return int(np.clip(int(np.sqrt(max(1, card))), 4, 128))
    return int(np.clip(int(round(1.8 * (card ** 0.25))), 4, 64))


def build_numeric_snapper(train_series: pd.Series, rare_min_count: int):
    """Frequent values: count >= rare_min_count in the fold's training rows. Rare values (and NaN) get the rare flag;
    rare finite values are snapped to the nearest frequent value."""
    s = pd.to_numeric(train_series, errors='coerce').astype(np.float32)
    vc = pd.Series(s).value_counts(dropna=False)
    frequent_vals = vc[vc >= rare_min_count].index.values
    frequent_vals = np.array([v for v in frequent_vals if pd.notna(v)], dtype=np.float32)
    if frequent_vals.size == 0:
        frequent_vals = np.array(pd.Series(s.dropna()).unique(), dtype=np.float32)
    frequent_vals = np.sort(frequent_vals)
    frequent_set = set(frequent_vals.tolist())

    def transform(series_any: pd.Series):
        x = pd.to_numeric(series_any, errors='coerce').astype(np.float32).values
        is_nan = np.isnan(x)
        is_rare = np.ones_like(x, dtype=np.int32)
        for i, v in enumerate(x):
            if np.isnan(v):
                is_rare[i] = 1
            else:
                is_rare[i] = 0 if (float(v) in frequent_set) else 1
        x_snapped = x.copy()
        if frequent_vals.size > 0:
            idx_snap = np.where((~is_nan) & (is_rare == 1))[0]
            if idx_snap.size > 0:
                v = x[idx_snap]
                pos = np.clip(np.searchsorted(frequent_vals, v), 0, len(frequent_vals) - 1)
                left = np.clip(pos - 1, 0, len(frequent_vals) - 1)
                left_vals, right_vals = frequent_vals[left], frequent_vals[pos]
                choose_right = (np.abs(v - right_vals) <= np.abs(v - left_vals))
                x_snapped[idx_snap] = np.where(choose_right, right_vals, left_vals).astype(np.float32)
        return x_snapped.astype(np.float32), is_rare.astype(np.int32)

    return transform


class TabMixDataset(Dataset):
    def __init__(self, X_cat, X_num, y=None):
        self.Xc = torch.as_tensor(X_cat, dtype=torch.long)
        self.Xn = torch.as_tensor(X_num, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.Xc.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.Xc[idx], self.Xn[idx]
        return self.Xc[idx], self.Xn[idx], self.y[idx]


class EmbMLP(nn.Module):
    def __init__(self, cat_cardinals, n_num, emb_dim_rule, layer_norm, hidden=(256, 128), emb_dropout=0.1,
                 mlp_dropout=0.2):
        super().__init__()
        self.emb_layers = nn.ModuleList()
        emb_out_dim = 0
        for card in cat_cardinals:
            d = emb_dim_from_card(card, emb_dim_rule)
            self.emb_layers.append(nn.Embedding(num_embeddings=card, embedding_dim=d))
            emb_out_dim += d
        self.emb_drop = nn.Dropout(emb_dropout)
        in_dim = emb_out_dim + n_num
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h)]
            if layer_norm:
                layers += [nn.LayerNorm(h)]
            layers += [nn.ReLU(inplace=True), nn.Dropout(mlp_dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)
        for emb in self.emb_layers:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, j]) for j, emb in enumerate(self.emb_layers)]
        z = torch.cat(embs, dim=1) if len(embs) else None
        if z is None:
            z = x_num
        else:
            z = torch.cat([self.emb_drop(z), x_num], dim=1)
        return self.mlp(z).squeeze(1)


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    out = []
    for batch in loader:
        xc, xn = batch[0], batch[1]
        logit = model(xc.to(DEVICE, non_blocking=True), xn.to(DEVICE, non_blocking=True))
        out.append(torch.sigmoid(logit).detach().cpu().numpy())
    return np.concatenate(out)


class SmoothBCE(nn.Module):
    def __init__(self, eps=0.02):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets * (1 - self.eps) + 0.5 * self.eps
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def train_one_fold(cfg, Xc_tr, Xn_tr, y_tr, Xc_va, Xn_va, y_va, cat_cardinals):
    lr = cfg['lr']
    model = EmbMLP(cat_cardinals, n_num=Xn_tr.shape[1], emb_dim_rule=cfg['emb_dim_rule'], layer_norm=cfg['layer_norm'],
                   hidden=HIDDEN, emb_dropout=EMB_DROPOUT, mlp_dropout=cfg['mlp_dropout']).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    loss_fn = SmoothBCE(0.02)
    loader_kw = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    dl_tr = DataLoader(TabMixDataset(Xc_tr, Xn_tr, y_tr), shuffle=True, **loader_kw)
    dl_va = DataLoader(TabMixDataset(Xc_va, Xn_va, y_va), shuffle=False, **loader_kw)
    sched = None
    if cfg['cosine']:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=lr * 0.05)

    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        if sched is not None and epoch <= WARMUP_EPOCHS:
            for pg in opt.param_groups:
                pg['lr'] = lr * (0.1 + 0.9 * (epoch / WARMUP_EPOCHS))
        running_loss, n_seen = 0.0, 0
        for xc, xn, yb in dl_tr:
            xc = xc.to(DEVICE, non_blocking=True)
            xn = xn.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xc, xn), yb)
            loss.backward()
            if cfg['grad_clip'] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step()
            running_loss += loss.item() * xc.size(0)
            n_seen += xc.size(0)
        if sched is not None and epoch > WARMUP_EPOCHS:
            sched.step()

        auc = roc_auc_score(y_va, predict_proba(model, dl_va))
        print(f'     epoch {epoch:03d} | loss {running_loss / max(1, n_seen):.5f} | val_auc {auc:.6f}', flush=True)
        if auc > best_auc + 1e-6:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def cross_validate(cfg, train, test, y, n_splits):
    if cfg['seed'] is not None:
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(train), dtype=np.float32)
    pred_test = np.zeros(len(test), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train)), y), 1):
        print(f'\n======== Fold {fold}/{n_splits} ========', flush=True)
        tr_df = train.iloc[tr_idx].reset_index(drop=True)
        va_df = train.iloc[va_idx].reset_index(drop=True)

        tr_cat_df = tr_df[CAT_FEATS].copy()
        va_cat_df = va_df[CAT_FEATS].copy()
        te_cat_df = test[CAT_FEATS].copy()
        for col in NUMS:
            snapper = build_numeric_snapper(tr_df[col], rare_min_count=RARE_MIN_COUNT)
            tr_snap, tr_israre = snapper(tr_df[col])
            va_snap, va_israre = snapper(va_df[col])
            te_snap, te_israre = snapper(test[col])
            tr_cat_df[f'{col}__cat'] = pd.Series(tr_snap).astype(str).values
            va_cat_df[f'{col}__cat'] = pd.Series(va_snap).astype(str).values
            te_cat_df[f'{col}__cat'] = pd.Series(te_snap).astype(str).values
            tr_cat_df[f'{col}__is_rare'] = tr_israre
            va_cat_df[f'{col}__is_rare'] = va_israre
            te_cat_df[f'{col}__is_rare'] = te_israre
        cat_all = list(tr_cat_df.columns)

        maps, sizes = make_vocab_maps(tr_cat_df, cat_all)
        cat_cardinals = [sizes[c] for c in cat_all]
        Xc_tr = encode_with_maps(tr_cat_df, cat_all, maps)
        Xc_va = encode_with_maps(va_cat_df, cat_all, maps)
        Xc_te = encode_with_maps(te_cat_df, cat_all, maps)

        scaler = StandardScaler()
        Xn_tr = scaler.fit_transform(tr_df[NUMS].astype(np.float32).values).astype(np.float32)
        Xn_va = scaler.transform(va_df[NUMS].astype(np.float32).values).astype(np.float32)
        Xn_te = scaler.transform(test[NUMS].astype(np.float32).values).astype(np.float32)

        model, best_auc = train_one_fold(cfg, Xc_tr, Xn_tr, y[tr_idx], Xc_va, Xn_va, y[va_idx], cat_cardinals)
        dl_va = DataLoader(TabMixDataset(Xc_va, Xn_va, y[va_idx]), batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                           pin_memory=True)
        oof[va_idx] = predict_proba(model, dl_va)
        dl_te = DataLoader(TabMixDataset(Xc_te, Xn_te, None), batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                           pin_memory=True)
        pred_test += predict_proba(model, dl_te).astype(np.float32) / n_splits
        print(f'[Fold {fold}] best val AUC: {best_auc:.6f}', flush=True)
    print(f'{n_splits}-fold OOF AUC: {roc_auc_score(y, oof):.6f}', flush=True)
    return oof, pred_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=sorted(PRESETS))
    pair = ap.parse_args().pair
    cfg = PRESETS[pair]

    train, test = read_train(), read_test()
    y = labels(train[TARGET]).astype(np.float32)
    oof, pred_test = cross_validate(cfg, train, test, y, 5)
    if cfg['test_folds'] != 5:
        _, pred_test = cross_validate(cfg, train, test, y, cfg['test_folds'])
    save_pair(MODEL_DIR, pair, oof, pred_test)


if __name__ == '__main__':
    main()
