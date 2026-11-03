#!/usr/bin/env python
"""MLP on one learned embedding per column pair (171 pairs of the 19 raw columns).

Inputs, built inside each fold from its training rows: every raw column as a string (NaN -> '__MISSING__'); values that
occur fewer than 25 times in the fold's training rows become '__RARE__'. Each of the 171 column pairs becomes one token
'a||b' with a vocabulary from the fold's training rows (unseen -> 0).
Network: one embedding table per pair (dim round(1.6 cardinality^0.25) clipped to 4-16, N(0, 0.02) init), concatenated
(dropout 0.1) -> MLP 512-256-128 (LayerNorm, ReLU, dropout 0.3) -> 1 logit.
Training: BCE on 0.02-smoothed labels, AdamW lr 2e-3, weight decay 2e-4, batch 1024, gradient clipping 1, up to 60
epochs with early stopping on the held-out AUC (patience 8); the best epoch predicts (sigmoid). numpy / torch are seeded
with 42 before the folds.
Usage: python train_pair_embedding_mlp.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, model_dir, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

PAIR = 'pair_embedding_mlp'
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
RANDOM_STATE = 42
DROP_COLS = ['customerID', 'id']
TEST_FOLDS = 50

USE_RARE_BUCKET = True
RARE_MIN_COUNT = 25
RARE_TOKEN = '__RARE__'
MISSING_TOKEN = '__MISSING__'
UNK_ID = 0

EPOCHS = 60
BATCH_SIZE = 1024
LR = 2e-3
WEIGHT_DECAY = 2e-4
PATIENCE = 8
LABEL_SMOOTH = 0.02
EMB_DROPOUT = 0.10
MLP_DROPOUT = 0.30
HIDDEN = (512, 256, 128)
MAX_EMB_DIM = 16


def _as_str_series(s: pd.Series) -> pd.Series:
    return s.astype('string').fillna(MISSING_TOKEN)


def build_rare_maps_from_train(tr_df: pd.DataFrame, cols, rare_min_count: int):
    """column -> set of values (as strings) that occur fewer than rare_min_count times in the fold's training rows."""
    rare_sets = {}
    for c in cols:
        vc = _as_str_series(tr_df[c]).value_counts(dropna=False)
        rare_sets[c] = set(vc[vc < rare_min_count].index.astype(str).tolist())
    return rare_sets


def apply_rare_bucket(df: pd.DataFrame, cols, rare_sets: dict):
    out = {}
    for c in cols:
        s = _as_str_series(df[c]).astype(str)
        out[c] = s.mask(s.isin(rare_sets[c]), RARE_TOKEN)
    return out


def make_pair_tokens(col_a: pd.Series, col_b: pd.Series) -> pd.Series:
    return col_a.astype(str) + '||' + col_b.astype(str)


def build_pair_vocab_from_train(tr_tokens: pd.Series):
    """token -> id (1..K) from the fold's training rows; 0 is reserved for unseen tokens."""
    uniq = pd.Index(tr_tokens.unique())
    token2id = pd.Series(np.arange(1, len(uniq) + 1, dtype=np.int64), index=uniq)
    return token2id, int(len(uniq) + 1)


def encode_tokens(tokens: pd.Series, token2id: pd.Series) -> np.ndarray:
    return tokens.map(token2id).fillna(UNK_ID).astype(np.int64).values


def emb_dim_from_card(card: int, max_dim: int = 16) -> int:
    return int(np.clip(int(round(1.6 * (card ** 0.25))), 4, max_dim))


class PairIdDataset(Dataset):
    def __init__(self, X_pair_ids: np.ndarray, y=None):
        self.X = torch.as_tensor(X_pair_ids, dtype=torch.long)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class SmoothBCE(nn.Module):
    def __init__(self, eps=0.02):
        super().__init__()
        self.eps = float(eps)

    def forward(self, logits, targets):
        t = targets * (1.0 - self.eps) + 0.5 * self.eps
        return nn.functional.binary_cross_entropy_with_logits(logits, t)


class PairEmbeddingMLP(nn.Module):
    def __init__(self, pair_cardinals, hidden=(512, 256, 128), emb_dropout=0.1, mlp_dropout=0.3, max_emb_dim=16):
        super().__init__()
        self.embs = nn.ModuleList()
        emb_out = 0
        for card in pair_cardinals:
            d = emb_dim_from_card(card, max_dim=max_emb_dim)
            self.embs.append(nn.Embedding(num_embeddings=card, embedding_dim=d))
            emb_out += d
        self.emb_drop = nn.Dropout(emb_dropout)
        layers, in_dim = [], emb_out
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(inplace=True), nn.Dropout(mlp_dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)
        for e in self.embs:
            nn.init.normal_(e.weight, mean=0.0, std=0.02)

    def forward(self, x_pair_ids):
        z = torch.cat([emb(x_pair_ids[:, j]) for j, emb in enumerate(self.embs)], dim=1)
        return self.mlp(self.emb_drop(z)).squeeze(1)


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    out = []
    for batch in loader:
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        out.append(torch.sigmoid(model(xb.to(DEVICE, non_blocking=True))).detach().cpu().numpy())
    return np.concatenate(out)


def train_one_fold(X_tr_ids, y_tr, X_va_ids, y_va, pair_cardinals):
    model = PairEmbeddingMLP(pair_cardinals=pair_cardinals, hidden=HIDDEN, emb_dropout=EMB_DROPOUT,
                             mlp_dropout=MLP_DROPOUT, max_emb_dim=MAX_EMB_DIM).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = SmoothBCE(LABEL_SMOOTH)
    dl_tr = DataLoader(PairIdDataset(X_tr_ids, y_tr), batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                       pin_memory=True, persistent_workers=True, prefetch_factor=4)
    dl_va = DataLoader(PairIdDataset(X_va_ids, y_va), batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                       pin_memory=True)

    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in dl_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)

        auc = roc_auc_score(y_va, predict_proba(model, dl_va))
        print(f'     epoch {epoch:03d} | loss {running / max(1, seen):.5f} | val_auc {auc:.6f}', flush=True)
        if auc > best_auc + 1e-6:
            best_auc = float(auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def cross_validate(train, test, y_all, features, pair_cols, n_splits):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    n, n_pair = len(train), len(pair_cols)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(n, dtype=np.float32)
    pred_test = np.zeros(len(test), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y_all), 1):
        print(f'\n======== Fold {fold}/{n_splits} ========', flush=True)
        tr_df = train.iloc[tr_idx].reset_index(drop=True)
        va_df = train.iloc[va_idx].reset_index(drop=True)
        te_df = test.reset_index(drop=True)

        if USE_RARE_BUCKET:
            rare_sets = build_rare_maps_from_train(tr_df, features, rare_min_count=RARE_MIN_COUNT)
            tr_cols = apply_rare_bucket(tr_df, features, rare_sets)
            va_cols = apply_rare_bucket(va_df, features, rare_sets)
            te_cols = apply_rare_bucket(te_df, features, rare_sets)
        else:
            tr_cols = {c: _as_str_series(tr_df[c]).astype(str) for c in features}
            va_cols = {c: _as_str_series(va_df[c]).astype(str) for c in features}
            te_cols = {c: _as_str_series(te_df[c]).astype(str) for c in features}

        X_tr_ids = np.zeros((len(tr_df), n_pair), dtype=np.int64)
        X_va_ids = np.zeros((len(va_df), n_pair), dtype=np.int64)
        X_te_ids = np.zeros((len(te_df), n_pair), dtype=np.int64)
        pair_cardinals = []
        for j, (f1, f2) in enumerate(pair_cols):
            tr_tok = make_pair_tokens(tr_cols[f1], tr_cols[f2])
            token2id, card = build_pair_vocab_from_train(tr_tok)
            pair_cardinals.append(card)
            X_tr_ids[:, j] = encode_tokens(tr_tok, token2id)
            X_va_ids[:, j] = encode_tokens(make_pair_tokens(va_cols[f1], va_cols[f2]), token2id)
            X_te_ids[:, j] = encode_tokens(make_pair_tokens(te_cols[f1], te_cols[f2]), token2id)

        y_tr = y_all[tr_idx].astype(np.float32)
        y_va = y_all[va_idx].astype(np.float32)
        model, best_auc = train_one_fold(X_tr_ids, y_tr, X_va_ids, y_va, pair_cardinals)
        dl_va = DataLoader(PairIdDataset(X_va_ids, None), batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                           pin_memory=True)
        p_va = predict_proba(model, dl_va).astype(np.float32)
        oof[va_idx] = p_va
        print(f'[Fold {fold}] AUC: {roc_auc_score(y_va, p_va):.6f} | best_auc: {best_auc:.6f}', flush=True)
        dl_te = DataLoader(PairIdDataset(X_te_ids, None), batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                           pin_memory=True)
        pred_test += predict_proba(model, dl_te).astype(np.float32) / n_splits
    print(f'{n_splits}-fold OOF AUC: {roc_auc_score(y_all, oof):.6f}', flush=True)
    return oof, pred_test


def main():
    train, test = read_train(), read_test()
    features = [c for c in train.columns if c not in DROP_COLS + [TARGET]]
    y_all = (train[TARGET].astype(str).str.strip().str.lower() == 'yes').astype(np.int64).values
    pair_cols = [(features[i], features[j]) for i in range(len(features)) for j in range(i + 1, len(features))]
    print(f'{len(features)} columns, {len(pair_cols)} pairs', flush=True)

    oof, pred_test = cross_validate(train, test, y_all, features, pair_cols, 5)
    if TEST_FOLDS != 5:
        _, pred_test = cross_validate(train, test, y_all, features, pair_cols, TEST_FOLDS)
    save_pair(MODEL_DIR, PAIR, oof, pred_test)


if __name__ == '__main__':
    main()
