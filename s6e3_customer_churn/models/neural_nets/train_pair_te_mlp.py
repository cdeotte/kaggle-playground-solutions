#!/usr/bin/env python
"""MLP on cuML target encodings of all 171 column pairs.

Features: the 19 raw columns are label-encoded on train + test (values as strings). Inside each fold, every pair of
columns is target-encoded with cuML `TargetEncoder` (smooth 0, mean, random split, seed 42 + fold number 1-5): the
fold's training rows get out-of-fold encodings from inner folds (`fit_transform`), the held-out and test rows the
encoding fitted on all of the fold's training rows. The 171 encodings are standardized on the fold's training rows.
Network: MLP 512-256-128 (LayerNorm, ReLU, dropout 0.3) -> 1 logit.
Training: BCE on 0.02-smoothed labels, AdamW lr 2e-3, weight decay 2e-4, batch 2048, gradient clipping 1, up to 80
epochs; lr constant for 2 epochs, then cosine annealing to 1e-4 (T_max 79); early stopping on the held-out AUC (patience
10) and the best epoch predicts (sigmoid). numpy / torch are seeded with 42 before the folds.
Usage: python train_pair_te_mlp.py"""
import sys
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, model_dir, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

PAIR = 'mlp_pair_target_encoding'
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
RANDOM_STATE = 42
DROP_COLS = ['customerID', 'id']

TE_N_FOLDS = 5
TEST_FOLDS = 50
TEST_TE_N_FOLDS = 10

EPOCHS = 80
BATCH_SIZE = 2048
LR = 2e-3
WEIGHT_DECAY = 2e-4
PATIENCE = 10
LABEL_SMOOTH = 0.02
HIDDEN = (512, 256, 128)
DROPOUT = 0.30
WARMUP_EPOCHS = 1
ETA_MIN = LR * 0.05


def label_encode_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, cols):
    train_out, test_out = train_df.copy(), test_df.copy()
    for c in cols:
        tr_s = train_out[c].astype('string').fillna('__MISSING__')
        te_s = test_out[c].astype('string').fillna('__MISSING__')
        uniq = pd.Index(pd.concat([tr_s, te_s], axis=0).unique())
        mapping = pd.Series(np.arange(len(uniq), dtype=np.int32), index=uniq)
        train_out[c] = tr_s.map(mapping).astype(np.int32)
        test_out[c] = te_s.map(mapping).astype(np.int32)
    return train_out, test_out


class FloatDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
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


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    out = []
    for batch in loader:
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        out.append(torch.sigmoid(model(xb.to(DEVICE, non_blocking=True))).detach().cpu().numpy())
    return np.concatenate(out)


def train_mlp_one_fold(X_tr, y_tr, X_va, y_va):
    model = MLP(in_dim=X_tr.shape[1], hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = SmoothBCE(LABEL_SMOOTH)
    dl_tr = DataLoader(FloatDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
                       persistent_workers=True, prefetch_factor=4)
    dl_va = DataLoader(FloatDataset(X_va, y_va), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=ETA_MIN)

    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        if epoch <= WARMUP_EPOCHS:
            for pg in opt.param_groups:
                pg['lr'] = LR * (0.1 + 0.9 * (epoch / max(1, WARMUP_EPOCHS)))
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
        if epoch > WARMUP_EPOCHS:
            sched.step()

        auc = roc_auc_score(y_va, predict_proba(model, dl_va))
        print(f'     epoch {epoch:03d} | lr {opt.param_groups[0]["lr"]:.2e} | loss {running / max(1, seen):.5f} '
              f'| val_auc {auc:.6f}', flush=True)
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


def cross_validate(X_all_g, X_test_g, y_all, pair_cols, n_test, n_splits, te_n_folds):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    n, n_pair = len(y_all), len(pair_cols)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(n, dtype=np.float32)
    pred_test = np.zeros(n_test, dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(np.zeros(n), y_all), 1):
        print(f'\n[Fold {fold:02d}/{n_splits}] tr={len(tr_idx)} va={len(va_idx)}', flush=True)
        X_tr_g = X_all_g.take(cudf.Series(tr_idx))
        X_va_g = X_all_g.take(cudf.Series(va_idx))
        y_tr = y_all[tr_idx].astype(np.int32)
        y_va = y_all[va_idx].astype(np.int32)
        y_tr_g = cudf.Series(y_tr)

        tr_pair = np.zeros((len(tr_idx), n_pair), dtype=np.float32)
        va_pair = np.zeros((len(va_idx), n_pair), dtype=np.float32)
        te_pair = np.zeros((n_test, n_pair), dtype=np.float32)
        te = TargetEncoder(n_folds=te_n_folds, smooth=0, seed=RANDOM_STATE + fold, split_method='random', stat='mean',
                           output_type='cupy')
        for t, (f1, f2) in enumerate(pair_cols):
            X_tr_ij_g = X_tr_g[[f1, f2]]
            tr_pair[:, t] = cp.asnumpy(te.fit_transform(X_tr_ij_g, y_tr_g)).ravel().astype(np.float32)
            te.fit(X_tr_ij_g, y_tr_g)
            va_pair[:, t] = cp.asnumpy(te.transform(X_va_g[[f1, f2]])).ravel().astype(np.float32)
            te_pair[:, t] = cp.asnumpy(te.transform(X_test_g[[f1, f2]])).ravel().astype(np.float32)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr_pair).astype(np.float32)
        X_va = scaler.transform(va_pair).astype(np.float32)
        X_te = scaler.transform(te_pair).astype(np.float32)

        model, best_auc = train_mlp_one_fold(X_tr, y_tr.astype(np.float32), X_va, y_va.astype(np.float32))
        dl_va = DataLoader(FloatDataset(X_va, None), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        p_va = predict_proba(model, dl_va).astype(np.float32)
        oof[va_idx] = p_va
        dl_te = DataLoader(FloatDataset(X_te, None), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        pred_test += predict_proba(model, dl_te).astype(np.float32) / n_splits
        print(f'  Fold {fold:02d} AUC: {roc_auc_score(y_va, p_va):.6f} | best_epoch_auc: {best_auc:.6f}', flush=True)
    print(f'{n_splits}-fold OOF AUC: {roc_auc_score(y_all, oof):.6f}', flush=True)
    return oof, pred_test


def main():
    train, test = read_train(), read_test()
    features = [c for c in train.columns if c not in DROP_COLS + [TARGET]]
    y_all = (train[TARGET].astype(str).str.strip().str.lower() == 'yes').astype(np.int32).values

    train_enc, test_enc = label_encode_train_test(train[features], test[features], features)
    X_all_g = cudf.DataFrame({f: cudf.Series(train_enc[f].values).astype('int32') for f in features})
    X_test_g = cudf.DataFrame({f: cudf.Series(test_enc[f].values).astype('int32') for f in features})
    pair_cols = [(features[i], features[j]) for i in range(len(features)) for j in range(i + 1, len(features))]
    print(f'{len(features)} columns, {len(pair_cols)} pairs', flush=True)

    oof, pred_test = cross_validate(X_all_g, X_test_g, y_all, pair_cols, len(test), 5, TE_N_FOLDS)
    if TEST_FOLDS != 5:
        _, pred_test = cross_validate(X_all_g, X_test_g, y_all, pair_cols, len(test), TEST_FOLDS, TEST_TE_N_FOLDS)
    save_pair(MODEL_DIR, PAIR, oof, pred_test)


if __name__ == '__main__':
    main()
