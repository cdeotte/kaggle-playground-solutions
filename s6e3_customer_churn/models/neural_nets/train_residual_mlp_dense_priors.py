#!/usr/bin/env python
"""Residual MLP on a dense block of engineered features, original-data priors and out-of-fold target encodings.

Needs `models/snap_features` (cache/combo.csv) first. Features (`lib/dense_priors_features.py`), built on train + test:
- the snap / decimal-string columns of combo.csv (MonthlyCharges_snap, TotalCharges_snap, the decimal digits mc0, mc1,
  tc0, tc1, ...);
- frequency / percentile rank / log1p / sqrt / 1 / (1 + x) / missing flag of tenure, MonthlyCharges, TotalCharges and
  the two snap values; 9 charge relationships; service yes / no / other counts and balance, has-internet / has-phone;
  yes / no / other flags of 11 columns;
- tenure bins and 14-quantile bins of the charges and their snap values; 16 pair and 4 triple crosses; integer and
  decimal digits of the charges; 32 n-gram crosses of the top categoricals;
- count and rare flag (count <= 20) and count of every categorical-like column; mean and difference of the 3 numerics
  within Contract / InternetService / PaymentMethod / tenure bin; original-data churn prior and count of the 16
  categoricals and standardized distances of the numerics to the original class means; train / test frequency ratio of
  the first 120 categorical-like columns; distances to the original class means of the 3 numerics (4 columns);
- smoothed (m = 20) out-of-fold target means of the categoricals, bins, 3 crosses and mc0 / tc0, from the 5 folds of
  StratifiedKFold(5, shuffle, seed 42) on the train rows (test = mean of the 5 fold encodings).
The network takes 555 of these columns: the first 64 categorical-like columns (base categoricals, bins, n-grams and
names with '__') as ordinal-encoded embeddings (fitted on the fold's training rows, unseen -> 0) and the other 491 as
numerics (NaN -> 0, standardized on the fold's training rows).
Network: BatchNorm of the numerics + embeddings (dim min(64, max(8, round(2 sqrt(cardinality + 1))))) -> Linear(384) ->
4 residual blocks (Linear(768), GELU, dropout 0.15, Linear(384), dropout, + skip, LayerNorm) -> Linear(192) -> GELU ->
dropout -> Linear -> 1 logit.
Training: BCE, AdamW lr 1e-3, weight decay 1e-4, batch 1024, up to 40 epochs; ReduceLROnPlateau on the held-out AUC
(factor 0.5, patience 3); early stopping (patience 8, min delta 1e-5) and the best epoch predicts (sigmoid). Python /
numpy / torch are seeded with 42 before the folds.
Usage: python train_residual_mlp_dense_priors.py"""
import copy
import gc
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import model_dir  # noqa: E402
from common.pairs import save_pair  # noqa: E402
from models.neural_nets.lib.dense_priors_features import (  # noqa: E402
    BASE_CATS, N_FOLDS, SEED, TARGET, add_orig_aux_model_features, build_feature_frames, choose_feature_columns,
    make_target_encoding_features, read_data, seed_everything)

PAIR = 'residual_mlp_dense_priors'
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TabDataset(Dataset):
    def __init__(self, x_num, x_cat, y=None):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x_num[idx], self.x_cat[idx]
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim),
                                 nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class TabularNet(nn.Module):
    def __init__(self, cat_cardinalities, n_num, hidden=384, depth=4, dropout=0.15):
        super().__init__()
        self.emb_layers = nn.ModuleList()
        emb_dim_total = 0
        for card in cat_cardinalities:
            emb_dim = min(64, max(8, int(round(math.sqrt(card + 1) * 2))))
            self.emb_layers.append(nn.Embedding(int(card) + 1, emb_dim))
            emb_dim_total += emb_dim
        self.num_bn = nn.BatchNorm1d(n_num)
        self.input = nn.Linear(n_num + emb_dim_total, hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden // 2, 1))

    def forward(self, x_num, x_cat):
        x = torch.cat([self.num_bn(x_num)] + [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)], dim=1)
        return self.head(self.blocks(self.input(x))).squeeze(1)


def evaluate(model, loader):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            preds.append(torch.sigmoid(model(x_num.to(DEVICE), x_cat.to(DEVICE))).detach().cpu().numpy())
            ys.append(y.numpy())
    preds, ys = np.concatenate(preds), np.concatenate(ys)
    return roc_auc_score(ys, preds), preds


def predict_loader(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for x_num, x_cat in loader:
            preds.append(torch.sigmoid(model(x_num.to(DEVICE), x_cat.to(DEVICE))).detach().cpu().numpy())
    return np.concatenate(preds)


def build_inputs():
    train, test, orig, y, y_orig = read_data()
    tr, te = build_feature_frames(train, test, orig, y_orig)
    aux_tr, aux_te = add_orig_aux_model_features(tr, te, orig.drop(columns=[TARGET], errors='ignore'), y_orig)
    tr = pd.concat([tr, aux_tr], axis=1)
    te = pd.concat([te, aux_te], axis=1)

    te_cols = [c for c in tr.columns if c in BASE_CATS or c.endswith('_bin') or c in [
        'Contract__InternetService', 'Contract__PaymentMethod', 'InternetService__TechSupport', 'mc0', 'tc0']]
    te_oof, te_test = make_target_encoding_features(tr, te, y, te_cols, n_splits=N_FOLDS, smooth_list=(20.0,))
    tr = pd.concat([tr, te_oof], axis=1)
    te = pd.concat([te, te_test], axis=1)

    use_cols = choose_feature_columns(tr)
    X, Xtest = tr[use_cols].copy(), te[use_cols].copy()
    cat_cols = []
    for c in use_cols:
        dt = str(X[c].dtype)
        if c in BASE_CATS or c.endswith('_bin') or c.startswith('NGRAM_') or '__' in c or dt in ('object', 'string',
                                                                                                 'category'):
            cat_cols.append(c)
    cat_cols = list(dict.fromkeys(cat_cols))[:64]
    num_cols = [c for c in use_cols if c not in cat_cols]
    print(f'{len(use_cols)} features: {len(cat_cols)} categorical, {len(num_cols)} numeric', flush=True)
    return X, Xtest, y, cat_cols, num_cols


def main():
    seed_everything(SEED)
    X, Xtest, y, cat_cols, num_cols = build_inputs()

    oof = np.zeros(len(X), dtype=np.float32)
    pred = np.zeros(len(Xtest), dtype=np.float32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (itr, iva) in enumerate(skf.split(X, y), 1):
        Xtr_num = X.iloc[itr][num_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
        Xva_num = X.iloc[iva][num_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
        Xte_num = Xtest[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
        scaler = StandardScaler()
        Xtr_num = scaler.fit_transform(Xtr_num)
        Xva_num = scaler.transform(Xva_num)
        Xte_num = scaler.transform(Xte_num)

        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        Xtr_cat = (enc.fit_transform(X.iloc[itr][cat_cols].astype('string').fillna('NA')) + 1).astype(np.int64)
        Xva_cat = (enc.transform(X.iloc[iva][cat_cols].astype('string').fillna('NA')) + 1).astype(np.int64)
        Xte_cat = (enc.transform(Xtest[cat_cols].astype('string').fillna('NA')) + 1).astype(np.int64)
        card = np.maximum(Xtr_cat.max(axis=0), np.maximum(Xva_cat.max(axis=0), Xte_cat.max(axis=0))).astype(int)

        train_loader = DataLoader(TabDataset(Xtr_num, Xtr_cat, y[itr]), batch_size=1024, shuffle=True, num_workers=0,
                                  pin_memory=True)
        valid_loader = DataLoader(TabDataset(Xva_num, Xva_cat, y[iva]), batch_size=4096, shuffle=False, num_workers=0,
                                  pin_memory=True)
        test_loader = DataLoader(TabDataset(Xte_num, Xte_cat, None), batch_size=4096, shuffle=False, num_workers=0,
                                 pin_memory=True)

        model = TabularNet(card, len(num_cols)).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
        loss_fn = nn.BCEWithLogitsLoss()

        best_auc, best_state, bad = -1.0, None, 0
        for epoch in range(40):
            model.train()
            for x_num, x_cat, yy in train_loader:
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(x_num.to(DEVICE), x_cat.to(DEVICE)), yy.to(DEVICE))
                loss.backward()
                opt.step()
            va_auc, _ = evaluate(model, valid_loader)
            sched.step(va_auc)
            print(f'  epoch {epoch + 1:02d} | val_auc {va_auc:.6f}', flush=True)
            if va_auc > best_auc + 1e-5:
                best_auc, best_state, bad = va_auc, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= 8:
                    break

        model.load_state_dict(best_state)
        va_auc, va_pred = evaluate(model, valid_loader)
        oof[iva] = va_pred.astype('float32')
        pred += predict_loader(model, test_loader).astype('float32') / N_FOLDS
        print(f'Fold {fold} AUC: {va_auc:.6f}', flush=True)

        del train_loader, valid_loader, test_loader, model, opt, sched
        gc.collect()
        torch.cuda.empty_cache()
    print(f'OOF AUC: {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(MODEL_DIR, PAIR, oof, pred)


if __name__ == '__main__':
    main()
