#!/usr/bin/env python
"""Full-batch 3-layer GraphSAGE on a KNN graph over all train + test rows (transductive), with rule-score features and
original-data class priors.

Features, built on train + test (the original rows are a reference only, not training rows):
- the 8 categoricals (label-encoded on train + test) and the 11 numerics;
- rule score 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10] -
  2 [stage Harvest] - 2 [stage Sowing] - [mulch Yes], is_low (score <= 0) and is_high (score > 3);
- per class k: the original-data share of class k in every categorical level (24 columns; unseen -> original prior).
The 38 numeric columns are standardized with train-row statistics.
Graph: cuML NearestNeighbors (k = 20) on the standardized numeric columns of all 900,000 train + test rows; node i sends
messages to its 20 neighbours (18M edges). Every fold trains on the whole graph; only the loss is restricted to the
fold's training rows.
Model: categorical embeddings (dim round(1.8 card^0.25) clipped to 4-32, N(0, 0.02) init) + numerics ->
Linear(256) + ReLU -> 3 SAGEConv(256) layers, each LayerNorm -> ReLU -> dropout 0.2 plus a residual connection ->
head 128 (ReLU, dropout 0.2) -> 3 logits.
Training: class-weighted cross-entropy (weights n / (3 n_class) from all train rows) with label smoothing 0.01, AdamW
lr 1e-3, weight decay 1e-4, one full-graph step per epoch, up to 80 epochs; the held-out balanced accuracy is checked at
epoch 1 and every 5 epochs (early stopping after 10 checks without a gain) and the best checkpoint predicts the held-out
and test rows. Seed 42, and torch seed 42 + fold (folds numbered 1-5) before each fold. Test = mean of the 5 fold
models. Needs RAPIDS cuML and torch_geometric; the whole graph is kept on the GPU.
Usage: python train_graphsage_full.py [--folds 0,1,2,3,4]"""
import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from cuml.neighbors import NearestNeighbors
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, ID, TARGET, TARGET_MAP, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'graphsage_transductive_knn_graph_b'
MODEL_DIR = model_dir('neural_nets')
SEED = 42
K_NEIGHBORS = 20
HIDDEN = 256
EPOCHS = 80
PATIENCE = 10
LR = 1e-3
WD = 1e-4
DROPOUT = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CATS = list(CATEGORICALS)
LABEL = 'target'


def emb_dim_from_card(card):
    return int(np.clip(round(1.8 * card ** 0.25), 4, 32))


class CatEmbed(nn.Module):
    def __init__(self, cardinals):
        super().__init__()
        self.embs = nn.ModuleList()
        out_dim = 0
        for card in cardinals:
            d = emb_dim_from_card(max(2, card))
            self.embs.append(nn.Embedding(max(2, card), d))
            out_dim += d
        self.out_dim = out_dim
        for e in self.embs:
            nn.init.normal_(e.weight, 0, 0.02)

    def forward(self, x_cat):
        return torch.cat([emb(x_cat[:, j]) for j, emb in enumerate(self.embs)], dim=1)


class SAGEWithCats(nn.Module):
    def __init__(self, num_in, cat_cardinals, hidden=256, dropout=0.2, n_classes=3):
        super().__init__()
        self.cat = CatEmbed(cat_cardinals)
        self.lin_in = nn.Linear(num_in + self.cat.out_dim, hidden)
        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, data):
        x = torch.cat([data.x_num, self.cat(data.x_cat)], dim=1)
        x = F.relu(self.lin_in(x))
        h = self.norm1(self.conv1(x, data.edge_index))
        h = F.dropout(F.relu(h), self.dropout, self.training) + x
        h2 = self.norm2(self.conv2(h, data.edge_index))
        h2 = F.dropout(F.relu(h2), self.dropout, self.training) + h
        h3 = self.norm3(self.conv3(h2, data.edge_index))
        h3 = F.dropout(F.relu(h3), self.dropout, self.training) + h2
        return self.head(h3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train, test, orig = read_train(), read_test(), read_original()
    train[LABEL] = train[TARGET].map(TARGET_MAP).astype(np.int32)
    orig[LABEL] = orig[TARGET].map(TARGET_MAP).astype(np.int32)
    y = train[LABEL].values
    y_orig = orig[LABEL].values
    cls_counts = np.bincount(y)
    cls_weights_t = torch.tensor(cls_counts.sum() / (3 * cls_counts), dtype=torch.float32).to(DEVICE)
    for df in [train, test]:
        df.drop(columns=[c for c in [ID, TARGET] if c in df.columns], inplace=True)
    n_train, n_test = len(train), len(test)
    print(f'Train: {n_train}, Test: {n_test}, Device: {DEVICE}', flush=True)

    for df in [train, test]:
        SM, RM, TC = df['Soil_Moisture'], df['Rainfall_mm'], df['Temperature_C']
        WS, CGS, MU = df['Wind_Speed_kmh'], df['Crop_Growth_Stage'], df['Mulching_Used']
        score = ((SM < 25) * 2 + (RM < 300) * 2 + (TC > 30) + (WS > 10) - (CGS == 'Harvest') * 2 - (CGS == 'Sowing') * 2
                 - (MU == 'Yes'))
        df['formula_score'] = score.astype(np.float32)
        df['is_low'] = (score <= 0).astype(np.float32)
        df['is_high'] = (score > 3).astype(np.float32)
    for c in CATS:
        for k in range(3):
            m = orig.groupby(c).apply(lambda g, kk=k: (g[LABEL] == kk).mean())
            gm = float((y_orig == k).mean())
            for df in [train, test]:
                df[f'ORIG_k{k}_{c}'] = df[c].map(m).fillna(gm).astype(np.float32)

    num_cols = [c for c in train.columns if c != LABEL and train[c].dtype != 'object']
    cat_cols = [c for c in train.columns if c != LABEL and train[c].dtype == 'object']
    cat_cardinals = []
    combined_cat = pd.concat([train[cat_cols], test[cat_cols]], ignore_index=True)
    for c in cat_cols:
        le = LabelEncoder()
        le.fit(combined_cat[c].astype(str).fillna('missing'))
        train[c] = le.transform(train[c].astype(str).fillna('missing'))
        test[c] = le.transform(test[c].astype(str).fillna('missing'))
        cat_cardinals.append(int(train[c].max()) + 1)

    scaler = StandardScaler()
    Xn_tr = scaler.fit_transform(train[num_cols].fillna(0).values).astype(np.float32)
    Xn_te = scaler.transform(test[num_cols].fillna(0).values).astype(np.float32)
    Xn_all = np.vstack([Xn_tr, Xn_te])
    Xc_all = np.vstack([train[cat_cols].values.astype(np.int64), test[cat_cols].values.astype(np.int64)])
    print(f'Num features: {len(num_cols)}, Cat features: {len(cat_cols)}', flush=True)

    print(f'Building cuML KNN graph (K={K_NEIGHBORS}) ...', flush=True)
    t0 = time.time()
    knn = NearestNeighbors(n_neighbors=K_NEIGHBORS)
    knn.fit(Xn_all)
    _, idx = knn.kneighbors(Xn_all)
    src_nodes = np.repeat(np.arange(len(Xn_all)), K_NEIGHBORS)
    edge_index = torch.tensor(np.vstack([src_nodes, idx.reshape(-1)]), dtype=torch.long)
    print(f'Graph: {len(Xn_all)} nodes, {edge_index.shape[1]} edges ({time.time() - t0:.1f}s)', flush=True)

    data = Data(
        x_num=torch.tensor(Xn_all, dtype=torch.float32),
        x_cat=torch.tensor(Xc_all, dtype=torch.long),
        edge_index=edge_index,
        y=torch.tensor(np.concatenate([y, np.full(n_test, -1, np.int32)]), dtype=torch.long),
    ).to(DEVICE)

    fold_ids = folds(y)
    for k in fold_list(args.folds):
        fold = k + 1
        print(f'\n=== Fold {fold}/5 ===', flush=True)
        torch.manual_seed(SEED + fold)
        t0 = time.time()
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        train_mask = torch.zeros(len(Xn_all), dtype=torch.bool)
        val_mask = torch.zeros(len(Xn_all), dtype=torch.bool)
        test_mask = torch.zeros(len(Xn_all), dtype=torch.bool)
        train_mask[tr_idx] = True
        val_mask[va_idx] = True
        test_mask[n_train:] = True
        data.train_mask = train_mask.to(DEVICE)
        data.val_mask = val_mask.to(DEVICE)
        data.test_mask = test_mask.to(DEVICE)

        model = SAGEWithCats(num_in=len(num_cols), cat_cardinals=cat_cardinals, hidden=HIDDEN, dropout=DROPOUT,
                             n_classes=3).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        criterion = nn.CrossEntropyLoss(weight=cls_weights_t, label_smoothing=0.01)

        best_bac = 0.0
        best_state = None
        patience_left = PATIENCE
        for epoch in range(1, EPOCHS + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 or epoch == 1:
                model.eval()
                with torch.no_grad():
                    va_probs = torch.softmax(model(data)[data.val_mask], dim=1).cpu().numpy()
                bac = balanced_accuracy_score(y[va_idx], np.argmax(va_probs, axis=1))
                if bac > best_bac:
                    best_bac = bac
                    best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
                    patience_left = PATIENCE
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(f'  Early stop ep {epoch} (best={best_bac:.5f})', flush=True)
                        break
                if epoch % 10 == 0 or epoch == 1:
                    print(f'  Ep {epoch:3d} loss={loss.item():.4f} BAC={bac:.5f} best={best_bac:.5f}', flush=True)

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            all_logits = model(data)
            va_probs = torch.softmax(all_logits[data.val_mask], dim=1).cpu().numpy()
            te_probs = torch.softmax(all_logits[data.test_mask], dim=1).cpu().numpy()
        print(f'  Fold {fold} BAC={best_bac:.5f} ({time.time() - t0:.0f}s)', flush=True)
        save_part(MODEL_DIR, PAIR, k, va_probs, te_probs)
        del model, best_state
        gc.collect()
        torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
