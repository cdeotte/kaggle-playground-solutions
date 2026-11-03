#!/usr/bin/env python
"""Mini-batch 2-layer GraphSAGE on sampled subgraphs of a cuML KNN graph over all train + test rows (transductive).

Node features (22 categoricals, 3 numerics; the original rows are not used):
- the 16 categoricals (stripped strings); numerics with missing -> train median;
- per numeric: the value snapped to the nearest value that occurs at least 25 times in train (frequent values kept;
  as a category) and a rare flag (fewer than 25 train occurrences);
- tenure, MonthlyCharges, TotalCharges, standardized with train statistics.
Graph: every row is linked to its 8 nearest rows (cuML NearestNeighbors) in [one-hot of the 16 categoricals (fitted on
train + test), 3 x the standardized numerics].
Batches: 8192 seed rows; their 2-hop neighbourhood takes 6 then 4 neighbours per node from a window of the 8 that
shifts with the epoch (offset epoch mod 8, +1 for the second hop); the subgraph keeps every KNN edge between its nodes.
Model: categorical embeddings (codes from train + test; dim round(1.8 card^0.25) clipped to 4-24, N(0, 0.02) init)
concatenated with the numerics -> Linear(128) + ReLU + dropout 0.2 -> 2 SAGEConv(128) layers, each LayerNorm -> ReLU ->
dropout 0.2 and a residual x + 0.5 h -> head Linear 64, ReLU, dropout 0.2 -> 1 logit.
Training: BCE with labels smoothed by 0.01 on the seed rows, AdamW lr 1e-3, weight decay 3e-4, fp16 autocast with
gradient scaling, gradient-norm clip 1, up to 80 epochs; the held-out AUC is checked at epoch 1 and every 2 epochs
(early stopping after 8 checks without a gain, not before epoch 6); the best weights predict the held-out and test
rows (subgraphs with offset 0, batches of 16384). Seed 42 (python / numpy / torch) once at the start, so the pair is
reproduced only when the 5 folds run in order in one process. Test = mean of the 5 fold models.
Needs RAPIDS cuML and torch_geometric.
Usage: python train_graphsage_sampled.py [--folds 0,1,2,3,4]"""
import argparse
import gc
import random
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, ID, NUMERICS, TARGET, folds, model_dir, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'graphsage_sampled_snapped_rare_flags_ohe_knn_graph'
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
RARE_MIN_COUNT = 25

K = 8
EPOCHS = 80
PATIENCE = 8  # validation checks
VAL_EVERY = 2
MIN_EPOCHS = 6
GRAPH_NUMERIC_MULTIPLIER = 3.0

BATCH_SIZE = 8192
INFER_BATCH_SIZE = 16384
FANOUTS = [6, 4]  # 2 hops for 2 GraphSAGE layers
USE_AMP = True

BASE_NUMS = list(NUMERICS)
BASE_CATS = list(CATEGORICALS)
CAT_COLS = BASE_CATS + [f'{c}__cat' for c in BASE_NUMS] + [f'{c}__is_rare' for c in BASE_NUMS]  # 22
NUM_COLS = BASE_NUMS[:]  # 3


def preprocess_base(train_df, test_df):
    tr = train_df.copy()
    te = test_df.copy()
    for df in (tr, te):
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip().replace('', np.nan),
                                           errors='coerce')
        for c in BASE_NUMS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float32)
        for c in BASE_CATS:
            df[c] = df[c].astype(str).fillna('missing').str.strip()
    # numeric NaNs -> train medians
    for c in BASE_NUMS:
        med = float(np.nanmedian(tr[c].to_numpy(np.float32)))
        if not np.isfinite(med):
            med = 0.0
        tr[c] = tr[c].fillna(med).astype(np.float32)
        te[c] = te[c].fillna(med).astype(np.float32)
    return tr, te


def build_numeric_snapper(train_series, rare_min_count):
    """Values seen at least rare_min_count times in train are kept; rarer values are snapped to the nearest of them
    and flagged."""
    s = pd.to_numeric(train_series, errors='coerce').astype(np.float32)
    vc = pd.Series(s).value_counts(dropna=False)
    frequent_vals = vc[vc >= rare_min_count].index.values
    frequent_vals = np.array([v for v in frequent_vals if pd.notna(v)], dtype=np.float32)
    if frequent_vals.size == 0:
        frequent_vals = np.array(pd.Series(s.dropna()).unique(), dtype=np.float32)
    frequent_vals = np.sort(frequent_vals)
    frequent_set = set(frequent_vals.tolist())

    def transform(series_any):
        x = pd.to_numeric(series_any, errors='coerce').astype(np.float32).values
        is_nan = np.isnan(x)
        is_rare = np.ones_like(x, dtype=np.int32)
        for i, v in enumerate(x):
            is_rare[i] = 1 if np.isnan(v) else (0 if float(v) in frequent_set else 1)
        x_snapped = x.copy()
        if frequent_vals.size > 0:
            idx_snap = np.where((~is_nan) & (is_rare == 1))[0]
            if idx_snap.size > 0:
                v = x[idx_snap]
                pos = np.clip(np.searchsorted(frequent_vals, v), 0, len(frequent_vals) - 1)
                left = np.clip(pos - 1, 0, len(frequent_vals) - 1)
                right = pos
                left_vals = frequent_vals[left]
                right_vals = frequent_vals[right]
                choose_right = np.abs(v - right_vals) <= np.abs(v - left_vals)
                x_snapped[idx_snap] = np.where(choose_right, right_vals, left_vals).astype(np.float32)
        return x_snapped.astype(np.float32), is_rare.astype(np.int32)

    return transform


def make_simple_features(train_df, test_df):
    tr = train_df.copy()
    te = test_df.copy()
    for col in BASE_NUMS:
        snapper = build_numeric_snapper(tr[col], rare_min_count=RARE_MIN_COUNT)
        tr_snap, tr_israre = snapper(tr[col])
        te_snap, te_israre = snapper(te[col])
        tr[f'{col}__cat'] = pd.Series(tr_snap).astype(str).values
        te[f'{col}__cat'] = pd.Series(te_snap).astype(str).values
        tr[f'{col}__is_rare'] = pd.Series(tr_israre).astype(str).values
        te[f'{col}__is_rare'] = pd.Series(te_israre).astype(str).values
    for df in (tr, te):
        for c in CAT_COLS:
            df[c] = df[c].astype(str).fillna('missing')
    return tr, te


def encode_cats(train_df, test_df, cat_cols):
    """Codes in order of first appearance over train + test."""
    tr_codes, te_codes, cardinals = [], [], []
    for c in cat_cols:
        all_vals = pd.concat([train_df[c].astype(str), test_df[c].astype(str)], axis=0, ignore_index=True)
        mapping = {v: i for i, v in enumerate(all_vals.unique().tolist())}
        tr_codes.append(train_df[c].astype(str).map(mapping).fillna(0).astype(np.int64).values)
        te_codes.append(test_df[c].astype(str).map(mapping).fillna(0).astype(np.int64).values)
        cardinals.append(len(mapping))
    Xc_tr = np.stack(tr_codes, axis=1) if tr_codes else np.zeros((len(train_df), 0), dtype=np.int64)
    Xc_te = np.stack(te_codes, axis=1) if te_codes else np.zeros((len(test_df), 0), dtype=np.int64)
    return Xc_tr, Xc_te, cardinals


def build_graph():
    """Node features (pinned CPU tensors) and the KNN neighbour table of all train + test rows."""
    train = read_train().drop(columns=[ID])
    test = read_test().drop(columns=[ID])
    print('train:', train.shape, 'test:', test.shape, flush=True)
    y = train[TARGET].astype(str).str.strip().map({'Yes': 1, 'No': 0}).values.astype(np.float32)

    train_base, test_base = preprocess_base(train, test)
    train_fe, test_fe = make_simple_features(train_base, test_base)
    Xc_train, Xc_test, cat_cardinals = encode_cats(train_fe, test_fe, CAT_COLS)

    for c in NUM_COLS:
        train_fe[c] = pd.to_numeric(train_fe[c], errors='coerce').fillna(0).astype(np.float32)
        test_fe[c] = pd.to_numeric(test_fe[c], errors='coerce').fillna(0).astype(np.float32)
    node_scaler = StandardScaler()
    Xn_train = node_scaler.fit_transform(train_fe[NUM_COLS].values.astype(np.float32)).astype(np.float32)
    Xn_test = node_scaler.transform(test_fe[NUM_COLS].values.astype(np.float32)).astype(np.float32)
    Xn_all = np.vstack([Xn_train, Xn_test])
    Xc_all = np.vstack([Xc_train, Xc_test])
    n_test = len(test_fe)
    print('x_num:', Xn_all.shape, 'x_cat:', Xc_all.shape, flush=True)

    # graph space: one-hot categoricals + GRAPH_NUMERIC_MULTIPLIER x standardized numerics
    graph_all_cat = pd.concat([train_fe[BASE_CATS].astype(str), test_fe[BASE_CATS].astype(str)], axis=0,
                              ignore_index=True)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
    X_graph_cat = ohe.fit_transform(graph_all_cat).astype(np.float32)
    graph_train_num = train_fe[BASE_NUMS].copy()
    graph_test_num = test_fe[BASE_NUMS].copy()
    for c in BASE_NUMS:
        graph_train_num[c] = pd.to_numeric(graph_train_num[c], errors='coerce').fillna(0).astype(np.float32)
        graph_test_num[c] = pd.to_numeric(graph_test_num[c], errors='coerce').fillna(0).astype(np.float32)
    graph_scaler = StandardScaler()
    X_graph_num_train = graph_scaler.fit_transform(graph_train_num.values.astype(np.float32)).astype(np.float32)
    X_graph_num_test = graph_scaler.transform(graph_test_num.values.astype(np.float32)).astype(np.float32)
    X_graph_num = np.vstack([X_graph_num_train, X_graph_num_test]).astype(np.float32)
    X_graph_num *= GRAPH_NUMERIC_MULTIPLIER
    X_graph = np.concatenate([X_graph_cat, X_graph_num], axis=1).astype(np.float32)
    print(f'cuML KNN graph (k = {K}) on a {X_graph.shape} matrix ...', flush=True)

    knn = NearestNeighbors(n_neighbors=K)
    knn.fit(X_graph)
    _, idx = knn.kneighbors(X_graph)
    neighbors = idx.astype(np.int32)
    del X_graph, X_graph_cat, X_graph_num, idx, knn
    gc.collect()

    x_num_cpu = torch.tensor(Xn_all, dtype=torch.float32).pin_memory()
    x_cat_cpu = torch.tensor(Xc_all, dtype=torch.long).pin_memory()
    y_all = np.concatenate([y, np.full(n_test, -1, np.float32)]).astype(np.float32)
    y_cpu = torch.tensor(y_all, dtype=torch.float32).pin_memory()
    return x_num_cpu, x_cat_cpu, y_cpu, y, neighbors, cat_cardinals, n_test


def emb_dim_from_card(card):
    return int(np.clip(int(round(1.8 * (card ** 0.25))), 4, 24))


class CatEmbed(nn.Module):
    def __init__(self, cardinals):
        super().__init__()
        self.embs = nn.ModuleList()
        out_dim = 0
        for card in cardinals:
            card = max(2, int(card))
            d = emb_dim_from_card(card)
            self.embs.append(nn.Embedding(card, d))
            out_dim += d
        self.out_dim = out_dim
        for e in self.embs:
            nn.init.normal_(e.weight, mean=0.0, std=0.02)

    def forward(self, x_cat):
        return torch.cat([emb(x_cat[:, j]) for j, emb in enumerate(self.embs)], dim=1)


class SAGEWithCats(nn.Module):
    def __init__(self, num_in, cat_cardinals, hidden=128, dropout=0.20):
        super().__init__()
        self.cat = CatEmbed(cat_cardinals)
        self.lin_in = nn.Linear(num_in + self.cat.out_dim, hidden)
        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x = torch.cat([data.x_num, self.cat(data.x_cat)], dim=1)
        x = F.relu(self.lin_in(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x1 = F.relu(self.norm1(self.conv1(x, data.edge_index)))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x = x + 0.5 * x1
        x2 = F.relu(self.norm2(self.conv2(x, data.edge_index)))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x = x + 0.5 * x2
        return self.head(x).squeeze(-1)


class SmoothBCE(nn.Module):
    def __init__(self, eps=0.01):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets * (1 - self.eps) + 0.5 * self.eps)


def build_subgraph_from_seeds(seed_nodes, neighbors, x_num_cpu, x_cat_cpu, y_cpu, fanouts, device, pos_buf, offset=0):
    """Seed rows + their sampled 2-hop neighbourhood, with every KNN edge between the collected nodes.
    pos_buf: int32 array of -1 (one entry per node), used as a global -> local index map and reset afterwards."""
    seed_nodes = np.asarray(seed_nodes, dtype=np.int32)
    frontier = seed_nodes
    collected = [seed_nodes]
    for hop, fanout in enumerate(fanouts):
        nbr = neighbors[frontier]  # [num_frontier, K]
        start = (offset + hop) % nbr.shape[1]
        cols = (np.arange(fanout) + start) % nbr.shape[1]
        frontier = np.unique(nbr[:, cols].reshape(-1))
        collected.append(frontier)

    nodes = np.unique(np.concatenate(collected))
    m = len(nodes)
    pos_buf[nodes] = np.arange(m, dtype=np.int32)
    seed_local = pos_buf[seed_nodes]

    sub_nbr = neighbors[nodes]
    dst_local_all = pos_buf[sub_nbr]
    mask = dst_local_all >= 0
    src_local = np.repeat(np.arange(m, dtype=np.int64), sub_nbr.shape[1])[mask.reshape(-1)]
    dst_local = dst_local_all[mask].astype(np.int64)
    edge_index = torch.tensor(np.vstack([src_local, dst_local]), dtype=torch.long, device=device)

    batch = Data(
        x_num=x_num_cpu[nodes].to(device, non_blocking=True),
        x_cat=x_cat_cpu[nodes].to(device, non_blocking=True),
        y=y_cpu[nodes].to(device, non_blocking=True),
        edge_index=edge_index,
    )
    batch.seed_local = torch.tensor(seed_local, dtype=torch.long, device=device)
    pos_buf[nodes] = -1
    return batch


def iterate_seed_batches(seed_nodes, batch_size, shuffle):
    arr = np.asarray(seed_nodes, dtype=np.int32).copy()
    if shuffle:
        np.random.shuffle(arr)
    for i in range(0, len(arr), batch_size):
        yield arr[i:i + batch_size]


@torch.no_grad()
def predict_seed_nodes(model, seed_nodes, graph, batch_size, offset=0):
    neighbors, x_num_cpu, x_cat_cpu, y_cpu, pos_buf = graph
    model.eval()
    out = np.zeros(len(seed_nodes), dtype=np.float32)
    pos = 0
    for batch_seed_nodes in iterate_seed_batches(seed_nodes, batch_size, shuffle=False):
        batch = build_subgraph_from_seeds(batch_seed_nodes, neighbors, x_num_cpu, x_cat_cpu, y_cpu, FANOUTS, DEVICE,
                                          pos_buf, offset=offset)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(USE_AMP and DEVICE == 'cuda')):
            logits = model(batch)
        probs = torch.sigmoid(logits[batch.seed_local]).float().cpu().numpy()
        out[pos:pos + len(batch_seed_nodes)] = probs.astype(np.float32)
        pos += len(batch_seed_nodes)
        del batch, logits, probs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    x_num_cpu, x_cat_cpu, y_cpu, y, neighbors, cat_cardinals, n_test = build_graph()
    n_train = len(y)
    n_all = n_train + n_test
    pos_buf = np.full(n_all, -1, dtype=np.int32)
    graph = (neighbors, x_num_cpu, x_cat_cpu, y_cpu, pos_buf)
    loss_fn = SmoothBCE(eps=0.01)
    test_nodes = np.arange(n_train, n_all, dtype=np.int32)

    fold_ids = folds(y)
    for k in fold_list(args.folds):
        print(f'\n================ Fold {k + 1}/5 ================', flush=True)
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        model = SAGEWithCats(num_in=x_num_cpu.shape[1], cat_cardinals=cat_cardinals, hidden=128,
                             dropout=0.20).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and DEVICE == 'cuda'))

        best_auc = -1.0
        best_state = None
        bad = 0
        t_fold = time.time()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            losses = []
            offset = epoch % K
            for batch_seed_nodes in iterate_seed_batches(tr_idx, BATCH_SIZE, shuffle=True):
                batch = build_subgraph_from_seeds(batch_seed_nodes, neighbors, x_num_cpu, x_cat_cpu, y_cpu, FANOUTS,
                                                  DEVICE, pos_buf, offset=offset)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(USE_AMP and DEVICE == 'cuda')):
                    logits = model(batch)
                    loss = loss_fn(logits[batch.seed_local], batch.y[batch.seed_local])
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
                del batch, logits, loss

            if (epoch % VAL_EVERY == 0) or (epoch == 1):
                val_pred = predict_seed_nodes(model, va_idx, graph, INFER_BATCH_SIZE, offset=offset)
                auc = roc_auc_score(y[va_idx], val_pred)
                print(f'epoch {epoch:03d} | loss {np.mean(losses):.5f} | val_auc {auc:.6f}', flush=True)
                if auc > best_auc + 1e-6:
                    best_auc = auc
                    best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if epoch >= MIN_EPOCHS and bad >= PATIENCE:
                        break
            elif epoch % 5 == 0:
                print(f'epoch {epoch:03d} | loss {np.mean(losses):.5f}', flush=True)

        model.load_state_dict(best_state)
        model.eval()
        val_pred = predict_seed_nodes(model, va_idx, graph, INFER_BATCH_SIZE, offset=0)
        test_pred = predict_seed_nodes(model, test_nodes, graph, INFER_BATCH_SIZE, offset=0)
        print(f'[Fold {k + 1}] best AUC: {best_auc:.6f} | AUC {roc_auc_score(y[va_idx], val_pred):.6f} | '
              f'fold time {time.time() - t_fold:.1f}s', flush=True)
        save_part(MODEL_DIR, PAIR, k, val_pred, test_pred)

        del model, opt, scaler, best_state, val_pred, test_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
