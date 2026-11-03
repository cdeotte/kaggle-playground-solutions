#!/usr/bin/env python
"""Mini-batch GraphSAGE on a KNN graph over train + test, with categorical embeddings and dense engineered features.

Features, built on train + test (no original rows):
- the 8 categoricals and the 11 numerics (missing -> train median);
- per numeric: log1p, sqrt and 1 / (1 + x) of max(x, 0), and a missing flag;
- water_total, |Soil_Moisture - Rainfall_mm - Previous_Irrigation_mm|, Rainfall / (Previous + 1), Previous / (Rainfall + 1),
  mean water input, water x area, water / (area + 1), Temperature_C - Humidity / 10;
- management flags: mulch yes / no, non-rain water source, drip, sprinkler, summer, early stage, and their count;
- train + test frequency of every categorical;
- fixed-edge bins of Rainfall, Previous_Irrigation, Soil_Moisture, Humidity, Temperature and Field_Area;
- 12 categorical pairs (6 of them with a bin) and 2 triples.
String columns (28) are factorized on train + test (code + 1); every other column is standardized with train-row
statistics.
Graph: cuML NearestNeighbors (k = 20) on the standardized dense features of all 900,000 train + test rows; node i sends
messages to its 20 neighbours.
Model: categorical embeddings (dim round(1.8 card^0.25) clipped to 4-24) + dense features -> Linear(256) + ReLU ->
2 SAGEConv(256) layers, each LayerNorm + ReLU + dropout and added with weight 0.5 -> head 64 -> 3 logits; dropout 0.25.
Training: seed nodes = the fold's training rows in shuffled batches of 8192; each batch is the 2-hop neighbourhood
sampled with fanouts 6 and 4 (a rotating window of the neighbour list, shifted every epoch) with all KNN edges among the
sampled nodes. Cross-entropy, AdamW lr 1e-3, weight decay 3e-4, fp16 autocast, gradient clip 1, up to 80 epochs; the
held-out balanced accuracy is checked at epoch 1 and every 2 epochs (early stopping after 8 checks without a gain, not
before epoch 6) and the best checkpoint predicts the held-out and test rows (neighbour window 0). Seed 42 (python /
numpy / torch) once before the folds. Test = mean of the 5 fold models.
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import (CATEGORICALS, CLASSES, ID, NUMERICS, TARGET, TARGET_MAP, folds, model_dir,  # noqa: E402
                         read_test, read_train)
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'graphsage_sampled_knn_graph_a'
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
GRAPH_KNN_K = 20
WINDOW_PERIOD = 8     # the neighbour-window shift of epoch e is e % WINDOW_PERIOD
EPOCHS = 80
PATIENCE = 8
VAL_EVERY = 2
MIN_EPOCHS = 6
BATCH_SIZE = 8192
INFER_BATCH_SIZE = 16384
FANOUTS = [6, 4]
USE_AMP = True

BASE_NUMS = list(NUMERICS)
BASE_CATS = list(CATEGORICALS)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fe_all(train_df, test_df):
    tr = train_df.copy()
    te = test_df.copy()
    for df in (tr, te):
        for c in BASE_NUMS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float32)
        for c in BASE_CATS:
            if c in df.columns:
                df[c] = df[c].astype(str).fillna('missing').str.strip()

    med_map = {}
    for c in BASE_NUMS:
        if c in tr.columns:
            med = float(np.nanmedian(tr[c].to_numpy(np.float32)))
            if not np.isfinite(med):
                med = 0.0
            med_map[c] = med
            tr[c] = tr[c].fillna(med).astype(np.float32)
        if c in te.columns:
            te[c] = te[c].fillna(med_map.get(c, 0.0)).astype(np.float32)

    def add_num_transforms(df):
        new_cols = {}
        for c in BASE_NUMS:
            if c not in df.columns:
                continue
            v = pd.to_numeric(df[c], errors='coerce').astype(np.float32).to_numpy(np.float32)
            miss = (~np.isfinite(v)).astype(np.int8)
            v = np.where(np.isfinite(v), v, 0.0).astype(np.float32)
            v0 = np.clip(v, 0.0, None)
            new_cols[f'LOG1P_{c}'] = np.log1p(v0).astype(np.float32)
            new_cols[f'SQRT_{c}'] = np.sqrt(v0).astype(np.float32)
            new_cols[f'INV1P_{c}'] = (1.0 / (1.0 + v0)).astype(np.float32)
            new_cols[f'MISS_{c}'] = miss
        return new_cols

    tr = pd.concat([tr, pd.DataFrame(add_num_transforms(tr))], axis=1)
    te = pd.concat([te, pd.DataFrame(add_num_transforms(te))], axis=1)

    def add_interactions(df):
        rain = df['Rainfall_mm'].to_numpy(np.float32)
        prev = df['Previous_Irrigation_mm'].to_numpy(np.float32)
        moist = df['Soil_Moisture'].to_numpy(np.float32)
        temp = df['Temperature_C'].to_numpy(np.float32)
        hum = df['Humidity'].to_numpy(np.float32)
        area = df['Field_Area_hectare'].to_numpy(np.float32)
        return {
            'water_total': (rain + prev).astype(np.float32),
            'abs_water_dev': np.abs(moist - rain - prev).astype(np.float32),
            'rain_to_prev_ratio': (rain / (prev + 1.0)).astype(np.float32),
            'prev_to_rain_ratio': (prev / (rain + 1.0)).astype(np.float32),
            'avg_water_input': ((rain + prev) / 2.0).astype(np.float32),
            'water_x_area': ((rain + prev) * area).astype(np.float32),
            'water_per_area': ((rain + prev) / (area + 1.0)).astype(np.float32),
            'heat_humidity_gap': (temp - hum / 10.0).astype(np.float32),
        }

    tr = pd.concat([tr, pd.DataFrame(add_interactions(tr))], axis=1)
    te = pd.concat([te, pd.DataFrame(add_interactions(te))], axis=1)

    def add_management_flags(df):
        mulch = df['Mulching_Used'].astype(str)
        irr = df['Irrigation_Type'].astype(str)
        water = df['Water_Source'].astype(str)
        season = df['Season'].astype(str)
        stage = df['Crop_Growth_Stage'].astype(str)
        new_cols = {
            'mulch_yes': (mulch == 'Yes').to_numpy(np.float32),
            'mulch_no': (mulch == 'No').to_numpy(np.float32),
            'has_non_rain_source': (water != 'Rainwater').to_numpy(np.float32),
            'is_drip_irrigation': irr.isin(['Drip', 'drip']).to_numpy(np.float32),
            'is_sprinkler_irrigation': irr.isin(['Sprinkler', 'sprinkler']).to_numpy(np.float32),
            'is_summer': season.isin(['Summer', 'summer']).to_numpy(np.float32),
            'is_early_stage': stage.isin(['Seedling', 'Vegetative', 'Early']).to_numpy(np.float32),
        }
        new_cols['management_count'] = (new_cols['mulch_yes'] + new_cols['has_non_rain_source']
                                         + new_cols['is_drip_irrigation']
                                         + new_cols['is_sprinkler_irrigation']).astype(np.float32)
        return new_cols

    tr = pd.concat([tr, pd.DataFrame(add_management_flags(tr))], axis=1)
    te = pd.concat([te, pd.DataFrame(add_management_flags(te))], axis=1)

    # frequency of every categorical over train + test
    both = pd.concat([tr[BASE_CATS], te[BASE_CATS]], axis=0, ignore_index=True)
    freq_tr, freq_te = {}, {}
    for c in BASE_CATS:
        vc = both[c].astype(str).value_counts()
        freq_tr[f'FREQ_{c}'] = tr[c].astype(str).map(vc).fillna(0).astype(np.float32).to_numpy()
        freq_te[f'FREQ_{c}'] = te[c].astype(str).map(vc).fillna(0).astype(np.float32).to_numpy()
    tr = pd.concat([tr, pd.DataFrame(freq_tr)], axis=1)
    te = pd.concat([te, pd.DataFrame(freq_te)], axis=1)

    bin_specs = {
        'Rainfall_bin': ('Rainfall_mm', [-np.inf, 10, 20, 35, 50, 70, 95, 130, np.inf]),
        'PrevIrr_bin': ('Previous_Irrigation_mm', [-np.inf, 5, 10, 20, 30, 45, 65, 90, np.inf]),
        'SoilMoisture_bin': ('Soil_Moisture', [-np.inf, 15, 25, 35, 45, 55, 65, 75, np.inf]),
        'Humidity_bin': ('Humidity', [-np.inf, 25, 40, 55, 65, 75, 85, 95, np.inf]),
        'Temperature_bin': ('Temperature_C', [-np.inf, 10, 18, 24, 30, 36, 42, np.inf]),
        'FieldArea_bin': ('Field_Area_hectare', [-np.inf, 1, 2, 4, 6, 9, 13, 20, np.inf]),
    }
    pairs = [
        ('Crop_Type', 'Season'),
        ('Irrigation_Type', 'Water_Source'),
        ('Crop_Type', 'Irrigation_Type'),
        ('Season', 'Crop_Growth_Stage'),
        ('Soil_Type', 'Crop_Type'),
        ('Region', 'Crop_Type'),
        ('Crop_Type', 'Rainfall_bin'),
        ('Irrigation_Type', 'PrevIrr_bin'),
        ('Water_Source', 'SoilMoisture_bin'),
        ('Season', 'Humidity_bin'),
        ('Crop_Growth_Stage', 'Temperature_bin'),
        ('Region', 'FieldArea_bin'),
    ]
    out = []
    for df in (tr, te):
        df = df.copy()
        for out_col, (src_col, bins) in bin_specs.items():
            df[out_col] = pd.cut(df[src_col], bins=bins, include_lowest=True).astype(str)
        for a, b in pairs:
            df[f'{a}__{b}'] = df[a].astype(str) + '|' + df[b].astype(str)
        df['Crop_Type__Season__Irrigation_Type'] = (df['Crop_Type'].astype(str) + '|' + df['Season'].astype(str) + '|'
                                                    + df['Irrigation_Type'].astype(str))
        df['Soil_Type__Crop_Type__Region'] = (df['Soil_Type'].astype(str) + '|' + df['Crop_Type'].astype(str) + '|'
                                              + df['Region'].astype(str))
        for c in BASE_CATS + list(bin_specs) + [c for c in df.columns if '__' in c]:
            df[c] = df[c].astype(str).fillna('missing')
        out.append(df)
    return out[0], out[1]


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

    def forward(self, x_cat):
        return torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)


class SAGEWithCats(nn.Module):
    def __init__(self, num_in, cat_cardinals, hidden=128, dropout=0.20, n_classes=3):
        super().__init__()
        self.cat = CatEmbed(cat_cardinals)
        self.lin_in = nn.Linear(num_in + self.cat.out_dim, hidden)
        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(64, n_classes))

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
        return self.head(x)


def build_subgraph(seed_nodes, neighbors, pos_buf, x_num_cpu, x_cat_cpu, y_cpu, fanouts, device, offset=0):
    """Seed nodes + their sampled 2-hop neighbourhood, with every KNN edge among these nodes. pos_buf (all -1) maps a
    global node id to its local index while the batch is built and is reset afterwards."""
    seed_nodes = np.asarray(seed_nodes, dtype=np.int32)
    frontier = seed_nodes
    collected = [seed_nodes]
    for hop, fanout in enumerate(fanouts):
        nbr = neighbors[frontier]
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
    batch = Data(x_num=x_num_cpu[nodes].to(device, non_blocking=True),
                 x_cat=x_cat_cpu[nodes].to(device, non_blocking=True),
                 y=y_cpu[nodes].to(device, non_blocking=True),
                 edge_index=edge_index)
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
    model.eval()
    out = np.zeros((len(seed_nodes), len(CLASSES)), dtype=np.float32)
    pos = 0
    for batch_seed_nodes in iterate_seed_batches(seed_nodes, batch_size, shuffle=False):
        batch = build_subgraph(batch_seed_nodes, fanouts=FANOUTS, device=DEVICE, offset=offset, **graph)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(USE_AMP and DEVICE == 'cuda')):
            logits = model(batch)
        probs = torch.softmax(logits[batch.seed_local], dim=1).float().cpu().numpy()
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

    train = read_train().drop(columns=[ID])
    test = read_test().drop(columns=[ID])
    y = train[TARGET].map(TARGET_MAP).values.astype(np.int64)
    train_fe, test_fe = fe_all(train.drop(columns=[TARGET]).copy(), test.copy())

    cat_cols = [c for c in train_fe.columns if train_fe[c].dtype == object]
    num_cols = [c for c in train_fe.columns if c not in cat_cols]
    all_cat_cardinals = []
    for c in cat_cols:
        both = pd.concat([train_fe[c], test_fe[c]], axis=0, ignore_index=True).astype(str)
        codes, uniques = pd.factorize(both, sort=False)
        train_fe[c] = codes[:len(train_fe)].astype(np.int64)
        test_fe[c] = codes[len(train_fe):].astype(np.int64)
        all_cat_cardinals.append(len(uniques) + 1)
    for c in num_cols:
        train_fe[c] = pd.to_numeric(train_fe[c], errors='coerce').fillna(0).astype(np.float32)
        test_fe[c] = pd.to_numeric(test_fe[c], errors='coerce').fillna(0).astype(np.float32)
    log(f'{len(num_cols)} dense columns, {len(cat_cols)} categorical columns')

    scaler = StandardScaler()
    Xn_train = scaler.fit_transform(train_fe[num_cols].values.astype(np.float32)).astype(np.float32)
    Xn_test = scaler.transform(test_fe[num_cols].values.astype(np.float32)).astype(np.float32)
    Xn_all = np.vstack([Xn_train, Xn_test]).astype(np.float32)
    Xc_all = np.vstack([train_fe[cat_cols].values.astype(np.int64) + 1,
                        test_fe[cat_cols].values.astype(np.int64) + 1]).astype(np.int64)

    log(f'KNN graph (k = {GRAPH_KNN_K}) on {len(Xn_all)} rows ...')
    knn = NearestNeighbors(n_neighbors=GRAPH_KNN_K)
    knn.fit(Xn_all)
    neighbors = knn.kneighbors(Xn_all, return_distance=False).astype(np.int32)
    del knn
    gc.collect()

    n_train, n_test = len(train_fe), len(test_fe)
    n_all = n_train + n_test
    y_all = np.concatenate([y, np.full(n_test, -1, dtype=np.int64)]).astype(np.int64)
    graph = dict(neighbors=neighbors, pos_buf=np.full(n_all, -1, dtype=np.int32),
                 x_num_cpu=torch.tensor(Xn_all, dtype=torch.float32).pin_memory(),
                 x_cat_cpu=torch.tensor(Xc_all, dtype=torch.long).pin_memory(),
                 y_cpu=torch.tensor(y_all, dtype=torch.long).pin_memory())
    test_nodes = np.arange(n_train, n_all, dtype=np.int32)

    fold_ids = folds(y)
    for k in fold_list(args.folds):
        fold = k + 1
        log(f'================ Fold {fold}/5 ================')
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        model = SAGEWithCats(num_in=Xn_all.shape[1], cat_cardinals=all_cat_cardinals, hidden=256, dropout=0.25).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
        grad_scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and DEVICE == 'cuda'))

        best_bac = -1.0
        best_state = None
        bad = 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            losses = []
            offset = epoch % WINDOW_PERIOD
            for batch_seed_nodes in iterate_seed_batches(tr_idx, BATCH_SIZE, shuffle=True):
                batch = build_subgraph(batch_seed_nodes, fanouts=FANOUTS, device=DEVICE, offset=offset, **graph)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(USE_AMP and DEVICE == 'cuda')):
                    logits = model(batch)
                    loss = F.cross_entropy(logits[batch.seed_local], batch.y[batch.seed_local])
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(opt)
                grad_scaler.update()
                losses.append(loss.item())
                del batch, logits, loss

            if (epoch % VAL_EVERY == 0) or (epoch == 1):
                val_pred = predict_seed_nodes(model, va_idx, graph, INFER_BATCH_SIZE, offset=offset)
                bac = balanced_accuracy_score(y[va_idx], val_pred.argmax(axis=1))
                log(f'fold {fold} epoch {epoch:03d} | loss {np.mean(losses):.5f} | val_bac {bac:.6f}')
                if bac > best_bac + 1e-6:
                    best_bac = bac
                    best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                if epoch >= MIN_EPOCHS and bad >= PATIENCE:
                    log(f'fold {fold}: early stopping at epoch {epoch}')
                    break

        model.load_state_dict(best_state)
        p_va = predict_seed_nodes(model, va_idx, graph, INFER_BATCH_SIZE, offset=0)
        p_te = predict_seed_nodes(model, test_nodes, graph, INFER_BATCH_SIZE, offset=0)
        log(f'Fold {fold} BAC: {balanced_accuracy_score(y[va_idx], p_va.argmax(axis=1)):.6f}')
        save_part(MODEL_DIR, PAIR, k, p_va, p_te)
        del model, opt, grad_scaler, p_va, p_te, best_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
