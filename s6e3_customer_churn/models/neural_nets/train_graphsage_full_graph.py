#!/usr/bin/env python
"""Full-batch 3-layer GraphSAGE on a cuML KNN graph over all train + test rows (transductive).

Two presets; the original rows are not used.

engineered (24 categoricals, 28 numerics; graph on the numerics):
- the 16 categoricals (stripped strings); numerics with missing -> train median;
- log1p, sqrt and 1 / (1 + x) of max(x, 0) and a missing flag for tenure, MonthlyCharges, TotalCharges;
- 7 charge relations: TotalCharges - tenure x MonthlyCharges and its absolute value, MonthlyCharges / (TotalCharges +
  1), TotalCharges / (MonthlyCharges + 1), TotalCharges / (tenure + 1), tenure x MonthlyCharges, tenure x TotalCharges;
- service counts over the 8 service columns (Yes, No, 'No phone / internet service', Yes again as service_count),
  has_internet, has_phone;
- categorical bins: tenure at fixed edges (0, 1, 3, 6, 12, 24, 36, 48, 60, 72, 10000), 40-quantile bins of
  MonthlyCharges and 60-quantile bins of TotalCharges (edges from train + test);
- 4 categorical pairs (Contract x InternetService, PaymentMethod x Contract, InternetService x TechSupport,
  PaymentMethod x PaperlessBilling) and the triple Contract x InternetService x PaymentMethod.
  The 28 numerics are standardized with train statistics; the graph links every row to its 20 nearest rows in that
  standardized numeric space. Hidden 256, lr 1.5e-3, up to 80 epochs, patience 10; no seed is set.
simple (22 categoricals, 3 numerics; graph on one-hot categoricals + numerics):
- the 16 categoricals; numerics with missing -> train median;
- per numeric: the value snapped to the nearest value that occurs at least 25 times in train (frequent values kept;
  as a category) and a rare flag (fewer than 25 train occurrences);
- the 3 numerics, standardized with train statistics.
  The graph links every row to its 20 nearest rows in [one-hot of the 16 categoricals (fitted on train + test),
  3 x the standardized numerics]. Hidden 192, lr 1e-3, up to 200 epochs, patience 20; numpy / torch seed 42 once at
  the start.

Model (both): categorical embeddings (codes from train + test; dim round(1.8 card^0.25) clipped to 4-32, N(0, 0.02)
init) concatenated with the numerics -> Linear(hidden) + ReLU + dropout 0.25 -> 3 SAGEConv(hidden) layers, each
LayerNorm -> ReLU -> dropout 0.25 and a residual x + 0.5 h -> head Linear 128, ReLU, dropout 0.25 -> 1 logit.
Training: every fold trains on the whole graph with the loss on the fold's training rows only: BCE with labels
smoothed by 0.01, AdamW (weight decay 3e-4), gradient-norm clip 1, one full-graph step per epoch; the held-out AUC is
checked after every epoch and the predictions (sigmoid) of the best epoch for the held-out and test nodes are kept
(early stopping after `patience` epochs without a gain). Test = mean of the 5 fold models. Needs RAPIDS cuML and
torch_geometric; the whole graph (849k nodes, 17M edges) stays on the GPU.
Usage: python train_graphsage_full_graph.py {engineered,simple} [--folds 0,1,2,3,4]"""
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, ID, NUMERICS, TARGET, folds, model_dir, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PRESETS = {
    'engineered': dict(pair='graphsage_engineered_numeric_knn_graph', hidden=256, lr=1.5e-3, epochs=80, patience=10,
                       seed=None),
    'simple': dict(pair='graphsage_snapped_rare_flags_ohe_knn_graph', hidden=192, lr=1e-3, epochs=200, patience=20,
                   seed=42),
}
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
K = 20
DROPOUT = 0.25
WEIGHT_DECAY = 3e-4
LABEL_SMOOTHING = 0.01

BASE_NUMS = list(NUMERICS)
BASE_CATS = list(CATEGORICALS)

# simple preset
RARE_MIN_COUNT = 25
GRAPH_NUMERIC_MULTIPLIER = 3.0
SIMPLE_CAT_COLS = BASE_CATS + [f'{c}__cat' for c in BASE_NUMS] + [f'{c}__is_rare' for c in BASE_NUMS]  # 22

# engineered preset
ENGINEERED_CAT_COLS = BASE_CATS + [
    'tenure_bin', 'MonthlyCharges_bin', 'TotalCharges_bin',
    'Contract__InternetService', 'PaymentMethod__Contract',
    'InternetService__TechSupport', 'PaymentMethod__PaperlessBilling',
    'Contract__InternetService__PaymentMethod',
]  # 24
SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']


def load_train_test():
    train = read_train().drop(columns=[ID])
    test = read_test().drop(columns=[ID])
    print('train:', train.shape, 'test:', test.shape, flush=True)
    y = train[TARGET].astype(str).str.strip().map({'Yes': 1, 'No': 0}).values.astype(np.float32)
    return train, test, y


# ---------------------------------------------------------------- engineered preset
def fe_engineered(train_df, test_df):
    tr = train_df.copy()
    te = test_df.copy()
    for df in (tr, te):
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
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

    def add_num_transforms(df):
        new_cols = {}
        for c in BASE_NUMS:
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
        ten = df['tenure'].to_numpy(np.float32)
        mc = df['MonthlyCharges'].to_numpy(np.float32)
        tc = df['TotalCharges'].to_numpy(np.float32)
        new_cols = {}
        new_cols['charges_deviation'] = (tc - ten * mc).astype(np.float32)
        new_cols['abs_charges_dev'] = np.abs(new_cols['charges_deviation']).astype(np.float32)
        new_cols['monthly_to_total_ratio'] = (mc / (tc + 1.0)).astype(np.float32)
        new_cols['total_to_monthly_ratio'] = (tc / (mc + 1.0)).astype(np.float32)
        new_cols['avg_monthly_charges'] = (tc / (ten + 1.0)).astype(np.float32)
        new_cols['tenure_x_monthly'] = (ten * mc).astype(np.float32)
        new_cols['tenure_x_total'] = (ten * tc).astype(np.float32)
        return new_cols

    tr = pd.concat([tr, pd.DataFrame(add_interactions(tr))], axis=1)
    te = pd.concat([te, pd.DataFrame(add_interactions(te))], axis=1)

    def add_service_counts(df):
        yes = np.zeros(len(df), dtype=np.float32)
        no = np.zeros(len(df), dtype=np.float32)
        oth = np.zeros(len(df), dtype=np.float32)
        for c in SERVICE_COLS:
            s = df[c].astype(str)
            yes += (s == 'Yes').to_numpy(np.float32)
            no += (s == 'No').to_numpy(np.float32)
            oth += s.isin(['No phone service', 'No internet service']).to_numpy(np.float32)
        return {
            'service_yes_count': yes.astype(np.float32),
            'service_no_count': no.astype(np.float32),
            'service_other_count': oth.astype(np.float32),
            'service_count': yes.astype(np.float32),
            'has_internet': (df['InternetService'].astype(str) != 'No').to_numpy(np.float32),
            'has_phone': (df['PhoneService'].astype(str) == 'Yes').to_numpy(np.float32),
        }

    tr = pd.concat([tr, pd.DataFrame(add_service_counts(tr))], axis=1)
    te = pd.concat([te, pd.DataFrame(add_service_counts(te))], axis=1)

    tenure_bins = [0, 1, 3, 6, 12, 24, 36, 48, 60, 72, 10_000]
    tr['tenure_bin'] = pd.cut(tr['tenure'], bins=tenure_bins, include_lowest=True).astype(str)
    te['tenure_bin'] = pd.cut(te['tenure'], bins=tenure_bins, include_lowest=True).astype(str)

    def qcut_edges_safe(x, q, fallback_q):
        x = pd.to_numeric(x, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < 100:
            return np.unique(np.quantile(x.to_numpy(np.float32), np.linspace(0, 1, min(fallback_q, 10))))
        try:
            return pd.qcut(x, q=q, retbins=True, duplicates='drop')[1]
        except Exception:
            return np.unique(np.quantile(x.to_numpy(np.float32), np.linspace(0, 1, min(fallback_q, 20))))

    all_mc = pd.concat([tr['MonthlyCharges'], te['MonthlyCharges']], axis=0, ignore_index=True)
    all_tc = pd.concat([tr['TotalCharges'], te['TotalCharges']], axis=0, ignore_index=True)
    mc_edges = qcut_edges_safe(all_mc, q=40, fallback_q=20)
    tc_edges = qcut_edges_safe(all_tc, q=60, fallback_q=30)
    for df in (tr, te):
        df['MonthlyCharges_bin'] = pd.cut(df['MonthlyCharges'], bins=mc_edges, include_lowest=True).astype(str)
        df['TotalCharges_bin'] = pd.cut(df['TotalCharges'], bins=tc_edges, include_lowest=True).astype(str)

    for df in (tr, te):
        df['Contract__InternetService'] = (df['Contract'].astype(str) + '|' + df['InternetService'].astype(str))
        df['PaymentMethod__Contract'] = (df['PaymentMethod'].astype(str) + '|' + df['Contract'].astype(str))
        df['InternetService__TechSupport'] = (df['InternetService'].astype(str) + '|' + df['TechSupport'].astype(str))
        df['PaymentMethod__PaperlessBilling'] = (df['PaymentMethod'].astype(str) + '|'
                                                 + df['PaperlessBilling'].astype(str))
        df['Contract__InternetService__PaymentMethod'] = (df['Contract'].astype(str) + '|'
                                                          + df['InternetService'].astype(str) + '|'
                                                          + df['PaymentMethod'].astype(str))

    for df in (tr, te):
        for c in ENGINEERED_CAT_COLS:
            df[c] = df[c].astype(str).fillna('missing').astype('category')
    return tr.copy(), te.copy()


# ---------------------------------------------------------------- simple preset
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


def fe_simple(train_df, test_df):
    tr, te = preprocess_base(train_df, test_df)
    for col in BASE_NUMS:
        snapper = build_numeric_snapper(tr[col], rare_min_count=RARE_MIN_COUNT)
        tr_snap, tr_israre = snapper(tr[col])
        te_snap, te_israre = snapper(te[col])
        tr[f'{col}__cat'] = pd.Series(tr_snap).astype(str).values
        te[f'{col}__cat'] = pd.Series(te_snap).astype(str).values
        tr[f'{col}__is_rare'] = pd.Series(tr_israre).astype(str).values
        te[f'{col}__is_rare'] = pd.Series(te_israre).astype(str).values
    for df in (tr, te):
        for c in SIMPLE_CAT_COLS:
            df[c] = df[c].astype(str).fillna('missing')
    return tr, te


def ohe_numeric_graph_matrix(train_fe, test_fe):
    """[one-hot of the 16 categoricals, GRAPH_NUMERIC_MULTIPLIER x standardized numerics] of train + test."""
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
    return np.concatenate([X_graph_cat, X_graph_num], axis=1).astype(np.float32)


# ---------------------------------------------------------------- shared
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


def scaled_numerics(train_fe, test_fe, num_cols):
    for c in num_cols:
        train_fe[c] = pd.to_numeric(train_fe[c], errors='coerce').fillna(0).astype(np.float32)
        test_fe[c] = pd.to_numeric(test_fe[c], errors='coerce').fillna(0).astype(np.float32)
    scaler = StandardScaler()
    Xn_train = scaler.fit_transform(train_fe[num_cols].values.astype(np.float32)).astype(np.float32)
    Xn_test = scaler.transform(test_fe[num_cols].values.astype(np.float32)).astype(np.float32)
    return np.vstack([Xn_train, Xn_test])


def build_graph_data(preset):
    """Node features of all train + test rows and the KNN edge list."""
    train, test, y = load_train_test()
    if preset == 'engineered':
        train_fe, test_fe = fe_engineered(train, test)
        cat_cols = [c for c in ENGINEERED_CAT_COLS if c in train_fe.columns and c in test_fe.columns]
        num_cols = [c for c in train_fe.columns if c != TARGET and c not in cat_cols
                    and (pd.api.types.is_numeric_dtype(train_fe[c].dtype)
                         or pd.api.types.is_bool_dtype(train_fe[c].dtype))]
        num_cols = [c for c in num_cols if c in test_fe.columns]
    else:
        train_fe, test_fe = fe_simple(train, test)
        cat_cols = SIMPLE_CAT_COLS
        num_cols = list(BASE_NUMS)
    print(f'{len(num_cols)} numeric, {len(cat_cols)} categorical node features', flush=True)

    Xc_train, Xc_test, cat_cardinals = encode_cats(train_fe, test_fe, cat_cols)
    Xn_all = scaled_numerics(train_fe, test_fe, num_cols)
    Xc_all = np.vstack([Xc_train, Xc_test])
    n_test = len(test_fe)

    X_graph = Xn_all if preset == 'engineered' else ohe_numeric_graph_matrix(train_fe, test_fe)
    print(f'cuML KNN graph (k = {K}) on a {X_graph.shape} matrix ...', flush=True)
    knn = NearestNeighbors(n_neighbors=K)
    knn.fit(X_graph)
    _, idx = knn.kneighbors(X_graph)
    src = np.repeat(np.arange(len(X_graph)), K)
    dst = idx.reshape(-1)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    data = Data(
        x_num=torch.tensor(Xn_all, dtype=torch.float32),
        x_cat=torch.tensor(Xc_all, dtype=torch.long),
        edge_index=edge_index,
        y=torch.tensor(np.concatenate([y, np.full(n_test, -1, np.float32)]), dtype=torch.float32),
    ).to(DEVICE)
    return data, y, cat_cardinals, Xn_all.shape[1], n_test


def emb_dim_from_card(card):
    return int(np.clip(int(round(1.8 * (card ** 0.25))), 4, 32))


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
        if x_cat.size(1) == 0:
            return None
        return torch.cat([emb(x_cat[:, j]) for j, emb in enumerate(self.embs)], dim=1)


class SAGEWithCats(nn.Module):
    def __init__(self, num_in, cat_cardinals, hidden=256, dropout=0.25):
        super().__init__()
        self.cat = CatEmbed(cat_cardinals)
        in_dim = num_in + self.cat.out_dim
        self.lin_in = nn.Linear(in_dim, hidden)
        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, data):
        zc = self.cat(data.x_cat)
        x = data.x_num if zc is None else torch.cat([data.x_num, zc], dim=1)
        x = F.relu(self.lin_in(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv, norm in ((self.conv1, self.norm1), (self.conv2, self.norm2), (self.conv3, self.norm3)):
            h = F.relu(norm(conv(x, data.edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + 0.5 * h
        return self.head(x).squeeze(-1)


class SmoothBCE(nn.Module):
    def __init__(self, eps=0.01):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets * (1 - self.eps) + 0.5 * self.eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preset', choices=sorted(PRESETS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    cfg = PRESETS[args.preset]
    pair = cfg['pair']

    if cfg['seed'] is not None:
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])

    data, y, cat_cardinals, n_num, n_test = build_graph_data(args.preset)
    n_train = len(y)
    n_all = n_train + n_test
    loss_fn = SmoothBCE(eps=LABEL_SMOOTHING)

    fold_ids = folds(y)
    for k in fold_list(args.folds):
        print(f'\n================ Fold {k + 1}/5 ================', flush=True)
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        train_mask = torch.zeros(n_all, dtype=torch.bool)
        train_mask[tr_idx] = True
        train_mask = train_mask.to(DEVICE)

        model = SAGEWithCats(num_in=n_num, cat_cardinals=cat_cardinals, hidden=cfg['hidden'],
                             dropout=DROPOUT).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=WEIGHT_DECAY)

        best_auc = -1.0
        best_probs = None
        bad = 0
        t_fold = time.time()
        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            logits = model(data)
            loss = loss_fn(logits[train_mask], data.y[train_mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(data)).detach().cpu().numpy()
            auc = roc_auc_score(y[va_idx], probs[va_idx])
            if epoch == 1 or epoch % 5 == 0:
                print(f'epoch {epoch:03d} | loss {loss.item():.5f} | val_auc {auc:.6f}', flush=True)
            if auc > best_auc + 1e-6:
                best_auc = auc
                best_probs = probs.copy()
                bad = 0
            else:
                bad += 1
                if bad >= cfg['patience']:
                    break

        print(f'[Fold {k + 1}] best AUC: {best_auc:.6f} | fold time {time.time() - t_fold:.1f}s', flush=True)
        save_part(MODEL_DIR, pair, k, best_probs[va_idx], best_probs[n_train:])
        del model, opt, best_probs
        torch.cuda.empty_cache()
        gc.collect()
    assemble(MODEL_DIR, pair, fold_ids, y)


if __name__ == '__main__':
    main()
