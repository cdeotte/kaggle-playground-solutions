#!/usr/bin/env python
"""Residual embedding MLP on dense F1 features, original-data priors and fold-local inner target encodings.

Features (built once on train + test + original rows; the original rows are never trained on):
- the 14 columns: Driver, Compound, Race, Year, PitStop, LapNumber, Stint, TyreLife, Position, Position_Change and any
  other numeric with < 100 unique values in train + test are categoricals, the other numerics stay numeric;
- 18 F1 relationships (tyre life / lap, lap / stint, degradation per tyre lap and per lap, delta per lap time, lap time -
  delta, progress products, position pressure, a 3-flag pit-window proxy, ...);
- log1p / sqrt / 1 / (1 + |x|) / percentile rank / frequency of the numerics and relationships (up to 36); round(0, 1, 2),
  floor, fractional part and 5 decimal-digit categoricals of 8 artifact numerics; 12-quantile bins of the high-cardinality
  numerics and 6 more; two- and three-way categorical crosses;
- for the first 80 categoricals: log count over train + test + original and log(1 + train frequency / original
  frequency); for the first 50: original-data target mean (smoothed toward the original mean with weight 25) and log
  count;
- distances to the two original class centroids (10 numerics scaled by the original std), their gap and softmax.
Inside each outer fold the first 60 categoricals get target encodings (smoothing 20): 5-fold inner cross-fitting on the
outer-train rows (StratifiedKFold seed 142 + fold, folds numbered 1-5) and the map of all outer-train rows for the
held-out and test rows. The first 80 categorical-flagged columns are ordinal-encoded into embeddings, all other columns
are standardized on the outer-train rows.

Model: embeddings (dim min(64, max(8, round(2 sqrt(cardinality + 1))))) + batch-normed numerics -> Linear 384 -> 4
residual blocks (Linear 768, GELU, dropout 0.15, Linear 384, dropout, post-LayerNorm) -> Linear 192, GELU, dropout ->
1 logit. BCE with label smoothing 0.02, AdamW lr 1e-3, weight decay 1e-4, batch 2048, gradient clipping 1.0,
ReduceLROnPlateau on the validation AUC (factor 0.5, patience 3), up to 40 epochs with early stopping (patience 8) on the
held-out fold; the best epoch predicts the held-out and test rows. Seed 42 + fold (folds numbered 1-5).
Test = mean of the 5 fold models (probabilities).
Usage: python train_residual_prior_nn.py [--folds 0,1,2,3,4]"""
import argparse
import gc
import math
import random
import sys
import warnings
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
from common.data import ID, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

PAIR = 'residual_embedding_nn_original_priors_te'
MODEL_DIR = model_dir('neural_nets')
INNER_FOLDS = 5
RANDOM_SEED = 42
MAX_CAT_COLS = 80
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PARAMS = {
    'hidden': 384,
    'depth': 4,
    'dropout': 0.15,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'max_epochs': 40,
    'patience': 8,
    'batch_size': 2048,
    'eval_batch_size': 8192,
    'label_smoothing': 0.02,
}


def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def to_binary_target(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(np.int8)
    return s.astype(str).str.lower().map({'false': 0, 'true': 1, '0': 0, '1': 1}).astype(np.int8)


def safe_name(c):
    return ''.join(ch if ch.isalnum() else '_' for ch in str(c)).strip('_')


def dedup_keep_order(seq):
    out = []
    seen = set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def finite_float32(s):
    arr = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
    return arr


def infer_groups(train, test, feature_cols):
    """Object columns and numerics with < 100 unique values (train + test) are categorical; numeric_like = all numerics."""
    cats, nums, numeric_like = [], [], []
    both = pd.concat([train[feature_cols], test[feature_cols]], axis=0, ignore_index=True)
    for c in feature_cols:
        if pd.api.types.is_object_dtype(both[c]) or str(both[c].dtype).startswith('string') or str(both[c].dtype) == 'category':
            cats.append(c)
        else:
            numeric_like.append(c)
            nun = both[c].nunique(dropna=False)
            if nun < 100:
                cats.append(c)
            else:
                nums.append(c)
    return cats, nums, numeric_like


def qcut_labels(s, q=12):
    s = pd.to_numeric(s, errors='coerce').astype('float32')
    try:
        return pd.qcut(s.rank(method='first'), q=q, labels=False, duplicates='drop').astype('float32').fillna(-1).astype('int16').astype(str)
    except Exception:
        return pd.Series(['-1'] * len(s), index=s.index, dtype='string')


def add_round_digit_features(all_df, cols):
    out = pd.DataFrame(index=all_df.index)
    for c in cols:
        if c not in all_df.columns:
            continue
        sn = safe_name(c)
        s = finite_float32(all_df[c])
        out[f'ROUND0_{sn}'] = s.round(0).astype('float32')
        out[f'ROUND1_{sn}'] = s.round(1).astype('float32')
        out[f'ROUND2_{sn}'] = s.round(2).astype('float32')
        out[f'FLOOR_{sn}'] = np.floor(s).astype('float32')
        out[f'FRAC_{sn}'] = (s - np.floor(s)).astype('float32')
        sf = s.map(lambda x: f'{float(x):.3f}')
        parts = sf.str.split('.', n=1, expand=True)
        left = parts[0].str.replace('-', '', regex=False).str.zfill(5)
        right = parts[1].str.pad(3, side='right', fillchar='0')
        out[f'DIG_{sn}_i0'] = left.str[-1].astype('string')
        out[f'DIG_{sn}_i1'] = left.str[-2].astype('string')
        out[f'DIG_{sn}_f0'] = right.str[0].astype('string')
        out[f'DIG_{sn}_f1'] = right.str[1].astype('string')
        out[f'DIG_{sn}_bucket'] = (left.str[-2:] + '_' + right.str[:1]).astype('string')
    return out


def add_numeric_transforms(all_df, cols):
    out = pd.DataFrame(index=all_df.index)
    for c in cols:
        if c not in all_df.columns:
            continue
        sn = safe_name(c)
        v = finite_float32(all_df[c])
        out[f'LOG1P_{sn}'] = np.log1p(v.clip(lower=0)).astype('float32')
        out[f'SQRT_{sn}'] = np.sqrt(v.clip(lower=0)).astype('float32')
        out[f'INV1P_{sn}'] = (1.0 / (1.0 + np.abs(v))).astype('float32')
        out[f'RANK_{sn}'] = v.rank(method='average', pct=True).astype('float32')
        vc = v.value_counts(normalize=True, dropna=False)
        out[f'FREQ_{sn}'] = v.map(vc).fillna(0).astype('float32')
    return out


def add_f1_relationships(all_df):
    def num(c):
        return finite_float32(all_df[c]) if c in all_df.columns else pd.Series(0, index=all_df.index, dtype='float32')
    lap = num('LapNumber')
    tyre = num('TyreLife')
    stint = num('Stint')
    pit = num('PitStop')
    pos = num('Position')
    poschg = num('Position_Change')
    lt = num('LapTime (s)')
    delta = num('LapTime_Delta')
    degr = num('Cumulative_Degradation')
    prog = num('RaceProgress')
    year = num('Year')
    out = pd.DataFrame(index=all_df.index)
    out['tyre_per_lap'] = (tyre / (lap + 1)).astype('float32')
    out['lap_per_stint'] = (lap / (stint + 1)).astype('float32')
    out['tyre_per_stint'] = (tyre / (stint + 1)).astype('float32')
    out['pit_adjusted_tyre'] = (tyre / (pit + 1)).astype('float32')
    out['degr_per_tyre'] = (degr / (tyre + 1)).astype('float32')
    out['degr_per_lap'] = (degr / (lap + 1)).astype('float32')
    out['delta_per_laptime'] = (delta / (np.abs(lt) + 1)).astype('float32')
    out['delta_per_degr'] = (delta / (np.abs(degr) + 1)).astype('float32')
    out['laptime_per_tyre'] = (lt / (tyre + 1)).astype('float32')
    out['laptime_minus_delta'] = (lt - delta).astype('float32')
    out['degr_x_progress'] = (degr * prog).astype('float32')
    out['tyre_x_progress'] = (tyre * prog).astype('float32')
    out['laptime_x_progress'] = (lt * prog).astype('float32')
    out['position_pressure'] = (pos + poschg.abs()).astype('float32')
    out['position_gain_risk'] = (-poschg / (pos + 1)).astype('float32')
    out['late_old_tyre'] = (prog * tyre).astype('float32')
    out['pit_window_proxy'] = ((tyre > 12).astype('float32') + (prog > 0.45).astype('float32') + (degr > degr.median()).astype('float32')).astype('float32')
    out['modern_year'] = (year - year.min()).astype('float32') if len(year) else 0
    return out


def build_feature_frames(train_x, test_x, orig_x, y_orig, base_cats, base_nums, numeric_like):
    n_tr, n_te, n_or = len(train_x), len(test_x), len(orig_x)
    all_df = pd.concat([train_x, test_x, orig_x], axis=0, ignore_index=True)
    for c in base_cats:
        if c in all_df.columns:
            all_df[c] = all_df[c].astype(str).fillna('NA')
    for c in numeric_like:
        if c in all_df.columns:
            all_df[c] = finite_float32(all_df[c])

    blocks = [all_df.copy()]
    artifact_nums = [c for c in ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'TyreLife', 'LapNumber'] if c in all_df.columns]
    rel = add_f1_relationships(all_df)
    blocks.append(rel)
    aug = pd.concat([all_df, rel], axis=1)
    trans_cols = dedup_keep_order(base_nums + artifact_nums + list(rel.columns))[:36]
    blocks.append(add_numeric_transforms(aug, trans_cols))
    blocks.append(add_round_digit_features(aug, artifact_nums + ['degr_per_tyre', 'late_old_tyre']))

    bin_cols = dedup_keep_order(base_nums + ['TyreLife', 'LapNumber', 'Position', 'Position_Change', 'degr_per_tyre', 'late_old_tyre'])
    bins = pd.DataFrame(index=all_df.index)
    for c in bin_cols:
        if c in aug.columns:
            bins[f'{safe_name(c)}_bin'] = qcut_labels(aug[c], q=12).astype('string')
    blocks.append(bins)

    cat_src = pd.concat([all_df, bins], axis=1)
    pair_specs = [
        ('Driver', 'Race'), ('Driver', 'Compound'), ('Race', 'Compound'), ('Year', 'Race'),
        ('Stint', 'Compound'), ('PitStop', 'Stint'), ('Position', 'Position_Change'),
        ('TyreLife_bin', 'Compound'), ('LapNumber_bin', 'Driver'), ('RaceProgress_bin', 'Compound'),
        ('Cumulative_Degradation_bin', 'Compound'), ('LapTime_s_bin', 'Race'), ('degr_per_tyre_bin', 'Compound'),
    ]
    crosses = pd.DataFrame(index=all_df.index)
    for a, b in pair_specs:
        if a in cat_src.columns and b in cat_src.columns:
            crosses[f'{a}__{b}'] = cat_src[a].astype(str).fillna('NA') + '__' + cat_src[b].astype(str).fillna('NA')
    triple_specs = [('Driver', 'Race', 'Compound'), ('Year', 'Race', 'Driver'), ('Race', 'Compound', 'Stint'), ('Compound', 'TyreLife_bin', 'RaceProgress_bin')]
    for a, b, c in triple_specs:
        if a in cat_src.columns and b in cat_src.columns and c in cat_src.columns:
            crosses[f'{a}__{b}__{c}'] = cat_src[a].astype(str).fillna('NA') + '__' + cat_src[b].astype(str).fillna('NA') + '__' + cat_src[c].astype(str).fillna('NA')
    blocks.append(crosses)

    all_feat = pd.concat(blocks, axis=1)
    all_feat = all_feat.loc[:, ~all_feat.columns.duplicated()].copy()

    train_feat = all_feat.iloc[:n_tr].reset_index(drop=True).copy()
    test_feat = all_feat.iloc[n_tr:n_tr+n_te].reset_index(drop=True).copy()
    orig_feat = all_feat.iloc[n_tr+n_te:n_tr+n_te+n_or].reset_index(drop=True).copy()

    # counts over train + test + original and train / original frequency drift
    cat_candidates = dedup_keep_order(base_cats + list(bins.columns) + list(crosses.columns) + [c for c in all_feat.columns if c.startswith('DIG_')])
    count_cols = cat_candidates[:80]
    for c in count_cols:
        vc_all = all_feat[c].astype(str).value_counts(dropna=False)
        train_feat[f'CNT_{safe_name(c)}'] = np.log1p(train_feat[c].astype(str).map(vc_all).fillna(0)).astype('float32')
        test_feat[f'CNT_{safe_name(c)}'] = np.log1p(test_feat[c].astype(str).map(vc_all).fillna(0)).astype('float32')
        orig_feat[f'CNT_{safe_name(c)}'] = np.log1p(orig_feat[c].astype(str).map(vc_all).fillna(0)).astype('float32')
        vc_tr = train_feat[c].astype(str).value_counts(normalize=True, dropna=False)
        vc_or = orig_feat[c].astype(str).value_counts(normalize=True, dropna=False)
        denom_tr = train_feat[c].astype(str).map(vc_or).fillna(1e-6) + 1e-6
        denom_te = test_feat[c].astype(str).map(vc_or).fillna(1e-6) + 1e-6
        denom_or = orig_feat[c].astype(str).map(vc_or).fillna(1e-6) + 1e-6
        train_feat[f'DRIFT_{safe_name(c)}'] = np.log1p(train_feat[c].astype(str).map(vc_tr).fillna(0) / denom_tr).astype('float32')
        test_feat[f'DRIFT_{safe_name(c)}'] = np.log1p(test_feat[c].astype(str).map(vc_tr).fillna(0) / denom_te).astype('float32')
        orig_feat[f'DRIFT_{safe_name(c)}'] = np.log1p(orig_feat[c].astype(str).map(vc_tr).fillna(0) / denom_or).astype('float32')

    # smoothed original-data target priors
    global_mean = float(np.mean(y_orig)) if y_orig is not None and len(y_orig) else 0.5
    for c in count_cols[:50]:
        stat = pd.DataFrame({'key': orig_feat[c].astype(str), 'y': y_orig}).groupby('key')['y'].agg(['mean', 'count'])
        smooth = (stat['mean'] * stat['count'] + global_mean * 25.0) / (stat['count'] + 25.0)
        for df in (train_feat, test_feat, orig_feat):
            df[f'ORIG_mean_{safe_name(c)}'] = df[c].astype(str).map(smooth).fillna(global_mean).astype('float32')
            df[f'ORIG_logcnt_{safe_name(c)}'] = np.log1p(df[c].astype(str).map(stat['count']).fillna(0)).astype('float32')

    # distances to the original-data class centroids (numerics scaled by the original std)
    aux_nums = [c for c in ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'TyreLife', 'LapNumber', 'Stint', 'PitStop', 'Position', 'Position_Change'] if c in orig_feat.columns]
    if aux_nums and y_orig is not None and len(y_orig):
        ref = orig_feat[aux_nums].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
        sd = ref.std().replace(0, 1).fillna(1).values.astype('float32')
        mus = []
        for cls in [0, 1]:
            if np.any(y_orig == cls):
                mus.append(ref.loc[y_orig == cls, aux_nums].mean().values.astype('float32'))
            else:
                mus.append(np.zeros(len(aux_nums), dtype='float32'))
        for df in (train_feat, test_feat, orig_feat):
            vals = df[aux_nums].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32').values
            d0 = np.sqrt((((vals - mus[0]) / sd) ** 2).sum(axis=1))
            d1 = np.sqrt((((vals - mus[1]) / sd) ** 2).sum(axis=1))
            logits = np.column_stack([-d0, -d1]).astype('float32')
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            df['AUX_dist0'] = d0.astype('float32')
            df['AUX_dist1'] = d1.astype('float32')
            df['AUX_prob_like_1'] = probs[:, 1].astype('float32')
            df['AUX_gap10'] = (d1 - d0).astype('float32')

    train_feat = train_feat.loc[:, ~train_feat.columns.duplicated()].copy()
    test_feat = test_feat.loc[:, train_feat.columns].copy()
    orig_feat = orig_feat.loc[:, train_feat.columns].copy()
    cat_candidates = [c for c in cat_candidates if c in train_feat.columns]
    return train_feat, test_feat, orig_feat, cat_candidates


def add_fold_te(X_tr, y_tr, X_val, X_te, te_cols, fold_seed):
    """Target encodings (smoothing 20): inner 5-fold cross-fitting on the outer-train rows, full map for the others."""
    te_cols = [c for c in te_cols if c in X_tr.columns]
    prior = float(np.mean(y_tr))
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=fold_seed)
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_te = X_te.copy()
    for c in te_cols:
        name = f'TE_{safe_name(c)}'
        X_tr[name] = np.nan
        key_all = X_tr[c].astype(str).fillna('NA')
        for it, iv in inner.split(X_tr, y_tr):
            stats = pd.DataFrame({'key': key_all.iloc[it].values, 'y': y_tr[it]}).groupby('key')['y'].agg(['mean', 'count'])
            enc = (stats['mean'] * stats['count'] + prior * 20.0) / (stats['count'] + 20.0)
            X_tr.loc[X_tr.index[iv], name] = key_all.iloc[iv].map(enc).fillna(prior).astype('float32').values
        stats = pd.DataFrame({'key': key_all.values, 'y': y_tr}).groupby('key')['y'].agg(['mean', 'count'])
        enc = (stats['mean'] * stats['count'] + prior * 20.0) / (stats['count'] + 20.0)
        X_val[name] = X_val[c].astype(str).fillna('NA').map(enc).fillna(prior).astype('float32')
        X_te[name] = X_te[c].astype(str).fillna('NA').map(enc).fillna(prior).astype('float32')
        X_tr[name] = X_tr[name].fillna(prior).astype('float32')
    return X_tr, X_val, X_te, [f'TE_{safe_name(c)}' for c in te_cols]


class TabDataset(Dataset):
    def __init__(self, x_num, x_cat, y=None):
        self.x_num = torch.tensor(np.asarray(x_num), dtype=torch.float32)
        self.x_cat = torch.tensor(np.asarray(x_cat), dtype=torch.long)
        self.y = None if y is None else torch.tensor(np.asarray(y), dtype=torch.float32)

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x_num[idx], self.x_cat[idx]
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class TabularNet(nn.Module):
    def __init__(self, cat_cardinalities, n_num, hidden=384, depth=4, dropout=0.15):
        super().__init__()
        self.emb_layers = nn.ModuleList()
        emb_dim_total = 0
        for card in cat_cardinalities:
            emb_dim = min(64, max(8, int(round(math.sqrt(int(card) + 1) * 2))))
            self.emb_layers.append(nn.Embedding(int(card) + 1, emb_dim))
            emb_dim_total += emb_dim
        self.num_bn = nn.BatchNorm1d(max(1, n_num))
        self.input = nn.Linear(max(1, n_num) + emb_dim_total, hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x_num, x_cat):
        if x_num.shape[1] == 0:
            x_num = torch.zeros((x_num.shape[0], 1), dtype=x_num.dtype, device=x_num.device)
        x_num = self.num_bn(x_num)
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)] if len(self.emb_layers) else []
        x = torch.cat([x_num] + embs, dim=1)
        x = self.input(x)
        x = self.blocks(x)
        return self.head(x).squeeze(1)


class SmoothBCE(nn.Module):
    def __init__(self, eps=0.02):
        super().__init__()
        self.eps = eps
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        targets = targets.float() * (1.0 - self.eps) + 0.5 * self.eps
        return self.loss(logits, targets)


def predict_valid(model, loader):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x_num, x_cat, yb in loader:
            x_num = x_num.to(DEVICE, non_blocking=True)
            x_cat = x_cat.to(DEVICE, non_blocking=True)
            p = torch.sigmoid(model(x_num, x_cat)).detach().cpu().numpy()
            preds.append(p)
            ys.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def predict_test(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for x_num, x_cat in loader:
            x_num = x_num.to(DEVICE, non_blocking=True)
            x_cat = x_cat.to(DEVICE, non_blocking=True)
            p = torch.sigmoid(model(x_num, x_cat)).detach().cpu().numpy()
            preds.append(p)
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    run_folds = fold_list(ap.parse_args().folds)

    seed_everything(RANDOM_SEED)
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = to_binary_target(train[TARGET])
    orig[TARGET] = to_binary_target(orig[TARGET])
    y = train[TARGET].to_numpy(dtype=np.int8)
    y_orig = orig[TARGET].to_numpy(dtype=np.int8)
    feature_cols = [c for c in train.columns if c not in [ID, TARGET]]
    base_cats, base_nums, numeric_like = infer_groups(train, test, feature_cols)
    # F1 categoricals first
    base_cats = dedup_keep_order([c for c in ['Driver', 'Compound', 'Race', 'Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position', 'Position_Change'] if c in feature_cols] + base_cats)
    base_nums = [c for c in feature_cols if c in base_nums]
    for df in (train, test, orig):
        for c in numeric_like:
            if c in df.columns:
                df[c] = finite_float32(df[c])
        for c in base_cats:
            if c in df.columns:
                df[c] = df[c].astype(str).fillna('NA')

    train_fe, test_fe, orig_fe, cat_candidates = build_feature_frames(
        train[feature_cols].copy(), test[feature_cols].copy(), orig[feature_cols].copy(), y_orig, base_cats, base_nums, numeric_like
    )
    keep_cols = list(dict.fromkeys(train_fe.columns))
    train_fe = train_fe.loc[:, keep_cols].copy()
    test_fe = test_fe.loc[:, keep_cols].copy()
    orig_fe = orig_fe.loc[:, keep_cols].copy()

    te_cols = dedup_keep_order([c for c in cat_candidates if c in train_fe.columns])[:60]
    base_cat_cols = []
    for c in train_fe.columns:
        dt = str(train_fe[c].dtype)
        if c in cat_candidates or c in base_cats or c.endswith('_bin') or c.startswith('DIG_') or '__' in c or dt in ('object', 'string', 'category'):
            base_cat_cols.append(c)
    base_cat_cols = dedup_keep_order([c for c in base_cat_cols if c in train_fe.columns])[:MAX_CAT_COLS]
    print(f'Feature frames: train={train_fe.shape}, test={test_fe.shape}, base_cat_cols={len(base_cat_cols)}, te_cols={len(te_cols)}', flush=True)

    fold_ids = folds(y)
    for k in run_folds:
        fold = k + 1  # folds numbered 1-5 in the seeds
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        print('=' * 70, flush=True)
        print(f'Fold {k}', flush=True)
        seed_everything(RANDOM_SEED + fold)
        X_tr = train_fe.iloc[tr_idx].reset_index(drop=True).copy()
        X_va = train_fe.iloc[va_idx].reset_index(drop=True).copy()
        X_te = test_fe.reset_index(drop=True).copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        X_tr, X_va, X_te, _ = add_fold_te(X_tr, y_tr, X_va, X_te, te_cols, RANDOM_SEED + 100 + fold)
        features = list(X_tr.columns)
        cat_cols = [c for c in base_cat_cols if c in features]
        num_cols = [c for c in features if c not in cat_cols]

        Xtr_num = X_tr[num_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
        Xva_num = X_va[num_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
        Xte_num = X_te[num_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
        scaler = StandardScaler()
        Xtr_num = scaler.fit_transform(Xtr_num).astype('float32')
        Xva_num = scaler.transform(Xva_num).astype('float32')
        Xte_num = scaler.transform(Xte_num).astype('float32')

        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        if cat_cols:
            Xtr_cat = enc.fit_transform(X_tr[cat_cols].astype(str).fillna('NA'))
            Xva_cat = enc.transform(X_va[cat_cols].astype(str).fillna('NA'))
            Xte_cat = enc.transform(X_te[cat_cols].astype(str).fillna('NA'))
            Xtr_cat = (Xtr_cat + 1).astype(np.int64)
            Xva_cat = (Xva_cat + 1).astype(np.int64)
            Xte_cat = (Xte_cat + 1).astype(np.int64)
            card = [len(cats) + 1 for cats in enc.categories_]
        else:
            Xtr_cat = np.zeros((len(X_tr), 0), dtype=np.int64)
            Xva_cat = np.zeros((len(X_va), 0), dtype=np.int64)
            Xte_cat = np.zeros((len(X_te), 0), dtype=np.int64)
            card = []

        train_loader = DataLoader(TabDataset(Xtr_num, Xtr_cat, y_tr), batch_size=MODEL_PARAMS['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        valid_loader = DataLoader(TabDataset(Xva_num, Xva_cat, y_va), batch_size=MODEL_PARAMS['eval_batch_size'], shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(TabDataset(Xte_num, Xte_cat, None), batch_size=MODEL_PARAMS['eval_batch_size'], shuffle=False, num_workers=0, pin_memory=True)

        model = TabularNet(card, len(num_cols), hidden=MODEL_PARAMS['hidden'], depth=MODEL_PARAMS['depth'], dropout=MODEL_PARAMS['dropout']).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=MODEL_PARAMS['lr'], weight_decay=MODEL_PARAMS['weight_decay'])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
        loss_fn = SmoothBCE(MODEL_PARAMS['label_smoothing'])
        best_auc = -1.0
        best_state = None
        best_preds = None
        bad = 0

        for epoch in range(MODEL_PARAMS['max_epochs']):
            model.train()
            train_loss = 0.0
            n_seen = 0
            for xb_num, xb_cat, yb in train_loader:
                xb_num = xb_num.to(DEVICE, non_blocking=True)
                xb_cat = xb_cat.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(xb_num, xb_cat)
                loss = loss_fn(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_loss += float(loss.item()) * len(yb)
                n_seen += len(yb)
            va_pred, va_y = predict_valid(model, valid_loader)
            va_auc = roc_auc_score(va_y, va_pred)
            sched.step(va_auc)
            if epoch == 0 or (epoch + 1) % 5 == 0:
                print(f'Fold {k} epoch {epoch+1:02d}: loss={train_loss/max(1,n_seen):.6f} auc={va_auc:.6f}', flush=True)
            if va_auc > best_auc + 1e-7:
                best_auc = va_auc
                best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
                best_preds = va_pred.astype('float32')
                bad = 0
            else:
                bad += 1
                if bad >= MODEL_PARAMS['patience']:
                    print(f'Fold {k} early stop at epoch {epoch+1}', flush=True)
                    break

        model.load_state_dict(best_state)
        test_pred = predict_test(model, test_loader).astype('float32')
        save_part(MODEL_DIR, PAIR, k, best_preds, test_pred)
        print(f'Fold {k} AUC: {best_auc:.9f}', flush=True)

        del X_tr, X_va, X_te, Xtr_num, Xva_num, Xte_num, Xtr_cat, Xva_cat, Xte_cat
        del train_loader, valid_loader, test_loader, model, opt, sched
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
