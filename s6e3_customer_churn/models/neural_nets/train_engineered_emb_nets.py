#!/usr/bin/env python
"""Embedding MLP / residual MLP / TabTransformer on engineered features: numeric transforms, charge relationships,
service counts, bins and crosses, category counts, original-data target means and train-vs-original drift ratios.
Three presets share the features and the training loop and differ only in the network, lr and weight decay.

Features, built once on train + test + original (no train labels; the original labels only feed the original means):
- TotalCharges NaN -> the column median; the 3 numerics as float32;
- FREQ (share of rows with that value) and percentile RANK of the 3 numerics over train + test + original;
  log1p / sqrt / 1 / (1 + x) transforms and missing flags;
- charge relationships: TotalCharges - tenure x MonthlyCharges (and its absolute value), MonthlyCharges /
  (TotalCharges + 1), TotalCharges / (MonthlyCharges + 1), TotalCharges / (tenure + 1), tenure x MonthlyCharges,
  tenure x TotalCharges;
- service yes / no / other counts over 8 service columns, has-internet / has-phone flags, and Yes / No / other one-hot
  flags of 11 yes-no columns;
- categorical bins: fixed tenure bins (0, 1, 3, 6, 12, 24, 36, 48, 60, 72), 40 / 60 quantile bins of MonthlyCharges /
  TotalCharges (edges from train + test + original), 6 categorical pairs and 1 triple;
- count and rare flag (count <= 50) over train + test + original of the 16 categoricals and the 10 bin / cross columns;
- original-data churn mean (unseen -> 0.5) of those 26 columns and of the 3 numerics as strings;
- drift: log1p(train share / (original share + 1e-6)) of the 26 columns.
Per fold (from the fold's training rows): for the 3 numerics, the value snapped to the nearest value that occurs at
least 25 times (ties to the larger value) and a rare flag, as 6 more categoricals; vocabularies of the 32 categoricals
(unseen -> 0); the 3 numerics + 171 engineered numerics standardized.

Networks (categorical embeddings: dim round(1.8 cardinality^0.25) clipped to 4-64, N(0, 0.02) init):
- emb_mlp_engineered_original_priors_drift: embeddings (dropout 0.1) + numerics -> MLP 512-256 (LayerNorm, ReLU,
  dropout 0.3) -> 1 logit. lr 2.5e-5, weight decay 3e-4.
- emb_resmlp_engineered_original_priors_drift: embeddings (dropout 0.1) + numerics -> 2 residual blocks (512-512 and
  256-256: Linear, BatchNorm, SiLU, dropout 0.15, Linear, BatchNorm, + projected shortcut, SiLU, dropout) -> 1 logit.
  lr 2.5e-5, weight decay 3e-4.
- tabtransformer_engineered_original_priors_drift: one 128-d token per categorical plus one linear token of all
  numerics, 5 post-norm Transformer encoder layers (8 heads, FFN 512, ReLU, dropout 0.2), flattened -> 512-256 head
  (BatchNorm, ReLU, dropout 0.2) -> 1 logit. lr 1e-3, weight decay 1e-5.
Training (all): BCE on 0.02-smoothed labels, AdamW, batch 256, gradient clipping 1, up to 80 epochs; lr constant for
2 epochs, then cosine annealing to 0.05 lr (T_max 79); early stopping on the held-out AUC (patience 10) and the best
epoch predicts (sigmoid). numpy / torch are seeded with 42 before the folds.
Usage: python train_engineered_emb_nets.py {emb_mlp_engineered_original_priors_drift,
    emb_resmlp_engineered_original_priors_drift,tabtransformer_engineered_original_priors_drift}"""
import argparse
import gc
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
from common.data import TARGET, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

PRESETS = {
    'emb_mlp_engineered_original_priors_drift': dict(model='mlp', lr=2.5e-5, weight_decay=3e-4, test_folds=50),
    'emb_resmlp_engineered_original_priors_drift': dict(model='resmlp', lr=2.5e-5, weight_decay=3e-4, test_folds=50),
    'tabtransformer_engineered_original_priors_drift': dict(model='tabtransformer', lr=1e-3, weight_decay=1e-5,
                                                            test_folds=5),
}
MODEL_DIR = model_dir('neural_nets')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
RANDOM_STATE = 42

BASE_CATS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
]
BASE_NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
RARE_MIN_COUNT = 25

EPOCHS = 80
BATCH_SIZE = 256
PATIENCE = 10
EMB_DROPOUT = 0.10
MLP_DROPOUT = 0.30
HIDDEN = (512, 256)
WARMUP_EPOCHS = 1
TABTRANSFORMER_PARAMS = dict(embed_dim=128, num_heads=8, num_transformer_layers=5, ff_hidden_dim=512, dropout_rate=0.20)

TENURE_BINS = [0, 1, 3, 6, 12, 24, 36, 48, 60, 72, 10_000]
MC_BINS = 40
TC_BINS = 60
CROSS_PAIRS = [
    ('Contract', 'InternetService'),
    ('PaymentMethod', 'Contract'),
    ('InternetService', 'OnlineSecurity'),
    ('PaymentMethod', 'PaperlessBilling'),
    ('Contract', 'PaperlessBilling'),
    ('InternetService', 'TechSupport'),
]
TRIPLE = [('Contract', 'InternetService', 'PaymentMethod')]
YN_COLS = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']


# ----------------------------------------------------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------------------------------------------------
def to_binary_target(series: pd.Series) -> np.ndarray:
    return (series.astype(str).str.strip().str.lower() == 'yes').astype(np.int32).values


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out['TotalCharges'].dtype == 'object':
        s = out['TotalCharges'].astype(str)
        s = s.mask(s.str.strip().eq(''), np.nan)
        out['TotalCharges'] = pd.to_numeric(s, errors='coerce')
    out['TotalCharges'] = out['TotalCharges'].astype('float32')
    out['TotalCharges'] = out['TotalCharges'].fillna(out['TotalCharges'].median())
    return out


def safe_str(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna('missing').str.strip()
    return df


def add_global_rank_and_freq(train_df, test_df, orig_df, num_cols):
    all_num = pd.concat([train_df[num_cols], test_df[num_cols], orig_df[num_cols]], axis=0, ignore_index=True)
    for c in num_cols:
        freq = all_num[c].value_counts(normalize=True)
        for df in (train_df, test_df, orig_df):
            df[f'FREQ_{c}'] = df[c].map(freq).fillna(0).astype('float32')
    n_tr, n_te = len(train_df), len(test_df)
    for c in num_cols:
        r = all_num[c].rank(method='average', pct=True)
        train_df[f'RANK_{c}'] = r.iloc[:n_tr].astype('float32').values
        test_df[f'RANK_{c}'] = r.iloc[n_tr:n_tr + n_te].astype('float32').values
        orig_df[f'RANK_{c}'] = r.iloc[n_tr + n_te:].astype('float32').values
    return train_df, test_df, orig_df


def add_num_transforms(train_df, test_df, orig_df, num_cols):
    for df in (train_df, test_df, orig_df):
        for c in num_cols:
            v = df[c].astype('float32')
            df[f'LOG1P_{c}'] = np.log1p(v.clip(lower=0)).astype('float32')
            df[f'SQRT_{c}'] = np.sqrt(v.clip(lower=0)).astype('float32')
            df[f'INV1P_{c}'] = (1.0 / (1.0 + v.clip(lower=0))).astype('float32')
            df[f'MISS_{c}'] = pd.to_numeric(df[c], errors='coerce').isna().astype('int8')
    return train_df, test_df, orig_df


def add_charge_relationships(train_df, test_df, orig_df):
    for df in (train_df, test_df, orig_df):
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['abs_charges_dev'] = np.abs(df['charges_deviation']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['total_to_monthly_ratio'] = (df['TotalCharges'] / (df['MonthlyCharges'] + 1)).astype('float32')
        df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
        df['tenure_x_monthly'] = (df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['tenure_x_total'] = (df['tenure'] * df['TotalCharges']).astype('float32')
    return train_df, test_df, orig_df


def add_service_counts(train_df, test_df, orig_df):
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for df in (train_df, test_df, orig_df):
        df = safe_str(df, service_cols + ['InternetService'])
        df['service_yes_count'] = (df[service_cols] == 'Yes').sum(axis=1).astype('float32')
        df['service_no_count'] = (df[service_cols] == 'No').sum(axis=1).astype('float32')
        df['service_other_count'] = (df[service_cols].isin(['No phone service', 'No internet service'])).sum(
            axis=1).astype('float32')
        df['service_count'] = df['service_yes_count'].astype('float32')
        df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
    return train_df, test_df, orig_df


def add_yes_no_other_flags(train_df, test_df, orig_df):
    for df in (train_df, test_df, orig_df):
        for c in YN_COLS:
            if c in df.columns:
                s = df[c].astype(str)
                df[f'ISYES_{c}'] = (s == 'Yes').astype('float32')
                df[f'ISNO_{c}'] = (s == 'No').astype('float32')
                df[f'ISOTHER_{c}'] = (~s.isin(['Yes', 'No'])).astype('float32')
    return train_df, test_df, orig_df


def add_bins_and_cross_cats(train_df, test_df, orig_df):
    for df in (train_df, test_df, orig_df):
        df['tenure_bin'] = pd.cut(df['tenure'], bins=TENURE_BINS, include_lowest=True).astype(str)
    all_mc = pd.concat([train_df['MonthlyCharges'], test_df['MonthlyCharges'], orig_df['MonthlyCharges']], axis=0)
    all_tc = pd.concat([train_df['TotalCharges'], test_df['TotalCharges'], orig_df['TotalCharges']], axis=0)
    mc_bins = pd.qcut(all_mc, q=MC_BINS, retbins=True, duplicates='drop')[1]
    tc_bins = pd.qcut(all_tc, q=TC_BINS, retbins=True, duplicates='drop')[1]
    for df in (train_df, test_df, orig_df):
        df['MonthlyCharges_bin'] = pd.cut(df['MonthlyCharges'], bins=mc_bins, include_lowest=True).astype(str)
        df['TotalCharges_bin'] = pd.cut(df['TotalCharges'], bins=tc_bins, include_lowest=True).astype(str)
    for a, b in CROSS_PAIRS:
        if a in train_df.columns and b in train_df.columns:
            for df in (train_df, test_df, orig_df):
                df[f'{a}__{b}'] = (df[a].astype(str) + '|' + df[b].astype(str))
    for a, b, c in TRIPLE:
        if a in train_df.columns and b in train_df.columns and c in train_df.columns:
            for df in (train_df, test_df, orig_df):
                df[f'{a}__{b}__{c}'] = (df[a].astype(str) + '|' + df[b].astype(str) + '|' + df[c].astype(str))
    return train_df, test_df, orig_df


def add_cat_counts_and_rare(train_df, test_df, orig_df, cat_cols, rare_thresh=50):
    all_cat = pd.concat([train_df[cat_cols], test_df[cat_cols], orig_df[cat_cols]], axis=0, ignore_index=True)
    for c in cat_cols:
        vc = all_cat[c].astype('string').value_counts(dropna=False)
        for df in (train_df, test_df, orig_df):
            cnt = df[c].astype('string').map(vc)
            df[f'CAT_CNT_{c}'] = pd.to_numeric(cnt, errors='coerce').fillna(0).astype('float32')
            df[f'CAT_RARE_{c}'] = (df[f'CAT_CNT_{c}'] <= rare_thresh).astype('float32')
    return train_df, test_df, orig_df


def add_orig_proba_features(train_df, test_df, orig_df, cols, y_orig_bin):
    orig_tmp = orig_df.copy()
    orig_tmp['_y_'] = y_orig_bin.astype('float32')
    for c in cols:
        if c not in orig_tmp.columns:
            continue
        m = orig_tmp.groupby(c, observed=False)['_y_'].mean()
        train_df[f'ORIG_proba_{c}'] = train_df[c].map(m).fillna(0.5).astype('float32')
        test_df[f'ORIG_proba_{c}'] = test_df[c].map(m).fillna(0.5).astype('float32')
    return train_df, test_df


def add_drift_ratio(train_df, test_df, orig_df, cols, eps=1e-6):
    new_cols = []
    for c in cols:
        if c not in train_df.columns or c not in orig_df.columns:
            continue
        p_tr = train_df[c].value_counts(normalize=True)
        p_or = orig_df[c].value_counts(normalize=True)
        name = f'DRIFT_{c}'
        train_df[name] = np.log1p(train_df[c].map(p_tr).fillna(0) / (train_df[c].map(p_or).fillna(eps) + eps)).astype(
            'float32')
        test_df[name] = np.log1p(test_df[c].map(p_tr).fillna(0) / (test_df[c].map(p_or).fillna(eps) + eps)).astype(
            'float32')
        new_cols.append(name)
    return train_df, test_df, new_cols


def build_features():
    train = clean_total_charges(read_train())
    test = clean_total_charges(read_test())
    orig = clean_total_charges(read_original())
    y = to_binary_target(train[TARGET]).astype(np.float32)
    y_orig = to_binary_target(orig[TARGET])

    train = safe_str(train, BASE_CATS)
    test = safe_str(test, BASE_CATS)
    orig = safe_str(orig, BASE_CATS)
    for df in (train, test, orig):
        for c in BASE_NUMS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
        df['MonthlyCharges'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].median())
        df['tenure'] = df['tenure'].fillna(df['tenure'].median())

    train, test, orig = add_global_rank_and_freq(train, test, orig, BASE_NUMS)
    train, test, orig = add_num_transforms(train, test, orig, BASE_NUMS)
    train, test, orig = add_charge_relationships(train, test, orig)
    train, test, orig = add_service_counts(train, test, orig)
    train, test, orig = add_yes_no_other_flags(train, test, orig)
    train, test, orig = add_bins_and_cross_cats(train, test, orig)

    engine_cats = ['tenure_bin', 'MonthlyCharges_bin', 'TotalCharges_bin'] + [f'{a}__{b}' for a, b in CROSS_PAIRS] + [
        f'{a}__{b}__{c}' for a, b, c in TRIPLE]
    engine_cats = [c for c in engine_cats if c in train.columns]
    train = safe_str(train, engine_cats)
    test = safe_str(test, engine_cats)
    orig = safe_str(orig, engine_cats)

    all_for_counts = BASE_CATS + engine_cats
    train, test, orig = add_cat_counts_and_rare(train, test, orig, all_for_counts, rare_thresh=50)

    for df in (train, test, orig):
        df['CAT_tenure'] = df['tenure'].astype(str)
        df['CAT_MonthlyCharges'] = df['MonthlyCharges'].astype(str)
        df['CAT_TotalCharges'] = df['TotalCharges'].astype(str)
    orig_proba_keys = BASE_CATS + engine_cats + ['CAT_tenure', 'CAT_MonthlyCharges', 'CAT_TotalCharges']
    train, test = add_orig_proba_features(train, test, orig, orig_proba_keys, y_orig)
    train, test, drift_cols = add_drift_ratio(train, test, orig, BASE_CATS + engine_cats, eps=1e-6)

    cat_base_for_nn = BASE_CATS + engine_cats
    num_engine = []
    for c in BASE_NUMS:
        num_engine += [f'FREQ_{c}', f'RANK_{c}', f'LOG1P_{c}', f'SQRT_{c}', f'INV1P_{c}', f'MISS_{c}']
    num_engine += ['charges_deviation', 'abs_charges_dev', 'monthly_to_total_ratio', 'total_to_monthly_ratio',
                   'avg_monthly_charges', 'tenure_x_monthly', 'tenure_x_total', 'service_yes_count',
                   'service_no_count', 'service_other_count', 'service_count', 'has_internet', 'has_phone']
    for c in YN_COLS:
        if c in train.columns:
            num_engine += [f'ISYES_{c}', f'ISNO_{c}', f'ISOTHER_{c}']
    for c in all_for_counts:
        num_engine += [f'CAT_CNT_{c}', f'CAT_RARE_{c}']
    for c in orig_proba_keys:
        if f'ORIG_proba_{c}' in train.columns:
            num_engine.append(f'ORIG_proba_{c}')
    num_engine += drift_cols
    num_engine = [c for c in num_engine if c in train.columns]
    nums_for_nn = BASE_NUMS + num_engine
    print(f'{len(cat_base_for_nn)} categorical (+6 per fold) and {len(nums_for_nn)} numeric columns', flush=True)
    return train, test, y, cat_base_for_nn, nums_for_nn


def emb_dim_from_card(card: int) -> int:
    return int(np.clip(int(round(1.8 * (card ** 0.25))), 4, 64))


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


# ----------------------------------------------------------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------------------------------------------------------
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
    def __init__(self, cat_cardinals, n_num, hidden=(256, 128), emb_dropout=0.1, mlp_dropout=0.2):
        super().__init__()
        self.emb_layers = nn.ModuleList()
        emb_out_dim = 0
        for card in cat_cardinals:
            d = emb_dim_from_card(card)
            self.emb_layers.append(nn.Embedding(num_embeddings=card, embedding_dim=d))
            emb_out_dim += d
        self.emb_drop = nn.Dropout(emb_dropout)
        in_dim = emb_out_dim + n_num
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(inplace=True), nn.Dropout(mlp_dropout)]
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


class ResMLPBlock(nn.Module):
    """y = Linear(u1) -> BN -> SiLU -> dropout -> Linear(u2) -> BN; out = SiLU(shortcut + y) -> dropout, where the
    shortcut is x or, when the widths differ, Linear(u2) -> BN of x."""

    def __init__(self, in_dim: int, u1: int, u2: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, u1)
        self.bn1 = nn.BatchNorm1d(u1)
        self.fc2 = nn.Linear(u1, u2)
        self.bn2 = nn.BatchNorm1d(u2)
        self.use_proj = (in_dim != u2)
        if self.use_proj:
            self.proj = nn.Linear(in_dim, u2)
            self.bn_proj = nn.BatchNorm1d(u2)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.drop(self.act(self.bn1(self.fc1(x))))
        y = self.bn2(self.fc2(y))
        shortcut = self.bn_proj(self.proj(x)) if self.use_proj else x
        return self.drop(self.act(shortcut + y))


class EmbResMLP(nn.Module):
    """Embeddings + numerics -> residual blocks over (u1, u2) pairs of res_units -> 1 logit."""

    def __init__(self, cat_cardinals, n_num: int, res_units=(128, 128, 128, 128), emb_dropout=0.10, block_dropout=0.10):
        super().__init__()
        self.n_cat = len(cat_cardinals)
        self.emb_layers = nn.ModuleList()
        emb_out_dim = 0
        for card in cat_cardinals:
            d = emb_dim_from_card(card)
            self.emb_layers.append(nn.Embedding(num_embeddings=card, embedding_dim=d))
            emb_out_dim += d
        self.emb_drop = nn.Dropout(emb_dropout)
        units = list(res_units)
        if len(units) == 1:
            units = [units[0], units[0]]
        blocks, cur = [], emb_out_dim + n_num
        for i in range(0, len(units), 2):
            u1 = units[i]
            u2 = units[i + 1] if (i + 1) < len(units) else units[i]
            blocks.append(ResMLPBlock(cur, u1, u2, dropout=block_dropout))
            cur = u2
        self.resmlp = nn.Sequential(*blocks)
        self.out = nn.Linear(cur, 1)
        for emb in self.emb_layers:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, x_cat, x_num):
        if self.n_cat > 0:
            z = torch.cat([emb(x_cat[:, j]) for j, emb in enumerate(self.emb_layers)], dim=1)
            z = torch.cat([self.emb_drop(z), x_num], dim=1)
        else:
            z = x_num
        return self.out(self.resmlp(z)).squeeze(1)


class TabTransformer(nn.Module):
    """One token per categorical (embedding) plus one token for all numerics (linear projection), a Transformer encoder
    over the tokens, the flattened tokens -> BatchNorm / ReLU MLP head -> 1 logit."""

    def __init__(self, n_num: int, cat_cardinals, embed_dim=128, num_heads=8, num_transformer_layers=5,
                 ff_hidden_dim=512, dropout_rate=0.2):
        super().__init__()
        self.n_cat = len(cat_cardinals)
        self.cat_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=card, embedding_dim=embed_dim)
                                             for card in cat_cardinals])
        self.numerical_proj = nn.Linear(int(n_num), embed_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_hidden_dim,
                                               dropout=dropout_rate, batch_first=True, activation='relu',
                                               norm_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=num_transformer_layers)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim * (1 + self.n_cat), ff_hidden_dim),
            nn.BatchNorm1d(ff_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_hidden_dim, ff_hidden_dim // 2),
            nn.BatchNorm1d(ff_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_hidden_dim // 2, 1),
        )
        for emb in self.cat_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, x_cat, x_num):
        num_tok = self.numerical_proj(x_num).unsqueeze(1)
        if self.n_cat > 0:
            cat_toks = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
            tokens = torch.cat([num_tok, cat_toks], dim=1)
        else:
            tokens = num_tok
        out = self.transformer_encoder(tokens)
        return self.mlp_head(out.reshape(out.size(0), -1)).squeeze(1)


def make_model(kind, cat_cardinals, n_num):
    if kind == 'mlp':
        return EmbMLP(cat_cardinals, n_num=n_num, hidden=HIDDEN, emb_dropout=EMB_DROPOUT, mlp_dropout=MLP_DROPOUT)
    if kind == 'resmlp':
        res_units = tuple([h for h in HIDDEN for _ in (0, 1)])
        return EmbResMLP(cat_cardinals, n_num=n_num, res_units=res_units, emb_dropout=EMB_DROPOUT,
                         block_dropout=max(0.0, MLP_DROPOUT * 0.5))
    return TabTransformer(n_num=n_num, cat_cardinals=cat_cardinals, **TABTRANSFORMER_PARAMS)


# ----------------------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------------------
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
    model = make_model(cfg['model'], cat_cardinals, Xn_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    loss_fn = SmoothBCE(0.02)
    loader_kw = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    dl_tr = DataLoader(TabMixDataset(Xc_tr, Xn_tr, y_tr), shuffle=True, **loader_kw)
    dl_va = DataLoader(TabMixDataset(Xc_va, Xn_va, y_va), shuffle=False, **loader_kw)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=lr * 0.05)

    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        if epoch <= WARMUP_EPOCHS:
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += loss.item() * xc.size(0)
            n_seen += xc.size(0)
        if epoch > WARMUP_EPOCHS:
            sched.step()

        auc = roc_auc_score(y_va, predict_proba(model, dl_va))
        print(f'     epoch {epoch:03d} | lr {opt.param_groups[0]["lr"]:.2e} | loss {running_loss / max(1, n_seen):.5f} '
              f'| val_auc {auc:.6f}', flush=True)
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


def cross_validate(cfg, train, test, y, cat_base_for_nn, nums_for_nn, n_splits):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(train), dtype=np.float32)
    pred_test = np.zeros(len(test), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train)), y), 1):
        print(f'\n======== Fold {fold}/{n_splits} ========', flush=True)
        tr_df = train.iloc[tr_idx].reset_index(drop=True)
        va_df = train.iloc[va_idx].reset_index(drop=True)

        tr_cat_df = tr_df[cat_base_for_nn].copy()
        va_cat_df = va_df[cat_base_for_nn].copy()
        te_cat_df = test[cat_base_for_nn].copy()
        for col in BASE_NUMS:
            snapper = build_numeric_snapper(tr_df[col], rare_min_count=RARE_MIN_COUNT)
            tr_snap, tr_israre = snapper(tr_df[col])
            va_snap, va_israre = snapper(va_df[col])
            te_snap, te_israre = snapper(test[col])
            tr_cat_df[f'{col}__cat'] = pd.Series(tr_snap).astype(str).values
            va_cat_df[f'{col}__cat'] = pd.Series(va_snap).astype(str).values
            te_cat_df[f'{col}__cat'] = pd.Series(te_snap).astype(str).values
            tr_cat_df[f'{col}__is_rare'] = tr_israre.astype(np.int32)
            va_cat_df[f'{col}__is_rare'] = va_israre.astype(np.int32)
            te_cat_df[f'{col}__is_rare'] = te_israre.astype(np.int32)
        cat_all = list(tr_cat_df.columns)

        maps, sizes = make_vocab_maps(tr_cat_df, cat_all)
        cat_cardinals = [sizes[c] for c in cat_all]
        Xc_tr = encode_with_maps(tr_cat_df, cat_all, maps)
        Xc_va = encode_with_maps(va_cat_df, cat_all, maps)
        Xc_te = encode_with_maps(te_cat_df, cat_all, maps)

        scaler = StandardScaler()
        Xn_tr = scaler.fit_transform(tr_df[nums_for_nn].astype(np.float32).values).astype(np.float32)
        Xn_va = scaler.transform(va_df[nums_for_nn].astype(np.float32).values).astype(np.float32)
        Xn_te = scaler.transform(test[nums_for_nn].astype(np.float32).values).astype(np.float32)

        model, best_auc = train_one_fold(cfg, Xc_tr, Xn_tr, y[tr_idx], Xc_va, Xn_va, y[va_idx], cat_cardinals)
        dl_va = DataLoader(TabMixDataset(Xc_va, Xn_va, y[va_idx]), batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                           pin_memory=True)
        oof[va_idx] = predict_proba(model, dl_va)
        dl_te = DataLoader(TabMixDataset(Xc_te, Xn_te, None), batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                           pin_memory=True)
        pred_test += predict_proba(model, dl_te).astype(np.float32) / n_splits
        print(f'[Fold {fold}] best val AUC: {best_auc:.6f}', flush=True)

        del tr_df, va_df, tr_cat_df, va_cat_df, te_cat_df, Xc_tr, Xc_va, Xc_te, Xn_tr, Xn_va, Xn_te, model, dl_va, dl_te
        gc.collect()
    print(f'{n_splits}-fold OOF AUC: {roc_auc_score(y, oof):.6f}', flush=True)
    return oof, pred_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=sorted(PRESETS))
    pair = ap.parse_args().pair
    cfg = PRESETS[pair]

    train, test, y, cat_base_for_nn, nums_for_nn = build_features()
    oof, pred_test = cross_validate(cfg, train, test, y, cat_base_for_nn, nums_for_nn, 5)
    if cfg['test_folds'] != 5:
        _, pred_test = cross_validate(cfg, train, test, y, cat_base_for_nn, nums_for_nn, cfg['test_folds'])
    save_pair(MODEL_DIR, pair, oof, pred_test)


if __name__ == '__main__':
    main()
