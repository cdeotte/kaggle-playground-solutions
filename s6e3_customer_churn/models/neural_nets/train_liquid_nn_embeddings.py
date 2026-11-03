#!/usr/bin/env python
"""Gated liquid neural network with categorical embeddings on decimal-digit, bin, pair and original-data churn-rate
features.

MonthlyCharges / TotalCharges are read as strings (their decimal digits are features), stripped and parsed; blank
TotalCharges -> tenure x MonthlyCharges. Features (the same function on the original, train and test rows; the
original rows are a reference only, not training rows):
- the 19 columns;
- decimal digits: first and second decimal digit and number of decimals of MonthlyCharges and TotalCharges (as
  written in the csv; -1 if absent);
- service counts over the 9 service columns: Yes / Fiber optic / DSL, No, No internet service, No phone service;
- 10 charge relations (TotalCharges / (tenure + 1), TotalCharges - tenure x MonthlyCharges with its absolute and
  relative value, tenure x MonthlyCharges, MonthlyCharges / (tenure + 1), TotalCharges / (MonthlyCharges + 1),
  MonthlyCharges - TotalCharges / (tenure + 1), SeniorCitizen x MonthlyCharges, SeniorCitizen x TotalCharges) and
  log1p / sqrt / square of the 3 numerics;
- categorical bins at fixed edges (tenure, MonthlyCharges, TotalCharges, TotalCharges / (tenure + 1)), SeniorCitizen as
  a category and floor bins (tenure / 3, / 6; MonthlyCharges / 5, / 10; TotalCharges / 50, / 100;
  TotalCharges / (tenure + 1) / 5);
- 12 string pairs: Contract x InternetService, Contract x PaymentMethod, Partner x Dependents, OnlineSecurity x
  TechSupport, StreamingTV x StreamingMovies, InternetService x MonthlyCharges bin, Contract x tenure bin,
  PaymentMethod x MonthlyCharges bin, and 4 digit pairs;
- original-data churn mean (unseen -> global mean) and row count of 29 categoricals / bins / digits and of the 8
  column pairs (the reference is the original data after the same feature function).
Inputs per fold: the columns that are not constant on the training rows; categoricals ordinal-encoded on the training
rows (0 = unknown) and embedded (dim round(1.6 sqrt(card)) clipped to 4-32, dropout 0.05); continuous columns with
missing -> training median, min-max scaled (scaler fitted on the training values + N(0, 1e-5) noise, noise seed 42).
Model: continuous projection Linear -> SiLU -> LayerNorm -> dropout 0.075 (width 192); [embeddings, projection] ->
Linear(384) -> SiLU -> LayerNorm -> dropout 0.15 -> 4 liquid blocks (gated liquid cell h <- h + g (SiLU(W x + U h) -
h) / (softplus(tau) + 1e-4), g = sigmoid(G [x, h]); x <- LayerNorm(x + dropout(P h))) -> head 192 (SiLU, LayerNorm,
dropout 0.15) -> 1 logit.
Training: BCE with logits on labels smoothed by 0.01, AdamW lr 3e-4, weight decay 1e-5, linear warm-up (8% of the
steps) then cosine decay to 3e-5 (stepped per batch), batch 1024, gradient-norm clip 1, bf16 autocast, up to 40
epochs with early stopping on the held-out AUC (patience 6); the best epoch predicts (sigmoid, fp32). Seed 42 once
before the folds, so the pair is reproduced only when the 5 folds run in order in one process. Test = mean of the 5
fold models.
Usage: python train_liquid_nn_embeddings.py [--folds 0,1,2,3,4]"""
import argparse
import gc
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import DATA_DIR, ID, ORIGINAL_FILE, TARGET, folds, model_dir  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'liquid_nn_embeddings_digits_orig_priors'
MODEL_DIR = model_dir('neural_nets')
SEED = 42
STRING_DTYPES = {'TotalCharges': 'string', 'MonthlyCharges': 'string'}


class CFG:
    # model
    hidden_dim = 384
    num_layers = 4
    dropout_prob = 0.15
    emb_dropout = 0.05
    # training
    learning_rate = 3e-4
    min_lr = 3e-5
    weight_decay = 1e-5
    warmup_ratio = 0.08
    batch_size = 1024
    max_epochs = 40
    patience = 6
    max_norm = 1.0
    label_smoothing = 0.01
    # preprocessing
    noise_std = 1e-5
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- data
def map_target_yes_no(s):
    return s.astype('string').str.strip().map({'Yes': 1, 'No': 0})


def clean_charges(df):
    df = df.copy()
    for c in ['TotalCharges', 'MonthlyCharges']:
        df[c] = df[c].astype('string').str.strip()
    # keep the exact strings for the digit features
    df['MonthlyCharges_str'] = df['MonthlyCharges'].copy()
    df['TotalCharges_str'] = df['TotalCharges'].copy()
    df['TotalCharges'] = df['TotalCharges'].replace('', pd.NA).replace(' ', pd.NA)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    if df['TotalCharges'].isna().any():
        df['TotalCharges'] = df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'])
    return df


def strip_string_cols(df):
    df = df.copy()
    for c in df.select_dtypes(include=['object', 'string']).columns.tolist():
        df[c] = df[c].astype('string').str.strip()
    return df


BASE_SERVICE_COLS = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
PAIR_COLS = [
    ('Contract', 'InternetService'),
    ('Contract', 'PaymentMethod'),
    ('Partner', 'Dependents'),
    ('OnlineSecurity', 'TechSupport'),
    ('StreamingTV', 'StreamingMovies'),
    ('InternetService', 'MonthlyCharges_bin_'),
    ('Contract', 'tenure_bin_'),
    ('PaymentMethod', 'MonthlyCharges_bin_'),
]
ORIG_MAP_COLS = [
    'Contract', 'InternetService', 'PaymentMethod', 'Partner', 'Dependents',
    'OnlineSecurity', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling', 'gender',
    'mc0_', 'mc1_', 'mc_len_', 'tc0_', 'tc1_', 'tc_len_',
    'tenure_bin_', 'MonthlyCharges_bin_', 'TotalCharges_bin_', 'avg_monthly_bin_',
    'SeniorCitizen_cat_', 'tenure_div3_', 'tenure_div6_',
    'MonthlyCharges_div5_', 'MonthlyCharges_div10_',
    'TotalCharges_div50_', 'TotalCharges_div100_', 'avg_monthly_div5_',
]


def add_digit_features(df):
    df = df.copy()
    mc_dec = df['MonthlyCharges_str'].astype('string').str.split('.').str[1].fillna('')
    df['mc0_'] = mc_dec.str[0].fillna('-1').replace('', '-1').astype('string')
    df['mc1_'] = mc_dec.str[1].fillna('-1').replace('', '-1').astype('string')
    df['mc_len_'] = mc_dec.str.len().fillna(0).astype('int32').astype('string')
    tc_dec = df['TotalCharges_str'].astype('string').str.split('.').str[1].fillna('')
    df['tc0_'] = tc_dec.str[0].fillna('-1').replace('', '-1').astype('string')
    df['tc1_'] = tc_dec.str[1].fillna('-1').replace('', '-1').astype('string')
    df['tc_len_'] = tc_dec.str.len().fillna(0).astype('int32').astype('string')
    return df


def add_service_features(df):
    df = df.copy()
    tmp = pd.DataFrame(index=df.index)
    for c in BASE_SERVICE_COLS:
        s = df[c].astype('string')
        tmp[f'{c}_yes'] = s.isin(['Yes', 'Fiber optic', 'DSL']).astype('float32')
        tmp[f'{c}_no'] = s.eq('No').astype('float32')
        tmp[f'{c}_no_internet'] = s.eq('No internet service').astype('float32')
        tmp[f'{c}_no_phone'] = s.eq('No phone service').astype('float32')
    df['_service_yes_count'] = tmp[[c for c in tmp.columns if c.endswith('_yes')]].sum(axis=1).astype('float32')
    df['_service_no_count'] = tmp[[c for c in tmp.columns if c.endswith('_no')]].sum(axis=1).astype('float32')
    df['_service_no_internet_count'] = tmp[[c for c in tmp.columns if c.endswith('_no_internet')]].sum(axis=1).astype(
        'float32')
    df['_service_no_phone_count'] = tmp[[c for c in tmp.columns if c.endswith('_no_phone')]].sum(axis=1).astype(
        'float32')
    return df


def add_numeric_features(df):
    df = df.copy()
    t = df['tenure'].astype('float32')
    mc = df['MonthlyCharges'].astype('float32')
    tc = df['TotalCharges'].astype('float32')
    sc = df['SeniorCitizen'].astype('float32')
    df['_avg_monthly_from_total'] = (tc / (t + 1.0)).astype('float32')
    df['_charge_gap'] = (tc - t * mc).astype('float32')
    df['_charge_gap_abs'] = np.abs(df['_charge_gap']).astype('float32')
    df['_charge_gap_rel'] = (df['_charge_gap'] / (tc + 1.0)).astype('float32')
    df['_tenure_x_monthly'] = (t * mc).astype('float32')
    df['_monthly_per_tenure'] = (mc / (t + 1.0)).astype('float32')
    df['_total_per_monthly'] = (tc / (mc + 1.0)).astype('float32')
    df['_monthly_minus_avg'] = (mc - df['_avg_monthly_from_total']).astype('float32')
    df['_senior_x_monthly'] = (sc * mc).astype('float32')
    df['_senior_x_total'] = (sc * tc).astype('float32')
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        x = df[col].astype('float32')
        df[f'_{col}_log1p'] = np.log1p(np.clip(x, 0, None)).astype('float32')
        df[f'_{col}_sqrt'] = np.sqrt(np.clip(x, 0, None)).astype('float32')
        df[f'_{col}_sq'] = np.square(x).astype('float32')
    return df


def add_bins_and_cats(df):
    df = df.copy()
    df['tenure_bin_'] = pd.cut(df['tenure'], bins=[-np.inf, 1, 6, 12, 24, 36, 48, 60, 72, np.inf],
                               labels=False).fillna(-1).astype('int32').astype('string')
    df['MonthlyCharges_bin_'] = pd.cut(df['MonthlyCharges'], bins=[-np.inf, 20, 35, 50, 65, 80, 95, 110, np.inf],
                                       labels=False).fillna(-1).astype('int32').astype('string')
    df['TotalCharges_bin_'] = pd.cut(df['TotalCharges'], bins=[-np.inf, 100, 500, 1000, 2000, 3000, 4500, 6000, np.inf],
                                     labels=False).fillna(-1).astype('int32').astype('string')
    df['avg_monthly_bin_'] = pd.cut(df['_avg_monthly_from_total'],
                                    bins=[-np.inf, 20, 35, 50, 65, 80, 95, 110, np.inf],
                                    labels=False).fillna(-1).astype('int32').astype('string')
    df['SeniorCitizen_cat_'] = df['SeniorCitizen'].astype('int32').astype('string')
    df['tenure_div3_'] = np.floor(df['tenure'] / 3).astype('int32').astype('string')
    df['tenure_div6_'] = np.floor(df['tenure'] / 6).astype('int32').astype('string')
    df['MonthlyCharges_div5_'] = np.floor(df['MonthlyCharges'] / 5).astype('int32').astype('string')
    df['MonthlyCharges_div10_'] = np.floor(df['MonthlyCharges'] / 10).astype('int32').astype('string')
    df['TotalCharges_div50_'] = np.floor(df['TotalCharges'] / 50).astype('int32').astype('string')
    df['TotalCharges_div100_'] = np.floor(df['TotalCharges'] / 100).astype('int32').astype('string')
    df['avg_monthly_div5_'] = np.floor(df['_avg_monthly_from_total'] / 5).astype('int32').astype('string')
    return df


def add_pair_features(df):
    df = df.copy()
    for a, b in PAIR_COLS:
        if a in df.columns and b in df.columns:
            df[f'{a}__{b}_'] = df[a].astype('string') + '|' + df[b].astype('string')
    df['mc0_tc0_'] = df['mc0_'].astype('string') + '|' + df['tc0_'].astype('string')
    df['mc1_tc1_'] = df['mc1_'].astype('string') + '|' + df['tc1_'].astype('string')
    df['Contract_mc0_'] = df['Contract'].astype('string') + '|' + df['mc0_'].astype('string')
    df['InternetService_tc0_'] = df['InternetService'].astype('string') + '|' + df['tc0_'].astype('string')
    return df


def add_orig_maps(df, orig_df):
    """Original-data churn mean / row count of every column in ORIG_MAP_COLS and of the column pairs that exist in
    both frames."""
    df = df.copy()
    global_mean = float(orig_df[TARGET].mean())
    for c in ORIG_MAP_COLS:
        if c in df.columns and c in orig_df.columns:
            grp_mean = orig_df.groupby(c, dropna=False)[TARGET].mean()
            grp_size = orig_df.groupby(c, dropna=False).size()
            df[f'_orig_mean__{c}'] = df[c].map(grp_mean).fillna(global_mean).astype('float32')
            df[f'_orig_size__{c}'] = df[c].map(grp_size).fillna(0).astype('float32')
    for a, b in PAIR_COLS:
        if a in df.columns and b in df.columns and a in orig_df.columns and b in orig_df.columns:
            key_df = df[a].astype('string') + '|' + df[b].astype('string')
            key_orig = orig_df[a].astype('string') + '|' + orig_df[b].astype('string')
            grp_mean = orig_df.groupby(key_orig, dropna=False)[TARGET].mean()
            grp_size = orig_df.groupby(key_orig, dropna=False).size()
            name = f'{a}__{b}'
            df[f'_orig_mean__{name}'] = key_df.map(grp_mean).fillna(global_mean).astype('float32')
            df[f'_orig_size__{name}'] = key_df.map(grp_size).fillna(0).astype('float32')
    return df


def feature_engineering(df, orig_df):
    df = df.copy()
    df = add_digit_features(df)
    df = add_service_features(df)
    df = add_numeric_features(df)
    df = add_bins_and_cats(df)
    df = add_pair_features(df)
    df = add_orig_maps(df, orig_df)
    return df


def load_data():
    train = pd.read_csv(DATA_DIR / 'train.csv', dtype=STRING_DTYPES)
    test = pd.read_csv(DATA_DIR / 'test.csv', dtype=STRING_DTYPES)
    orig = pd.read_csv(DATA_DIR / ORIGINAL_FILE, dtype=STRING_DTYPES)
    print(f'train {train.shape} | test {test.shape} | original {orig.shape}', flush=True)

    for df in (train, orig):
        if df[TARGET].dtype == 'O' or str(df[TARGET].dtype).startswith('string'):
            df[TARGET] = map_target_yes_no(df[TARGET])
        df[TARGET] = df[TARGET].astype('int64')

    train = strip_string_cols(clean_charges(train))
    test = strip_string_cols(clean_charges(test))
    orig = strip_string_cols(clean_charges(orig))

    # the original rows go through the same features first (their own reference is the raw original frame);
    # the featured original frame is then the reference for train and test
    orig = feature_engineering(orig, orig)
    X = feature_engineering(train.drop([ID, TARGET], axis=1).copy(), orig)
    X_test = feature_engineering(test.drop([ID], axis=1).copy(), orig)
    X = X.drop(columns=['MonthlyCharges_str', 'TotalCharges_str'])
    X_test = X_test.drop(columns=['MonthlyCharges_str', 'TotalCharges_str'])
    y = train[TARGET].copy()

    cat_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    print(f'X {X.shape} | X_test {X_test.shape} | {len(cat_cols)} categorical, {X.shape[1] - len(cat_cols)} '
          f'continuous', flush=True)
    return X, y, X_test


# ---------------------------------------------------------------- model
def get_emb_dim(cardinality: int) -> int:
    return int(min(32, max(4, round(1.6 * (cardinality ** 0.5)))))


class LiquidNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.tau = nn.Parameter(torch.ones(hidden_dim))
        self.gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h):
        dx = F.silu(self.W(x) + self.U(h))
        g = torch.sigmoid(self.gate(torch.cat([x, h], dim=-1)))
        dh = (dx - h) / (F.softplus(self.tau) + 1e-4)
        return h + g * dh


class LiquidBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.liquid = LiquidNeuron(dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h):
        new_h = self.liquid(x, h)
        x = self.norm(x + self.dropout(self.proj(new_h)))
        return x, new_h


class LiquidEmbeddingNetwork(nn.Module):
    def __init__(self, cat_cardinalities: List[int], num_cont: int, hidden_dim: int = 256, num_layers: int = 4,
                 dropout_prob: float = 0.15, emb_dropout: float = 0.05):
        super().__init__()
        self.num_cont = num_cont
        self.hidden_dim = hidden_dim
        self.emb_dims = [get_emb_dim(c) for c in cat_cardinalities]
        self.emb_layers = nn.ModuleList([nn.Embedding(card, emb_dim)
                                         for card, emb_dim in zip(cat_cardinalities, self.emb_dims)])
        self.emb_dropout = nn.Dropout(emb_dropout)

        cont_hidden = max(64, hidden_dim // 2)
        self.cont_proj = nn.Sequential(
            nn.Linear(num_cont, cont_hidden),
            nn.SiLU(),
            nn.LayerNorm(cont_hidden),
            nn.Dropout(dropout_prob / 2),
        ) if num_cont > 0 else None

        input_dim = sum(self.emb_dims) + (cont_hidden if num_cont > 0 else 0)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob),
        )
        self.blocks = nn.ModuleList([LiquidBlock(hidden_dim, hidden_dim, dropout=dropout_prob)
                                     for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_cat, x_cont):
        feats = []
        if len(self.emb_layers) > 0:
            x_emb = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)], dim=1)
            feats.append(self.emb_dropout(x_emb))
        if self.num_cont > 0:
            feats.append(self.cont_proj(x_cont))
        x = self.input_proj(torch.cat(feats, dim=1))
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        for block in self.blocks:
            x, h = block(x, h)
        return self.head(x).squeeze(-1)


def build_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = min_lr / CFG.learning_rate
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class LNNClassifier:
    def __init__(self, random_state: int = SEED, device: str = CFG.device):
        self.random_state = random_state
        self.device = torch.device(device)

    def _preprocess_data(self, X: pd.DataFrame, y: Optional[np.ndarray], training: bool):
        X = X.copy()
        if training:
            self._non_constant_columns = X.columns[X.nunique(dropna=False) > 1].tolist()
        X = X[self._non_constant_columns].copy()

        if training:
            self._categorical_features = [col for col in X.columns
                                          if X[col].dtype.name in ['object', 'category', 'string']]
            self._continuous_features = [col for col in X.columns if col not in self._categorical_features]
        categorical_features = self._categorical_features
        continuous_features = self._continuous_features

        X_cat_df = X[categorical_features].copy() if len(categorical_features) else pd.DataFrame(index=X.index)
        X_cont_df = X[continuous_features].copy() if len(continuous_features) else pd.DataFrame(index=X.index)
        for c in categorical_features:
            X_cat_df[c] = X_cat_df[c].astype('string').fillna('missing')
        if len(continuous_features) > 0:
            if training:
                self._cont_medians = X_cont_df.median()
            X_cont_df = X_cont_df.fillna(self._cont_medians)

        if training:
            self._categorical_encoders = []
            self._cat_cardinalities = []
            for c in categorical_features:
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1)
                enc.fit(X_cat_df[[c]].astype(str).values)
                self._categorical_encoders.append(enc)
                self._cat_cardinalities.append(len(enc.categories_[0]) + 2)  # index 0 = unknown / missing

        if len(categorical_features) > 0:
            cat_arrays = []
            for i, c in enumerate(categorical_features):
                arr = self._categorical_encoders[i].transform(X_cat_df[[c]].astype(str).values).astype(np.int64).ravel()
                cat_arrays.append(np.where(arr < 0, 0, arr + 1)[:, None])
            X_cat = np.concatenate(cat_arrays, axis=1).astype(np.int64)
        else:
            X_cat = np.zeros((len(X), 0), dtype=np.int64)

        if len(continuous_features) > 0:
            X_cont = X_cont_df.to_numpy(dtype=np.float32)
            if training:
                noise = (np.random.default_rng(self.random_state).normal(0.0, CFG.noise_std, X_cont.shape)
                         .astype(np.float32))
                self._cont_feature_preprocessor = MinMaxScaler().fit(X_cont + noise)
            X_cont = self._cont_feature_preprocessor.transform(X_cont).astype(np.float32)
        else:
            X_cont = np.zeros((len(X), 0), dtype=np.float32)

        X_cat = torch.tensor(X_cat, dtype=torch.long, device=self.device)
        X_cont = torch.tensor(X_cont, dtype=torch.float32, device=self.device)
        if y is not None:
            y = torch.tensor(y.astype(np.float32), dtype=torch.float32, device=self.device)
        return X_cat, X_cont, y

    def fit(self, X: pd.DataFrame, y: np.ndarray, eval_set: Tuple[pd.DataFrame, np.ndarray]):
        X_cat_train, X_cont_train, y_train = self._preprocess_data(X, y, training=True)
        X_cat_val, X_cont_val, y_val = self._preprocess_data(eval_set[0], eval_set[1], training=False)

        self.model = LiquidEmbeddingNetwork(cat_cardinalities=self._cat_cardinalities, num_cont=X_cont_train.shape[1],
                                            hidden_dim=CFG.hidden_dim, num_layers=CFG.num_layers,
                                            dropout_prob=CFG.dropout_prob, emb_dropout=CFG.emb_dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
        steps_per_epoch = math.ceil(len(y_train) / CFG.batch_size)
        total_steps = CFG.max_epochs * steps_per_epoch
        scheduler = build_cosine_scheduler(optimizer, warmup_steps=int(total_steps * CFG.warmup_ratio),
                                           total_steps=total_steps, min_lr=CFG.min_lr)
        use_amp = self.device.type == 'cuda'
        best = {'epoch': -1, 'val_auc': -np.inf, 'state': None}
        remaining_patience = CFG.patience

        for epoch in range(CFG.max_epochs):
            self.model.train()
            train_losses = []
            for batch_idx in torch.randperm(len(y_train), device=self.device).split(CFG.batch_size):
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    logits = self.model(X_cat_train[batch_idx], X_cont_train[batch_idx]).float()
                    yb = y_train[batch_idx]
                    if CFG.label_smoothing > 0:
                        yb = yb * (1 - CFG.label_smoothing) + 0.5 * CFG.label_smoothing
                    loss = F.binary_cross_entropy_with_logits(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CFG.max_norm)
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

            self.model.eval()
            val_losses, val_probs = [], []
            with torch.no_grad():
                for start in range(0, len(y_val), CFG.batch_size):
                    end = start + CFG.batch_size
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                        logits = self.model(X_cat_val[start:end], X_cont_val[start:end]).float()
                        loss = F.binary_cross_entropy_with_logits(logits, y_val[start:end])
                        probs = torch.sigmoid(logits)
                    val_losses.append(loss.item())
                    val_probs.append(probs.detach().cpu().numpy())
            val_probs = np.concatenate(val_probs)
            val_auc = roc_auc_score(y_val.detach().cpu().numpy(), val_probs)
            print(f'Epoch {epoch + 1} | Train Loss: {float(np.mean(train_losses)):.6f} | '
                  f'Val Loss: {float(np.mean(val_losses)):.6f} | Val AUC: {val_auc:.6f}', flush=True)
            if val_auc > best['val_auc'] + 1e-6:
                best = {'epoch': epoch, 'val_auc': val_auc,
                        'state': {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}}
                remaining_patience = CFG.patience
            else:
                remaining_patience -= 1
            if remaining_patience == 0:
                print('Early stopping triggered.', flush=True)
                break

        self.model.load_state_dict(best['state'])
        print(f'Best epoch: {best["epoch"] + 1} | best val AUC: {best["val_auc"]:.6f}', flush=True)
        return self

    def predict_proba(self, X: pd.DataFrame, batch_size: Optional[int] = 8192) -> np.ndarray:
        X_cat, X_cont, _ = self._preprocess_data(X, y=None, training=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                end = start + batch_size
                logits = self.model(X_cat[start:end], X_cont[start:end]).float()
                preds.append(torch.sigmoid(logits).cpu().numpy())
        return np.clip(np.concatenate(preds), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(SEED)
    X, y, X_test = load_data()
    clean_memory()

    fold_ids = folds(y.to_numpy())
    for k in fold_list(args.folds):
        print('#' * 25, flush=True)
        print(f'### Fold {k + 1}/5 ...', flush=True)
        print('#' * 25, flush=True)
        train_idx, valid_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        model = LNNClassifier()
        model.fit(X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].to_numpy(),
                  eval_set=(X.iloc[valid_idx].reset_index(drop=True), y.iloc[valid_idx].to_numpy()))
        oof_pred = model.predict_proba(X.iloc[valid_idx].reset_index(drop=True))
        test_pred = model.predict_proba(X_test)
        print(f'\nFold {k + 1} | AUC: {roc_auc_score(y.iloc[valid_idx].to_numpy(), oof_pred):.6f}\n', flush=True)
        save_part(MODEL_DIR, PAIR, k, oof_pred, test_pred)
        del model, oof_pred, test_pred
        clean_memory()
    assemble(MODEL_DIR, PAIR, fold_ids, y.to_numpy())


if __name__ == '__main__':
    main()
