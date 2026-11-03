# Code of the public notebook "RealMLP --> 0.97014" (version 3):
# https://www.kaggle.com/code/nawfeelrahman1124444/realmlp-0-97014
# Everything before its global target encoding and CV loop (train.py replaces those). Importing this module seeds
# the RNGs (seed 63), builds the notebook's label-free features of train and test (X, X_test, X_raw, X_test_raw,
# cat_cols, target_enc_cols) and defines its RealMLP (RealMLP_TD_Classifier, CONFIG) and its target / frequency
# encoder (apply_encodings_cv). Only the two data paths were changed (data come from common.data).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.data import read_test, read_train  # noqa: E402

import math
import random
import warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

SEED = 63

def seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

seed_everything(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train = read_train()
test = read_test()

ID = 'id'
TARGET = 'addicted_label'
X = train.drop([ID, TARGET], axis=1)
train_id = train[ID]
y = train[TARGET].astype(int)
X_test = test.drop([ID], axis=1)
test_id = test[ID]
del train, test

initial_features = [c for c in X.columns if c != ID]

X_raw = X.copy()
X_test_raw = X_test.copy()

USE_LOG_FE = True
USE_ROUNDED_CAT_FE = True
USE_BIN_FE = True
USE_COMBO_FE = True
USE_COUNT_FE = False
USE_DIGIT_FE = True
USE_ACTIVITY_STAT_FE = True

EPS = 1e-6
extra_category_map = {}

def add_log_features(df):
    if not USE_LOG_FE:
        return df
    df['fe_log_weekend_screen_time'] = np.log1p(pd.to_numeric(df['weekend_screen_time'], errors='coerce').clip(lower=0)).astype('float32').fillna(0.0)
    df['fe_log_social_media_hours'] = np.log1p(pd.to_numeric(df['social_media_hours'], errors='coerce').clip(lower=0)).astype('float32').fillna(0.0)
    return df

def add_rounded_cat_features(df, fit=False):
    if not USE_ROUNDED_CAT_FE:
        return df
    round_cols = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours']
    for col in round_cols:
        name = f'{col}_rcat_'
        values = pd.to_numeric(df[col], errors='coerce').round(-1)
        raw = values.fillna(-999999).astype(str)
        if fit:
            codes, uniques = raw.factorize()
            extra_category_map[name] = uniques
        else:
            uniques = extra_category_map[name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = raw.map(code_map).fillna(-1).astype('int32')
        df[name] = codes
        df[name] = df[name].astype('category')
    return df

def add_bin_features(df, fit=False):
    if not USE_BIN_FE:
        return df
    bin_config = {
        'daily_screen_time_hours': 50,
        'social_media_hours': 40,
        'notifications_per_day': 40,
        'app_opens_per_day': 40,
        'weekend_screen_time': 40,
    }
    for col, n_bins in bin_config.items():
        name = f'{col}_{n_bins}_quantile_bin_'
        values = pd.to_numeric(df[col], errors='coerce').fillna(0.0).values.reshape(-1, 1)
        if fit:
            kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
            binned = kb.fit_transform(values).ravel().astype('int32')
            extra_category_map[name] = kb
        else:
            kb = extra_category_map[name]
            binned = kb.transform(values).ravel().astype('int32')
        df[name] = binned
        df[name] = df[name].astype('category')
    return df

def add_combo_features(df, fit=False):
    if not USE_COMBO_FE:
        return df
    combos = [('gender', 'stress_level'), ('gender', 'academic_work_impact'), ('stress_level', 'academic_work_impact')]
    for cols in combos:
        name = '_'.join(cols) + '_'
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + '_' + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            extra_category_map[name] = uniques
        else:
            uniques = extra_category_map[name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype('int32')
        df[name] = codes
        df[name] = df[name].astype('category')
    return df

digit_columns = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                  'work_study_hours', 'sleep_hours', 'weekend_screen_time',
                  'notifications_per_day', 'app_opens_per_day', 'age']

def add_digit_features(df, fit=False):
    if not USE_DIGIT_FE:
        return df
    for col in digit_columns:
        values = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float32')
        for position, multiplier in [('ones', 1), ('tenths', 10), ('hundredths', 100)]:
            df[f'fe_{col}_digit_{position}'] = (
                np.floor(np.abs(values) * multiplier).astype('Int32') % 10
            ).astype('float32')
    for col in digit_columns:
        values = pd.to_numeric(df[col], errors='coerce').astype('float32')
        for suffix, encoded in [('floorcat', np.floor(values)),
                                 ('round1cat', values.round(1) * 10),
                                 ('mod10cat', np.floor(np.abs(values)) % 10),
                                 ('frac20cat', np.floor((np.abs(values) % 1) * 20))]:
            name = f'{col}_{suffix}_'
            raw = pd.Series(encoded).fillna(-999999).astype('Int32').astype(str)
            if fit:
                codes, uniques = raw.factorize()
                extra_category_map[name] = uniques
            else:
                uniques = extra_category_map[name]
                code_map = {cat: i for i, cat in enumerate(uniques)}
                codes = raw.map(code_map).fillna(-1).astype('int32')
            df[name] = codes
            df[name] = df[name].astype('category')
    return df

activity_cols = ['daily_screen_time_hours', 'weekend_screen_time', 'social_media_hours', 'gaming_hours', 'work_study_hours']

def add_activity_stat_features(df, fit=False):
    if not USE_ACTIVITY_STAT_FE:
        return df
    activity_values = df[activity_cols].apply(lambda s: pd.to_numeric(s, errors='coerce'))
    df['fe_activity_mean'] = activity_values.mean(axis=1).astype('float32').fillna(0.0)
    df['fe_activity_std'] = activity_values.std(axis=1).astype('float32').fillna(0.0)
    df['fe_activity_min'] = activity_values.min(axis=1).astype('float32').fillna(0.0)
    df['fe_activity_max'] = activity_values.max(axis=1).astype('float32').fillna(0.0)
    df['fe_activity_range'] = (df['fe_activity_max'] - df['fe_activity_min']).astype('float32').fillna(0.0)
    raw_argmax = activity_values.fillna(-np.inf).to_numpy().argmax(axis=1).astype(str)
    name = 'activity_argmax_cat_'
    if fit:
        codes, uniques = pd.factorize(pd.Series(raw_argmax), sort=False)
        extra_category_map[name] = uniques
    else:
        uniques = extra_category_map[name]
        code_map = {cat: i for i, cat in enumerate(uniques)}
        codes = pd.Series(raw_argmax).map(code_map).fillna(-1).astype('int32')
    df[name] = codes
    df[name] = df[name].astype('category')
    return df

def add_count_features(df, cat_col_list, fit=False):
    if not USE_COUNT_FE:
        return df
    for col in cat_col_list:
        name = f'_{col}_count'
        if fit:
            count_map = df[col].astype(object).value_counts()
            extra_category_map[name] = count_map
        else:
            count_map = extra_category_map[name]
        df[name] = df[col].astype(object).map(count_map).fillna(0).astype('int32')
    return df

category_map = {}
for col in initial_features:
    X[col] = X[col].astype("str")
    X_test[col] = X_test[col].astype("str")
    codes, uniques = X[col].factorize()
    category_map[col] = uniques
    code_map = {cat: i for i, cat in enumerate(uniques)}
    X[col] = codes
    X_test[col] = X_test[col].map(code_map).fillna(-1).astype('int32')
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

def engineer_features(df):
    df = df.copy()

    df["missing_count"] = df[initial_features].apply(
        lambda row: row.astype(str).isin(["nan", "None", "<NA>"]).sum(), axis=1
    ).astype(float)

    def to_float(col_name):
        if col_name in df.columns:
            return pd.to_numeric(df[col_name].astype(str), errors="coerce")
        return pd.Series(np.nan, index=df.index)

    s_media = to_float("social_media_hours")
    gaming = to_float("gaming_hours")
    work = to_float("work_study_hours")
    daily_screen = to_float("daily_screen_time_hours")
    sleep = to_float("sleep_hours")
    weekend_screen = to_float("weekend_screen_time")
    app_opens = to_float("app_opens_per_day")
    notifications = to_float("notifications_per_day")

    df["total_breakdown_hours"] = s_media.fillna(0.0) + gaming.fillna(0.0) + work.fillna(0.0)
    df["social_ratio"] = s_media / (daily_screen + 1e-5)
    df["gaming_ratio"] = gaming / (daily_screen + 1e-5)
    df["work_ratio"] = work / (daily_screen + 1e-5)
    df["unaccounted_screen_time"] = daily_screen - df["total_breakdown_hours"]
    df["screen_to_sleep_ratio"] = daily_screen / (sleep + 1e-5)
    df["weekend_vs_daily_ratio"] = weekend_screen / (daily_screen + 1e-5)
    df["app_opens_per_hour"] = app_opens / (daily_screen + 1e-5)
    df["notifications_per_hour"] = notifications / (daily_screen + 1e-5)

    engineered_cont_cols = [
        "total_breakdown_hours", "social_ratio", "gaming_ratio", "work_ratio",
        "unaccounted_screen_time", "screen_to_sleep_ratio", "weekend_vs_daily_ratio",
        "app_opens_per_hour", "notifications_per_hour"
    ]
    df[engineered_cont_cols] = df[engineered_cont_cols].fillna(0.0)

    return df

X_raw = engineer_features(X_raw)
X_test_raw = engineer_features(X_test_raw)

engineered_cont_cols = [
    "missing_count", "total_breakdown_hours", "social_ratio", "gaming_ratio", "work_ratio",
    "unaccounted_screen_time", "screen_to_sleep_ratio", "weekend_vs_daily_ratio",
    "app_opens_per_hour", "notifications_per_hour"
]

for col in engineered_cont_cols:
    X[col] = X_raw[col].values
    X_test[col] = X_test_raw[col].values

cat_cols = list(initial_features)
num_cols = list(engineered_cont_cols)

X_raw = add_log_features(X_raw)
X_test_raw = add_log_features(X_test_raw)
X_raw = add_rounded_cat_features(X_raw, fit=True)
X_test_raw = add_rounded_cat_features(X_test_raw, fit=False)
X_raw = add_bin_features(X_raw, fit=True)
X_test_raw = add_bin_features(X_test_raw, fit=False)
X_raw = add_combo_features(X_raw, fit=True)
X_test_raw = add_combo_features(X_test_raw, fit=False)
X_raw = add_digit_features(X_raw, fit=True)
X_test_raw = add_digit_features(X_test_raw, fit=False)
X_raw = add_activity_stat_features(X_raw, fit=True)
X_test_raw = add_activity_stat_features(X_test_raw, fit=False)

new_num_cols = [c for c in X_raw.columns if c.startswith('fe_')]
new_cat_cols = [c for c in X_raw.columns if c.endswith('_') and c not in initial_features]

for col in new_num_cols:
    X[col] = X_raw[col].astype('float32').values
    X_test[col] = X_test_raw[col].astype('float32').values
    num_cols.append(col)

for col in new_cat_cols:
    codes_tr, uniques = X_raw[col].astype(str).factorize()
    code_map = {cat: i for i, cat in enumerate(uniques)}
    X[col] = codes_tr
    X_test[col] = X_test_raw[col].astype(str).map(code_map).fillna(-1).astype('int32')
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')
    cat_cols.append(col)

X_raw = add_count_features(X_raw, cat_cols, fit=True)
X_test_raw = add_count_features(X_test_raw, cat_cols, fit=False)
count_cols = [c for c in X_raw.columns if c.startswith('_') and c.endswith('_count')]
target_enc_cols = X_raw.columns
for col in count_cols:
    X[col] = X_raw[col].astype('float32').values
    X_test[col] = X_test_raw[col].astype('float32').values
    num_cols.append(col)

cat_cols = sorted(list(dict.fromkeys(cat_cols)))
num_cols = sorted(list(dict.fromkeys(num_cols)))

X = X.reindex(sorted(X.columns), axis=1)
X_test = X_test.reindex(sorted(X_test.columns), axis=1)

class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tfms):
        self._tfms = [t for t in tfms if t in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")]

    def fit(self, X, y=None):
        if "median_center" in self._tfms or "robust_scale" in self._tfms:
            self._median = np.median(X, axis=0)
            q_diff = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            zero_idx = q_diff == 0.0
            q_diff[zero_idx] = 0.5 * (X.max(axis=0)[zero_idx] - X.min(axis=0)[zero_idx])
            self._iqr_factors = 1.0 / (q_diff + 1e-30)
            self._iqr_factors[q_diff == 0.0] = 0.0
        return self

    def transform(self, X, y=None):
        X = X.copy().astype(np.float32)
        for tfm in self._tfms:
            if tfm == "median_center":
                X -= self._median[None, :]
            elif tfm == "robust_scale":
                X *= self._iqr_factors[None, :]
            elif tfm == "smooth_clip":
                X = X / np.sqrt(1 + (X / 3) ** 2)
            elif tfm == "l2_normalize":
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(norms == 0, 1.0, norms)
        return X

class CategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens, cat_dims, embed_dim=8, onehot_thresh=8):
        super().__init__()
        self.n_ens = n_ens
        self.cat_dims = cat_dims
        self.onehot_features = []
        self.embed_layers = nn.ModuleList()
        self._embed_feature_indices = []
        for i, dim in enumerate(cat_dims):
            if dim <= onehot_thresh:
                self.onehot_features.append(i)
            else:
                emb = nn.ModuleList([nn.Embedding(dim, embed_dim) for _ in range(n_ens)])
                self.embed_layers.append(emb)
                self._embed_feature_indices.append(i)

    def forward(self, x):
        batch_size, n_ens, _ = x.shape
        features = []
        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            total_oh = sum(onehot_dims)
            encoded = torch.zeros(batch_size, n_ens, total_oh, device=x.device)
            start = 0
            for idx, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, idx:idx+1].long()
                encoded.scatter_(2, pos + start, 1.0)
                start += dim
            features.append(encoded)
        for emb_list, feat_idx in zip(self.embed_layers, self._embed_feature_indices):
            feat_embs = []
            for model_idx in range(self.n_ens):
                indices = x[:, model_idx, feat_idx:feat_idx+1].long()
                feat_embs.append(emb_list[model_idx](indices))
            feat_combined = torch.cat(feat_embs, dim=1)
            features.append(feat_combined)
        return torch.cat(features, dim=2)

class ScalingLayer(nn.Module):
    def __init__(self, n_ens, n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x):
        return x * self.scale[None, :, :]

class NTPLinear(nn.Module):
    def __init__(self, n_ens, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

    def forward(self, x):
        x = torch.einsum("bki,kio->bko", x, self.weight) / math.sqrt(self.in_features)
        if self.bias is not None:
            x = x + self.bias
        return x

class PBLDEmbedding(nn.Module):
    def __init__(self, n_ens, n_features, hidden_dim=16, out_dim=4, freq_scale=0.1, activation=nn.GELU):
        super().__init__()
        self.n_ens = n_ens
        self.n_features = n_features
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim - 1))
        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)

    def forward(self, x):
        periodic = torch.cos(2 * math.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
        transformed = self.act(torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2) + self.b2.unsqueeze(0))
        feat = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        return feat.flatten(start_dim=2)

class RealMLP(nn.Module):
    def __init__(self, output_dim, cat_dims, n_numerical, cfg):
        super().__init__()
        n_ens = cfg["n_ens"]
        embed_dim = cfg["embed_dim"]
        self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(n_ens=n_ens, cat_dims=cat_dims, embed_dim=embed_dim, onehot_thresh=cfg["onehot_thresh"])
        self.num_embed = PBLDEmbedding(n_ens=n_ens, n_features=n_numerical, hidden_dim=cfg["pbld_hidden_dim"],
                                        out_dim=cfg["pbld_out_dim"], freq_scale=cfg["pbld_freq_scale"], activation=cfg["pbld_activation"])
        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(c if c <= cfg["onehot_thresh"] else embed_dim for c in cat_dims)
        total_dim = num_emb_dim + cat_emb_dim
        act = cfg["activation"]
        layers = []
        if cfg["add_front_scale"]:
            layers.append(ScalingLayer(n_ens=n_ens, n_features=total_dim))
        self._dropout_modules = []
        in_dim = total_dim
        for i, out_dim_h in enumerate(cfg["hidden_dims"]):
            linear = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=out_dim_h)
            if i == 0:
                self.first_linear = linear
            drop = nn.Dropout(cfg["dropout"])
            self._dropout_modules.append(drop)
            layers += [linear, act(), drop]
            in_dim = out_dim_h
        self.hidden = nn.Sequential(*layers)
        self.output_layer = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=output_dim)

    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x_cat = self.cate(x_cat)
        x = self.hidden(torch.cat([x_num, x_cat], dim=2))
        return F.softmax(self.output_layer(x), dim=2)

def apply_schedule(init_value, progress, sched, flat_ratio=0.3):
    if sched == "constant":
        return init_value
    elif sched == "cos":
        return init_value * (math.cos(math.pi * progress) + 1) / 2
    elif sched == "flat_cos":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (math.cos(math.pi * t) + 1) / 2
    elif sched == "flat_anneal":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (1 - t)
    elif sched == "sqrt_cos":
        return init_value * math.sqrt((math.cos(math.pi * progress) + 1) / 2)
    elif sched == "expm4t":
        return init_value * math.exp(-4 * progress)
    else:
        raise ValueError(f"Unknown schedule: '{sched}'")

def get_parameter_groups(model, p):
    first_linear_weight_id = id(model.first_linear.weight)
    scale_p, pbld_p, first_w_p, other_w_p, bias_p = [], [], [], [], []
    for name, param in model.named_parameters():
        if "num_embed" in name:
            pbld_p.append(param)
        elif "scale" in name:
            scale_p.append(param)
        elif id(param) == first_linear_weight_id:
            first_w_p.append(param)
        elif "bias" in name:
            bias_p.append(param)
        else:
            other_w_p.append(param)
    LR = p["lr"]
    WD = p["weight_decay"]
    return [
        {"params": scale_p, "lr": LR * p["lr_scale_mult"], "weight_decay": WD * p["wd_scale_mult"], "group": "scale"},
        {"params": pbld_p, "lr": LR * p["pbld_lr_factor"], "weight_decay": WD, "group": "pbld"},
        {"params": first_w_p, "lr": LR * p["first_layer_lr_factor"], "weight_decay": WD * p["first_layer_wd_factor"], "group": "first_w"},
        {"params": other_w_p, "lr": LR, "weight_decay": WD, "group": "other_w"},
        {"params": bias_p, "lr": LR * p["lr_bias_mult"], "weight_decay": WD * p["wd_bias_mult"], "group": "bias"},
    ]

def smooth_ce_loss(y_true, y_pred, ls=0.0, class_weights=None):
    n_classes = y_pred.size(1)
    y_smooth = torch.full_like(y_pred, ls / n_classes)
    y_smooth.scatter_(1, y_true.unsqueeze(1), 1.0 - ls + ls / n_classes)
    per_sample_loss = -(y_smooth * torch.log(y_pred.clamp(1e-15, 1))).sum(dim=1)
    if class_weights is not None:
        sample_weights = class_weights[y_true]
        return (per_sample_loss * sample_weights).sum() / sample_weights.sum()
    return per_sample_loss.mean()

class RealMLP_TD_Classifier(BaseEstimator):
    def __init__(self, **kwargs):
        self.params = {**CONFIG, **kwargs}

    def fit(self, X_train, y_train, X_val, y_val, cat_col_names=None, X_test=None):
        p = self.params
        dev = torch.device(p["device"] if torch.cuda.is_available() else "cpu")
        verbose = p["verbosity"]
        cat_col_names = cat_col_names or []
        num_col_names = [c for c in X_train.columns if c not in cat_col_names]

        X_tr_num = X_train[num_col_names].values.astype(np.float32)
        X_val_num = X_val[num_col_names].values.astype(np.float32)
        X_tr_cat = X_train[cat_col_names].values.astype(np.int64)
        X_val_cat = X_val[cat_col_names].values.astype(np.int64)
        y_tr = np.asarray(y_train)
        y_v = np.asarray(y_val)

        self.preprocessor_ = NumericalPreprocessor(p["tfms"])
        self.preprocessor_.fit(X_tr_num)
        X_tr_num = self.preprocessor_.transform(X_tr_num)
        X_val_num = self.preprocessor_.transform(X_val_num)

        self.cat_col_names_ = cat_col_names
        self.num_col_names_ = num_col_names
        if cat_col_names:
            X_tr_cat = X_tr_cat + 1
            X_val_cat = X_val_cat + 1
            X_tr_cat = np.where(X_tr_cat < 1, 0, X_tr_cat)
            X_val_cat = np.where(X_val_cat < 1, 0, X_val_cat)
            all_cat = [X_tr_cat, X_val_cat]
            if X_test is not None:
                X_test_cat = X_test[cat_col_names].values.astype(np.int64) + 1
                X_test_cat = np.where(X_test_cat < 1, 0, X_test_cat)
                all_cat.append(X_test_cat)
            cat_dims = (np.concatenate(all_cat, axis=0).max(axis=0) + 1).tolist()
        else:
            cat_dims = []
        self.cat_dims_ = cat_dims

        if cat_dims:
            cat_max = np.array(cat_dims) - 1
            X_tr_cat = np.clip(X_tr_cat, 0, cat_max)
            X_val_cat = np.clip(X_val_cat, 0, cat_max)

        classes = np.unique(y_tr)
        self.classes_ = classes
        weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)
        n_classes = len(classes)

        self.model_ = RealMLP(output_dim=n_classes, cat_dims=cat_dims, n_numerical=X_tr_num.shape[1], cfg=p).to(dev)

        param_groups = get_parameter_groups(self.model_, p)
        for g in param_groups:
            g["lr_base"] = g["lr"]
        optimizer = torch.optim.AdamW(param_groups, betas=(p["mom"], p["sq_mom"]))

        Xtn = torch.as_tensor(X_tr_num, dtype=torch.float32, device=dev)
        Xtc = torch.as_tensor(X_tr_cat, dtype=torch.long, device=dev)
        ytt = torch.as_tensor(y_tr, dtype=torch.long, device=dev)
        Xvn = torch.as_tensor(X_val_num, dtype=torch.float32, device=dev)
        Xvc = torch.as_tensor(X_val_cat, dtype=torch.long, device=dev)

        n_ens = p["n_ens"]
        train_bs = p["train_bs"]
        eval_bs = p["eval_bs"]
        epochs = p["epochs"]
        lr_sched = p["lr_sched"]
        flat_ratio = p["flat_ratio"]
        ema_decay = p["ema_decay"]
        total_steps = epochs * len(y_tr)
        train_order = np.arange(len(y_tr))

        best_score = -np.inf
        best_epoch = 0
        best_val_probs = None
        best_state = None
        ema_state = None
        if ema_decay > 0:
            ema_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}

        for epoch in range(epochs):
            self.model_.train()
            for start in range(0, len(y_tr), train_bs):
                progress = (epoch * len(y_tr) + start) / total_steps
                idx_batch = train_order[start:start + train_bs]
                for g in optimizer.param_groups:
                    g["lr"] = apply_schedule(g["lr_base"], progress, lr_sched, flat_ratio)
                optimizer.zero_grad()
                y_pred = self.model_(Xtn[idx_batch], Xtc[idx_batch])
                ls_val = apply_schedule(p["ls_eps"], progress, p["ls_eps_sched"], flat_ratio)
                drop_val = apply_schedule(p["dropout"], progress, p["p_drop_sched"], flat_ratio)
                for dm in self.model_._dropout_modules:
                    dm.p = drop_val
                loss = smooth_ce_loss(
                    ytt[idx_batch].repeat_interleave(n_ens),
                    y_pred.reshape(-1, n_classes),
                    ls=ls_val, class_weights=class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), p["grad_clip"])
                optimizer.step()
                if ema_state is not None:
                    with torch.no_grad():
                        for key, value in self.model_.state_dict().items():
                            if torch.is_floating_point(value):
                                ema_state[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
                            else:
                                ema_state[key].copy_(value)
            np.random.shuffle(train_order)

            live_state = None
            if ema_state is not None:
                live_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}

            self.model_.eval()
            if ema_state is not None:
                self.model_.load_state_dict(ema_state, strict=True)
            with torch.no_grad():
                val_probs = np.concatenate([
                    self.model_(Xvn[s:s + eval_bs], Xvc[s:s + eval_bs]).mean(dim=1).cpu().numpy()
                    for s in range(0, len(y_v), eval_bs)
                ], axis=0)
            epoch_score = roc_auc_score(y_v, val_probs[:, 1])
            improved = epoch_score > best_score
            if improved:
                best_score = epoch_score
                best_epoch = epoch + 1
                best_val_probs = val_probs.copy()
                state_src = ema_state if ema_state is not None else self.model_.state_dict()
                best_state = {k: v.detach().clone() for k, v in state_src.items()}
            if live_state is not None:
                self.model_.load_state_dict(live_state, strict=True)
            if verbose >= 2:
                print(f"  epoch {epoch + 1}/{epochs}  auc = {epoch_score:.5f}  best = {best_score:.5f}  ls = {ls_val:.4f}  drop = {drop_val:.4f}" + (" v" if improved else ""))
            if p["use_early_stopping"]:
                patience = (best_epoch * p["early_stopping_multiplicative_patience"] + p["early_stopping_additive_patience"])
                if (epoch + 1) > patience:
                    if verbose >= 1:
                        print(f"  Early stopping at epoch {epoch + 1} (best epoch {best_epoch})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state, strict=True)
        self.best_score_ = best_score
        self.best_val_probs_ = best_val_probs
        self._dev = dev
        if verbose >= 1:
            print(f"  best AUC: {best_score:.5f}  (epoch {best_epoch})")
        return self

    def predict_proba(self, X):
        eval_bs = self.params["eval_bs"]
        X_num = self.preprocessor_.transform(X[self.num_col_names_].values.astype(np.float32))
        X_cat = X[self.cat_col_names_].values.astype(np.int64) + 1
        X_cat = np.where(X_cat < 1, 0, X_cat)
        X_cat = np.clip(X_cat, 0, np.array(self.cat_dims_) - 1)
        Xn = torch.as_tensor(X_num, dtype=torch.float32, device=self._dev)
        Xc = torch.as_tensor(X_cat, dtype=torch.long, device=self._dev)
        self.model_.eval()
        with torch.no_grad():
            return np.concatenate([
                self.model_(Xn[s:s + eval_bs], Xc[s:s + eval_bs]).mean(dim=1).cpu().numpy()
                for s in range(0, len(X_num), eval_bs)
            ], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

CONFIG = {
    "n_ens": 16,
    "embed_dim": 16,
    "onehot_thresh": 8,
    "hidden_dims": [512, 512, 512],
    "dropout": 0.06,
    "p_drop_sched": "expm4t",
    "activation": nn.GELU,
    "add_front_scale": True,
    "pbld_hidden_dim": 20,
    "pbld_out_dim": 5,
    "pbld_freq_scale": 5.0,
    "pbld_activation": nn.PReLU,
    "pbld_lr_factor": 0.093,
    "lr": 0.01,
    "mom": 0.9,
    "sq_mom": 0.98,
    "lr_sched": "flat_cos",
    "flat_ratio": 0.35,
    "first_layer_lr_factor": 1.0,
    "first_layer_wd_factor": 0.1,
    "lr_scale_mult": 10.0,
    "lr_bias_mult": 0.1,
    "weight_decay": 0.013,
    "wd_scale_mult": 0.1,
    "wd_bias_mult": 0.5,
    "ema_decay": 0.997875,
    "grad_clip": 1.2,
    "ls_eps": 0.04,
    "ls_eps_sched": "cos",
    "tfms": ["median_center", "robust_scale"],
    "epochs": 4,
    "train_bs": 256,
    "eval_bs": 10240,
    "verbosity": 2,
    "use_early_stopping": False,
    "early_stopping_additive_patience": 10,
    "early_stopping_multiplicative_patience": 1,
    "device": "cuda",
    "random_state": 63,
}

FOLDS = 5
TE = True

USE_TE_MEAN = True
USE_TE_FREQ = True
USE_TE_MEDIAN = False
USE_TE_MIN = False
USE_TE_MAX = False
USE_TE_STD = False
USE_TE_SKEW = False

def apply_encodings_cv(train_df, test_df, cat_cols, target_col, n_splits=5, smoothing=10, random_state=42):
    skf_enc = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    stat_suffixes = []
    if USE_TE_MEAN:
        stat_suffixes.append("te")
    if USE_TE_FREQ:
        stat_suffixes.append("freq")
    if USE_TE_MEDIAN:
        stat_suffixes.append("te_median")
    if USE_TE_MIN:
        stat_suffixes.append("te_min")
    if USE_TE_MAX:
        stat_suffixes.append("te_max")
    if USE_TE_STD:
        stat_suffixes.append("te_std")
    if USE_TE_SKEW:
        stat_suffixes.append("te_skew")

    for col in cat_cols:
        for suffix in stat_suffixes:
            train_encoded[f"{col}_{suffix}"] = 0.0
            test_encoded[f"{col}_{suffix}"] = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf_enc.split(train_df, target_col)):
        tr_subset = train_df.iloc[train_idx]
        val_subset = train_df.iloc[val_idx]
        tr_target = target_col.iloc[train_idx]
        global_mean = float(tr_target.mean())
        global_median = float(tr_target.median())
        global_min = float(tr_target.min())
        global_max = float(tr_target.max())
        global_std = float(tr_target.std())
        global_skew = float(tr_target.skew())

        for col in cat_cols:
            if USE_TE_FREQ:
                counts = tr_subset[col].value_counts()
                val_freq = val_subset[col].map(counts).astype(float).fillna(0.0).values
                train_encoded.iloc[val_idx, train_encoded.columns.get_loc(f"{col}_freq")] = val_freq
                test_freq = test_df[col].map(counts).astype(float).fillna(0.0).values
                test_encoded[f"{col}_freq"] += test_freq / n_splits

            stats_df = pd.DataFrame({col: tr_subset[col].values, "_target": tr_target.values})
            agg_funcs = []
            if USE_TE_MEAN or True:
                agg_funcs.append("count")
                agg_funcs.append("mean")
            if USE_TE_MEDIAN:
                agg_funcs.append("median")
            if USE_TE_MIN:
                agg_funcs.append("min")
            if USE_TE_MAX:
                agg_funcs.append("max")
            if USE_TE_STD:
                agg_funcs.append("std")
            if USE_TE_SKEW:
                agg_funcs.append("skew")
            stats = stats_df.groupby(col, observed=False)["_target"].agg(agg_funcs)

            if USE_TE_MEAN:
                smoothed_te = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
                val_te = val_subset[col].map(smoothed_te).astype(float).fillna(global_mean).values
                train_encoded.iloc[val_idx, train_encoded.columns.get_loc(f"{col}_te")] = val_te
                test_te = test_df[col].map(smoothed_te).astype(float).fillna(global_mean).values
                test_encoded[f"{col}_te"] += test_te / n_splits

            if USE_TE_MEDIAN:
                val_v = val_subset[col].map(stats["median"]).astype(float).fillna(global_median).values
                train_encoded.iloc[val_idx, train_encoded.columns.get_loc(f"{col}_te_median")] = val_v
                test_v = test_df[col].map(stats["median"]).astype(float).fillna(global_median).values
                test_encoded[f"{col}_te_median"] += test_v / n_splits

            if USE_TE_MIN:
                val_v = val_subset[col].map(stats["min"]).astype(float).fillna(global_min).values
                train_encoded.iloc[val_idx, train_encoded.columns.get_loc(f"{col}_te_min")] = val_v
                test_v = test_df[col].map(stats["min"]).astype(float).fillna(global_min).values
                test_encoded[f"{col}_te_min"] += test_v / n_splits

            if USE_TE_MAX:
                val_v = val_subset[col].map(stats["max"]).astype(float).fillna(global_max).values
                train_encoded.iloc[val_idx, train_encoded.columns.get_loc(f"{col}_te_max")] = val_v
                test_v = test_df[col].map(stats["max"]).astype(float).fillna(global_max).values
                test_encoded[f"{col}_te_max"] += test_v / n_splits

            if USE_TE_STD:
                val_v = val_subset[col].map(stats["std"]).astype(float).fillna(global_std).values
                train_encoded.iloc[val_idx, train_encoded.columns.get_loc(f"{col}_te_std")] = val_v
                test_v = test_df[col].map(stats["std"]).astype(float).fillna(global_std).values
                test_encoded[f"{col}_te_std"] += test_v / n_splits

            if USE_TE_SKEW:
                val_v = val_subset[col].map(stats["skew"]).astype(float).fillna(global_skew).values
                train_encoded.iloc[val_idx, train_encoded.columns.get_loc(f"{col}_te_skew")] = val_v
                test_v = test_df[col].map(stats["skew"]).astype(float).fillna(global_skew).values
                test_encoded[f"{col}_te_skew"] += test_v / n_splits

    return train_encoded, test_encoded
