# Code of the public notebook "RealMLP for Predicting Smartphone Addiction" (version 1), cells 6-22, unchanged:
# https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction
# Importing this module seeds the RNGs (seed 42) and defines CONFIG, the feature engineering, the RealMLP and
# load_and_preprocess; the notebook's main() (cell 24) is replaced by train.py.
# notebook cell 6
import math
import random
import warnings
import time
import sys
import io
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')
# GPU
if not torch.cuda.is_available():
    torch.set_num_threads(max(1, torch.get_num_threads()))

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    # --- Model architecture ---
    "n_ens":            8,                
    "embed_dim":        8,
    "onehot_thresh":    8,
    "hidden_dims":      [512, 512, 512],  # Same as PDF
    "dropout":          0.06,
    "p_drop_sched":     "expm4t",
    "activation":       nn.SiLU,
    "add_front_scale":  True,

    # --- PBLD (periodic) embedding for numericals 
    "pbld_hidden_dim":  20,              
    "pbld_out_dim":     5,
    "pbld_freq_scale":  5.0,
    "pbld_activation":  nn.PReLU,
    "pbld_lr_factor":   0.093,

    # --- Optimizer ---
    "lr":               0.01,            
    "mom":              0.9,
    "sq_mom":           0.98,
    "lr_sched":         "flat_cos",
    "flat_ratio":       0.3,
    "first_layer_lr_factor": 1.0,
    "first_layer_wd_factor": 0.1,
    "lr_scale_mult":    10.0,
    "lr_bias_mult":     0.1,
    "weight_decay":     0.013,            
    "wd_scale_mult":    0.1,
    "wd_bias_mult":     0.5,
    "ema_decay":        0.997875,
    "grad_clip":        1.2,

    # --- Label smoothing ---
    "ls_eps":           0.04,
    "ls_eps_sched":     "cos",

    # --- Preprocessing ---
    "tfms":             ["median_center", "robust_scale"],

    # --- Training loop ---
    "epochs":           8,               # GPU: more epochs for better convergence
    "train_bs":         256,              # GPU: larger batch size (RTX 5060 8GB VRAM)
    "eval_bs":          10240,            # GPU: larger eval batch
    "verbosity":        2,

    # --- Early stopping ---
    "use_early_stopping":                  False,
    "early_stopping_additive_patience":    10,
    "early_stopping_multiplicative_patience": 1,

    # --- Device ---
    "device":           "cuda",           # GPU/CUDA
    "random_state":     42,
}

FOLDS = 5   
SEED = 42


def seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


seed_everything(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True   # Auto-tune cuDNN for fixed input sizes
    print(f"GPU detected: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CUDA version: {torch.version.cuda}", flush=True)
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)



# notebook cell 8
# Feature Engineering 
# ============================================================
def feature_engineering(df, cat_cols, num_cols, category_map=None,
                        fit=False, median_values=None):
    """Feature engineering adapted from PDF template.

    Key differences from naive PDF copy:
    1. Missing indicators added BEFORE fill (captures missingness pattern)
    2. Median fill instead of 0.0 (preserves feature distribution)
    3. Selective binning: only 2 features, not all 9
    """
    if category_map is None:
        category_map = {}

    # ---- STEP 1: Missing indicators (BEFORE filling NaN) ----
    # Add binary missing indicators as categorical features.
    # These capture the missingness pattern, which is predictive
    # (4-19% missing per feature in this dataset).
    new_cat_cols = []
    for col in num_cols:
        miss_name = f"_miss_{col}"
        df[miss_name] = df[col].isnull().astype('int32')
        new_cat_cols.append(miss_name)

    # ---- STEP 2: Fill NaNs ----
    # Categoricals: fill with "missing" 
    for col in cat_cols:
        df[col] = df[col].fillna("missing")

    # Numericals: fill with MEDIAN (FIX: PDF uses 0.0 which creates outliers)
    for col in num_cols:
        if median_values and col in median_values:
            fill_val = median_values[col]
        else:
            fill_val = df[col].median()
        df[col] = df[col].fillna(fill_val)

    # ---- STEP 3: Factorize string categoricals ----
    for col in cat_cols:
        if fit:
            codes, uniques = df[col].factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = df[col].map(code_map).fillna(-1).astype('int32')
        df[col] = codes.astype('int32')

    # ---- STEP 4: Create _cat_ columns from ALL numericals  ----
    # With median fill, _cat_ correctly groups similar values.
    # With 0.0 fill (old code), all NaN got same code = meaningless.
    for col in num_cols:
        cat_name = f"{col}_cat_"
        if fit:
            codes, uniques = df[col].factorize()
            category_map[cat_name] = uniques
        else:
            uniques = category_map[cat_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = df[col].map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes.astype('int32')
        new_cat_cols.append(cat_name)

    # ---- STEP 5: Selective binning  ----
    # PDF: bin_config = {'exercise_duration': [10], 'water_intake': [10]}
    # We bin the 2 most addiction-relevant continuous features.
    bin_config = {
        'daily_screen_time_hours': [10],
        'social_media_hours': [10],
    }
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            for strategy in ['quantile']:
                bin_name = f"{col}_{n_bins}_{strategy}_bin_"
                if fit:
                    kb = KBinsDiscretizer(
                        n_bins=n_bins, encode='ordinal',
                        strategy=strategy, subsample=None
                    )
                    binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype('int32')
                df[bin_name] = binned
                new_cat_cols.append(bin_name)

    return df, new_cat_cols



# notebook cell 15
# Model Components 
# ============================================================
class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tfms):
        self._tfms = [t for t in tfms
                      if t in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")]

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
    def __init__(self, n_ens, cat_dims, embed_dim=8, onehot_thresh=8, device=None):
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
                emb = nn.ModuleList(
                    [nn.Embedding(dim, embed_dim) for _ in range(n_ens)]
                )
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
                pos = onehot_x[:, :, idx: idx + 1].long()
                encoded.scatter_(2, pos + start, 1.0)
                start += dim
            features.append(encoded)
        for emb_list, feat_idx in zip(self.embed_layers, self._embed_feature_indices):
            feat_embs = []
            for model_idx in range(self.n_ens):
                indices = x[:, model_idx, feat_idx: feat_idx + 1].long()
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
    """Periodic Basis with Learned Decay embedding for numerical features."""
    def __init__(self, n_ens, n_features, hidden_dim=16, out_dim=4,
                 freq_scale=0.1, activation=nn.GELU):
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
        periodic = torch.cos(
            2 * math.pi * (
                x.unsqueeze(-1) * self.w1.unsqueeze(0)
                + self.b1.unsqueeze(0)
            )
        )
        transformed = self.act(
            torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2)
            + self.b2.unsqueeze(0)
        )
        feat = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        return feat.flatten(start_dim=2)


class RealMLP(nn.Module):
    def __init__(self, output_dim, cat_dims, n_numerical, cfg):
        super().__init__()
        n_ens = cfg["n_ens"]
        embed_dim = cfg["embed_dim"]
        self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(
            n_ens=n_ens, cat_dims=cat_dims, embed_dim=embed_dim,
            onehot_thresh=cfg["onehot_thresh"],
        )
        self.num_embed = PBLDEmbedding(
            n_ens=n_ens,
            n_features=n_numerical,
            hidden_dim=cfg["pbld_hidden_dim"],
            out_dim=cfg["pbld_out_dim"],
            freq_scale=cfg["pbld_freq_scale"],
            activation=cfg["pbld_activation"],
        )
        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(
            c if c <= cfg["onehot_thresh"] else embed_dim for c in cat_dims
        )
        total_dim = num_emb_dim + cat_emb_dim
        hidden_dims = cfg["hidden_dims"]
        act = cfg["activation"]

        layers = []
        if cfg["add_front_scale"]:
            layers.append(ScalingLayer(n_ens=n_ens, n_features=total_dim))
        self._dropout_modules = []
        in_dim = total_dim
        for i, out_dim_h in enumerate(hidden_dims):
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
        combined = torch.cat([x_num, x_cat], dim=2)
        x = self.hidden(combined)
        x = self.output_layer(x)
        return F.softmax(x, dim=2)



# notebook cell 20
# Schedule helpers 
# ============================================================
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


# ============================================================
# Parameter groups 
# ============================================================
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
        {"params": scale_p,   "lr": LR * p["lr_scale_mult"],          "weight_decay": WD * p["wd_scale_mult"]},
        {"params": pbld_p,    "lr": LR * p["pbld_lr_factor"],          "weight_decay": WD},
        {"params": first_w_p, "lr": LR * p["first_layer_lr_factor"],   "weight_decay": WD * p["first_layer_wd_factor"]},
        {"params": other_w_p, "lr": LR,                                "weight_decay": WD},
        {"params": bias_p,    "lr": LR * p["lr_bias_mult"],            "weight_decay": WD * p["wd_bias_mult"]},
    ]


# ============================================================
# Label-smoothed cross-entropy 
# ============================================================
def smooth_ce_loss(y_true, y_pred, ls=0.0, class_weights=None):
    n_classes = y_pred.size(1)
    y_smooth = torch.full_like(y_pred, ls / n_classes)
    y_smooth.scatter_(1, y_true.unsqueeze(1), 1.0 - ls + ls / n_classes)
    per_sample_loss = -(y_smooth * torch.log(y_pred.clamp(1e-15, 1))).sum(dim=1)
    if class_weights is not None:
        sample_weights = class_weights[y_true]
        return (per_sample_loss * sample_weights).sum() / sample_weights.sum()
    return per_sample_loss.mean()


# ============================================================
# Sklearn-compatible wrapper (adapted for AUC metric)
# ============================================================
class RealMLP_TD_Classifier(BaseEstimator):
    def __init__(self, **kwargs):
        self.params = {**CONFIG, **kwargs}

    def fit(self, X_train, y_train, X_val, y_val, cat_col_names=None, X_test=None):
        p = self.params
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        verbose = p["verbosity"]
        cat_col_names = cat_col_names or []
        num_col_names = [c for c in X_train.columns if c not in cat_col_names]

        X_tr_num = X_train[num_col_names].values.astype(np.float32)
        X_val_num = X_val[num_col_names].values.astype(np.float32)
        X_tr_cat = X_train[cat_col_names].values.astype(np.int64)
        X_val_cat = X_val[cat_col_names].values.astype(np.int64)
        y_tr = np.asarray(y_train)
        y_v = np.asarray(y_val)

        # Numerical preprocessing
        self.preprocessor_ = NumericalPreprocessor(p["tfms"])
        self.preprocessor_.fit(X_tr_num)
        X_tr_num = self.preprocessor_.transform(X_tr_num)
        X_val_num = self.preprocessor_.transform(X_val_num)

        # Categorical dims
        self.cat_col_names_ = cat_col_names
        self.num_col_names_ = num_col_names
        if cat_col_names:
            all_cat = [X_tr_cat, X_val_cat]
            if X_test is not None:
                all_cat.append(X_test[cat_col_names].values.astype(np.int64))
            cat_dims = (np.concatenate(all_cat, axis=0).max(axis=0) + 1).tolist()
        else:
            cat_dims = []
        self.cat_dims_ = cat_dims

        if cat_dims:
            cat_max = np.array(cat_dims) - 1
            X_tr_cat = np.clip(X_tr_cat, 0, cat_max)
            X_val_cat = np.clip(X_val_cat, 0, cat_max)

        # Class weights
        classes = np.unique(y_tr)
        self.classes_ = classes
        weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)

        # Build model
        n_classes = len(classes)
        self.model_ = RealMLP(
            output_dim=n_classes, cat_dims=cat_dims,
            n_numerical=X_tr_num.shape[1], cfg=p,
        ).to(dev)

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
                idx_batch = train_order[start: start + train_bs]
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
                    ls=ls_val, class_weights=class_weights,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), p["grad_clip"])
                optimizer.step()
                if ema_state is not None:
                    with torch.no_grad():
                        model_state = self.model_.state_dict()
                        for key, value in model_state.items():
                            if torch.is_floating_point(value):
                                ema_state[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
                            else:
                                ema_state[key].copy_(value)
            np.random.shuffle(train_order)

            self.model_.eval()
            if ema_state is not None:
                live_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                self.model_.load_state_dict(ema_state, strict=True)

            with torch.no_grad():
                val_probs = np.concatenate([
                    self.model_(Xvn[s: s + eval_bs], Xvc[s: s + eval_bs])
                        .mean(dim=1).cpu().numpy()
                    for s in range(0, len(y_v), eval_bs)
                ], axis=0)

            # Use AUC for binary classification
            if n_classes == 2:
                epoch_score = roc_auc_score(y_v, val_probs[:, 1])
            else:
                from sklearn.metrics import balanced_accuracy_score
                epoch_score = balanced_accuracy_score(y_v, np.argmax(val_probs, axis=1))

            improved = epoch_score > best_score
            if improved:
                best_score = epoch_score
                best_epoch = epoch + 1
                best_val_probs = val_probs.copy()
                state_src = ema_state if ema_state is not None else self.model_.state_dict()
                best_state = {k: v.detach().clone() for k, v in state_src.items()}

            if verbose >= 2:
                print(f"  epoch {epoch+1}/{epochs}  auc={epoch_score:.5f}  best={best_score:.5f}  "
                      f"ls={ls_val:.4f}  drop={drop_val:.4f}" + (" *" if improved else ""), flush=True)

            if p["use_early_stopping"]:
                patience = (best_epoch * p["early_stopping_multiplicative_patience"]
                            + p["early_stopping_additive_patience"])
                if (epoch + 1) > patience:
                    if verbose >= 1:
                        print(f"  Early stopping at epoch {epoch+1} (best epoch {best_epoch})", flush=True)
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state, strict=True)
        self.best_score_ = best_score
        self.best_val_probs_ = best_val_probs
        self._dev = dev
        if verbose >= 1:
            print(f"  -> best AUC: {best_score:.5f}  (epoch {best_epoch})", flush=True)
        return self

    def predict_proba(self, X):
        eval_bs = self.params["eval_bs"]
        X_num = self.preprocessor_.transform(X[self.num_col_names_].values.astype(np.float32))
        X_cat = X[self.cat_col_names_].values.astype(np.int64)
        X_cat = np.clip(X_cat, 0, np.array(self.cat_dims_) - 1)
        Xn = torch.as_tensor(X_num, dtype=torch.float32, device=self._dev)
        Xc = torch.as_tensor(X_cat, dtype=torch.long, device=self._dev)
        self.model_.eval()
        with torch.no_grad():
            return np.concatenate([
                self.model_(Xn[s: s + eval_bs], Xc[s: s + eval_bs])
                    .mean(dim=1).cpu().numpy()
                for s in range(0, len(X_num), eval_bs)
            ], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# notebook cell 22
# Data Loading & Preprocessing
# ============================================================
def load_and_preprocess(train_path, test_path):
    """Load data and apply feature engineering with data-specific fixes."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    ID = 'id'
    TARGET = 'addicted_label'
    X = train.drop([ID, TARGET], axis=1).copy()
    train_id = train[ID]
    y = train[TARGET]
    X_test = test.drop([ID], axis=1).copy()
    test_id = test[ID]

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    print(f"X init shape: {X.shape}", flush=True)
    print(f"X_test init shape: {X_test.shape}", flush=True)
    print(f"init cat_cols ({len(cat_cols)}): {cat_cols}", flush=True)
    print(f"init num_cols ({len(num_cols)}): {num_cols}", flush=True)

    # Compute medians from TRAINING data only (prevents leakage)
    median_values = {col: X[col].median() for col in num_cols}
    print(f"\nMedian values for fill:", flush=True)
    for col, med in median_values.items():
        missing_pct = X[col].isnull().mean() * 100
        print(f"  {col}: median={med:.2f}, missing={missing_pct:.1f}%", flush=True)

    category_map = {}
    X, new_cat_cols = feature_engineering(
        X, list(cat_cols), list(num_cols), category_map,
        fit=True, median_values=median_values
    )
    X_test, _ = feature_engineering(
        X_test, list(cat_cols), list(num_cols), category_map,
        fit=False, median_values=median_values
    )

    cat_cols = sorted(cat_cols + new_cat_cols)
    X = X.reindex(sorted(X.columns), axis=1)
    X_test = X_test.reindex(sorted(X_test.columns), axis=1)

    # num_cols stays the same (original numericals)
    # TE columns (added per-fold) will also be numerical
    print(f"\nprep cat_cols ({len(cat_cols)}): {cat_cols}", flush=True)
    print(f"prep num_cols ({len(num_cols)}): {num_cols}", flush=True)
    print(f"X prep shape: {X.shape}", flush=True)
    print(f"X_test prep shape: {X_test.shape}", flush=True)

    return X, y, X_test, test_id, cat_cols, num_cols
