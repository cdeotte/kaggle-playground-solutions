#!/usr/bin/env python
"""PyTorch RealMLP (8 vectorized members) in nested 5 x 5 cross-validation, with inner-fold multiclass target encodings
of the high-cardinality codes (pair realmlp_torch_nested_combo_te_b).

Adapted from yunsuxiaozi's public notebook "RealMLP from scratch"
(https://www.kaggle.com/code/yunsuxiaozi/realmlp-from-scratchcv-0-91908).

Features (competition rows only; bins and codes fitted on the train rows and applied to the test rows):
- numerics (19): the 11 raw numerics, moisture / rainfall, temperature / humidity, moisture / temperature, wind speed
  squared, the two decimals of Soil_Moisture and Temperature_C, Rainfall_mm integer part mod 10 and mod 50;
- integer codes (31): the 8 categoricals (order of first appearance, unseen -> 0), high temperature (>= 30), tens digit
  of Rainfall_mm, Rainfall_mm multiple of 10, KBinsDiscretizer quantile and k-means bins of Rainfall_mm (10, 5),
  Soil_Moisture (10), Temperature_C (5), every numeric factorized (Rainfall_mm and Previous_Irrigation_mm rounded to
  integers; unseen -> 0), the Crop_Type x Season x Region code.
Nested CV: outer folds = the 5 folds of common/data.py; inside each outer training set an inner
StratifiedKFold(5, shuffle=True, random_state=1000 + outer fold). Per inner fold: scikit-learn TargetEncoder (multiclass,
cv 5) of the codes with more than 8 levels, fitted on the inner training rows and appended to the numerics; robust
scaling + smooth clipping fitted on the inner training rows.
Network: PBLD periodic embeddings of the numerics (16 frequencies, 5 outputs per feature), one-hot (<= 8 levels) or
6-d embeddings of the codes, a learnable feature scaling, NTP linear layers 512 -> 64 -> 128 -> 3 (GELU, dropout
0.05), 8 members trained jointly. AdamW (lr 0.01 for weights, x10 for the scaling, x0.1 for biases) with a flat-anneal
schedule, label smoothing 0.04 (cosine-annealed), balanced class weights, gradient clipping 1, batch 4096, 5 epochs;
the epoch with the best inner-validation balanced accuracy is kept. Members' softmax probabilities are averaged.
OOF slice k (shape (5, 630000, 3)): the inner-validation predictions for outer fold k's training rows and the mean of
the 5 inner models for its held-out rows. Test = mean of the 25 models.
Usage: python train_torch_nested_combo_te_b.py"""
import copy
import gc
import math
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.utils.class_weight import compute_class_weight

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, model_dir, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

FOLDER = "realmlp"
PAIR = "realmlp_torch_nested_combo_te_b"

N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
EPOCHS = 5
EARLY_STOP = 5
TRAIN_BS = 256 * 16
EVAL_BS = 4096 * 4
EMBED_DIM = 6
LR = 0.01
N_ENS = 8

CATS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season", "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
NUMS = ["Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity", "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]

category_map = {}
important_combos = [("Crop_Type", "Season", "Region")]


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def feature_engineering(df, fit=False):
    """New code columns end with '_', new numeric columns start with '_'. fit=True fits the bins / codes."""
    df["_Moisture_/_Rainfall"] = (df["Soil_Moisture"] / (df["Rainfall_mm"] + 1e-6)).astype("float32")
    df["_Temp_/_Humidity"] = (df["Temperature_C"] / (df["Humidity"] + 1e-6)).astype("float32")
    df["_Moisture_to_Temp"] = (df["Soil_Moisture"] / (df["Temperature_C"] + 1e-6)).astype("float32")
    df["_Wind_Speed_sq"] = (df["Wind_Speed_kmh"] ** 2).astype("float32")
    df["is_high_temp_"] = (df["Temperature_C"] >= 30).astype(int)

    for col in ["Rainfall_mm"]:
        k = -1
        digit_name = f"{col}_d{k}_"
        df[digit_name] = ((df[col] * 10 ** k) % 10).astype(int)

    for col in ["Soil_Moisture", "Temperature_C"]:
        decimal_name = f"_{col}_decimal"
        df[decimal_name] = (df[col] % 1).round(2).astype("float32")

    df["_Rainfall_mod10"] = (np.floor(df["Rainfall_mm"]) % 10).astype("float32")
    df["_Rainfall_mod50"] = (np.floor(df["Rainfall_mm"]) % 50).astype("float32")
    df["Rainfall_is_multiple_10_"] = (np.floor(df["Rainfall_mm"]) % 10 == 0).astype(int)

    bin_config = {"Rainfall_mm": [10, 5], "Soil_Moisture": [10], "Temperature_C": [5]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            for strategy in ["quantile", "kmeans"]:
                bin_name = f"{col}_{n_bins}_{strategy}_bin_"
                if fit:
                    kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy, subsample=None)
                    binned = kb.fit_transform(df[[col]]).ravel().astype(int)
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype(int)
                df[bin_name] = binned

    for col in NUMS:
        cat_name = f"{col}_cat_"
        round_level = 0
        if fit:
            round_flag = col in ["Rainfall_mm", "Previous_Irrigation_mm"]
            series = df[col].round(round_level) if round_flag else df[col]
            codes, uniques = series.factorize()
            category_map[col] = {"uniques": uniques, "round_flag": round_flag}
            df[cat_name] = codes + 1
        else:
            round_flag = category_map[col]["round_flag"]
            uniques = category_map[col]["uniques"]
            series = df[col].round(round_level) if round_flag else df[col]
            code_map = {cat: i + 1 for i, cat in enumerate(uniques)}
            df[cat_name] = series.map(code_map).fillna(0).astype(int)

    combo_names = []
    for col1, col2, col3 in important_combos:
        combo_name = f"{col1}_{col2}_{col3}_"
        combo_names.append(combo_name)
        combo_series = df[col1].astype(str) + "_" + df[col2].astype(str) + "_" + df[col3].astype(str)
        if fit:
            codes, uniques = combo_series.factorize()
            category_map[combo_name] = uniques
            df[combo_name] = codes + 1
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i + 1 for i, cat in enumerate(uniques)}
            df[combo_name] = combo_series.map(code_map).fillna(0).astype(int)

    new_cat_cols = [col for col in df.columns if col.endswith("_")]
    new_num_cols = [col for col in df.columns if col.startswith("_")]
    return df, new_cat_cols, new_num_cols, combo_names


class RobustScaleSmoothClipTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self._median = np.median(X, axis=-2)
        quant_diff = np.quantile(X, 0.75, axis=-2) - np.quantile(X, 0.25, axis=-2)
        idxs = quant_diff == 0.0
        quant_diff[idxs] = 0.5 * (np.max(X, axis=-2)[idxs] - np.min(X, axis=-2)[idxs])
        factors = 1.0 / (quant_diff + 1e-30)
        factors[quant_diff == 0.0] = 0.0
        self._factors = factors
        return self

    def transform(self, X, y=None):
        x_scaled = self._factors[None, :] * (X - self._median[None, :])
        return x_scaled / np.sqrt(1 + (x_scaled / 3) ** 2)


class CategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens: int, cat_dims, embed_dim=8, device=None):
        super().__init__()
        self.n_ens = n_ens
        self.cat_dims = cat_dims
        self.onehot_features = []
        self.embed_layers = []
        for i, dim in enumerate(self.cat_dims):
            if dim <= 8:
                self.onehot_features.append(i)
            else:
                emb = nn.ModuleList([nn.Embedding(dim, embed_dim).to(device) for _ in range(n_ens)])
                self.embed_layers.append((i, emb))

    def forward(self, x):
        batch_size, n_ens, n_cat = x.shape
        features = []
        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            total_onehot_dim = sum(onehot_dims)
            onehot_encoded = torch.zeros(batch_size, n_ens, total_onehot_dim, device=x.device)
            start = 0
            for idx, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, idx:idx + 1].long()
                onehot_encoded.scatter_(2, pos + start, 1.0)
                start += dim
            features.append(onehot_encoded)
        for feat_idx, embed_list in self.embed_layers:
            feat_embs = []
            for model_idx in range(self.n_ens):
                indices = x[:, model_idx, feat_idx:feat_idx + 1].long()
                emb = embed_list[model_idx](indices)
                feat_embs.append(emb)
            features.append(torch.cat(feat_embs, dim=1))
        return torch.cat(features, dim=2)


class ScalingLayer(nn.Module):
    def __init__(self, n_ens=4, n_features=128):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale[None, :, :]


class NTPLinear(nn.Module):
    def __init__(self, n_ens: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1)
        x = x @ self.weight / np.sqrt(self.in_features)
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class PBLDEmbedding(nn.Module):
    def __init__(self, n_ens: int, n_features: int, hidden_dim: int = 16, out_dim: int = 4, freq_scale: float = 0.1):
        super().__init__()
        self.n_ens = n_ens
        self.n_features = n_features
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) * (1.0 / np.sqrt(hidden_dim)))
        self.b2 = nn.Parameter(torch.randn(n_ens, n_features, out_dim - 1))
        self.act = nn.GELU()
        nn.init.uniform_(self.b1, -np.pi, np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_ens, n_features = x.shape
        results = []
        for i in range(self.n_features):
            xi = x[:, :, i]
            w1_i, b1_i = self.w1[:, i, :], self.b1[:, i, :]
            periodic = torch.cos(2 * np.pi * (xi.unsqueeze(-1) * w1_i.unsqueeze(0) + b1_i.unsqueeze(0)))
            w2_i, b2_i = self.w2[:, i, :, :], self.b2[:, i, :]
            transformed = self.act(torch.einsum("bkh,khd->bkd", periodic, w2_i) + b2_i.unsqueeze(0))
            feat = torch.cat([xi.unsqueeze(-1), transformed], dim=2)
            results.append(feat)
        return torch.cat(results, dim=2)


class RealMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim=3, cat_dims=[], n_numerical=None, n_ens=8, embed_dim=10):
        super().__init__()
        act = nn.GELU
        self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(n_ens=self.n_ens, cat_dims=cat_dims, embed_dim=embed_dim, device=device)
        self.num_embed = PBLDEmbedding(n_features=n_numerical, hidden_dim=16, out_dim=5, freq_scale=1.0, n_ens=self.n_ens)

        num_emb_dim = n_numerical * 5
        cat_emb_dim = sum([c if c <= 8 else embed_dim for c in cat_dims])
        total_dim = num_emb_dim + cat_emb_dim

        self.dropout = nn.Dropout(0.05)
        self.model = nn.Sequential(
            ScalingLayer(n_ens=self.n_ens, n_features=total_dim),
            NTPLinear(n_ens=self.n_ens, in_features=total_dim, out_features=512), act(), self.dropout,
            NTPLinear(n_ens=self.n_ens, in_features=512, out_features=64), act(),
            NTPLinear(n_ens=self.n_ens, in_features=64, out_features=128), act(), self.dropout,
        )
        self.output = NTPLinear(n_ens=self.n_ens, in_features=128, out_features=output_dim)

    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x_cat = self.cate(x_cat)
        combined = torch.cat([x_num, x_cat], dim=2)
        x = self.model(combined)
        return self.output(x)


def get_parameter_groups_3class(model):
    scale_params, weight_params, bias_params = [], [], []
    for name, param in model.named_parameters():
        if "scale" in name:
            scale_params.append(param)
        elif "bias" in name:
            bias_params.append(param)
        else:
            weight_params.append(param)
    return scale_params, weight_params, bias_params


def flat_anneal(init_value, progress, flat_ratio=0.5):
    if progress < flat_ratio:
        return init_value
    decay_progress = (progress - flat_ratio) / (1 - flat_ratio)
    return init_value * (1 - decay_progress)


def cos_anneal(init_value, progress):
    return init_value * (math.cos(math.pi * progress) + 1) / 2


def predict_in_batches(model, x_num, x_cat):
    model.eval()
    probs = []
    x_num_t = torch.as_tensor(x_num, dtype=torch.float32, device=device)
    x_cat_t = torch.as_tensor(x_cat, dtype=torch.long, device=device)
    with torch.no_grad():
        for idx in range(0, len(x_num), EVAL_BS):
            bx_num = x_num_t[idx:idx + EVAL_BS]
            bx_cat = x_cat_t[idx:idx + EVAL_BS]
            p = F.softmax(model(bx_num, bx_cat), dim=2).mean(dim=1)
            probs.append(p.cpu().numpy())
    return np.concatenate(probs, axis=0)


def main():
    print(f"Using device: {device}")
    print("Loading data...")
    train = read_train().set_index(ID)
    test = read_test().set_index(ID)
    train[TARGET] = train[TARGET].map(TARGET_MAP)

    y = train[TARGET].astype(int).values
    X = train.drop([TARGET], axis=1)
    X_test = test.copy()

    print("Feature Engineering...")
    cat_cols = CATS.copy()
    num_cols = NUMS.copy()
    X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, fit=True)
    X_test, new_cat_cols, new_num_cols, combo_names = feature_engineering(X_test, fit=False)
    cat_cols += new_cat_cols
    num_cols += new_num_cols

    for c in CATS:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            mapping = {val: i + 1 for i, val in enumerate(X[c].unique())}
            X[c] = X[c].map(mapping).fillna(0).astype(int)
            X_test[c] = X_test[c].map(mapping).fillna(0).astype(int)

    X_num = X[num_cols].values.astype(np.float32)
    X_cat = X[cat_cols].values.astype(np.int64)
    testX_num = X_test[num_cols].values.astype(np.float32)
    testX_cat = X_test[cat_cols].values.astype(np.int64)

    # category sizes (embedding / one-hot dimensions)
    global_cat_dims = X_cat.max(axis=0) + 1

    print(f"\nTraining RealMLP with {N_OUTER_SPLITS}x{N_INNER_SPLITS} Nested CV...")
    n_train = len(X_num)
    n_test = len(testX_num)

    oof_matrix = np.zeros((N_OUTER_SPLITS, n_train, 3), dtype=np.float32)
    test_pred = np.zeros((n_test, 3), dtype=np.float32)

    fold_id = folds(y)
    outer_va_scores = []
    outer_full_scores = []

    for outer_fold in range(1, N_OUTER_SPLITS + 1):
        outer_tr_idx = np.where(fold_id != outer_fold - 1)[0]
        outer_va_idx = np.where(fold_id == outer_fold - 1)[0]
        print(f"\n{'=' * 80}")
        print(f"OUTER FOLD {outer_fold}/{N_OUTER_SPLITS}")
        print(f"outer_tr size = {len(outer_tr_idx):,}")
        print(f"outer_va size = {len(outer_va_idx):,}")
        print(f"{'-' * 80}")

        X_num_outer_tr, X_cat_outer_tr, y_outer_tr = X_num[outer_tr_idx], X_cat[outer_tr_idx], y[outer_tr_idx]
        X_cat_outer_va, y_outer_va = X_cat[outer_va_idx], y[outer_va_idx]
        X_num_outer_va = X_num[outer_va_idx]

        outer_full_pred = np.zeros((n_train, 3), dtype=np.float32)
        outer_va_pred_sum = np.zeros((len(outer_va_idx), 3), dtype=np.float32)

        skf_inner = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=1000 + outer_fold)
        inner_scores = []

        for inner_fold, (inner_tr_idx_local, inner_va_idx_local) in enumerate(skf_inner.split(X_num_outer_tr, y_outer_tr), start=1):
            global_inner_va_idx = outer_tr_idx[inner_va_idx_local]

            X_num_tr, X_cat_tr, y_tr = X_num_outer_tr[inner_tr_idx_local], X_cat_outer_tr[inner_tr_idx_local], y_outer_tr[inner_tr_idx_local]
            X_num_va, X_cat_va, y_va = X_num_outer_tr[inner_va_idx_local], X_cat_outer_tr[inner_va_idx_local], y_outer_tr[inner_va_idx_local]

            X_num_outer_va_fold = X_num_outer_va.copy()
            X_num_test_fold = testX_num.copy()

            # balanced class weights of the inner training rows
            classes = np.unique(y_tr)
            class_weights_arr = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
            weight_dict = dict(zip(classes, class_weights_arr))
            sample_weights_train = np.array([weight_dict[label] for label in y_tr])

            # target encoding of the high-cardinality codes, fitted on the inner training rows
            cols_to_encode = [i for i in range(X_cat_tr.shape[1]) if X_cat_tr[:, i].max() > 8]
            if cols_to_encode:
                te = TargetEncoder(target_type="multiclass", cv=5)
                X_cat_tr_encoded = te.fit_transform(X_cat_tr[:, cols_to_encode], y_tr)
                X_cat_va_encoded = te.transform(X_cat_va[:, cols_to_encode])
                X_cat_outer_va_encoded = te.transform(X_cat_outer_va[:, cols_to_encode])
                X_cat_test_encoded = te.transform(testX_cat[:, cols_to_encode])

                X_num_tr = np.concatenate([X_num_tr, X_cat_tr_encoded], axis=1)
                X_num_va = np.concatenate([X_num_va, X_cat_va_encoded], axis=1)
                X_num_outer_va_fold = np.concatenate([X_num_outer_va_fold, X_cat_outer_va_encoded], axis=1)
                X_num_test_fold = np.concatenate([X_num_test_fold, X_cat_test_encoded], axis=1)

            # robust scaling + smooth clipping
            rssc = RobustScaleSmoothClipTransform()
            rssc.fit(X_num_tr)
            X_num_tr = rssc.transform(X_num_tr)
            X_num_va = rssc.transform(X_num_va)
            X_num_outer_va_fold = rssc.transform(X_num_outer_va_fold)
            X_num_test_fold = rssc.transform(X_num_test_fold)

            realmlp = RealMLP(
                input_dim=X_num_tr.shape[1] + X_cat_tr.shape[1] * EMBED_DIM,
                output_dim=3,
                cat_dims=global_cat_dims,
                n_numerical=X_num_tr.shape[1],
                n_ens=N_ENS,
                embed_dim=EMBED_DIM,
            ).to(device)

            scale_params, weights, biases = get_parameter_groups_3class(realmlp)
            optimizer = torch.optim.AdamW([
                {"params": scale_params, "name": "scale", "lr": LR * 10.0, "weight_decay": 1e-3},
                {"params": weights, "name": "weights", "lr": LR, "weight_decay": 1e-2},
                {"params": biases, "name": "biases", "lr": LR * 0.1, "weight_decay": 5e-3},
            ], betas=(0.9, 0.98))

            X_num_tr_t = torch.as_tensor(X_num_tr, dtype=torch.float32, device=device)
            X_cat_tr_t = torch.as_tensor(X_cat_tr, dtype=torch.long, device=device)
            y_tr_t = torch.as_tensor(y_tr, dtype=torch.long, device=device)
            weights_t = torch.as_tensor(sample_weights_train, dtype=torch.float32, device=device)

            train_idx_shuffled = np.arange(len(y_tr))
            best_score = 0.0
            best_state = None
            patience_counter = 0

            for epoch in range(EPOCHS):
                np.random.shuffle(train_idx_shuffled)
                realmlp.train()

                for idx in range(0, len(y_tr), TRAIN_BS):
                    progress = epoch / EPOCHS + idx / (len(y_tr) * EPOCHS)
                    bs_idx = train_idx_shuffled[idx:idx + TRAIN_BS]

                    optimizer.param_groups[0]["lr"] = flat_anneal(LR * 10.0, progress)
                    optimizer.param_groups[1]["lr"] = flat_anneal(LR, progress)
                    optimizer.param_groups[2]["lr"] = flat_anneal(LR * 0.1, progress)
                    optimizer.zero_grad()

                    batch_x_num = X_num_tr_t[bs_idx]
                    batch_x_cat = X_cat_tr_t[bs_idx]
                    batch_y = y_tr_t[bs_idx]
                    batch_weights = weights_t[bs_idx]

                    logits = realmlp(batch_x_num, batch_x_cat)
                    logits_reshaped = logits.reshape(-1, 3)
                    targets_expanded = batch_y.repeat_interleave(N_ENS)
                    weights_expanded = batch_weights.repeat_interleave(N_ENS)

                    loss = F.cross_entropy(logits_reshaped, targets_expanded, reduction="none", label_smoothing=cos_anneal(0.04, progress))
                    loss = (loss * weights_expanded).mean()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(realmlp.parameters(), max_norm=1.0)
                    optimizer.step()

                # inner validation
                val_probs = predict_in_batches(realmlp, X_num_va, X_cat_va)
                score = balanced_accuracy_score(y_va, np.argmax(val_probs, axis=1))

                if score > best_score:
                    best_score = score
                    best_state = copy.deepcopy(realmlp.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= EARLY_STOP:
                    break

            # restore the best epoch
            realmlp.load_state_dict(best_state)

            # 1. inner validation rows (outer training rows of this slice)
            p_inner_va = predict_in_batches(realmlp, X_num_va, X_cat_va)
            outer_full_pred[global_inner_va_idx] = p_inner_va
            inner_acc = balanced_accuracy_score(y_va, np.argmax(p_inner_va, axis=1))
            inner_scores.append(float(inner_acc))
            print(f"Outer {outer_fold} | Inner {inner_fold}/{N_INNER_SPLITS} -> Balanced ACC = {inner_acc:.5f}")

            # 2. outer validation rows (averaged over the 5 inner models)
            p_outer_va = predict_in_batches(realmlp, X_num_outer_va_fold, X_cat_outer_va)
            outer_va_pred_sum += p_outer_va

            # 3. test rows (averaged over all 25 models)
            p_test = predict_in_batches(realmlp, X_num_test_fold, testX_cat)
            test_pred += p_test / (N_OUTER_SPLITS * N_INNER_SPLITS)

            del realmlp, optimizer, X_num_tr_t, X_cat_tr_t, y_tr_t, weights_t, best_state
            gc.collect()
            torch.cuda.empty_cache()

        outer_va_pred = outer_va_pred_sum / N_INNER_SPLITS
        outer_full_pred[outer_va_idx] = outer_va_pred
        oof_matrix[outer_fold - 1, :, :] = outer_full_pred

        outer_va_acc = balanced_accuracy_score(y_outer_va, np.argmax(outer_va_pred, axis=1))
        outer_full_acc = balanced_accuracy_score(y, np.argmax(outer_full_pred, axis=1))
        outer_va_scores.append(float(outer_va_acc))
        outer_full_scores.append(float(outer_full_acc))

        print(f"{'-' * 80}")
        print(f"Outer {outer_fold} Mean Inner ACC = {np.mean(inner_scores):.5f}")
        print(f"Outer {outer_fold} OUTER-VA ACC   = {outer_va_acc:.5f}")
        print(f"Outer {outer_fold} FULL-ROW ACC   = {outer_full_acc:.5f}")

    oof_mean = oof_matrix.mean(axis=0).astype(np.float32)
    oof_mean_acc = balanced_accuracy_score(y, np.argmax(oof_mean, axis=1))

    print(f"\n{'=' * 80}")
    print("NESTED TRAINING FINISHED")
    print(f"Outer-fold OUTER-VA ACCs        = {[round(x, 5) for x in outer_va_scores]}")
    print(f"Outer-fold FULL-ROW ACCs        = {[round(x, 5) for x in outer_full_scores]}")
    print(f"Mean OUTER-VA ACC               = {np.mean(outer_va_scores):.5f} +/- {np.std(outer_va_scores):.5f}")
    print(f"Mean-row OOF ACC                = {oof_mean_acc:.5f}")

    save_pair(model_dir(FOLDER), PAIR, oof_matrix, test_pred)


if __name__ == "__main__":
    main()
