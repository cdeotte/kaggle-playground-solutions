"""RealMLP written from scratch in PyTorch (the model code of the public notebook "RealMLP from scratchCV:0.91908").
https://www.kaggle.com/code/yunsuxiaozi/realmlp-from-scratchcv-0-91908

n_ens ensemble members are trained at once (every weight tensor has a leading member dimension): numeric features go
through a PBLD periodic embedding, categoricals with at most 8 levels are one-hot encoded and larger ones embedded,
then a learnable per-feature scaling layer and NTP (neural-tangent-parametrized) linear layers 512-64-128 with GELU and
dropout 0.05, and a 2-class softmax head per member."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RobustScaleSmoothClipTransform(BaseEstimator, TransformerMixin):
    """(x - median) / IQR (half the range where the IQR is 0), then smoothly clipped: x / sqrt(1 + (x / 3)^2)."""

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
        self.device = device
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
                feat_embs.append(embed_list[model_idx](indices))
            features.append(torch.cat(feat_embs, dim=1))
        return torch.cat(features, dim=2)


class ScalingLayer(nn.Module):
    def __init__(self, n_ens=4, n_features: int = 128):
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
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) * (1.0 / np.sqrt(hidden_dim)))
        self.b2 = nn.Parameter(torch.randn(n_ens, n_features, out_dim - 1))
        self.act = nn.GELU()
        nn.init.uniform_(self.b1, -np.pi, np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        results = []
        for i in range(self.n_features):
            xi = x[:, :, i]
            w1_i, b1_i = self.w1[:, i, :], self.b1[:, i, :]
            periodic = torch.cos(2 * np.pi * (xi.unsqueeze(-1) * w1_i.unsqueeze(0) + b1_i.unsqueeze(0)))
            w2_i, b2_i = self.w2[:, i, :, :], self.b2[:, i, :]
            transformed = self.act(torch.einsum('bkh,khd->bkd', periodic, w2_i) + b2_i.unsqueeze(0))
            results.append(torch.cat([xi.unsqueeze(-1), transformed], dim=2))
        return torch.cat(results, dim=2)


class RealMLP(torch.nn.Module):
    def __init__(self, input_dim=32, output_dim=2, cat_dims=[], n_numerical=None, n_ens=8, embed_dim=10):
        super().__init__()
        act = nn.GELU
        self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(n_ens=self.n_ens, cat_dims=cat_dims, embed_dim=embed_dim, device=DEVICE)
        self.num_embed = PBLDEmbedding(n_features=n_numerical, hidden_dim=16, out_dim=5, freq_scale=1.0,
                                       n_ens=self.n_ens)
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
        x = self.output(x)
        return F.softmax(x, dim=2)


def get_parameter_groups_3class(model):
    """(scale, weight, bias) parameters, for per-group learning rates and weight decays."""
    scale_params, weight_params, bias_params = [], [], []
    for name, param in model.named_parameters():
        if 'scale' in name:
            scale_params.append(param)
        elif 'bias' in name:
            bias_params.append(param)
        else:
            weight_params.append(param)
    return scale_params, weight_params, bias_params


def flat_anneal(init_value, progress, flat_ratio=0.5):
    if progress < flat_ratio:
        return init_value
    decay_progress = (progress - flat_ratio) / (1 - flat_ratio)
    return init_value * (1 - decay_progress)


def cos(init_value, progress):
    return init_value * (math.cos(math.pi * progress) + 1) / 2


def cross_entropy_loss(y_true, y_pred, label_smoothing=0.04, eps=1e-15):
    p = y_pred[:, 1]
    p = torch.clip(p, eps, 1 - eps)
    y_true_smooth = y_true * (1 - label_smoothing) + label_smoothing / 2
    ce_loss = -(y_true_smooth * torch.log(p) + (1 - y_true_smooth) * torch.log(1 - p))
    return torch.mean(ce_loss)
