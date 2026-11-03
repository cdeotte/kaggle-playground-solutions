# Code of the public notebook "Multi-Level Deep Univariate Spline Transformer" (version 5):
# https://www.kaggle.com/code/ern711/multi-level-deep-univariate-spline-transformer
# Everything before its five-fold cross-validation loop (train.py runs the folds with train_one_fold). Importing
# this module seeds the RNGs, builds the label-free features of train and test and defines the model and the
# per-fold training. Only the two data paths were changed (data come from common.data).
"""Playground Series S6E8: Multi-Level Deep Univariate Spline Transformer.

This model is built around a simple design principle: learn strong feature-local
representations before allowing cross-feature interaction.

Numerical features are first standardized and passed through a small nonlinear
pre-spline context module. Each numerical feature is then modeled independently
by a feature-specific mixture of spline, MLP, and raw-value experts with local
residual depth. The resulting numerical tokens are combined by a shallow
self-attention interaction block.

The model produces four supervised prediction logits at different representation
depths:

1. ADD  - additive prediction from per-feature univariate token scores.
2. UNI  - pooled prediction from the univariate token bank.
3. ATTN - pooled prediction from post-attention interaction tokens.
4. FINAL - direct prediction from the complete learned representation.

A small row-dependent convex mixer combines these four logits into the final
Stage-1 prediction. The mixer is initialized to favor the FINAL logit so training
starts close to the original direct-final-head model.

Stage 2 freezes the complete Stage-1 network and trains a compact hypernetwork
that applies bounded row-dependent corrections at three sites: numerical tokens,
post-attention tokens, and the normalized input to the final MLP. The same four
prediction heads and learned mixer remain active after these corrections.

All target/frequency encodings, numerical scaling, and spline knots are fitted
inside each outer training fold. Target encodings for outer-training rows are
constructed with inner-fold out-of-fold mappings to avoid target leakage.

The script is configured for five-fold stratified cross-validation. Test
predictions are averaged across the five restored fold models.
"""


# ==============================================================================
# IMPORTS AND CONFIGURATION
# ==============================================================================
# Reproducibility, paths, optimization, and architecture configuration.

import gc
import copy
import random
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.data import DATA_DIR  # noqa: E402
TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'
TARGET = 'addicted_label'
ID_COL = 'id'
SEED = 21
N_SPLITS = 5
OUTER_SPLIT_SEED = 21
INNER_TE_SPLITS = 5
TE_SMOOTHING = 10.0
BATCH_SIZE = 4096
EPOCHS = 35
PATIENCE = 7
LR = 0.0007
WEIGHT_DECAY = 0.0005
LABEL_SMOOTHING_EPS = 0.005
GRAD_CLIP = 5.0
MIX_TOKEN_DIM = 64
LOCAL_EXPANSION = 2.0
LOCAL_DROPOUT = 0.05
INITIAL_LOCAL_SCALE = 0.2
EXPERT_GATE_HIDDEN = 32
EXPERT_GATE_SCALE_INIT = 0.1
EXPERT_GATE_TEMPERATURE = 1.0
RESOLUTION_GATE_HIDDEN = 32
RESOLUTION_GATE_SCALE_INIT = 0.1
RESOLUTION_GATE_TEMPERATURE = 1.0
ATTENTION_DIM = 64
ATTENTION_HEADS = 8
ATTENTION_LAYERS = 1
ATTENTION_FF_MULT = 2
ATTENTION_DROPOUT = 0.05
RESIDUAL_HIDDEN = 384
RESIDUAL_DROPOUT = 0.05
INITIAL_INTERACTION_SCALE = 0.1
FINAL_LOSS_WEIGHT = 1.0
ADDITIVE_AUX_WEIGHT = 0.3

# Small nonlinear cross-feature correction applied before univariate encoding.
#
# The context network is intentionally tiny. Its purpose is not to replace the
# feature-local spline/MoE encoders, but to make a small row-dependent adjustment
# to each standardized scalar before that scalar is modeled independently.
PRE_SPLINE_CONTEXT_HIDDEN = 32
INITIAL_PRE_SPLINE_CONTEXT_SCALE = 0.02

HYPER_EPOCHS = 20
HYPER_PATIENCE = 5
HYPER_LR = 5e-4
HYPER_WEIGHT_DECAY = 1e-4

# Small multi-site controller: keep capacity close to the successful token-only model.
HYPER_BOTTLENECK = 32
HYPER_HIDDEN = 192

# Main token adaptation.
HYPER_TOKEN_SCALE_LIMIT = 0.15
HYPER_TOKEN_SHIFT_LIMIT = 0.15

# Secondary sites are deliberately conservative.
HYPER_ATTN_SCALE_LIMIT = 0.05
HYPER_HEAD_SCALE_LIMIT = 0.05

HYPER_SEED_OFFSET = 20000

# ==============================================================================
# MULTI-LEVEL OUTPUT HEADS AND LEARNED LOGIT MIXER
# ==============================================================================
# Four supervised prediction logits are produced at increasing representation
# depth: ADD, UNI, ATTN, and FINAL. A small row-dependent softmax mixer combines
# them into the Stage-1 prediction.
#
# This publication version runs all outer folds.
MULTIHEAD_FOLDS_TO_RUN = list(range(N_SPLITS))

# Pooled auxiliary classifiers operating on univariate and attention tokens.
UNI_HEAD_HIDDEN = 128
ATTN_HEAD_HIDDEN = 128
AUX_HEAD_DROPOUT = 0.05

# Auxiliary loss weights. ADDITIVE_AUX_WEIGHT is defined above.
UNI_AUX_WEIGHT = 0.10
ATTN_AUX_WEIGHT = 0.10

# The mixer sees the four logits plus all six pairwise absolute disagreements.
# Its output layer is zero-initialized and biased toward FINAL at initialization.
MIXER_HIDDEN = 16
MIXER_FINAL_BIAS = 2.0
MIXER_ENTROPY_WEIGHT = 1e-5

# Runtime device.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything(SEED)


# ==============================================================================
# DATA LOADING AND FEATURE ENGINEERING
# ==============================================================================
# Build raw, engineered, encoded-source, and categorical feature lists.

train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)
print('Train shape:', train_raw.shape)
print('Test shape :', test_raw.shape)
initial_features = [col for col in train_raw.columns if col not in [TARGET, ID_COL]]
RAW_NUMERICALS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time']

def engineer_features(df):
    """Create the continuous engineered features used by the model.

    The transformations are intentionally simple and domain-driven. Infinite
    values produced by ratios are replaced with zero after feature creation.
    """
    df = df.copy()
    df['missing_count'] = df[initial_features].isna().sum(axis=1).astype(np.float32)

    def num(col):
        return pd.to_numeric(df[col], errors='coerce')
    social = num('social_media_hours')
    gaming = num('gaming_hours')
    work = num('work_study_hours')
    screen = num('daily_screen_time_hours')
    sleep = num('sleep_hours')
    weekend = num('weekend_screen_time')
    app_opens = num('app_opens_per_day')
    notifications = num('notifications_per_day')
    df['total_breakdown_hours'] = social.fillna(0.0) + gaming.fillna(0.0) + work.fillna(0.0)
    df['social_ratio'] = social / (screen + 1e-05)
    df['gaming_ratio'] = gaming / (screen + 1e-05)
    df['work_ratio'] = work / (screen + 1e-05)
    df['unaccounted_screen_time'] = screen - df['total_breakdown_hours']
    df['screen_to_sleep_ratio'] = screen / (sleep + 1e-05)
    df['weekend_vs_daily_ratio'] = weekend / (screen + 1e-05)
    df['app_opens_per_hour'] = app_opens / (screen + 1e-05)
    df['notifications_per_hour'] = notifications / (screen + 1e-05)
    engineered = ['total_breakdown_hours', 'social_ratio', 'gaming_ratio', 'work_ratio', 'unaccounted_screen_time', 'screen_to_sleep_ratio', 'weekend_vs_daily_ratio', 'app_opens_per_hour', 'notifications_per_hour']
    df[engineered] = df[engineered].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df
train_full = engineer_features(train_raw)
test_full = engineer_features(test_raw)
for col in initial_features:
    train_full[f'cat__{col}'] = train_full[col].astype(str)
    test_full[f'cat__{col}'] = test_full[col].astype(str)
TARGET_ENC_CATS = [f'cat__{col}' for col in initial_features]
ENGINEERED_CONTINUOUS = ['missing_count', 'total_breakdown_hours', 'social_ratio', 'gaming_ratio', 'work_ratio', 'unaccounted_screen_time', 'screen_to_sleep_ratio', 'weekend_vs_daily_ratio', 'app_opens_per_hour', 'notifications_per_hour']
TE_FEATURES = [f'{col}_te' for col in TARGET_ENC_CATS]
FREQ_FEATURES = [f'{col}_freq' for col in TARGET_ENC_CATS]
NUMERICALS = RAW_NUMERICALS + ENGINEERED_CONTINUOUS + TE_FEATURES + FREQ_FEATURES
MISSING_CATS = [f'missing__{col}' for col in RAW_NUMERICALS]
CATS = TARGET_ENC_CATS + MISSING_CATS
print()
print('Numerical features :', len(NUMERICALS))
print('Categorical features:', len(CATS))
FEATURE_CAPACITY = {'age': {'width': 48, 'depth': 2}, 'daily_screen_time_hours': {'width': 96, 'depth': 4}, 'social_media_hours': {'width': 96, 'depth': 4}, 'gaming_hours': {'width': 80, 'depth': 3}, 'work_study_hours': {'width': 96, 'depth': 4}, 'sleep_hours': {'width': 80, 'depth': 3}, 'notifications_per_day': {'width': 64, 'depth': 3}, 'app_opens_per_day': {'width': 64, 'depth': 3}, 'weekend_screen_time': {'width': 96, 'depth': 4}}
for col in ENGINEERED_CONTINUOUS:
    FEATURE_CAPACITY[col] = {'width': 64, 'depth': 3}
for col in TE_FEATURES:
    FEATURE_CAPACITY[col] = {'width': 64, 'depth': 3}
for col in FREQ_FEATURES:
    FEATURE_CAPACITY[col] = {'width': 40, 'depth': 2}
RAW_FEATURE_SPLINE_SPECS = {'age': [(8, 16), (16, 24), (19, 32)], 'daily_screen_time_hours': [(32, 16), (64, 16), (128, 32), (256, 64), (512, 96)], 'social_media_hours': [(32, 16), (64, 16), (128, 32), (256, 64), (512, 96)], 'gaming_hours': [(32, 16), (64, 16), (128, 32), (256, 64)], 'work_study_hours': [(32, 16), (64, 16), (128, 32), (256, 64), (512, 96)], 'sleep_hours': [(32, 16), (64, 16), (128, 32), (256, 64)], 'notifications_per_day': [(32, 16), (64, 16), (128, 32), (232, 64)], 'app_opens_per_day': [(32, 16), (64, 16), (128, 32), (167, 64)], 'weekend_screen_time': [(32, 16), (64, 16), (128, 32), (256, 64), (512, 96)]}


# ==============================================================================
# ENCODING AND SPLINE HELPERS
# ==============================================================================
# Fold-safe encodings and quantile-knot construction.

def build_unique_quantile_knots(values, requested_knots):
    """Build approximately quantile-spaced spline knots with duplicate removal.

    The requested number of knots is capped by the number of unique values.
    Constant features receive a tiny two-knot fallback interval.
    """
    values = np.asarray(values, dtype=np.float32)
    unique_values = np.unique(values)
    if len(unique_values) == 1:
        value = float(unique_values[0])
        return np.array([value - 1e-05, value + 1e-05], dtype=np.float32)
    n_knots = min(requested_knots, len(unique_values))
    q = np.linspace(0.0, 1.0, n_knots)
    knots = np.quantile(values, q).astype(np.float32)
    knots = np.unique(knots)
    if len(knots) < 2:
        knots = unique_values
    return knots.astype(np.float32)

def choose_engineered_specs(values):
    """Choose spline resolutions for a feature from its observed cardinality."""
    n_unique = len(np.unique(values))
    if n_unique <= 2:
        return [(2, 16)]
    if n_unique <= 4:
        return [(4, 16)]
    if n_unique <= 16:
        return [(4, 16), (8, 16), (16, 24)]
    if n_unique <= 64:
        return [(8, 16), (16, 16), (32, 24), (64, 32)]
    return [(16, 16), (32, 16), (64, 32), (128, 64)]

def apply_nested_encodings(train_df, val_df, test_df, cat_cols, target_col, n_inner_splits, smoothing, random_state):
    """Create leakage-safe target and frequency encodings for one outer CV fold.

    The outer-training rows receive inner-fold out-of-fold encodings, so a row's
    target is never used to construct its own target encoding. Validation and
    test encodings are averaged across mappings fitted exclusively on the
    corresponding inner-training partitions.
    """
    train_encoded = train_df.copy()
    val_encoded = val_df.copy()
    test_encoded = test_df.copy()
    for col in cat_cols:
        train_encoded[f'{col}_te'] = 0.0
        train_encoded[f'{col}_freq'] = 0.0
        val_encoded[f'{col}_te'] = 0.0
        val_encoded[f'{col}_freq'] = 0.0
        test_encoded[f'{col}_te'] = 0.0
        test_encoded[f'{col}_freq'] = 0.0
    y_inner = train_df[target_col].values
    inner_skf = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
    for inner_train_idx, inner_oof_idx in inner_skf.split(train_df, y_inner):
        inner_train = train_df.iloc[inner_train_idx]
        inner_oof = train_df.iloc[inner_oof_idx]
        global_mean = float(inner_train[target_col].mean())
        for col in cat_cols:
            counts = inner_train[col].value_counts(dropna=False)
            train_encoded.loc[inner_oof_idx, f'{col}_freq'] = inner_oof[col].map(counts).astype(float).fillna(0.0).values
            val_encoded[f'{col}_freq'] += val_df[col].map(counts).astype(float).fillna(0.0).values / n_inner_splits
            test_encoded[f'{col}_freq'] += test_df[col].map(counts).astype(float).fillna(0.0).values / n_inner_splits
            stats = inner_train.groupby(col, observed=False)[target_col].agg(['count', 'mean'])
            mapping = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
            train_encoded.loc[inner_oof_idx, f'{col}_te'] = inner_oof[col].map(mapping).astype(float).fillna(global_mean).values
            val_encoded[f'{col}_te'] += val_df[col].map(mapping).astype(float).fillna(global_mean).values / n_inner_splits
            test_encoded[f'{col}_te'] += test_df[col].map(mapping).astype(float).fillna(global_mean).values / n_inner_splits
    return (train_encoded, val_encoded, test_encoded)

def default_emb_dim(cardinality):
    """Return a bounded embedding dimension for a categorical feature."""
    return int(min(32, max(4, round(1.6 * cardinality ** 0.5))))


# ==============================================================================
# DEEP UNIVARIATE SPLINE ENCODER
# ==============================================================================
# Model each numerical feature independently before cross-feature mixing.

class DynamicGate(nn.Module):
    """Global mixture weights with a small row-dependent adjustment."""

    def __init__(self, n_outputs, hidden_dim, scale_init, temperature):
        super().__init__()

        self.temperature = temperature

        self.global_logits = nn.Parameter(
            torch.zeros(n_outputs)
        )

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_outputs),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.scale_raw = nn.Parameter(
            torch.tensor(
                float(np.arctanh(scale_init))
            )
        )

    def forward(self, x):
        dynamic = self.net(
            x.unsqueeze(1)
        )

        scale = torch.tanh(
            self.scale_raw
        )

        logits = (
            self.global_logits.unsqueeze(0)
            + scale * dynamic
        )

        weights = torch.softmax(
            logits / self.temperature,
            dim=1,
        )

        return weights, scale

class UnivariateSpline(nn.Module):
    """Learn a vector-valued piecewise-linear function of one scalar."""

    def __init__(self, knots, d_emb):
        super().__init__()

        self.register_buffer(
            "knots",
            knots.clone(),
        )

        self.values = nn.Parameter(
            torch.randn(
                len(knots),
                d_emb,
            ) * 0.02
        )

    def forward(self, x):
        x = torch.clamp(
            x,
            min=self.knots[0],
            max=self.knots[-1],
        )

        right = torch.searchsorted(
            self.knots,
            x,
            right=True,
        )

        right = torch.clamp(
            right,
            1,
            len(self.knots) - 1,
        )

        left = right - 1

        x0 = self.knots[left]
        x1 = self.knots[right]

        alpha = (
            (x - x0)
            / (x1 - x0).clamp_min(1e-8)
        )

        v0 = self.values[left]
        v1 = self.values[right]

        return (
            v0
            + alpha.unsqueeze(1)
            * (v1 - v0)
        )

class MultiResolutionSplineExpert(nn.Module):
    """Combine several spline resolutions with a dynamic gate."""

    def __init__(self, knots_list, specs, output_dim):
        super().__init__()

        self.splines = nn.ModuleList()
        self.projections = nn.ModuleList()

        for knots, (_, d_emb) in zip(
            knots_list,
            specs,
        ):
            self.splines.append(
                UnivariateSpline(
                    knots,
                    d_emb,
                )
            )

            self.projections.append(
                nn.Linear(
                    d_emb,
                    output_dim,
                )
            )

        self.gate = DynamicGate(
            n_outputs=len(specs),
            hidden_dim=RESOLUTION_GATE_HIDDEN,
            scale_init=RESOLUTION_GATE_SCALE_INIT,
            temperature=RESOLUTION_GATE_TEMPERATURE,
        )

    def forward(self, x):
        tokens = []

        for spline, projection in zip(
            self.splines,
            self.projections,
        ):
            tokens.append(
                projection(
                    spline(x)
                )
            )

        tokens = torch.stack(
            tokens,
            dim=1,
        )

        weights, scale = self.gate(x)

        token = (
            tokens
            * weights.unsqueeze(2)
        ).sum(
            dim=1
        )

        return token, weights, scale

class CoarseSplineExpert(nn.Module):
    """Low-resolution smooth spline expert."""

    def __init__(self, knots, output_dim):
        super().__init__()

        d_emb = min(
            32,
            output_dim,
        )

        self.spline = UnivariateSpline(
            knots,
            d_emb,
        )

        self.projection = nn.Linear(
            d_emb,
            output_dim,
        )

    def forward(self, x):
        return self.projection(
            self.spline(x)
        )

class TinyMLPExpert(nn.Module):
    """Small nonlinear neural expert for one scalar input."""

    def __init__(self, output_dim):
        super().__init__()

        hidden = max(
            16,
            output_dim // 2,
        )

        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(
                hidden,
                output_dim,
            ),
        )

    def forward(self, x):
        return self.net(
            x.unsqueeze(1)
        )

class RawPathExpert(nn.Module):
    """Direct gated projection of the standardized scalar."""

    def __init__(self, output_dim):
        super().__init__()

        self.projection = nn.Linear(
            1,
            output_dim,
        )

        self.gate = nn.Linear(
            1,
            output_dim,
        )

    def forward(self, x):
        scalar = x.unsqueeze(1)

        return (
            self.projection(scalar)
            * torch.sigmoid(
                self.gate(scalar)
            )
        )

class SwiGLU(nn.Module):
    """SwiGLU feed-forward layer used in local and interaction blocks."""

    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.value = nn.Linear(dim, hidden_dim)
        self.gate = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        h = self.value(x) * F.silu(self.gate(x))
        h = self.dropout(h)
        return self.output(h)

class LocalSwiGLUBlock(nn.Module):
    """Residual feature-local feed-forward block with a learned residual scale."""

    def __init__(self, dim):
        super().__init__()
        hidden = max(16, int(round(dim * LOCAL_EXPANSION)))
        self.norm = nn.LayerNorm(dim)
        self.ff = SwiGLU(dim, hidden, LOCAL_DROPOUT)
        self.scale_raw = nn.Parameter(torch.tensor(float(np.arctanh(INITIAL_LOCAL_SCALE))))

    def forward(self, x):
        return x + torch.tanh(self.scale_raw) * self.ff(self.norm(x))

class DeepUnivariateFeatureEncoder(nn.Module):
    """Encode one numerical feature before any cross-feature interaction.

    Four expert families are mixed dynamically, processed by feature-specific
    local depth, and projected to a common token dimension.
    """

    def __init__(self, feature_name, knots_list, spline_specs, coarse_knots):
        super().__init__()
        cfg = FEATURE_CAPACITY[feature_name]
        self.width = cfg['width']
        self.depth = cfg['depth']
        self.multi_spline = MultiResolutionSplineExpert(knots_list=knots_list, specs=spline_specs, output_dim=self.width)
        self.coarse_spline = CoarseSplineExpert(knots=coarse_knots, output_dim=self.width)
        self.tiny_mlp = TinyMLPExpert(self.width)
        self.raw_path = RawPathExpert(self.width)
        self.expert_gate = DynamicGate(n_outputs=4, hidden_dim=EXPERT_GATE_HIDDEN, scale_init=EXPERT_GATE_SCALE_INIT, temperature=EXPERT_GATE_TEMPERATURE)
        self.local_blocks = nn.ModuleList([LocalSwiGLUBlock(self.width) for _ in range(self.depth)])
        self.output_projection = nn.Sequential(nn.LayerNorm(self.width), nn.Linear(self.width, MIX_TOKEN_DIM))

    def forward(self, x):
        spline_token, resolution_weights, resolution_scale = self.multi_spline(x)
        coarse_token = self.coarse_spline(x)
        mlp_token = self.tiny_mlp(x)
        raw_token = self.raw_path(x)
        experts = torch.stack([spline_token, coarse_token, mlp_token, raw_token], dim=1)
        expert_weights, expert_scale = self.expert_gate(x)
        token = (experts * expert_weights.unsqueeze(2)).sum(dim=1)
        for block in self.local_blocks:
            token = block(token)
        token = self.output_projection(token)
        return {'token': token, 'expert_weights': expert_weights, 'expert_scale': expert_scale, 'resolution_weights': resolution_weights, 'resolution_scale': resolution_scale}

class DeepUnivariateEncoder(nn.Module):
    """Apply an independent deep univariate encoder to every numerical feature."""

    def __init__(self, feature_knots, feature_specs, coarse_knots):
        super().__init__()
        self.encoders = nn.ModuleDict({col: DeepUnivariateFeatureEncoder(feature_name=col, knots_list=feature_knots[col], spline_specs=feature_specs[col], coarse_knots=coarse_knots[col]) for col in NUMERICALS})

    def forward(self, x):
        tokens = []
        diagnostics = {}
        for j, col in enumerate(NUMERICALS):
            out = self.encoders[col](x[:, j])
            tokens.append(out['token'])
            diagnostics[col] = out
        return (torch.stack(tokens, dim=1), diagnostics)


# ==============================================================================
# MULTI-LEVEL PREDICTION AND SHALLOW FEATURE INTERACTION
# ==============================================================================
# Build predictions at several representation depths and mix them adaptively.

class AdditiveHead(nn.Module):
    """Score each numerical token with a single linear feature head."""

    def __init__(self):
        super().__init__()

        self.feature_heads = nn.ModuleList(
            [
                nn.Linear(
                    MIX_TOKEN_DIM,
                    1,
                )
                for _ in NUMERICALS
            ]
        )

        self.bias = nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, tokens):
        scores = []

        for j, head in enumerate(
            self.feature_heads
        ):
            scores.append(
                head(
                    tokens[:, j, :]
                ).squeeze(1)
            )

        feature_scores = torch.stack(
            scores,
            dim=1,
        )

        additive_logit = (
            self.bias
            + feature_scores.sum(
                dim=1
            )
        )

        return (
            additive_logit,
            feature_scores,
        )

class SmallAttentionBlock(nn.Module):
    """Pre-normalized self-attention block with a SwiGLU feed-forward stage."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(ATTENTION_DIM)
        self.attention = nn.MultiheadAttention(embed_dim=ATTENTION_DIM, num_heads=ATTENTION_HEADS, dropout=ATTENTION_DROPOUT, batch_first=True)
        self.attention_scale = nn.Parameter(torch.tensor(0.2))
        self.norm2 = nn.LayerNorm(ATTENTION_DIM)
        self.ff = SwiGLU(ATTENTION_DIM, ATTENTION_DIM * ATTENTION_FF_MULT, ATTENTION_DROPOUT)
        self.ff_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        n = self.norm1(x)
        attended, _ = self.attention(n, n, n, need_weights=False)
        x = x + torch.tanh(self.attention_scale) * attended
        x = x + torch.tanh(self.ff_scale) * self.ff(self.norm2(x))
        return x

class InteractionEncoder(nn.Module):
    """Mix numerical feature tokens with self-attention after univariate encoding."""

    def __init__(self, n_features):
        super().__init__()
        self.input_projection = nn.Linear(MIX_TOKEN_DIM, ATTENTION_DIM)
        self.feature_identity = nn.Parameter(torch.empty(1, n_features, ATTENTION_DIM))
        nn.init.normal_(self.feature_identity, std=0.02)
        # Global interaction is intentionally shallow: one residual attention block
        # is applied only after each numerical feature has already been modeled
        # independently by its deep univariate encoder.
        self.layers = nn.ModuleList([SmallAttentionBlock() for _ in range(ATTENTION_LAYERS)])
        self.norm = nn.LayerNorm(ATTENTION_DIM)

    def forward(self, tokens):
        x = self.input_projection(tokens) + self.feature_identity
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class NonlinearPreSplineContext(nn.Module):
    """Apply a small nonlinear context correction to every numerical feature.

    The module observes all standardized numerical features in the row and
    produces one scalar correction per feature:

        h = SiLU(W1 x + b1)
        delta = W2 h + b2
        x_contextual = x + scale * delta

    The final projection is zero-initialized, so the model starts with exactly
    zero context correction. Each numerical feature also has its own learned
    residual scale, allowing training to decide how much contextualization is
    useful for that feature.

    This keeps the overall architecture "univariate first": the expensive
    spline/MoE encoder for feature i still receives only one scalar value.
    Context changes that scalar slightly, but does not replace the independent
    feature-specific representation learning.
    """

    def __init__(self, n_features):
        super().__init__()

        self.input = nn.Linear(
            n_features,
            PRE_SPLINE_CONTEXT_HIDDEN,
        )

        self.output = nn.Linear(
            PRE_SPLINE_CONTEXT_HIDDEN,
            n_features,
        )

        # Start exactly at the no-context baseline.
        nn.init.zeros_(
            self.output.weight
        )
        nn.init.zeros_(
            self.output.bias
        )

        self.scale_raw = nn.Parameter(
            torch.full(
                (n_features,),
                float(
                    np.arctanh(
                        INITIAL_PRE_SPLINE_CONTEXT_SCALE
                    )
                ),
            )
        )

    def forward(self, x):
        """Return contextualized scalars, learned scales, and corrections."""
        hidden = F.silu(
            self.input(
                x
            )
        )

        delta = self.output(
            hidden
        )

        scales = torch.tanh(
            self.scale_raw
        )

        correction = (
            delta
            * scales.unsqueeze(0)
        )

        contextualized = (
            x
            + correction
        )

        return (
            contextualized,
            scales,
            correction,
        )



class PooledTokenHead(nn.Module):
    """Predict from a token bank using permutation-invariant mean/max pooling.

    The head summarizes all feature tokens with their element-wise mean and max,
    concatenates both summaries, and maps the result to one binary-classification
    logit. The same module is used for the UNI and ATTN prediction paths.
    """
    def __init__(self, token_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * token_dim),
            nn.Linear(2 * token_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(AUX_HEAD_DROPOUT),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens):
        pooled = torch.cat(
            [
                tokens.mean(dim=1),
                tokens.max(dim=1).values,
            ],
            dim=1,
        )
        return self.net(pooled).squeeze(1)


class MultiLogitMixer(nn.Module):
    """Combine ADD, UNI, ATTN, and FINAL logits with row-dependent convex weights.

    The mixer receives the four logits together with the six pairwise absolute
    logit disagreements. A softmax produces four non-negative weights summing to
    one. The output layer starts with zero weights and a positive bias on FINAL,
    so the initial mixture strongly favors the strongest original prediction path
    while still allowing training to discover complementary intermediate heads.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, MIXER_HIDDEN),
            nn.SiLU(),
            nn.Linear(MIXER_HIDDEN, 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[3] = MIXER_FINAL_BIAS

    def forward(self, add_logit, uni_logit, attn_logit, final_logit):
        logits = torch.stack(
            [add_logit, uni_logit, attn_logit, final_logit],
            dim=1,
        )
        diffs = torch.stack(
            [
                (add_logit - uni_logit).abs(),
                (add_logit - attn_logit).abs(),
                (add_logit - final_logit).abs(),
                (uni_logit - attn_logit).abs(),
                (uni_logit - final_logit).abs(),
                (attn_logit - final_logit).abs(),
            ],
            dim=1,
        )
        mixer_input = torch.cat([logits, diffs], dim=1)
        weights = torch.softmax(self.net(mixer_input), dim=1)
        mixed = (weights * logits).sum(dim=1)
        return mixed, weights


class SplineTransformerModel(nn.Module):
    """Deep-univariate classifier with four supervised output heads and a mixer.

    Pipeline:
        standardized numericals
            -> small nonlinear pre-spline context correction
            -> independent deep univariate spline/MoE encoders
            -> shallow self-attention interaction encoder
            -> multi-level predictions

    The four prediction paths are:

        ADD:
            Sum of one learned scalar score per numerical feature token.

        UNI:
            Small classifier over mean/max pooled univariate tokens.

        ATTN:
            Small classifier over mean/max pooled post-attention tokens.

        FINAL:
            Direct MLP prediction from raw standardized numericals, flattened
            univariate tokens, flattened attention tokens, categorical embeddings,
            per-feature additive scores, and the additive probability.

    MultiLogitMixer then forms a row-dependent convex combination of the four
    logits. The model therefore preserves intermediate predictive signals instead
    of forcing every useful decision rule through only the deepest classifier.
    """

    def __init__(
        self,
        feature_knots,
        feature_specs,
        coarse_knots,
        cat_sizes,
    ):
        super().__init__()

        self.n_num = len(
            NUMERICALS
        )

        self.pre_spline_context = NonlinearPreSplineContext(
            self.n_num
        )

        self.univariate_encoder = DeepUnivariateEncoder(
            feature_knots,
            feature_specs,
            coarse_knots,
        )

        # Four prediction paths are supervised independently before mixing.
        self.additive_head = AdditiveHead()

        self.uni_output_head = PooledTokenHead(
            MIX_TOKEN_DIM,
            UNI_HEAD_HIDDEN,
        )

        self.attn_output_head = PooledTokenHead(
            ATTENTION_DIM,
            ATTN_HEAD_HIDDEN,
        )

        self.logit_mixer = MultiLogitMixer()

        self.interaction_encoder = InteractionEncoder(
            self.n_num
        )

        self.cat_embeddings = nn.ModuleDict()

        cat_dim = 0

        for col in CATS:
            cardinality = cat_sizes[col]
            dim = default_emb_dim(
                cardinality
            )

            self.cat_embeddings[col] = nn.Embedding(
                cardinality,
                dim,
            )

            cat_dim += dim

        final_input_dim = (
            self.n_num
            + self.n_num * MIX_TOKEN_DIM
            + self.n_num * ATTENTION_DIM
            + cat_dim
            + self.n_num
            + 1
        )

        self.final_norm = nn.LayerNorm(
            final_input_dim
        )

        self.final_head = nn.Sequential(
            nn.Linear(
                final_input_dim,
                RESIDUAL_HIDDEN,
            ),
            nn.LayerNorm(
                RESIDUAL_HIDDEN
            ),
            nn.SiLU(),
            nn.Dropout(
                RESIDUAL_DROPOUT
            ),
            nn.Linear(
                RESIDUAL_HIDDEN,
                RESIDUAL_HIDDEN // 2,
            ),
            nn.LayerNorm(
                RESIDUAL_HIDDEN // 2
            ),
            nn.SiLU(),
            nn.Dropout(
                RESIDUAL_DROPOUT
            ),
            nn.Linear(
                RESIDUAL_HIDDEN // 2,
                1,
            ),
        )

    def cat_flat(self, x_cat):
        """Flatten all learned categorical embeddings into one row vector."""
        embedded = []

        for j, col in enumerate(
            CATS
        ):
            embedded.append(
                self.cat_embeddings[col](
                    x_cat[:, j]
                )
            )

        return torch.cat(
            embedded,
            dim=1,
        )

    def forward(
        self,
        x_num,
        x_cat,
    ):
        # A tiny nonlinear network uses the complete numerical row to make a
        # small correction to each scalar before feature-local modeling.
        (
            contextualized_num,
            pre_context_scales,
            pre_context_correction,
        ) = self.pre_spline_context(
            x_num
        )

        # Each feature is still modeled independently after contextualization.
        # The spline/MoE encoder for feature i sees only contextualized_num[:, i].
        tokens, diagnostics = self.univariate_encoder(
            contextualized_num
        )

        # The additive path remains an auxiliary supervision signal and an
        # explicit decomposition available to the final joint predictor.
        (
            additive_logit,
            feature_scores,
        ) = self.additive_head(
            tokens
        )

        additive_probability = torch.sigmoid(
            additive_logit
        )

        uni_logit = self.uni_output_head(
            tokens
        )

        # Cross-feature interaction is learned only after univariate encoding.
        interaction_tokens = self.interaction_encoder(
            tokens
        )

        attn_logit = self.attn_output_head(
            interaction_tokens
        )

        cat_features = self.cat_flat(
            x_cat
        )

        # Deepest prediction path: direct classification from the complete
        # representation. No additive-logit-plus-residual constraint is imposed.
        final_input = torch.cat(
            [
                x_num,
                tokens.flatten(1),
                interaction_tokens.flatten(1),
                cat_features,
                feature_scores,
                additive_probability.unsqueeze(1),
            ],
            dim=1,
        )

        final_input = self.final_norm(
            final_input
        )

        final_input = F.normalize(
            final_input,
            p=2,
            dim=1,
        )

        base_final_logit = self.final_head(
            final_input
        ).squeeze(1)

        # Final Stage-1 prediction: adaptive convex mixture of all four logits.
        final_logit, mixer_weights = self.logit_mixer(
            additive_logit,
            uni_logit,
            attn_logit,
            base_final_logit,
        )

        return {
            "final_logit": final_logit,
            "base_final_logit": base_final_logit,
            "uni_logit": uni_logit,
            "attn_logit": attn_logit,
            "mixer_weights": mixer_weights,
            "final_probability": torch.sigmoid(
                final_logit
            ),
            "additive_logit": additive_logit,
            "additive_probability": additive_probability,
            "feature_scores": feature_scores,
            "pre_context_scales": pre_context_scales,
            "pre_context_correction": pre_context_correction,
            "diagnostics": diagnostics,
        }



# ==============================================================================
# LOSS, INFERENCE, AND FOLD PREPARATION
# ==============================================================================
# Training objective, batched inference, and leakage-safe fold construction.

def compute_loss(output, target):
    """Compute the mixed final loss plus supervision for intermediate heads.

    FINAL_LOSS_WEIGHT supervises the mixed output used for inference. ADD, UNI,
    and ATTN each receive auxiliary binary-cross-entropy losses so predictive
    signal is explicitly preserved at multiple representation depths. A very
    small entropy penalty discourages needlessly diffuse mixer weights.
    """
    final_loss = F.binary_cross_entropy_with_logits(
        output['final_logit'],
        target,
    )

    additive_loss = F.binary_cross_entropy_with_logits(
        output['additive_logit'],
        target,
    )

    uni_loss = F.binary_cross_entropy_with_logits(
        output['uni_logit'],
        target,
    )

    attn_loss = F.binary_cross_entropy_with_logits(
        output['attn_logit'],
        target,
    )

    mixer_weights = output['mixer_weights']
    mixer_entropy = -(
        mixer_weights
        * torch.log(mixer_weights.clamp_min(1e-8))
    ).sum(dim=1).mean()

    return (
        FINAL_LOSS_WEIGHT * final_loss
        + ADDITIVE_AUX_WEIGHT * additive_loss
        + UNI_AUX_WEIGHT * uni_loss
        + ATTN_AUX_WEIGHT * attn_loss
        + MIXER_ENTROPY_WEIGHT * mixer_entropy
    )

def count_params(model):
    """Count trainable model parameters."""
    return sum((p.numel() for p in model.parameters() if p.requires_grad))

@torch.no_grad()
def predict_model(model, Xn, Xc, batch_size=8192):
    """Run batched inference and return final and additive probabilities."""
    model.eval()

    final_parts = []
    add_parts = []

    for start in range(
        0,
        len(Xn),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(Xn),
        )

        out = model(
            Xn[start:end],
            Xc[start:end],
        )

        final_parts.append(
            out['final_probability']
            .detach()
            .cpu()
            .numpy()
        )

        add_parts.append(
            out['additive_probability']
            .detach()
            .cpu()
            .numpy()
        )

    return {
        'final': np.concatenate(
            final_parts
        ),
        'add': np.concatenate(
            add_parts
        ),
    }

def prepare_fold(fold, train_idx, val_idx):
    """Build one leakage-safe outer-fold dataset and all fold-specific spline metadata."""
    fold_train = train_full.iloc[train_idx].copy().reset_index(drop=True)
    fold_val = train_full.iloc[val_idx].copy().reset_index(drop=True)
    fold_test = test_full.copy().reset_index(drop=True)
    y_train = fold_train[TARGET].values.astype(np.float32)
    y_val = fold_val[TARGET].values.astype(np.float32)
    fold_train, fold_val, fold_test = apply_nested_encodings(train_df=fold_train, val_df=fold_val, test_df=fold_test, cat_cols=TARGET_ENC_CATS, target_col=TARGET, n_inner_splits=INNER_TE_SPLITS, smoothing=TE_SMOOTHING, random_state=SEED + fold)
    raw_train_fold = train_raw.iloc[train_idx].reset_index(drop=True)
    raw_val_fold = train_raw.iloc[val_idx].reset_index(drop=True)
    for col in RAW_NUMERICALS:
        missing_col = f'missing__{col}'
        fold_train[missing_col] = raw_train_fold[col].isna().astype('int8').astype(str).values
        fold_val[missing_col] = raw_val_fold[col].isna().astype('int8').astype(str).values
        fold_test[missing_col] = test_raw[col].isna().astype('int8').astype(str).values
    for col in NUMERICALS:
        for df in [fold_train, fold_val, fold_test]:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    for col in CATS:
        fold_train[col] = fold_train[col].astype(str)
        fold_val[col] = fold_val[col].astype(str)
        fold_test[col] = fold_test[col].astype(str)
    cat_maps = {}
    cat_sizes = {}
    for col in CATS:
        values = pd.concat([fold_train[col], fold_val[col], fold_test[col]], ignore_index=True)
        uniques = pd.Index(values.unique())
        mapping = {value: i + 1 for i, value in enumerate(uniques)}
        cat_maps[col] = mapping
        cat_sizes[col] = len(mapping) + 1

    def encode_cats(df):
        arrays = []
        for col in CATS:
            arrays.append(df[col].map(cat_maps[col]).fillna(0).astype(np.int64).values)
        return np.stack(arrays, axis=1)
    scaler = StandardScaler()
    Xn_train = scaler.fit_transform(fold_train[NUMERICALS].values).astype(np.float32)
    Xn_val = scaler.transform(fold_val[NUMERICALS].values).astype(np.float32)
    Xn_test = scaler.transform(fold_test[NUMERICALS].values).astype(np.float32)
    Xc_train = encode_cats(fold_train)
    Xc_val = encode_cats(fold_val)
    Xc_test = encode_cats(fold_test)
    feature_specs = {}
    for j, col in enumerate(NUMERICALS):
        if col in RAW_FEATURE_SPLINE_SPECS:
            feature_specs[col] = RAW_FEATURE_SPLINE_SPECS[col]
        else:
            feature_specs[col] = choose_engineered_specs(Xn_train[:, j])
    feature_knots = {}
    coarse_knots = {}
    for j, col in enumerate(NUMERICALS):
        values = Xn_train[:, j]
        feature_knots[col] = []
        for requested_knots, d_emb in feature_specs[col]:
            knots_np = build_unique_quantile_knots(values, requested_knots)
            feature_knots[col].append(torch.tensor(knots_np, dtype=torch.float32, device=DEVICE))
        n_unique = len(np.unique(values))
        coarse_n = min(32, max(4, n_unique))
        coarse_np = build_unique_quantile_knots(values, coarse_n)
        coarse_knots[col] = torch.tensor(coarse_np, dtype=torch.float32, device=DEVICE)
    return {'Xn_train': Xn_train, 'Xn_val': Xn_val, 'Xn_test': Xn_test, 'Xc_train': Xc_train, 'Xc_val': Xc_val, 'Xc_test': Xc_test, 'y_train': y_train, 'y_val': y_val, 'feature_knots': feature_knots, 'feature_specs': feature_specs, 'coarse_knots': coarse_knots, 'cat_sizes': cat_sizes}


# ==============================================================================
# TRAIN ONE OUTER FOLD
# ==============================================================================
# Train to the best validation AUC and restore that checkpoint.

class HyperAdaptableSplineTransformerModel(SplineTransformerModel):
    """Stage-1 model with three bounded adaptation sites exposed for Stage 2.

    During Stage 2 all Stage-1 parameters are frozen. The hypernetwork supplies
    row-dependent token scale/shift, post-attention scale, and final-input scale
    corrections. The four prediction heads and their learned mixer remain frozen
    and evaluate the corrected representations.
    """

    @torch.no_grad()
    def extract_controller_tokens(self, x_num):
        contextualized_num, _, _ = self.pre_spline_context(x_num)
        tokens, _ = self.univariate_encoder(contextualized_num)
        return tokens

    def forward_small_multisite(
        self,
        x_num,
        x_cat,
        token_scale,
        token_shift,
        attn_scale,
        head_scale,
    ):
        # Keep the strong pre-spline path untouched.
        contextualized_num, _, _ = self.pre_spline_context(x_num)
        tokens, _ = self.univariate_encoder(contextualized_num)

        # Site 1: main token adaptation.
        tokens = tokens * (1.0 + token_scale) + token_shift

        additive_logit, feature_scores = self.additive_head(tokens)
        additive_probability = torch.sigmoid(additive_logit)

        uni_logit = self.uni_output_head(
            tokens
        )

        interaction_tokens = self.interaction_encoder(tokens)

        # Site 2: small per-feature modulation after attention.
        interaction_tokens = interaction_tokens * (1.0 + attn_scale)

        attn_logit = self.attn_output_head(
            interaction_tokens
        )

        cat_features = self.cat_flat(x_cat)

        final_input = torch.cat([
            x_num,
            tokens.flatten(1),
            interaction_tokens.flatten(1),
            cat_features,
            feature_scores,
            additive_probability.unsqueeze(1),
        ], dim=1)

        final_input = self.final_norm(final_input)
        final_input = F.normalize(final_input, p=2, dim=1)

        # Site 3: small dimension-wise modulation before the frozen final head.
        final_input = final_input * (1.0 + head_scale)

        base_final_logit = self.final_head(final_input).squeeze(1)

        final_logit, mixer_weights = self.logit_mixer(
            additive_logit,
            uni_logit,
            attn_logit,
            base_final_logit,
        )

        return {
            'final_logit': final_logit,
            'base_final_logit': base_final_logit,
            'uni_logit': uni_logit,
            'attn_logit': attn_logit,
            'mixer_weights': mixer_weights,
            'final_probability': torch.sigmoid(final_logit),
            'additive_logit': additive_logit,
            'additive_probability': additive_probability,
            'feature_scores': feature_scores,
        }


class SmallMultiSiteHyperNetwork(nn.Module):
    """Compact row-conditioned controller used only in frozen-base Stage 2.

    The controller summarizes the frozen univariate token bank and predicts
    bounded corrections for three sites:

    1. per-feature token scale and directional token shift,
    2. per-feature post-attention scale,
    3. dimension-wise scale before the frozen FINAL MLP.

    All modulation heads are zero-initialized, so Stage 2 starts exactly at the
    restored Stage-1 prediction before learning corrections.
    """

    def __init__(
        self,
        n_num,
        token_dim,
        final_input_dim,
        bottleneck=HYPER_BOTTLENECK,
        hidden=HYPER_HIDDEN,
    ):
        super().__init__()

        self.local_summary = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, bottleneck),
            nn.SiLU(),
        )

        self.global_net = nn.Sequential(
            nn.LayerNorm(n_num * bottleneck),
            nn.Linear(n_num * bottleneck, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        self.token_scale_head = nn.Linear(hidden, n_num)
        self.token_shift_head = nn.Linear(hidden, n_num)
        self.attn_head = nn.Linear(hidden, n_num)
        self.final_head = nn.Linear(hidden, final_input_dim)

        for head in [
            self.token_scale_head,
            self.token_shift_head,
            self.attn_head,
            self.final_head,
        ]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.token_shift_direction = nn.Parameter(
            torch.randn(n_num, token_dim) * 0.02
        )

    def forward(self, frozen_tokens):
        h = self.local_summary(frozen_tokens).flatten(1)
        h = self.global_net(h)

        token_scale = (
            HYPER_TOKEN_SCALE_LIMIT
            * torch.tanh(self.token_scale_head(h))
        ).unsqueeze(2)

        shift_coeff = (
            HYPER_TOKEN_SHIFT_LIMIT
            * torch.tanh(self.token_shift_head(h))
        )

        shift_direction = F.normalize(
            self.token_shift_direction,
            p=2,
            dim=1,
        )

        token_shift = (
            shift_coeff.unsqueeze(2)
            * shift_direction.unsqueeze(0)
        )

        attn_scale = (
            HYPER_ATTN_SCALE_LIMIT
            * torch.tanh(self.attn_head(h))
        ).unsqueeze(2)

        head_scale = (
            HYPER_HEAD_SCALE_LIMIT
            * torch.tanh(self.final_head(h))
        )

        return {
            'token_scale': token_scale,
            'token_shift': token_shift,
            'token_shift_coeff': shift_coeff,
            'attn_scale': attn_scale,
            'head_scale': head_scale,
        }


@torch.no_grad()
def predict_hyper_adapted(base_model, hyper_model, Xn, Xc, batch_size=8192):
    base_model.eval()
    hyper_model.eval()

    final_parts = []
    add_parts = []

    for start in range(0, len(Xn), batch_size):
        end = min(start + batch_size, len(Xn))

        frozen_tokens = base_model.extract_controller_tokens(
            Xn[start:end]
        )

        mods = hyper_model(frozen_tokens)

        out = base_model.forward_small_multisite(
            Xn[start:end],
            Xc[start:end],
            mods['token_scale'],
            mods['token_shift'],
            mods['attn_scale'],
            mods['head_scale'],
        )

        final_parts.append(
            out['final_probability'].detach().cpu().numpy()
        )
        add_parts.append(
            out['additive_probability'].detach().cpu().numpy()
        )

    return {
        'final': np.concatenate(final_parts),
        'add': np.concatenate(add_parts),
    }



@torch.no_grad()
def evaluate_multilevel_heads(model, Xn, Xc, y, prefix='HEADS'):
    model.eval()

    add_all, uni_all, attn_all = [], [], []
    base_all, mix_all, w_all = [], [], []

    n = len(Xn)
    for start in range(0, n, 8192):
        end = min(start + 8192, n)

        xb_num = torch.from_numpy(
            Xn[start:end]
        ).to(
            DEVICE,
            non_blocking=True,
        )

        xb_cat = torch.from_numpy(
            Xc[start:end]
        ).to(
            DEVICE,
            non_blocking=True,
        )

        out = model(
            xb_num,
            xb_cat,
        )

        add_all.append(
            torch.sigmoid(out['additive_logit']).cpu().numpy()
        )
        uni_all.append(
            torch.sigmoid(out['uni_logit']).cpu().numpy()
        )
        attn_all.append(
            torch.sigmoid(out['attn_logit']).cpu().numpy()
        )
        base_all.append(
            torch.sigmoid(out['base_final_logit']).cpu().numpy()
        )
        mix_all.append(
            torch.sigmoid(out['final_logit']).cpu().numpy()
        )
        w_all.append(
            out['mixer_weights'].cpu().numpy()
        )

    add_auc = roc_auc_score(y, np.concatenate(add_all))
    uni_auc = roc_auc_score(y, np.concatenate(uni_all))
    attn_auc = roc_auc_score(y, np.concatenate(attn_all))
    base_auc = roc_auc_score(y, np.concatenate(base_all))
    mix_auc = roc_auc_score(y, np.concatenate(mix_all))

    mw = np.concatenate(w_all, axis=0).mean(axis=0)

    print(
        f'{prefix} | '
        f'ADD {add_auc:.6f} | '
        f'UNI {uni_auc:.6f} | '
        f'ATTN {attn_auc:.6f} | '
        f'BASEFINAL {base_auc:.6f} | '
        f'MIXED {mix_auc:.6f}'
    )
    print(
        f'{prefix} mixer mean | '
        f'ADD {mw[0]:.3f} | UNI {mw[1]:.3f} | '
        f'ATTN {mw[2]:.3f} | FINAL {mw[3]:.3f}'
    )


def train_one_fold(fold, train_idx, val_idx):
    print()
    print('=' * 130)
    print(f'FOLD {fold}')
    print('=' * 130)
    seed_everything(SEED + fold)
    data = prepare_fold(fold, train_idx, val_idx)

    Xn_train = torch.tensor(data['Xn_train'], dtype=torch.float32, device=DEVICE)
    Xn_val = torch.tensor(data['Xn_val'], dtype=torch.float32, device=DEVICE)
    Xn_test = torch.tensor(data['Xn_test'], dtype=torch.float32, device=DEVICE)
    Xc_train = torch.tensor(data['Xc_train'], dtype=torch.long, device=DEVICE)
    Xc_val = torch.tensor(data['Xc_val'], dtype=torch.long, device=DEVICE)
    Xc_test = torch.tensor(data['Xc_test'], dtype=torch.long, device=DEVICE)
    y_train_np = data['y_train']
    y_val_np = data['y_val']
    y_soft_np = y_train_np * (1.0 - LABEL_SMOOTHING_EPS) + 0.5 * LABEL_SMOOTHING_EPS
    y_soft = torch.tensor(y_soft_np, dtype=torch.float32, device=DEVICE)
    n_train = len(Xn_train)

    print()
    print('-' * 130)
    print('STAGE 1: BASE MODEL')
    print('-' * 130)
    model = HyperAdaptableSplineTransformerModel(
        feature_knots=data['feature_knots'],
        feature_specs=data['feature_specs'],
        coarse_knots=data['coarse_knots'],
        cat_sizes=data['cat_sizes'],
    ).to(DEVICE)
    print('Base parameters:', f'{count_params(model):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    best_add_auc = None
    best_context_mean = None
    best_context_max = None
    stale = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = torch.randperm(n_train, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0
        for batch_start in range(0, n_train, BATCH_SIZE):
            idx = permutation[batch_start:batch_start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            output = model(Xn_train[idx], Xc_train[idx])
            loss = compute_loss(output, y_soft[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        pred = predict_model(model, Xn_val, Xc_val)
        add_auc = roc_auc_score(y_val_np, pred['add'])
        final_auc = roc_auc_score(y_val_np, pred['final'])
        context_scales = torch.tanh(model.pre_spline_context.scale_raw).detach().cpu().numpy()
        context_mean = float(np.mean(np.abs(context_scales)))
        context_max = float(np.max(np.abs(context_scales)))
        print(
            f'BASE F{fold} E{epoch:02d} | Loss {epoch_loss / n_batches:.6f} '
            f'| ADD {add_auc:.6f} | FINAL {final_auc:.6f} '
            f'| CtxMean {context_mean:.4f} | CtxMax {context_max:.4f}'
        )
        if final_auc > best_auc:
            best_auc = final_auc
            best_epoch = epoch
            best_add_auc = add_auc
            best_context_mean = context_mean
            best_context_max = context_max
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            print(f'Fold {fold}: base early stopping.')
            break

    model.load_state_dict(best_state)
    evaluate_multilevel_heads(
        model,
        data['Xn_val'],
        data['Xc_val'],
        data['y_val'],
        prefix='BASE BEST HEADS',
    )
    model.eval()
    base_val = predict_model(model, Xn_val, Xc_val)
    base_test = predict_model(model, Xn_test, Xc_test)
    print()
    print(f'BASE BEST | E{best_epoch:02d} | ADD {best_add_auc:.6f} | FINAL {best_auc:.6f}')

    for p in model.parameters():
        p.requires_grad = False

    print()
    print('-' * 130)
    print('STAGE 2: SMALL MULTI-SITE HYPERNETWORK')
    print('-' * 130)

    seed_everything(SEED + HYPER_SEED_OFFSET + fold)

    final_input_dim = int(
        model.final_norm.normalized_shape[0]
    )

    hyper = SmallMultiSiteHyperNetwork(
        n_num=len(NUMERICALS),
        token_dim=MIX_TOKEN_DIM,
        final_input_dim=final_input_dim,
    ).to(DEVICE)

    print(
        'Hyper parameters:',
        f'{count_params(hyper):,}',
        '| limits:',
        f'tokenS={HYPER_TOKEN_SCALE_LIMIT:.2f}',
        f'tokenB={HYPER_TOKEN_SHIFT_LIMIT:.2f}',
        f'attn={HYPER_ATTN_SCALE_LIMIT:.2f}',
        f'head={HYPER_HEAD_SCALE_LIMIT:.2f}',
    )

    init_pred = predict_hyper_adapted(
        model, hyper, Xn_val, Xc_val
    )
    init_auc = roc_auc_score(
        y_val_np, init_pred['final']
    )
    init_max_diff = float(
        np.max(
            np.abs(
                init_pred['final'] - base_val['final']
            )
        )
    )

    print(
        f'HYPER INIT FINAL {init_auc:.6f} '
        f'| Base diff {init_auc - best_auc:+.8f} '
        f'| MaxPredDiff {init_max_diff:.8f}'
    )

    hyper_optimizer = torch.optim.AdamW(
        hyper.parameters(),
        lr=HYPER_LR,
        weight_decay=HYPER_WEIGHT_DECAY,
    )

    hyper_best_auc = best_auc
    hyper_best_epoch = 0
    hyper_best_state = copy.deepcopy(
        hyper.state_dict()
    )
    hyper_stale = 0

    for epoch in range(1, HYPER_EPOCHS + 1):
        hyper.train()
        model.eval()

        permutation = torch.randperm(
            n_train,
            device=DEVICE,
        )

        epoch_loss = 0.0
        n_batches = 0
        seen = 0

        sums = {
            'token_scale': 0.0,
            'token_shift': 0.0,
            'attn': 0.0,
            'head': 0.0,
        }
        maxs = {
            'token_scale': 0.0,
            'token_shift': 0.0,
            'attn': 0.0,
            'head': 0.0,
        }

        for batch_start in range(0, n_train, BATCH_SIZE):
            idx = permutation[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            hyper_optimizer.zero_grad(
                set_to_none=True
            )

            with torch.no_grad():
                frozen_tokens = model.extract_controller_tokens(
                    Xn_train[idx]
                )

            mods = hyper(frozen_tokens)

            output = model.forward_small_multisite(
                Xn_train[idx],
                Xc_train[idx],
                mods['token_scale'],
                mods['token_shift'],
                mods['attn_scale'],
                mods['head_scale'],
            )

            loss = F.binary_cross_entropy_with_logits(
                output['final_logit'],
                y_soft[idx],
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                hyper.parameters(),
                GRAD_CLIP,
            )

            hyper_optimizer.step()

            b = len(idx)
            seen += b
            epoch_loss += loss.item()
            n_batches += 1

            diagnostics = {
                'token_scale': mods['token_scale'],
                'token_shift': mods['token_shift_coeff'],
                'attn': mods['attn_scale'],
                'head': mods['head_scale'],
            }

            for name, tensor in diagnostics.items():
                a = tensor.detach().abs()
                sums[name] += float(a.mean().cpu()) * b
                maxs[name] = max(
                    maxs[name],
                    float(a.max().cpu()),
                )

        pred = predict_hyper_adapted(
            model, hyper, Xn_val, Xc_val
        )
        hyper_auc = roc_auc_score(
            y_val_np,
            pred['final'],
        )

        means = {
            k: v / max(seen, 1)
            for k, v in sums.items()
        }

        print(
            f'HYPER F{fold} E{epoch:02d} '
            f'| Loss {epoch_loss / n_batches:.6f} '
            f'| FINAL {hyper_auc:.6f} '
            f'| Gain {hyper_auc - best_auc:+.6f} '
            f'| TokS {means["token_scale"]:.4f}/{maxs["token_scale"]:.4f} '
            f'| TokB {means["token_shift"]:.4f}/{maxs["token_shift"]:.4f} '
            f'| Attn {means["attn"]:.4f}/{maxs["attn"]:.4f} '
            f'| Head {means["head"]:.4f}/{maxs["head"]:.4f}'
        )

        if hyper_auc > hyper_best_auc:
            hyper_best_auc = hyper_auc
            hyper_best_epoch = epoch
            hyper_best_state = copy.deepcopy(
                hyper.state_dict()
            )
            hyper_stale = 0
        else:
            hyper_stale += 1

        if hyper_stale >= HYPER_PATIENCE:
            print(
                f'Fold {fold}: hypernetwork early stopping.'
            )
            break

    hyper.load_state_dict(
        hyper_best_state
    )

    val_pred = predict_hyper_adapted(
        model, hyper, Xn_val, Xc_val
    )
    test_pred = predict_hyper_adapted(
        model, hyper, Xn_test, Xc_test
    )

    final_auc = roc_auc_score(
        y_val_np,
        val_pred['final'],
    )

    print()
    print(
        f'FOLD {fold} HYPER BEST '
        f'| Base E{best_epoch:02d} {best_auc:.6f} '
        f'| Hyper E{hyper_best_epoch:02d} {final_auc:.6f} '
        f'| Gain {final_auc - best_auc:+.6f}'
    )

    result = {
        'val_final': val_pred['final'],
        'val_add': base_val['add'],
        'test_final': test_pred['final'],
        'test_add': base_test['add'],
        'best_auc': final_auc,
        'best_epoch': hyper_best_epoch,
        'best_add_auc': best_add_auc,
        'best_context_mean': best_context_mean,
        'best_context_max': best_context_max,
    }

    del model, hyper, Xn_train, Xn_val, Xn_test, Xc_train, Xc_val, Xc_test, y_soft, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
