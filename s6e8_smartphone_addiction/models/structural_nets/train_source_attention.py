#!/usr/bin/env python
"""source_attention_net: mask-routed source-attention network over original-row donors (lib/attention_model.py).

Inputs per row: 404 context features (the 333-column fold-specific forensic matrix of ../lgbm_margin + the first 71
compact reconstruction columns of ../realmlp_repair), standardized with outer-train mean / std (NaN -> 0, clip +-10);
the 12 raw fields (numerics robust-scaled with the original data's median and (q75 - q25) / 1.349, clip +-8;
categoricals as original-level codes / (levels - 1)) with their observed / missing flags; the 6 x 16 donors of
cache/donors (build_donors.py) with the donors' original values, original label and severity ((Mild, Moderate, Severe)
-> (0, 0.5, 1), missing -> label); and the forensic column strong__selected_base_margin as the base margin (clip +-12).

Training: AdamW (lr 2e-3, weight decay 2e-4), one-cycle schedule (20% warm-up, div 10, final div 50), batch 2048, bf16
autocast, BCE with label smoothing 0.002, gradient clipping 2.0, + 0.015 x observed-numeric reconstruction MSE
(attention_aux). Protocol per outer fold k (seed 1701 + k): the outer-train rows are split by splitmix64(id) mod 5
(remainder 0 = selector, 20%); a model trains 8 epochs on the other 80% and the epoch with the best selector AUC is
kept; then a fresh model with the same seed is trained on all outer-train rows for that many epochs (new one-cycle
schedule, standardization refit on all outer-train rows) and predicts the held-out fold and the test rows.
Test = mean of the 5 fold models.

Variants (--variant): attention_aux (the pair, source_attention_net), attention_noaux (no auxiliary loss),
context_control (no donor attention).
Usage: python train_source_attention.py [--folds 0,1,2,3,4] [--variant attention_aux]"""
import argparse
import gc
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # deterministic cuBLAS; set before CUDA starts

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib.attention_model import VARIANTS, SourceAttentionModel  # noqa: E402
from lib.donors import CATEGORICAL, TOP_K, encode_queries, fit_retrieval_state, load_donors  # noqa: E402
from lib.inputs import (FORENSIC_WIDTH, RECONSTRUCTION_COMPACT, forensic_columns, load_forensic,  # noqa: E402
                        load_reconstruction_compact)

PAIR = {'attention_aux': 'source_attention_net', 'attention_noaux': 'source_attention_net_noaux',
        'context_control': 'source_context_net'}
BATCH_SIZE = 2048
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0002
LABEL_SMOOTHING = 0.002
SELECTOR_EPOCHS = 8
SEED_BASE = 1701
DEVICE = 'cuda'


def splitmix64(values, salt=0):
    with np.errstate(over='ignore'):
        value = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15 + salt)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return value ^ (value >> np.uint64(31))


@dataclass
class Context:
    """Fold context matrices per role ('train' = outer-train rows, 'valid' = held-out rows, 'test') and the global row
    ids of each role (train.csv rows for train / valid, test.csv rows for test)."""
    arrays: dict
    rows: dict
    base_margin_index: int

    def batch(self, role, local):
        forensic, reconstruction = self.arrays[role]
        return np.concatenate([np.asarray(forensic[local], dtype=np.float32),
                               np.asarray(reconstruction[local], dtype=np.float32)], axis=1)


def load_context(k, f, n_test):
    rows = {'train': np.flatnonzero(f != k), 'valid': np.flatnonzero(f == k), 'test': np.arange(n_test)}
    arrays = {}
    for role in ('train', 'valid', 'test'):
        forensic, reconstruction = load_forensic(k, role), load_reconstruction_compact(k, role)
        if len(forensic) != len(rows[role]) or len(reconstruction) != len(rows[role]):
            raise RuntimeError(f'fold {k} {role}: context rows do not match the fold')
        arrays[role] = (forensic, reconstruction)
    base_margin_index = forensic_columns(k, ['strong__selected_base_margin'])[0]
    return Context(arrays=arrays, rows=rows, base_margin_index=base_margin_index)


def fit_context_state(ctx, local_indices):
    """Mean / std of the 404 context columns over the given outer-train rows (finite values only)."""
    width = FORENSIC_WIDTH + RECONSTRUCTION_COMPACT
    sums, squares, counts = np.zeros(width), np.zeros(width), np.zeros(width, dtype=np.int64)
    for offset in range(0, len(local_indices), 32768):
        value = ctx.batch('train', local_indices[offset:offset + 32768]).astype(np.float64)
        finite = np.isfinite(value)
        safe = np.where(finite, value, 0.0)
        sums += safe.sum(axis=0)
        squares += np.square(safe).sum(axis=0)
        counts += finite.sum(axis=0)
    center = sums / np.maximum(counts, 1)
    variance = squares / np.maximum(counts, 1) - np.square(center)
    scale = np.sqrt(np.maximum(variance, 1e-8))
    center[~np.isfinite(center)] = 0.0
    scale[~np.isfinite(scale) | (scale < 1e-4)] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


@dataclass(frozen=True)
class RawArrays:
    numeric: np.ndarray
    numeric_observed: np.ndarray
    categorical_scaled: np.ndarray
    categorical_observed: np.ndarray
    missing: np.ndarray


def make_raw_arrays(frame, state):
    numeric, categorical = encode_queries(frame, state)
    numeric_observed = np.isfinite(numeric).astype(np.float32)
    numeric_scaled = (numeric - state.center[None, :]) / state.scale[None, :]
    numeric_scaled = np.where(np.isfinite(numeric_scaled), numeric_scaled, 0.0)
    numeric_scaled = np.clip(numeric_scaled, -8.0, 8.0).astype(np.float32)
    categorical_observed = (categorical >= 0).astype(np.float32)
    categorical_scaled = np.zeros_like(categorical, dtype=np.float32)
    for j, name in enumerate(CATEGORICAL):
        denominator = max(len(state.category_levels[name]) - 1, 1)
        categorical_scaled[:, j] = np.where(categorical[:, j] >= 0,
                                            categorical[:, j].astype(np.float32) / np.float32(denominator), 0.0)
    missing = np.concatenate([1.0 - numeric_observed, 1.0 - categorical_observed], axis=1).astype(np.float32)
    return RawArrays(numeric=numeric_scaled, numeric_observed=numeric_observed, categorical_scaled=categorical_scaled,
                     categorical_observed=categorical_observed, missing=missing)


def source_model_arrays(source, state):
    """Donor attributes: robust-scaled numerics, scaled categorical codes, original label, severity in [0, 1]."""
    source_numeric = (state.numeric - state.center[None, :]) / state.scale[None, :]
    source_numeric = np.clip(source_numeric, -8.0, 8.0).astype(np.float32)
    source_categorical = np.zeros_like(state.categorical, dtype=np.float32)
    for j, name in enumerate(CATEGORICAL):
        denominator = max(len(state.category_levels[name]) - 1, 1)
        source_categorical[:, j] = state.categorical[:, j] / np.float32(denominator)
    source_target = source[TARGET].to_numpy(np.float32)
    severity = source['addiction_level'].map({'Mild': 1.0, 'Moderate': 2.0, 'Severe': 3.0}).to_numpy(np.float32)
    fallback = np.float32(1.0) + np.float32(2.0) * source_target
    severity = np.where(np.isfinite(severity), severity, fallback)
    severity = ((severity - np.float32(1.0)) / np.float32(2.0)).astype(np.float32)
    return source_numeric, source_categorical, source_target, severity


def model_batch(ctx, role, local, center, scale, raw, donors, targets):
    raw_context = ctx.batch(role, local)
    base_margin = np.asarray(raw_context[:, ctx.base_margin_index], dtype=np.float32)
    base_margin = np.clip(np.where(np.isfinite(base_margin), base_margin, 0.0), -12.0, 12.0)
    context = (raw_context - center[None, :]) / scale[None, :]
    context = np.clip(np.where(np.isfinite(context), context, 0.0), -10.0, 10.0).astype(np.float32)
    rows = ctx.rows[role][local]

    def t(value, dtype=torch.float32):
        return torch.as_tensor(np.asarray(value), dtype=dtype, device=DEVICE)
    payload = {
        'context': t(context), 'base_margin': t(base_margin), 'raw_numeric': t(raw.numeric[rows]),
        'raw_numeric_observed': t(raw.numeric_observed[rows]), 'raw_categorical_scaled': t(raw.categorical_scaled[rows]),
        'raw_categorical_observed': t(raw.categorical_observed[rows]), 'missing_mask': t(raw.missing[rows]),
        'donor_indices': t(np.asarray(donors.indices[rows], dtype=np.int64), torch.long),
        'donor_score': t(np.asarray(donors.score[rows], dtype=np.float32)),
        'donor_exact': t(np.asarray(donors.exact[rows], dtype=np.float32)),
        'coverage': t(np.asarray(donors.coverage[rows], dtype=np.float32)),
    }
    y = None if targets is None else t(targets[rows])
    return payload, y


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')


def predict(model, ctx, role, local, center, scale, raw, donors):
    model.eval()
    parts = []
    with torch.no_grad():
        for offset in range(0, len(local), 4096):
            payload, _ = model_batch(ctx, role, local[offset:offset + 4096], center, scale, raw, donors, None)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, _ = model(**payload)
            parts.append(torch.sigmoid(logits).to(torch.float64).cpu().numpy())
    return np.concatenate(parts)


def train_epochs(model, auxiliary_weight, ctx, fit_local, epochs, seed, center, scale, raw, donors, y,
                 selector_local=None):
    """Trains `epochs` epochs on the outer-train rows `fit_local` (one-cycle schedule over exactly these epochs). With
    selector rows, scores them after every epoch and returns the epoch with the best selector AUC."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = int(np.ceil(len(fit_local) / BATCH_SIZE))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=steps_per_epoch * epochs,
                                                    pct_start=0.2, div_factor=10.0, final_div_factor=50.0)
    rng = np.random.default_rng(seed)
    best_auc, best_epoch = -np.inf, 1
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(fit_local)
        total, count = 0.0, 0
        for offset in range(0, len(order), BATCH_SIZE):
            local = order[offset:offset + BATCH_SIZE]
            payload, yb = model_batch(ctx, 'train', local, center, scale, raw, donors, y)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, reconstruction = model(**payload)
                smooth_y = yb * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
                loss = F.binary_cross_entropy_with_logits(logits, smooth_y)
                if reconstruction is not None and auxiliary_weight > 0.0:
                    observed = payload['raw_numeric_observed']
                    squared = torch.square(reconstruction - payload['raw_numeric']) * observed
                    loss = loss + auxiliary_weight * (squared.sum() / torch.clamp(observed.sum(), min=1.0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(local)
            count += len(local)
        message = f'  epoch {epoch}: train loss {total / max(count, 1):.5f}'
        if selector_local is not None:
            prediction = predict(model, ctx, 'train', selector_local, center, scale, raw, donors)
            auc = roc_auc_score(y[ctx.rows['train'][selector_local]], prediction)
            message += f', selector AUC {auc:.6f}'
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch
        print(message, flush=True)
    return best_epoch


def run_fold(k, f, y, variant, data):
    ctx = load_context(k, f, len(data['raw_test'].numeric))
    train_local = np.arange(len(ctx.rows['train']), dtype=np.int64)
    partition = splitmix64(data['ids'][ctx.rows['train']]) % np.uint64(5)
    selector_local, fit_local = train_local[partition == 0], train_local[partition != 0]
    seed = SEED_BASE + k
    auxiliary_weight = VARIANTS[variant]['auxiliary_weight']

    def new_model():
        tensors = [torch.as_tensor(value) for value in data['source_arrays']]
        return SourceAttentionModel(FORENSIC_WIDTH + RECONSTRUCTION_COMPACT, *tensors, variant=variant,
                                    top_k=TOP_K).to(DEVICE)

    # 1) epoch selection on the 80 / 20 id-hash split of the outer-train rows
    center, scale = fit_context_state(ctx, fit_local)
    seed_everything(seed)
    model = new_model()
    print(f'fold {k}: selection run ({len(fit_local)} fit / {len(selector_local)} selector rows)', flush=True)
    epochs = train_epochs(model, auxiliary_weight, ctx, fit_local, SELECTOR_EPOCHS, seed, center, scale, data['raw_train'],
                          data['donors_train'], y, selector_local=selector_local)
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # 2) refit on all outer-train rows for the selected number of epochs
    center, scale = fit_context_state(ctx, train_local)
    seed_everything(seed)
    model = new_model()
    print(f'fold {k}: refit on {len(train_local)} rows for {epochs} epochs', flush=True)
    train_epochs(model, auxiliary_weight, ctx, train_local, epochs, seed, center, scale, data['raw_train'],
                 data['donors_train'], y)
    pv = predict(model, ctx, 'valid', np.arange(len(ctx.rows['valid'])), center, scale, data['raw_train'],
                 data['donors_train'])
    pt = predict(model, ctx, 'test', np.arange(len(ctx.rows['test'])), center, scale, data['raw_test'],
                 data['donors_test'])
    save_part(HERE, PAIR[variant], k, pv, pt)
    print(f'fold {k}: AUC {roc_auc_score(y[f == k], pv):.6f} (epochs {epochs})', flush=True)
    del model
    torch.cuda.empty_cache()
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--variant', default='attention_aux', choices=list(VARIANTS))
    a = ap.parse_args()
    train, test, source = read_train(), read_test(), read_original(keep_level=True)
    y = labels(train[TARGET])
    f = folds(y)
    state = fit_retrieval_state(source)
    data = {
        'ids': train['id'].to_numpy(np.int64),
        'raw_train': make_raw_arrays(train, state),
        'raw_test': make_raw_arrays(test, state),
        'donors_train': load_donors('train'),
        'donors_test': load_donors('test'),
        'source_arrays': source_model_arrays(source, state),
    }
    for k in fold_list(a.folds):
        if have_part(HERE, PAIR[a.variant], k):
            print(f'fold {k}: done', flush=True)
            continue
        run_fold(k, f, y, a.variant, data)
    assemble(HERE, PAIR[a.variant], f, y)


if __name__ == '__main__':
    main()
