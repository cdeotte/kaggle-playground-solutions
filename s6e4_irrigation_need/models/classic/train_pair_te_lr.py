#!/usr/bin/env python
"""Pair target encodings -> logit polynomial expansion -> multinomial logistic regression (three presets).

All 171 pairs of the 19 columns (each column label-encoded over train + test; numerics by their exact value). Per
outer fold, per pair and per class: the mean of [class] over the pair's value combination (cuML TargetEncoder, no
smoothing, 5 inner folds: the outer-training rows get inner out-of-fold values, the held-out and test rows the means
over all outer-training rows) -> 513 columns, clipped to [1e-5, 1 - 1e-5], logit z, expanded to [z, z^2] or
[z, z^2, z^3], optionally + 9 threshold-score columns, standardized on the outer-training rows. Model: multinomial
logistic regression in PyTorch on the GPU, full batch, Adam lr 0.01, 1,000 epochs, class-balanced sample weights,
L2 weight decay 1 / (0.5 n) on the weights only. Test = mean of the 5 fold models. The original rows are not used.

Threshold score: 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10]
- 2 [stage Harvest] - 2 [stage Sowing] - [mulching used]. The 9 columns: score, score <= 0, score > 3, the 4 threshold
flags and two more numerics (ratio set: moisture / (rainfall + 1), temperature x wind; offset set: moisture - 25,
rainfall - 300).

Presets (fold = 1-5):
  lr_pair_te_logit3_a          stack A  [z, z^2, z^3] (1,539 columns); inner TE folds random, seed 42 + 10 fold + class;
                                        class weights of the outer-training rows; torch seed 42
  lr_pair_te_logit2_formula_a  stack A  [z, z^2] + 9 (ratio set) (1,035 columns); inner TE folds random,
                                        seed 42 + fold + class; class weights of the outer-training rows; torch seed 42
  lr_pair_te_logit3_formula_b  stack B  [z, z^2, z^3] + 9 (offset set) (1,548 columns); inner TE folds interleaved;
                                        class weights of all train rows; default torch seed
Usage: python train_pair_te_lr.py <preset> [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from cuml.preprocessing import TargetEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, ID, TARGET, folds, labels, model_dir, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

FOLDER = 'classic'
N_CLASSES = 3
SEED = 42
TE_N_FOLDS = 5
LR_C = 0.5
LR_EPOCHS = 1000
LR_LR = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# column order of the stack A presets (categoricals, then this numeric order); stack B uses the train.csv order
GROUPED_NUMS = ['Soil_pH', 'Organic_Carbon', 'Electrical_Conductivity', 'Sunlight_Hours', 'Wind_Speed_kmh',
                'Field_Area_hectare', 'Temperature_C', 'Soil_Moisture', 'Humidity', 'Previous_Irrigation_mm',
                'Rainfall_mm']

PRESETS = {
    'lr_pair_te_logit3_a': dict(
        column_order='grouped', degree=3, formula=None, class_weights='fold', torch_seed=SEED,
        te_kwargs=lambda fold, k: dict(seed=SEED + 10 * fold + k, split_method='random', stat='mean', output_type='cupy')),
    'lr_pair_te_logit2_formula_a': dict(
        column_order='grouped', degree=2, formula='ratio', class_weights='fold', torch_seed=SEED,
        te_kwargs=lambda fold, k: dict(seed=SEED + fold + k, split_method='random', stat='mean', output_type='cupy')),
    'lr_pair_te_logit3_formula_b': dict(
        column_order='csv', degree=3, formula='offset', class_weights='all', torch_seed=None,
        te_kwargs=lambda fold, k: dict()),
}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


class MultiLogReg(nn.Module):
    def __init__(self, in_features, out_features=3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


def threshold_score_features(df, kind):
    sm, rm, tc, ws = df['Soil_Moisture'], df['Rainfall_mm'], df['Temperature_C'], df['Wind_Speed_kmh']
    cgs = df['Crop_Growth_Stage'].astype(str)
    mu = df['Mulching_Used'].astype(str)
    score = ((sm < 25).astype(int) * 2 + (rm < 300).astype(int) * 2 + (tc > 30).astype(int) + (ws > 10).astype(int)
             - (cgs == 'Harvest').astype(int) * 2 - (cgs == 'Sowing').astype(int) * 2 - (mu == 'Yes').astype(int))
    cols = [score, score <= 0, score > 3, sm < 25, rm < 300, tc > 30, ws > 10]
    if kind == 'ratio':
        cols += [sm / (rm + 1.0), tc * ws]
    else:
        cols += [sm - 25, rm - 300]
    return np.column_stack([np.asarray(c, dtype=np.float32) for c in cols]).astype(np.float32)


def label_encode_combined(train_df, test_df, cols):
    """Codes in order of first appearance over train + test."""
    tr_out, te_out = train_df[cols].copy(), test_df[cols].copy()
    for c in cols:
        tr_s = train_df[c].astype(str).fillna('__MISSING__')
        te_s = test_df[c].astype(str).fillna('__MISSING__')
        uniq = pd.Index(pd.concat([tr_s, te_s], axis=0, ignore_index=True).unique())
        mapping = pd.Series(np.arange(len(uniq), dtype=np.int32), index=uniq)
        tr_out[c] = tr_s.map(mapping).astype(np.int32)
        te_out[c] = te_s.map(mapping).astype(np.int32)
    return tr_out, te_out


def logit_expand(arr, degree, eps=1e-5):
    arr = np.clip(arr, eps, 1.0 - eps)
    z = np.log(arr / (1.0 - arr)).astype(np.float32)
    return np.hstack([z ** p for p in range(1, degree + 1)]).astype(np.float32)


def train_pytorch_lr(X_tr, y_tr, sw, X_va, X_te, torch_seed):
    if torch_seed is not None:
        np.random.seed(torch_seed)
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    sw_t = torch.tensor(sw, dtype=torch.float32, device=DEVICE)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)

    model = MultiLogReg(in_features=X_tr.shape[1], out_features=N_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam([
        {'params': model.linear.weight, 'weight_decay': 1.0 / (LR_C * len(y_tr))},
        {'params': model.linear.bias, 'weight_decay': 0.0},
    ], lr=LR_LR)

    model.train()
    for _ in range(LR_EPOCHS):
        optimizer.zero_grad()
        loss = (criterion(model(X_tr_t), y_tr_t) * sw_t).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_probs = torch.softmax(model(X_va_t), dim=1).cpu().numpy().astype(np.float32)
        test_probs = torch.softmax(model(X_te_t), dim=1).cpu().numpy().astype(np.float32)
    del model, X_tr_t, y_tr_t, sw_t, X_va_t, X_te_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return val_probs, test_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preset', choices=sorted(PRESETS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    pair, cfg = args.preset, PRESETS[args.preset]
    out = model_dir(FOLDER)

    train, test = read_train(), read_test()
    y_train = labels(train[TARGET]).astype(np.int32)
    if cfg['column_order'] == 'grouped':
        features = list(CATEGORICALS) + GROUPED_NUMS
    else:
        features = [c for c in train.columns if c not in (ID, TARGET)]
    if cfg['formula'] is not None:
        formula_train = threshold_score_features(train, cfg['formula'])
        formula_test = threshold_score_features(test, cfg['formula'])
    if cfg['class_weights'] == 'all':
        cls_counts = np.bincount(y_train)
        sample_weights_all = (cls_counts.sum() / (N_CLASSES * cls_counts))[y_train]

    train_enc, test_enc = label_encode_combined(train, test, features)
    X_all_g = cudf.DataFrame({f: cudf.Series(train_enc[f].values).astype('int32') for f in features})
    X_test_g = cudf.DataFrame({f: cudf.Series(test_enc[f].values).astype('int32') for f in features})
    pair_cols = [(features[i], features[j]) for i in range(len(features)) for j in range(i + 1, len(features))]
    n_pair = len(pair_cols)
    log(f'{pair}: {len(features)} columns, {n_pair} pairs')

    fold_id = folds(y_train)
    for k in fold_list(args.folds):
        if have_part(out, pair, k):
            log(f'{pair} fold {k}: done')
            continue
        fold = k + 1
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        X_tr_g, X_va_g = X_all_g.iloc[tr_idx], X_all_g.iloc[va_idx]
        y_tr = y_train[tr_idx].astype(np.int32)

        tr_pair = np.zeros((len(tr_idx), n_pair * N_CLASSES), dtype=np.float32)
        va_pair = np.zeros((len(va_idx), n_pair * N_CLASSES), dtype=np.float32)
        te_pair = np.zeros((len(test), n_pair * N_CLASSES), dtype=np.float32)
        for t, (f1, f2) in enumerate(pair_cols):
            if t == 0 or (t + 1) % 50 == 0 or t + 1 == n_pair:
                log(f'  fold {k} pair {t + 1}/{n_pair}: {f1} x {f2}')
            X_tr_ij, X_va_ij, X_te_ij = X_tr_g[[f1, f2]], X_va_g[[f1, f2]], X_test_g[[f1, f2]]
            for c in range(N_CLASSES):
                y_c_g = cudf.Series((y_tr == c).astype(np.int32))
                te_enc = TargetEncoder(n_folds=TE_N_FOLDS, smooth=0, **cfg['te_kwargs'](fold, c))
                col = t * N_CLASSES + c
                tr_pair[:, col] = cp.asnumpy(te_enc.fit_transform(X_tr_ij, y_c_g)).ravel().astype(np.float32)
                te_enc.fit(X_tr_ij, y_c_g)
                va_pair[:, col] = cp.asnumpy(te_enc.transform(X_va_ij)).ravel().astype(np.float32)
                te_pair[:, col] = cp.asnumpy(te_enc.transform(X_te_ij)).ravel().astype(np.float32)

        X_tr = logit_expand(tr_pair, cfg['degree'])
        X_va = logit_expand(va_pair, cfg['degree'])
        X_te = logit_expand(te_pair, cfg['degree'])
        if cfg['formula'] is not None:
            X_tr = np.hstack([X_tr, formula_train[tr_idx]]).astype(np.float32)
            X_va = np.hstack([X_va, formula_train[va_idx]]).astype(np.float32)
            X_te = np.hstack([X_te, formula_test]).astype(np.float32)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_va = scaler.transform(X_va).astype(np.float32)
        X_te = scaler.transform(X_te).astype(np.float32)

        if cfg['class_weights'] == 'fold':
            classes = np.unique(y_tr)
            cw = dict(zip(classes, compute_class_weight('balanced', classes=classes, y=y_tr)))
            sw = np.array([cw[v] for v in y_tr], dtype=np.float32)
        else:
            sw = sample_weights_all[tr_idx]
        p_va, p_te = train_pytorch_lr(X_tr, y_tr, sw, X_va, X_te, cfg['torch_seed'])
        save_part(out, pair, k, p_va, p_te)
        log(f'{pair} fold {k}: {X_tr.shape[1]} columns, balanced accuracy '
            f'{balanced_accuracy_score(y_train[va_idx], p_va.argmax(axis=1)):.6f}')
        del X_tr, X_va, X_te, tr_pair, va_pair, te_pair, X_tr_g, X_va_g
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assemble(out, pair, fold_id, y_train)


if __name__ == '__main__':
    main()
