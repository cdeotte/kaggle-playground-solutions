#!/usr/bin/env python
"""FT-Transformer (pytorch-frame) on the raw columns + rule-score flags, categorical bigrams, floor-5 bins and
original-data class priors.

Features, built once on train + test (the original rows are a reference only, not training rows):
- the 8 categoricals and the 11 numerics;
- rule score 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10] -
  2 [stage Harvest] - 2 [stage Sowing] - [mulch Yes], is_low (score <= 0), is_high (score > 3) and the 4 threshold
  flags;
- 8 categorical bigrams; floor(x / 5) x 5 of Soil_Moisture, Rainfall_mm, Temperature_C and Wind_Speed_kmh as
  categoricals;
- per class k: the original-data share of class k in every categorical level (24 columns; unseen -> original prior).
42 numerical and 20 categorical columns. Column statistics are computed on each fold's training rows.
Model: torch_frame FTTransformer(channels 128, 4 layers, 3 outputs) with EmbeddingEncoder for the categoricals and
LinearPeriodicEncoder for the numericals.
Training: class-weighted cross-entropy (weights n / (3 n_class) from all train rows) with label smoothing 0.01, AdamW
lr 5e-4, weight decay 5e-5, OneCycle (12% warm-up, cosine, stepped per batch over 18 epochs), batch 1024, bf16 autocast,
gradient clip 1, up to 18 epochs with early stopping on the held-out balanced accuracy (patience 4); the best epoch
predicts (softmax). Seed 42 (torch / numpy) once before the folds. Test = mean of the 5 fold models.
Usage: python train_ft_transformer.py [--folds 0,1,2,3,4]"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch_frame import stype
from torch_frame.data import DataFrameToTensorFrameConverter
from torch_frame.data.loader import DataLoader
from torch_frame.data.stats import compute_col_stats
from torch_frame.nn import EmbeddingEncoder, FTTransformer, LinearPeriodicEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, ID, TARGET, TARGET_MAP, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'ft_transformer_formula_pairs_orig_priors_b'
MODEL_DIR = model_dir('neural_nets')
SEED = 42
CHANNELS = 128
LAYERS = 4
EPOCHS = 18
PATIENCE = 4
LR = 5e-4
WD = 5e-5
WARMUP = 0.12
BATCH = 1024
LABEL_SMOOTH = 0.01
GRAD_CLIP = 1.0
USE_AMP = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MISSING_TOKEN = '__MISSING__'

CATS = list(CATEGORICALS)
FORMULA_NUMS = ['Soil_Moisture', 'Rainfall_mm', 'Temperature_C', 'Wind_Speed_kmh']
LABEL = 'target'  # integer label column used by torch_frame


def to_safe_string_key(s):
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    return s.astype('str').fillna(MISSING_TOKEN)


def add_formula_features(df):
    SM, RM, TC, WS = df['Soil_Moisture'], df['Rainfall_mm'], df['Temperature_C'], df['Wind_Speed_kmh']
    CGS, MU = df['Crop_Growth_Stage'], df['Mulching_Used']
    score = ((SM < 25).astype(int) * 2
             + (RM < 300).astype(int) * 2
             + (TC > 30).astype(int)
             + (WS > 10).astype(int)
             - (CGS == 'Harvest').astype(int) * 2
             - (CGS == 'Sowing').astype(int) * 2
             - (MU == 'Yes').astype(int))
    df = df.copy()
    df['formula_score'] = score.astype(np.float32)
    df['is_low'] = (score <= 0).astype(np.float32)
    df['is_high'] = (score > 3).astype(np.float32)
    df['SM_below25'] = (SM < 25).astype(np.float32)
    df['RM_below300'] = (RM < 300).astype(np.float32)
    df['TC_above30'] = (TC > 30).astype(np.float32)
    df['WS_above10'] = (WS > 10).astype(np.float32)
    return df


def add_pair_categoricals(df):
    df = df.copy()
    pairs = [
        ('Crop_Growth_Stage', 'Mulching_Used'),
        ('Crop_Growth_Stage', 'Season'),
        ('Crop_Type', 'Soil_Type'),
        ('Crop_Type', 'Season'),
        ('Crop_Type', 'Crop_Growth_Stage'),
        ('Region', 'Season'),
        ('Irrigation_Type', 'Water_Source'),
        ('Soil_Type', 'Region'),
    ]
    for c1, c2 in pairs:
        df[f'bi_{c1}_{c2}'] = to_safe_string_key(df[c1].astype(str) + '__' + df[c2].astype(str))
    return df


def add_numeric_bins(df):
    df = df.copy()
    for col in FORMULA_NUMS:
        binned = pd.Series((np.floor(df[col].values / 5) * 5).astype(int).astype(str), index=df.index)
        df[f'{col}_floor5'] = to_safe_string_key(binned)
    return df


def add_orig_mappings(df, orig_df):
    df = df.copy()
    for c in CATS:
        for k in range(3):
            m = orig_df.groupby(c).apply(lambda g, kk=k: (g[LABEL] == kk).mean())
            global_mean = float((orig_df[LABEL] == k).mean())
            df[f'orig_mean_k{k}_{c}'] = df[c].map(m).fillna(global_mean).astype(np.float32)
    return df


def engineer_features(df, orig_df):
    df = add_formula_features(df)
    df = add_pair_categoricals(df)
    df = add_numeric_bins(df)
    return add_orig_mappings(df, orig_df)


def detect_col_types(df, target_col):
    num_cols, cat_cols = [], []
    for col in df.columns:
        if col == target_col:
            continue
        dt = df[col].dtype
        if isinstance(dt, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dt) or pd.api.types.is_string_dtype(dt):
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return num_cols, cat_cols


def build_model(col_stats, col_names_dict):
    model = FTTransformer(
        channels=CHANNELS,
        out_channels=3,
        num_layers=LAYERS,
        col_stats=col_stats,
        col_names_dict=col_names_dict,
        stype_encoder_dict={stype.categorical: EmbeddingEncoder(), stype.numerical: LinearPeriodicEncoder()},
    )
    return model.to(DEVICE)


def train_epoch(model, loader, optimizer, scheduler, scaler, cls_weights_t):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=USE_AMP):
            preds = model(batch)
            loss = F.cross_entropy(preds, batch.y.long().squeeze(-1), weight=cls_weights_t, label_smoothing=LABEL_SMOOTH)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs_list = []
    for batch in loader:
        batch = batch.to(DEVICE)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=USE_AMP):
            logits = model(batch)
        probs_list.append(torch.softmax(logits, dim=-1).float().cpu().numpy())
    return np.vstack(probs_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    train, test, orig = read_train(), read_test(), read_original()
    train[LABEL] = train[TARGET].map(TARGET_MAP)
    orig[LABEL] = orig[TARGET].map(TARGET_MAP)
    y = train[LABEL].values
    print(f'Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}, device {DEVICE}', flush=True)

    cls_counts = np.bincount(y)
    cls_weights = cls_counts.sum() / (3 * cls_counts)
    cls_weights_t = torch.tensor(cls_weights, dtype=torch.float32).to(DEVICE)
    print(f'Class weights: {cls_weights}', flush=True)

    all_data = pd.concat([train, test], axis=0, ignore_index=True).drop(columns=[TARGET, ID])
    all_data = engineer_features(all_data, orig)
    n_train = len(train)
    train_fe = all_data.iloc[:n_train].reset_index(drop=True)
    test_fe = all_data.iloc[n_train:].reset_index(drop=True)

    num_cols_fe, cat_cols_fe = detect_col_types(train_fe, LABEL)
    print(f'Numeric cols: {len(num_cols_fe)}, Categorical cols: {len(cat_cols_fe)}', flush=True)
    col_to_stype = {c: stype.numerical for c in num_cols_fe}
    col_to_stype.update({c: stype.categorical for c in cat_cols_fe})
    col_to_stype[LABEL] = stype.numerical

    test_fe_with_target = test_fe.copy()
    test_fe_with_target[LABEL] = 0

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fold_ids = folds(y)
    for k in fold_list(args.folds):
        print(f'\n=== Fold {k + 1}/5 ===', flush=True)
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        tr_df = train_fe.iloc[tr_idx].reset_index(drop=True)
        va_df = train_fe.iloc[va_idx].reset_index(drop=True)
        y_va = y[va_idx]

        # column statistics from the fold's training rows
        col_stats = {col: compute_col_stats(tr_df[col], st) for col, st in col_to_stype.items()}
        converter = DataFrameToTensorFrameConverter(col_to_stype=col_to_stype, col_stats=col_stats, target_col=LABEL)
        cols = list(col_to_stype.keys())
        loader_tr = DataLoader(converter(tr_df[cols]), batch_size=BATCH, shuffle=True)
        loader_va = DataLoader(converter(va_df[cols]), batch_size=BATCH * 4, shuffle=False)
        loader_test = DataLoader(converter(test_fe_with_target[cols]), batch_size=BATCH * 4, shuffle=False)

        model = build_model(col_stats, converter.col_names_dict)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = OneCycleLR(optimizer, max_lr=LR, total_steps=EPOCHS * len(loader_tr), pct_start=WARMUP,
                               anneal_strategy='cos')
        scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

        best_bac = 0.0
        best_state = None
        no_improve = 0
        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_epoch(model, loader_tr, optimizer, scheduler, scaler, cls_weights_t)
            va_bac = balanced_accuracy_score(y_va, np.argmax(predict(model, loader_va), axis=1))
            print(f'  Ep {epoch:2d}/{EPOCHS} | loss={tr_loss:.4f} | val_bac={va_bac:.5f}', flush=True)
            if va_bac > best_bac:
                best_bac = va_bac
                best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f'  Early stop at epoch {epoch} (best={best_bac:.5f})', flush=True)
                    break

        model.load_state_dict(best_state)
        va_probs = predict(model, loader_va)
        test_probs = predict(model, loader_test)
        print(f'  Fold {k + 1} BAC = {balanced_accuracy_score(y_va, np.argmax(va_probs, axis=1)):.5f}', flush=True)
        save_part(MODEL_DIR, PAIR, k, va_probs, test_probs)
        del model, best_state
        torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
