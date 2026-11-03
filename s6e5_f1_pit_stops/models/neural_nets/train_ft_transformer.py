#!/usr/bin/env python
"""FT-Transformer (pytorch_tabular) on the raw columns plus fold-local snapped-value categories and rare flags.

Features (competition rows only, no original rows; built inside each fold from the outer-train rows):
- 9 categoricals (strings): Driver, Compound, Race, Year, PitStop, LapNumber, Stint, Position, Position_Change;
- 5 continuous columns: TyreLife, LapTime_Delta, Cumulative_Degradation, LapTime (s), RaceProgress (missing -> outer-train
  median);
- for each of the 5 continuous columns, a categorical of the value snapped to the nearest value that occurs at least 25
  times in the outer-train rows (frequent values stay as they are; ties go to the larger value) and a categorical rare
  flag (value not frequent): 19 categorical + 5 continuous columns.
Model: pytorch_tabular FTTransformer (input_embed_dim 64, 4 attention blocks, 8 heads, attention and FF dropout 0.1, FF
hidden multiplier 4, other settings at the library defaults), AdamW lr 3e-4, batch 4096, up to 200 epochs with early
stopping on the held-out fold's loss (patience 20, min delta 1e-4); the best checkpoint (held-out loss) predicts.
pytorch_tabular seeds every fit with 42. Positive-class probabilities clipped to [1e-7, 1 - 1e-7]; test = mean of the
5 fold models.
Usage: python train_ft_transformer.py [--folds 0,1,2,3,4]"""
import argparse
import gc
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models.ft_transformer.config import FTTransformerConfig
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, cache_dir, folds, model_dir, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

PAIR = 'ft_transformer_snapped_simple_features'
MODEL_DIR = model_dir('neural_nets')
CKPT_DIR = cache_dir('neural_nets') / 'ft_transformer'  # pytorch_tabular checkpoints and Lightning logs
SEED = 42
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 4096
LEARNING_RATE = 3e-4
RARE_MIN_COUNT = 25

NUMS = ['TyreLife', 'LapTime_Delta', 'Cumulative_Degradation', 'LapTime (s)', 'RaceProgress']
CAT_BASE = ['Driver', 'Compound', 'Race', 'Year', 'PitStop', 'LapNumber', 'Stint', 'Position', 'Position_Change']
CAT_PROXY = [f'{c}__cat' for c in NUMS]
CAT_RARE = [f'{c}__is_rare' for c in NUMS]
CAT_COLS = CAT_BASE + CAT_PROXY + CAT_RARE
CONT_COLS = NUMS[:]


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def safe_set(obj, **kwargs):
    for k, v in kwargs.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


def strip_lr_from_optimizer_params(optimizer_config):
    if hasattr(optimizer_config, 'optimizer_params') and optimizer_config.optimizer_params is not None:
        if isinstance(optimizer_config.optimizer_params, dict) and 'lr' in optimizer_config.optimizer_params:
            optimizer_config.optimizer_params = {k: v for k, v in optimizer_config.optimizer_params.items() if k != 'lr'}


def preprocess_base(train: pd.DataFrame, test: pd.DataFrame):
    train = train.drop(columns=[ID])[CAT_BASE + NUMS + [TARGET]].copy()
    test = test.drop(columns=[ID])[CAT_BASE + NUMS].copy()
    train[TARGET] = train[TARGET].astype('int64')
    for df in (train, test):
        for c in NUMS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float32)
        for c in CAT_BASE:
            df[c] = df[c].astype('object').fillna('__MISSING__').astype(str)
    return train, test


def build_numeric_snapper(train_series: pd.Series, rare_min_count: int):
    """Frequent values: count >= rare_min_count in the outer-train rows. Rare values snap to the nearest frequent value."""
    s = pd.to_numeric(train_series, errors='coerce').astype(np.float32)
    vc = pd.Series(s).value_counts(dropna=False)
    frequent_vals = vc[vc >= rare_min_count].index.values
    frequent_vals = np.array([v for v in frequent_vals if pd.notna(v)], dtype=np.float32)
    if frequent_vals.size == 0:
        frequent_vals = np.array(pd.Series(s.dropna()).unique(), dtype=np.float32)
    frequent_vals = np.sort(frequent_vals)

    def transform(series_any: pd.Series):
        x = pd.to_numeric(series_any, errors='coerce').astype(np.float32).values
        is_nan = np.isnan(x)
        # rare = missing or not exactly one of the frequent values
        is_rare = (is_nan | ~np.isin(x, frequent_vals)).astype(np.int32)
        x_snapped = x.copy()
        if frequent_vals.size > 0:
            idx_snap = np.where((~is_nan) & (is_rare == 1))[0]
            if idx_snap.size > 0:
                v = x[idx_snap]
                pos = np.searchsorted(frequent_vals, v)
                pos = np.clip(pos, 0, len(frequent_vals) - 1)
                left = np.clip(pos - 1, 0, len(frequent_vals) - 1)
                right = pos
                left_vals = frequent_vals[left]
                right_vals = frequent_vals[right]
                choose_right = (np.abs(v - right_vals) <= np.abs(v - left_vals))
                x_snapped[idx_snap] = np.where(choose_right, right_vals, left_vals).astype(np.float32)
        return x_snapped.astype(np.float32), is_rare.astype(np.int32)

    return transform


def add_fold_features(tr_df: pd.DataFrame, va_df: pd.DataFrame, te_df: pd.DataFrame):
    tr_df = tr_df.copy()
    va_df = va_df.copy()
    te_df = te_df.copy()
    for c in NUMS:
        med = pd.to_numeric(tr_df[c], errors='coerce').median()
        med = 0.0 if pd.isna(med) else float(med)
        tr_df[c] = pd.to_numeric(tr_df[c], errors='coerce').fillna(med).astype(np.float32)
        va_df[c] = pd.to_numeric(va_df[c], errors='coerce').fillna(med).astype(np.float32)
        te_df[c] = pd.to_numeric(te_df[c], errors='coerce').fillna(med).astype(np.float32)
    for c in CAT_BASE:
        tr_df[c] = tr_df[c].astype('object').fillna('__MISSING__').astype(str)
        va_df[c] = va_df[c].astype('object').fillna('__MISSING__').astype(str)
        te_df[c] = te_df[c].astype('object').fillna('__MISSING__').astype(str)
    for col in NUMS:
        snapper = build_numeric_snapper(tr_df[col], rare_min_count=RARE_MIN_COUNT)
        tr_snap, tr_israre = snapper(tr_df[col])
        va_snap, va_israre = snapper(va_df[col])
        te_snap, te_israre = snapper(te_df[col])
        tr_df[f'{col}__cat'] = pd.Series(tr_snap).astype(str).values
        va_df[f'{col}__cat'] = pd.Series(va_snap).astype(str).values
        te_df[f'{col}__cat'] = pd.Series(te_snap).astype(str).values
        tr_df[f'{col}__is_rare'] = pd.Series(tr_israre).astype(str).values
        va_df[f'{col}__is_rare'] = pd.Series(va_israre).astype(str).values
        te_df[f'{col}__is_rare'] = pd.Series(te_israre).astype(str).values
    tr_out = tr_df[CAT_COLS + CONT_COLS + [TARGET]].copy()
    va_out = va_df[CAT_COLS + CONT_COLS + [TARGET]].copy()
    te_out = te_df[CAT_COLS + CONT_COLS].copy()
    return tr_out, va_out, te_out


def pick_prob_col(df_out: pd.DataFrame):
    prob_cols = [c for c in df_out.columns if 'prob' in c.lower()]
    if prob_cols:
        return prob_cols[-1]
    pred_cols = [c for c in df_out.columns if 'prediction' in c.lower()]
    if pred_cols:
        return pred_cols[-1]
    raise ValueError(f'Could not find prob/pred column in: {list(df_out.columns)}')


def make_common_configs(model_name: str, fold: int):
    data_config = DataConfig(target=[TARGET], categorical_cols=CAT_COLS, continuous_cols=CONT_COLS)
    trainer_config = TrainerConfig()
    safe_set(trainer_config, batch_size=BATCH_SIZE)
    safe_set(trainer_config, max_epochs=MAX_EPOCHS)
    safe_set(trainer_config, early_stopping='valid_loss')
    safe_set(trainer_config, early_stopping_patience=PATIENCE)
    safe_set(trainer_config, early_stopping_min_delta=1e-4)
    safe_set(trainer_config, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    safe_set(trainer_config, checkpoints='valid_loss')
    safe_set(trainer_config, load_best=True)
    safe_set(trainer_config, checkpoints_path=str(CKPT_DIR))
    safe_set(trainer_config, checkpoints_name=f'{model_name}_fold{fold}')
    safe_set(trainer_config, trainer_kwargs={'default_root_dir': str(CKPT_DIR)})
    optimizer_config = OptimizerConfig()
    safe_set(optimizer_config, optimizer='AdamW')
    strip_lr_from_optimizer_params(optimizer_config)
    return data_config, trainer_config, optimizer_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    run_folds = fold_list(ap.parse_args().folds)

    model_config = FTTransformerConfig(
        task='classification',
        learning_rate=LEARNING_RATE,
        input_embed_dim=64,
        num_attn_blocks=4,
        num_heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        ff_hidden_multiplier=4,
    )
    seed_everything(SEED)
    train, test = preprocess_base(read_train(), read_test())
    print(f'{len(CAT_COLS)} categorical + {len(CONT_COLS)} continuous columns; target mean {float(train[TARGET].mean()):.6f}', flush=True)
    y = train[TARGET].values

    fold_ids = folds(y)
    for k in range(5):
        if k not in run_folds:
            continue
        print(f'\n---- Fold {k} ----', flush=True)
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        tr_base = train.iloc[tr_idx].copy().reset_index(drop=True)
        va_base = train.iloc[va_idx].copy().reset_index(drop=True)
        te_base = test.copy().reset_index(drop=True)
        tr_df, va_df, te_df = add_fold_features(tr_base, va_base, te_base)

        data_config, trainer_config, optimizer_config = make_common_configs('ft_transformer', k)
        tabular_model = TabularModel(data_config=data_config, model_config=model_config, trainer_config=trainer_config, optimizer_config=optimizer_config)
        tabular_model.fit(train=tr_df, validation=va_df)
        va_out = tabular_model.predict(va_df.drop(columns=[TARGET]))
        te_out = tabular_model.predict(te_df)
        p_va = np.clip(va_out[pick_prob_col(va_out)].astype('float32').values, 1e-7, 1 - 1e-7)
        p_te = np.clip(te_out[pick_prob_col(te_out)].astype('float32').values, 1e-7, 1 - 1e-7)
        print(f'Fold {k} AUC = {roc_auc_score(y[va_idx], p_va):.6f}', flush=True)
        save_part(MODEL_DIR, PAIR, k, p_va, p_te)

        del tabular_model, tr_base, va_base, te_base, tr_df, va_df, te_df, va_out, te_out, p_va, p_te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
