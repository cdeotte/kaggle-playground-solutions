#!/usr/bin/env python
"""FT-Transformer (pytorch_tabular) on the raw columns plus fold-local snapped-value categories and rare flags.

Inputs (25), built inside each fold from its training rows:
- the 16 categoricals as strings (NaN -> '__MISSING__');
- tenure, MonthlyCharges and TotalCharges as continuous columns (missing -> the fold's training median);
- for each of the 3 numerics, a categorical of the value snapped to the nearest value that occurs at least 25 times in
  the fold's training rows (frequent values stay as they are; ties go to the larger value) and a categorical rare flag
  (value not frequent).
Model: pytorch_tabular FTTransformer (input_embed_dim 64, 4 attention blocks, 8 heads, attention and FF dropout 0.1, FF
hidden multiplier 4, other settings at the library defaults), AdamW lr 3e-4, batch 4096, up to 200 epochs with early
stopping on the held-out fold's loss (patience 20, min delta 1e-4); the best checkpoint (held-out loss) predicts.
Python / numpy / torch are seeded with 42 before the folds and pytorch_tabular seeds every fit with 42.
Usage: python train_pytorch_tabular_ft_transformer.py"""
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
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, cache_dir, model_dir, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

PAIR = 'pytorch_tabular_ft_transformer_snapped_numerics'
MODEL_DIR = model_dir('neural_nets')
CKPT_DIR = str(cache_dir('neural_nets') / 'pytorch_tabular_ft_transformer')
DROP_COLS = ['customerID', 'id']
N_FOLDS = 5
SEED = 42
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 4096
LEARNING_RATE = 3e-4
RARE_MIN_COUNT = 25

BASE_FEATS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
              'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
              'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
CAT_BASE = [c for c in BASE_FEATS if c not in NUMS]
CAT_PROXY = [f'{c}__cat' for c in NUMS]
CAT_RARE = [f'{c}__is_rare' for c in NUMS]
CAT_COLS_25 = CAT_BASE + CAT_PROXY + CAT_RARE
CONT_COLS_25 = NUMS[:]


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


def clean_total_charges(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().replace('', np.nan)
    return pd.to_numeric(s, errors='coerce').astype(np.float32)


def preprocess_base(train: pd.DataFrame, test: pd.DataFrame):
    train = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
    test = test.drop(columns=[c for c in DROP_COLS if c in test.columns])
    train = train[[c for c in BASE_FEATS + [TARGET] if c in train.columns]].copy()
    test = test[[c for c in BASE_FEATS if c in test.columns]].copy()
    if train[TARGET].dtype == 'O':
        train[TARGET] = train[TARGET].map({'No': 0, 'Yes': 1}).astype('int64')
    else:
        train[TARGET] = train[TARGET].astype('int64')
    for df in (train, test):
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').astype(np.float32)
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').astype(np.float32)
        df['TotalCharges'] = clean_total_charges(df['TotalCharges'])
    for c in CAT_BASE:
        for df in (train, test):
            df[c] = df[c].astype('object').fillna('__MISSING__').astype(str)
    return train, test


def build_numeric_snapper(train_series: pd.Series, rare_min_count: int):
    """Frequent values: count >= rare_min_count in the fold's training rows. Rare values (and NaN) get the rare flag;
    rare finite values are snapped to the nearest frequent value."""
    s = pd.to_numeric(train_series, errors='coerce').astype(np.float32)
    vc = pd.Series(s).value_counts(dropna=False)
    frequent_vals = vc[vc >= rare_min_count].index.values
    frequent_vals = np.array([v for v in frequent_vals if pd.notna(v)], dtype=np.float32)
    if frequent_vals.size == 0:
        frequent_vals = np.array(pd.Series(s.dropna()).unique(), dtype=np.float32)
    frequent_vals = np.sort(frequent_vals)
    frequent_set = set(frequent_vals.tolist())

    def transform(series_any: pd.Series):
        x = pd.to_numeric(series_any, errors='coerce').astype(np.float32).values
        is_nan = np.isnan(x)
        is_rare = np.ones_like(x, dtype=np.int32)
        for i, v in enumerate(x):
            if np.isnan(v):
                is_rare[i] = 1
            else:
                is_rare[i] = 0 if float(v) in frequent_set else 1
        x_snapped = x.copy()
        if frequent_vals.size > 0:
            idx_snap = np.where((~is_nan) & (is_rare == 1))[0]
            if idx_snap.size > 0:
                v = x[idx_snap]
                pos = np.clip(np.searchsorted(frequent_vals, v), 0, len(frequent_vals) - 1)
                left = np.clip(pos - 1, 0, len(frequent_vals) - 1)
                left_vals, right_vals = frequent_vals[left], frequent_vals[pos]
                choose_right = (np.abs(v - right_vals) <= np.abs(v - left_vals))
                x_snapped[idx_snap] = np.where(choose_right, right_vals, left_vals).astype(np.float32)
        return x_snapped.astype(np.float32), is_rare.astype(np.int32)

    return transform


def add_fold_features(tr_df: pd.DataFrame, va_df: pd.DataFrame, te_df: pd.DataFrame):
    tr_df, va_df, te_df = tr_df.copy(), va_df.copy(), te_df.copy()
    for c in NUMS:
        med = pd.to_numeric(tr_df[c], errors='coerce').median()
        med = 0.0 if pd.isna(med) else float(med)
        for df in (tr_df, va_df, te_df):
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(med).astype(np.float32)
    for c in CAT_BASE:
        for df in (tr_df, va_df, te_df):
            df[c] = df[c].astype('object').fillna('__MISSING__').astype(str)
    for col in NUMS:
        snapper = build_numeric_snapper(tr_df[col], rare_min_count=RARE_MIN_COUNT)
        for df in (tr_df, va_df, te_df):
            snap, israre = snapper(df[col])
            df[f'{col}__cat'] = pd.Series(snap).astype(str).values
            df[f'{col}__is_rare'] = pd.Series(israre).astype(str).values
    return (tr_df[CAT_COLS_25 + CONT_COLS_25 + [TARGET]].copy(), va_df[CAT_COLS_25 + CONT_COLS_25 + [TARGET]].copy(),
            te_df[CAT_COLS_25 + CONT_COLS_25].copy())


def pick_prob_col(df_out: pd.DataFrame):
    prob_cols = [c for c in df_out.columns if 'prob' in c.lower()]
    if prob_cols:
        return prob_cols[-1]
    pred_cols = [c for c in df_out.columns if 'prediction' in c.lower()]
    if pred_cols:
        return pred_cols[-1]
    raise ValueError(f'Could not find prob/pred column in: {list(df_out.columns)}')


def make_common_configs(fold: int):
    data_config = DataConfig(target=[TARGET], categorical_cols=CAT_COLS_25, continuous_cols=CONT_COLS_25)
    trainer_config = TrainerConfig()
    safe_set(trainer_config, batch_size=BATCH_SIZE)
    safe_set(trainer_config, max_epochs=MAX_EPOCHS)
    safe_set(trainer_config, early_stopping='valid_loss')
    safe_set(trainer_config, early_stopping_patience=PATIENCE)
    safe_set(trainer_config, early_stopping_min_delta=1e-4)
    safe_set(trainer_config, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    safe_set(trainer_config, checkpoints='valid_loss')
    safe_set(trainer_config, load_best=True)
    safe_set(trainer_config, checkpoints_path=CKPT_DIR)
    safe_set(trainer_config, checkpoints_name=f'ft_transformer_fold{fold}')
    safe_set(trainer_config, enable_checkpointing=True)
    optimizer_config = OptimizerConfig()
    safe_set(optimizer_config, optimizer='AdamW')
    strip_lr_from_optimizer_params(optimizer_config)
    return data_config, trainer_config, optimizer_config


def main():
    seed_everything(SEED)
    train, test = preprocess_base(read_train(), read_test())
    print(f'{len(CAT_COLS_25)} categorical + {len(CONT_COLS_25)} continuous inputs', flush=True)
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

    y = train[TARGET].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float32)
    test_pred = np.zeros(len(test), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), start=1):
        print(f'\n---- Fold {fold}/{N_FOLDS} ----', flush=True)
        tr_df, va_df, te_df = add_fold_features(train.iloc[tr_idx].copy().reset_index(drop=True),
                                                train.iloc[va_idx].copy().reset_index(drop=True),
                                                test.copy().reset_index(drop=True))
        data_config, trainer_config, optimizer_config = make_common_configs(fold)
        tabular_model = TabularModel(data_config=data_config, model_config=model_config, trainer_config=trainer_config,
                                     optimizer_config=optimizer_config)
        tabular_model.fit(train=tr_df, validation=va_df)
        va_out = tabular_model.predict(va_df.drop(columns=[TARGET]))
        te_out = tabular_model.predict(te_df)
        oof[va_idx] = va_out[pick_prob_col(va_out)].astype('float32').values
        test_pred += te_out[pick_prob_col(te_out)].astype('float32').values / N_FOLDS
        print(f'Fold {fold} AUC = {roc_auc_score(y[va_idx], oof[va_idx]):.6f}', flush=True)

        del tabular_model, tr_df, va_df, te_df, va_out, te_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f'OOF AUC = {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(MODEL_DIR, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
