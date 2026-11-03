#!/usr/bin/env python
"""GANDALF (pytorch_tabular) on the raw columns.

Inputs: the 15 string columns as categoricals (pytorch_tabular embeds them) and the 4 numeric columns (SeniorCitizen,
tenure, MonthlyCharges, TotalCharges) as float32 continuous columns (missing -> median over train + test). No feature
engineering, no original rows.
Model: pytorch_tabular GANDALF with the library defaults (6 GFLU stages, initial feature sparsity 0.3, learnable
sparsity), learning rate 3e-4, AdamW, batch 4096, up to 30 epochs with early stopping on the held-out fold's loss
(patience 5, min delta 1e-4). No best-checkpoint callback is set, so the model at the end of training predicts
P(Churn = Yes). Seed 42 once at the start; pytorch_tabular seeds every fit with its default seed (42).
Test = mean of the 5 fold models.
Lightning logs go to cache/gandalf_raw_columns/.
Usage: python train_gandalf_raw_columns.py [--folds 0,1,2,3,4]"""
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
from pytorch_tabular.models.gandalf.config import GANDALFConfig
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, cache_dir, folds, model_dir, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

PAIR = 'gandalf_raw_columns'
MODEL_DIR = model_dir('neural_nets')
LOG_DIR = cache_dir('neural_nets') / 'gandalf_raw_columns'
SEED = 42
MAX_EPOCHS = 30
BATCH_SIZE = 4096
LEARNING_RATE = 3e-4
PATIENCE = 5


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def infer_cat_cont_cols(df, target):
    cols = [c for c in df.columns if c != target]
    cat_cols, cont_cols = [], []
    for c in cols:
        if (pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(df[c])):
            cat_cols.append(c)
        else:
            cont_cols.append(c)
    return cat_cols, cont_cols


def preprocess(train, test):
    train = train.drop(columns=[ID])
    test = test.drop(columns=[ID])

    if train[TARGET].dtype == 'O':
        train[TARGET] = train[TARGET].map({'No': 0, 'Yes': 1}).astype('int64')
    else:
        train[TARGET] = train[TARGET].astype('int64')

    for df in (train, test):
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    tmp = pd.concat([train, test.assign(**{TARGET: 0})], axis=0, ignore_index=True)
    cat_cols, cont_cols = infer_cat_cont_cols(tmp, TARGET)

    for c in cat_cols:
        for df in (train, test):
            df[c] = df[c].astype('object').fillna('__MISSING__').astype(str)

    all_cont = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
    medians = {}
    for c in cont_cols:
        med = pd.to_numeric(all_cont[c], errors='coerce').median()
        medians[c] = 0.0 if pd.isna(med) else float(med)
    for c, med in medians.items():
        for df in (train, test):
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(med).astype('float32')
    return train, test, cat_cols, cont_cols


def pick_prob_col(df_out):
    """The P(class 1) column of a pytorch_tabular prediction frame (the last probability column)."""
    prob_cols = [c for c in df_out.columns if 'prob' in c.lower()]
    if prob_cols:
        return prob_cols[-1]
    pred_cols = [c for c in df_out.columns if 'prediction' in c.lower()]
    if pred_cols:
        return pred_cols[-1]
    raise ValueError(f'Could not find prob/pred column in: {list(df_out.columns)}')


def safe_set(obj, **kwargs):
    for k, v in kwargs.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


def make_common_configs(cat_cols, cont_cols):
    data_config = DataConfig(target=[TARGET], categorical_cols=cat_cols, continuous_cols=cont_cols)

    trainer_config = TrainerConfig()
    safe_set(trainer_config, batch_size=BATCH_SIZE)
    safe_set(trainer_config, max_epochs=MAX_EPOCHS)
    safe_set(trainer_config, checkpoints=None)
    safe_set(trainer_config, load_best=True)
    safe_set(trainer_config, early_stopping='valid_loss')
    safe_set(trainer_config, early_stopping_patience=PATIENCE)
    safe_set(trainer_config, early_stopping_min_delta=1e-4)
    safe_set(trainer_config, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    safe_set(trainer_config, trainer_kwargs={'default_root_dir': str(LOG_DIR)})

    optimizer_config = OptimizerConfig()
    safe_set(optimizer_config, optimizer='AdamW')
    # the learning rate is set on the model config, so optimizer_params must not carry one
    if isinstance(getattr(optimizer_config, 'optimizer_params', None), dict) and 'lr' in optimizer_config.optimizer_params:
        optimizer_config.optimizer_params = {k: v for k, v in optimizer_config.optimizer_params.items() if k != 'lr'}
    return data_config, trainer_config, optimizer_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(SEED)
    train, test = read_train(), read_test()
    print('train:', train.shape, 'test:', test.shape, flush=True)
    train, test, cat_cols, cont_cols = preprocess(train, test)
    print(f'categorical ({len(cat_cols)}): {cat_cols}', flush=True)
    print(f'continuous ({len(cont_cols)}): {cont_cols}', flush=True)

    data_config, trainer_config, optimizer_config = make_common_configs(cat_cols, cont_cols)
    model_config = GANDALFConfig(task='classification', learning_rate=LEARNING_RATE)
    y = train[TARGET].values
    fold_ids = folds(y)
    for k in fold_list(args.folds):
        print(f'\n---- Fold {k + 1}/5 ----', flush=True)
        tr_df = train.iloc[np.where(fold_ids != k)[0]].copy()
        va_idx = np.where(fold_ids == k)[0]
        va_df = train.iloc[va_idx].copy()

        tabular_model = TabularModel(data_config=data_config, model_config=model_config, trainer_config=trainer_config,
                                     optimizer_config=optimizer_config)
        tabular_model.fit(train=tr_df, validation=va_df)
        va_out = tabular_model.predict(va_df.drop(columns=[TARGET]))
        te_out = tabular_model.predict(test)
        va_pred = va_out[pick_prob_col(va_out)].astype('float32').values
        te_pred = te_out[pick_prob_col(te_out)].astype('float32').values
        print(f'Fold {k + 1} AUC = {roc_auc_score(y[va_idx], va_pred):.6f}', flush=True)
        save_part(MODEL_DIR, PAIR, k, va_pred, te_pred)

        del tabular_model, tr_df, va_df, va_out, te_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
