#!/usr/bin/env python
"""GANDALF (pytorch_tabular) on F1 state features, rounded / rank-bin keys of the high-cardinality numerics, combo keys
and original-data priors.

Features (built once on train + test, with the original rows as reference only; the original rows are not trained on):
- the 14 columns: Driver, Compound, Race and every numeric with < 100 distinct values in train + test are categoricals
  (strings; the numerics as their float32 values, e.g. '2022.0'); the other numerics stay continuous;
- 6 F1 state features: LapNumber - TyreLife, Cumulative_Degradation / (TyreLife + 1), LapTime_Delta / (|LapTime| + 1),
  Position / (RaceProgress + 0.02), PitStop + RaceProgress, Stint / (TyreLife + 1);
- for the high-cardinality numerics LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress: the value rounded to
  1 and to 2 decimals and a 32-bin rank key (bin edges from the ranks over train + test + original, applied to the ranks
  within each of the three frames), as categoricals;
- 6 combo keys: Driver|Compound, Driver|Race, Race|Year, Compound|TyreLife, Stint|TyreLife, Driver|PitStop;
- original-data target mean (unseen keys -> original prior) and log1p count of every categorical / key and of the 4
  high-cardinality numerics (all 101,371 original rows).
Continuous inputs: missing -> outer-train median.
Model: pytorch_tabular GANDALF with the library defaults (6 GFLU stages, feature-init sparsity 0.3, learnable sparsity,
no GFLU dropout), learning rate 3e-4, AdamW, batch 4096, up to 30 epochs with early stopping on the held-out fold's loss
(patience 5, min delta 1e-4); the best checkpoint (held-out loss) predicts. pytorch_tabular seeds every fit with 42.
Test = mean of the 5 fold models (positive-class probabilities).
Usage: python train_gandalf.py [--folds 0,1,2,3,4]"""
import argparse
import gc
import os
import random
import re
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
from common.data import ID, TARGET, cache_dir, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

PAIR = 'gandalf_artifact_combos_original_priors'
MODEL_DIR = model_dir('neural_nets')
CKPT_DIR = cache_dir('neural_nets') / 'gandalf'  # pytorch_tabular checkpoints and Lightning logs
SEED = 42
MAX_EPOCHS = 30
BATCH_SIZE = 4096
LEARNING_RATE = 3e-4
PATIENCE = 5
MISSING_TOKEN = '__MISSING__'


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def safe_name(x):
    return re.sub(r'[^0-9A-Za-z_]+', '_', str(x)).strip('_')


def safe_num(s):
    return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')


def safe_str(s):
    return s.astype('object').fillna(MISSING_TOKEN).astype(str)


def infer_groups(train, test, orig):
    """cats: string columns + numerics with < 100 distinct values in train + test; nums: the other numerics;
    numeric_like: every numeric column."""
    feature_cols = [c for c in train.columns if c not in [ID, TARGET] and c in test.columns and c in orig.columns]
    cats, nums, numeric_like = [], [], []
    both = pd.concat([train[feature_cols], test[feature_cols]], axis=0, ignore_index=True)
    for c in feature_cols:
        if train[c].dtype == object or test[c].dtype == object or orig[c].dtype == object:
            cats.append(c)
        else:
            numeric_like.append(c)
            if both[c].nunique(dropna=False) < 100:
                cats.append(c)
            else:
                nums.append(c)
    return cats, nums, numeric_like


def add_f1_features(train, test, orig, cats, nums, numeric_like, y_orig):
    """Adds the features to train, test and orig in place; returns train, test and the categorical / continuous columns."""
    for df in (train, test, orig):
        for c in cats:
            df[c] = safe_str(df[c])
        for c in numeric_like:  # low-cardinality numerics are back to float32 here (strings again at training time)
            df[c] = safe_num(df[c])

    cat_added = []
    for df in (train, test, orig):
        lap = safe_num(df['LapNumber'])
        tyre = safe_num(df['TyreLife'])
        lt = safe_num(df['LapTime (s)'])
        ld = safe_num(df['LapTime_Delta'])
        deg = safe_num(df['Cumulative_Degradation'])
        prog = safe_num(df['RaceProgress'])
        pos = safe_num(df['Position'])
        pit = safe_num(df['PitStop'])
        stint = safe_num(df['Stint'])
        df['lap_tyre_gap'] = (lap - tyre).astype('float32')
        df['degradation_per_tyre'] = (deg / (tyre + 1.0)).astype('float32')
        df['delta_per_laptime'] = (ld / (np.abs(lt) + 1.0)).astype('float32')
        df['position_progress'] = (pos / (prog + 0.02)).astype('float32')
        df['pit_progress'] = (pit + prog).astype('float32')
        df['stint_tyre_ratio'] = (stint / (tyre + 1.0)).astype('float32')

    high_card = [c for c in ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress'] if c in nums]
    all_ref = pd.concat([train, test, orig], axis=0, ignore_index=True)
    for col in high_card:
        base = safe_name(col)
        # low / moderate-cardinality keys only (no exact high-cardinality values as categories)
        for dec in [1, 2]:
            name = f'cat_{base}_r{dec}'
            for df in (train, test, orig):
                df[name] = safe_num(df[col]).round(dec).astype(str)
            cat_added.append(name)
        try:
            _, bins = pd.qcut(safe_num(all_ref[col]).rank(method='first'), q=32, labels=False, retbins=True, duplicates='drop')
        except Exception:
            bins = None
        qname = f'cat_{base}_q32'
        for df in (train, test, orig):
            if bins is None:
                df[qname] = '0'
            else:
                ranks = safe_num(df[col]).rank(method='first')
                df[qname] = pd.cut(ranks, bins=bins, labels=False, include_lowest=True).astype('Int16').astype(str)
        cat_added.append(qname)

    for a, b in [('Driver', 'Compound'), ('Driver', 'Race'), ('Race', 'Year'), ('Compound', 'TyreLife'), ('Stint', 'TyreLife'), ('Driver', 'PitStop')]:
        if a in train.columns and b in train.columns:
            name = f'combo_{safe_name(a)}__{safe_name(b)}'
            for df in (train, test, orig):
                df[name] = safe_str(df[a]) + '|' + safe_str(df[b])
            cat_added.append(name)

    prior = float(np.mean(y_orig))
    prior_cols = list(dict.fromkeys(cats + cat_added + high_card))
    for col in prior_cols:
        stats = pd.DataFrame({'key': orig[col].values, 'target': y_orig}).groupby('key', dropna=False)['target'].agg(['mean', 'count'])
        m = f'orig_mean_{safe_name(col)}'
        n = f'orig_count_{safe_name(col)}'
        for df in (train, test, orig):
            df[m] = df[col].map(stats['mean']).fillna(prior).astype('float32')
            df[n] = np.log1p(df[col].map(stats['count']).fillna(0).astype('float32'))

    cat_cols = list(dict.fromkeys(cats + cat_added))
    cont_cols = [c for c in train.columns if c not in [TARGET] + cat_cols and c in test.columns]
    return train, test, cat_cols, cont_cols


def safe_set(obj, **kwargs):
    for k, v in kwargs.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


def make_common_configs(cat_cols, cont_cols):
    data_config = DataConfig(target=[TARGET], categorical_cols=cat_cols, continuous_cols=cont_cols)
    trainer_config = TrainerConfig()
    safe_set(trainer_config, batch_size=BATCH_SIZE)
    safe_set(trainer_config, max_epochs=MAX_EPOCHS)
    safe_set(trainer_config, checkpoints='valid_loss')
    safe_set(trainer_config, load_best=True)
    safe_set(trainer_config, early_stopping='valid_loss')
    safe_set(trainer_config, early_stopping_patience=PATIENCE)
    safe_set(trainer_config, early_stopping_min_delta=1e-4)
    safe_set(trainer_config, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    safe_set(trainer_config, checkpoints_path=str(CKPT_DIR))
    safe_set(trainer_config, trainer_kwargs={'default_root_dir': str(CKPT_DIR)})
    optimizer_config = OptimizerConfig()
    safe_set(optimizer_config, optimizer='AdamW')
    if hasattr(optimizer_config, 'optimizer_params') and isinstance(optimizer_config.optimizer_params, dict):
        optimizer_config.optimizer_params = {k: v for k, v in optimizer_config.optimizer_params.items() if k != 'lr'}
    return data_config, trainer_config, optimizer_config


def extract_positive_prob(df_out: pd.DataFrame) -> np.ndarray:
    cols = list(df_out.columns)
    prob_cols = [c for c in cols if 'prob' in c.lower()]
    if not prob_cols:
        pred_cols = [c for c in cols if 'prediction' in c.lower()]
        prob_cols = pred_cols
    if len(prob_cols) == 1:
        return df_out[prob_cols[0]].astype('float32').to_numpy()
    if len(prob_cols) >= 2:
        def key(c):
            low = c.lower()
            if '1' in low or 'true' in low or 'pitnextlap' in low:
                return 1
            if '0' in low or 'false' in low:
                return 0
            return 0
        prob_cols = sorted(prob_cols, key=key)
        return df_out[prob_cols[-1]].astype('float32').to_numpy()
    raise ValueError(f'Could not find probability columns in output columns: {cols}')


def train_gandalf_cv(train: pd.DataFrame, test: pd.DataFrame, cat_cols, cont_cols, run_folds):
    data_config, trainer_config, optimizer_config = make_common_configs(cat_cols, cont_cols)
    y = train[TARGET].astype('int64').to_numpy()
    fold_ids = folds(y)
    for k in range(5):
        if k not in run_folds:
            continue
        print(f'\n---- Fold {k} ----', flush=True)
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        tr_df = train.iloc[tr_idx].copy()
        va_df = train.iloc[va_idx].copy()
        for c in cat_cols:
            tr_df[c] = safe_str(tr_df[c])
            va_df[c] = safe_str(va_df[c])
            test[c] = safe_str(test[c])
        for c in cont_cols:
            med = safe_num(tr_df[c]).median()
            tr_df[c] = safe_num(tr_df[c]).fillna(med).astype('float32')
            va_df[c] = safe_num(va_df[c]).fillna(med).astype('float32')
            test[c] = safe_num(test[c]).fillna(med).astype('float32')

        model_config = GANDALFConfig(task='classification', learning_rate=LEARNING_RATE)
        tabular_model = TabularModel(data_config=data_config, model_config=model_config, trainer_config=trainer_config, optimizer_config=optimizer_config)
        tabular_model.fit(train=tr_df, validation=va_df)
        va_prob = extract_positive_prob(tabular_model.predict(va_df.drop(columns=[TARGET])))
        te_prob = extract_positive_prob(tabular_model.predict(test))
        print(f'Fold {k} AUC = {roc_auc_score(y[va_idx], va_prob):.8f}', flush=True)
        save_part(MODEL_DIR, PAIR, k, va_prob, te_prob)
        del tabular_model, tr_df, va_df, va_prob, te_prob
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    run_folds = fold_list(ap.parse_args().folds)

    seed_everything(SEED)
    train, test, orig = read_train(), read_test(), read_original()
    y_orig = orig[TARGET].astype('int8').to_numpy()
    cats, nums, numeric_like = infer_groups(train, test, orig)
    train_x = train.drop(columns=[ID]).copy()
    test_x = test.drop(columns=[ID]).copy()
    orig_x = orig.copy()
    train_x, test_x, cat_cols, cont_cols = add_f1_features(train_x, test_x, orig_x, cats, nums, numeric_like, y_orig)
    train_x[TARGET] = train[TARGET].astype('int64').values  # target only in the train frame
    cont_cols = [c for c in cont_cols if c != TARGET and c in train_x.columns and c in test_x.columns]
    cat_cols = [c for c in cat_cols if c in train_x.columns and c in test_x.columns]
    print(f'{len(cat_cols)} categorical columns: {cat_cols[:15]}{" ..." if len(cat_cols) > 15 else ""}', flush=True)
    print(f'{len(cont_cols)} continuous columns', flush=True)

    train_gandalf_cv(train_x, test_x[cat_cols + cont_cols].copy(), cat_cols, cont_cols, run_folds)


if __name__ == '__main__':
    main()
