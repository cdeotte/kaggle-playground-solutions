#!/usr/bin/env python
"""Public notebook "3-Model Ensemble: RealMLP + CatBoost + XGBoost" (version 1), its RealMLP (Pipeline A), run on our 5
folds. https://www.kaggle.com/code/amanatar/3-model-ensemble-realmlp-catboost-xgboost

Features (the notebook's Pipeline A, fitted once on all competition training rows and applied to test and original
rows): LapNumber / RaceProgress and TyreLife / LapNumber; every numeric column and the two ratios floored and factorized
as string categories; counts of Driver / Compound / Race / Year / PitStop; 200 quantile bins of RaceProgress; Race x
Compound and Race x Year combo categories.
Original rows: the fold-matched 4/5 of the original dataset (same StratifiedKFold on its own target) is appended to each
fold's training rows.
Per fold: scikit-learn TargetEncoder (cv=5, smooth 'auto', shuffle, seed 42) of the two combo columns, fitted on the
training rows (competition + original).
Model (the notebook's): pytabkit RealMLP_TD_Classifier, 8 members, 512-256-128 SiLU, 4 epochs, batch 256, lr 0.03,
seed 42. The notebook seeds once (42) and trains its folds in one loop, so the 5 folds run in order in one process.
Test = mean of the 5 fold models; OOF and test are clipped to [1e-7, 1 - 1e-7].
Usage: python train.py"""
import gc
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, read_original, read_test, read_train  # noqa: E402
from lib.runner import orig_train_index, run_all_folds, work_dir, working_directory  # noqa: E402

FOLDER = 'amanatar_realmlp'
SEED, FOLDS = 42, 5
MLP_PARAMS = {
    'random_state': SEED,
    'verbosity': 2,
    'val_metric_name': '1-auc_ovr',
    'n_ens': 8,
    'n_epochs': 4,
    'batch_size': 256,
    'use_early_stopping': False,
    'lr': 0.03,
    'wd': 0.018,
    'sq_mom': 0.98,
    'lr_sched': 'lin_cos_log_15',
    'first_layer_lr_factor': 0.25,
    'embedding_size': 6,
    'max_one_hot_cat_size': 18,
    'hidden_sizes': [512, 256, 128],
    'act': 'silu',
    'p_drop': 0.05,
    'p_drop_sched': 'expm4t',
    'plr_hidden_1': 16,
    'plr_hidden_2': 8,
    'plr_act_name': 'gelu',
    'plr_lr_factor': 0.1151,
    'plr_sigma': 2.33,
    'ls_eps': 0.01,
    'ls_eps_sched': 'sqrt_cos',
    'add_front_scale': False,
    'bias_init_mode': 'neg-uniform-dynamic-2',
    'tfms': ['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding', 'l2_normalize'],
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pipeline_a_fe(df, fit=False, state=None):
    """Pipeline A feature engineering: categorize numerics, count-encode, quantile bins and combos."""
    df = df.copy()
    if state is None:
        state = {}

    cat_cols_base = df.select_dtypes(include=['object']).columns.tolist()
    num_cols_base = df.select_dtypes(exclude=['object']).columns.tolist()
    num_cols_base = [c for c in num_cols_base if c not in [ID, TARGET]]

    df['_LP_RP'] = (df['LapNumber'] / (df['RaceProgress'] + 1e-6)).astype('float32')
    df['_TL_LP'] = (df['TyreLife'] / df['LapNumber'].clip(lower=1)).astype('float32')

    for col in num_cols_base + ['_LP_RP', '_TL_LP']:
        cat_name = f'{col}_cat_' if col in num_cols_base else f'{col[1:]}_cat_'
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            state[col] = uniques
        else:
            code_map = {cat: i for i, cat in enumerate(state[col])}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes.astype(str)

    for col in cat_cols_base + ['Year_cat_', 'PitStop_cat_']:
        cnt_name = f'_{col}_count' if col in cat_cols_base else f'_{col[:-1]}_count'
        if fit:
            cm = df[col].value_counts()
            state[cnt_name] = cm
        else:
            cm = state[cnt_name]
        df[cnt_name] = df[col].map(cm).fillna(0).astype('int32')

    for col, nb in [('RaceProgress', 200)]:
        bn = f'{col}_{nb}_qbin_'
        if fit:
            kb = KBinsDiscretizer(n_bins=nb, encode='ordinal', strategy='quantile', subsample=None)
            df[bn] = kb.fit_transform(df[[col]]).ravel().astype('int32').astype(str)
            state[bn] = kb
        else:
            df[bn] = state[bn].transform(df[[col]]).ravel().astype('int32').astype(str)

    combos = [('Race', 'Compound'), ('Race', 'Year')]
    combo_names = []
    for cols in combos:
        cn = '_'.join(cols) + '_'
        combo_names.append(cn)
        s = df[cols[0]].astype(str)
        for c in cols[1:]:
            s = s + '_' + df[c].astype(str)
        if fit:
            codes, uniques = pd.factorize(s, sort=False)
            state[cn] = uniques
        else:
            code_map = {cat: i for i, cat in enumerate(state[cn])}
            codes = s.map(code_map).fillna(-1).astype('int32')
        df[cn] = codes.astype(str)

    return df, state, combo_names


def main():
    seed_everything(SEED)
    train_raw, test_raw, orig_raw = read_train(), read_test(), read_original()
    train_raw[TARGET] = pd.to_numeric(train_raw[TARGET]).astype('int8')
    orig_raw[TARGET] = pd.to_numeric(orig_raw[TARGET]).astype('int8')

    x_a = train_raw.drop([ID, TARGET], axis=1).copy()
    y_a = train_raw[TARGET].copy()
    xt_a = test_raw.drop([ID], axis=1).copy()
    orig_a = orig_raw.drop([TARGET], axis=1).copy()
    y_orig_a = orig_raw[TARGET].copy()

    x_a, st_a, combo_names = pipeline_a_fe(x_a, fit=True)
    xt_a, _, _ = pipeline_a_fe(xt_a, fit=False, state=st_a)
    orig_a, _, _ = pipeline_a_fe(orig_a, fit=False, state=st_a)
    print(f'Pipeline A features: {x_a.shape[1]}', flush=True)

    def fit_fold(fold, tri, vi):
        oti = orig_train_index(fold, y_orig_a.to_numpy())
        xtr = pd.concat([x_a.iloc[tri], orig_a.iloc[oti]], axis=0).reset_index(drop=True)
        ytr = pd.concat([y_a.iloc[tri], y_orig_a.iloc[oti]], axis=0).reset_index(drop=True)
        xvl = x_a.iloc[vi].copy()
        yvl = y_a.iloc[vi]
        xts = xt_a.copy()

        te = TargetEncoder(cv=FOLDS, smooth='auto', shuffle=True, random_state=SEED)
        te_names = [f'_{c}TE' for c in combo_names]
        xtr[te_names] = te.fit_transform(xtr[combo_names], ytr)
        xvl[te_names] = te.transform(xvl[combo_names])
        xts[te_names] = te.transform(xts[combo_names])

        with working_directory(work_dir(FOLDER)):  # pytabkit / lightning may write logs to the working directory
            model = RealMLP_TD_Classifier(**MLP_PARAMS)
            model.fit(xtr, ytr, xvl, yvl)
            valid_pred = model.predict_proba(xvl)[:, 1].astype('float32')
            test_pred = model.predict_proba(xts)[:, 1].astype('float32')
        del xtr, xvl, xts, ytr, yvl, model
        torch.cuda.empty_cache()
        gc.collect()
        return valid_pred, test_pred

    run_all_folds(FOLDER, FOLDER, fit_fold, y=y_a.to_numpy(), clip=(1e-7, 1 - 1e-7))


if __name__ == '__main__':
    main()
