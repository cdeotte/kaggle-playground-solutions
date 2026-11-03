#!/usr/bin/env python
"""Public notebook "PS|S6|E3: RealMLP · PyTabKit" (an earlier and a later version) run on our 5 folds.
https://www.kaggle.com/code/yekenot/ps-s6-e3-realmlp-pytabkit

Features (the notebook's, label-free, fitted once on all train rows and applied to test): MonthlyCharges / TotalCharges,
TotalCharges / tenure, MonthlyCharges / (TotalCharges / tenure), TotalCharges / MonthlyCharges, tenure^2; the
thousands digit of TotalCharges; quantile bins of TotalCharges and MonthlyCharges; every numeric column factorized as a
category (TotalCharges rounded to an integer first); the Contract x InternetService x PaymentMethod combo.
  mish (earlier version): bins TotalCharges 4000 / 500, MonthlyCharges 200 / 100.
  silu (later version): + tenure >= 24 flag, MonthlyCharges cents, TotalCharges mod 100 / mod 1000 and is-multiple-of-10
  flag; quantile and k-means bins of TotalCharges 5000 / 400 and MonthlyCharges 300.
Per fold: a scikit-learn TargetEncoder (cv = the number of folds, smooth 'auto', shuffle, seed 42) of the combo.
Model: pytabkit RealMLP_TD_Classifier, 8 members, 3 epochs, batch 256, lr 0.075, wd 0.0236, PLR embeddings (16 / 8,
sigma 2.33), one-hot up to 18 levels, label smoothing 0.01, early stopping on the validation fold's AUC, seed 42.
  mish: 4 x 352 mish layers, flat_cos dropout schedule, coslog4 label smoothing schedule, patience 20.
  silu: 512-256-128 SiLU layers, flat_anneal lr schedule, expm4t dropout schedule, sqrt_cos label smoothing schedule,
  patience 10.
The notebook seeds once (42) and trains its folds in one loop, so the folds run in order in one process.
Test = mean of the fold models.
Usage: python train.py --preset mish|silu"""
import argparse
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, read_test, read_train  # noqa: E402
from lib.runner import kfold, run_all_folds, split, work_dir, working_directory  # noqa: E402

warnings.filterwarnings('ignore')

FOLDER = 'yekenot_realmlp_pytabkit'
SEED = 42
IMPORTANT_COMBOS = [('Contract', 'InternetService', 'PaymentMethod')]

PARAMS_MISH = {
    'device': 'cuda',
    'random_state': 42,
    'verbosity': 2,
    'n_epochs': 3,
    'batch_size': 256,
    'n_ens': 8,
    'val_metric_name': '1-auc_ovr',
    'use_early_stopping': True,
    'early_stopping_additive_patience': 20,
    'early_stopping_multiplicative_patience': 1,
    'act': 'mish',
    'embedding_size': 6,
    'first_layer_lr_factor': 0.25,
    'hidden_sizes': 'rectangular',
    'hidden_width': 352,
    'lr': 0.075,
    'ls_eps': 0.01,
    'ls_eps_sched': 'coslog4',
    'max_one_hot_cat_size': 18,
    'n_hidden_layers': 4,
    'p_drop': 0.05,
    'p_drop_sched': 'flat_cos',
    'plr_hidden_1': 16,
    'plr_hidden_2': 8,
    'plr_lr_factor': 0.1151,
    'plr_sigma': 2.33,
    'scale_lr_factor': 2.24,
    'sq_mom': 0.988,
    'wd': 0.0236,
}

PARAMS_SILU = {
    'random_state': 42,
    'verbosity': 2,
    'val_metric_name': '1-auc_ovr',
    'n_ens': 8,
    'n_epochs': 3,
    'batch_size': 256,
    'use_early_stopping': True,
    'early_stopping_additive_patience': 10,
    'early_stopping_multiplicative_patience': 1,
    'lr': 0.075,
    'wd': 0.0236,
    'sq_mom': 0.988,
    'lr_sched': 'flat_anneal',
    'first_layer_lr_factor': 0.25,
    'add_front_scale': False,
    'bias_init_mode': 'neg-uniform-dynamic-2',
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
    'tfms': ['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding', 'l2_normalize'],
}


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _add_arithmetic(df):
    df['_MonthlyCharges_/_TotalCharges'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1e-6)).astype('float32')
    df['_TotalCharges_/_tenure'] = (df['TotalCharges'] / (df['tenure'] + 1e-6)).astype('float32')
    df['_Monthly_to_avg_ratio'] = (df['MonthlyCharges'] / (df['_TotalCharges_/_tenure'] + 1e-6)).astype('float32')
    df['_TotalCharges_/_MonthlyCharges'] = (df['TotalCharges'] / (df['MonthlyCharges'] + 1e-6)).astype('float32')
    df['_tenure_sq'] = (df['tenure'] ** 2).astype('float32')


def _combo_series(df, cols):
    col1, col2, col3 = cols
    return df[col1].astype(str) + '_' + df[col2].astype(str) + '_' + df[col3].astype(str)


def features_mish(df, num_cols, category_map, fit=False):
    """Feature engineering of the earlier version (fit=True on train fills category_map, fit=False applies it)."""
    df = df.copy()
    _add_arithmetic(df)

    for col in ['TotalCharges']:
        k = -3
        digit_name = f'{col}_d{k}_'
        df[digit_name] = ((df[col].astype('float32') * (10 ** k)) % 10).astype('int8')
        df[digit_name] = df[digit_name].astype('category')

    bin_config = {'TotalCharges': [4000, 500], 'MonthlyCharges': [200, 100]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            bin_name = f'{col}_{n_bins}_bin_'
            if fit:
                kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
                binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                category_map[bin_name] = kb
            else:
                binned = category_map[bin_name].transform(df[[col]]).ravel().astype('int32')
            df[bin_name] = binned.astype('int32')
            df[bin_name] = df[bin_name].astype('category')

    for col in num_cols:
        cat_name = f'{col}_cat_'
        if fit:
            round_flag = col == 'TotalCharges'
            series = df[col].round(0) if round_flag else df[col]
            codes, uniques = series.factorize()
            category_map[col] = {'uniques': uniques, 'round_flag': round_flag}
        else:
            round_flag = category_map[col]['round_flag']
            series = df[col].round(0) if round_flag else df[col]
            code_map = {cat: i for i, cat in enumerate(category_map[col]['uniques'])}
            codes = pd.to_numeric(series.map(code_map), errors='coerce').fillna(-1).astype('int32')
        df[cat_name] = pd.Series(codes, index=df.index).astype('int32')
        df[cat_name] = df[cat_name].astype('category')

    combo_names = []
    for cols in IMPORTANT_COMBOS:
        combo_name = '_'.join(cols) + '_'
        combo_names.append(combo_name)
        combo_series = _combo_series(df, cols)
        if fit:
            codes, uniques = combo_series.factorize()
            category_map[combo_name] = uniques
        else:
            code_map = {cat: i for i, cat in enumerate(category_map[combo_name])}
            codes = pd.to_numeric(combo_series.map(code_map), errors='coerce').fillna(-1).astype('int32')
        df[combo_name] = pd.Series(codes, index=df.index).astype('int32')
        df[combo_name] = df[combo_name].astype('category')
    return df, combo_names


def features_silu(df, num_cols, category_map, fit=False):
    """Feature engineering of the later version (fit=True on train fills category_map, fit=False applies it)."""
    _add_arithmetic(df)
    df['is_loyal_customer_'] = (df['tenure'] >= 24).astype('category')

    for col in ['TotalCharges']:
        k = -3
        digit_name = f'{col}_d{k}_'
        df[digit_name] = ((df[col] * 10 ** k) % 10).astype('int8')
        df[digit_name] = df[digit_name].astype('category')
    for col in ['MonthlyCharges']:
        df[f'_{col}_decimal'] = (df[col] % 1).round(2).astype('float32')
    df['_TotalCharges_mod100'] = (np.floor(df['TotalCharges']) % 100).astype('float32')
    df['_TotalCharges_mod1000'] = (np.floor(df['TotalCharges']) % 1000).astype('float32')
    df['TotalCharges_is_multiple_10_'] = (np.floor(df['TotalCharges']) % 10 == 0).astype('category')

    bin_config = {'TotalCharges': [5000, 400], 'MonthlyCharges': [300]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            for strategy in ['quantile', 'kmeans']:
                bin_name = f'{col}_{n_bins}_{strategy}_bin_'
                if fit:
                    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy, subsample=None)
                    binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                    category_map[bin_name] = kb
                else:
                    binned = category_map[bin_name].transform(df[[col]]).ravel().astype('int32')
                df[bin_name] = binned
                df[bin_name] = df[bin_name].astype('category')

    for col in num_cols:
        cat_name = f'{col}_cat_'
        if fit:
            round_flag = col == 'TotalCharges'
            series = df[col].round(0) if round_flag else df[col]
            codes, uniques = series.factorize()
            category_map[col] = {'uniques': uniques, 'round_flag': round_flag}
        else:
            round_flag = category_map[col]['round_flag']
            series = df[col].round(0) if round_flag else df[col]
            code_map = {cat: i for i, cat in enumerate(category_map[col]['uniques'])}
            codes = series.map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype('category')

    combo_names = []
    for cols in IMPORTANT_COMBOS:
        combo_name = '_'.join(cols) + '_'
        combo_names.append(combo_name)
        combo_series = _combo_series(df, cols)
        if fit:
            codes, uniques = combo_series.factorize()
            category_map[combo_name] = uniques
        else:
            code_map = {cat: i for i, cat in enumerate(category_map[combo_name])}
            codes = combo_series.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype('category')
    return df, combo_names


PRESETS = {
    'mish': {'features': features_mish, 'params': PARAMS_MISH, 'test_folds': 5},
    'silu': {'features': features_silu, 'params': PARAMS_SILU, 'test_folds': 51},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=sorted(PRESETS), required=True,
                    help='mish: the earlier notebook version, silu: the later one')
    name = ap.parse_args().preset
    preset, pair = PRESETS[name], f'{FOLDER}_{name}'
    print('PyTorch', torch.__version__, flush=True)

    train, test = read_train(), read_test()
    y = (train[TARGET].astype(str).str.strip().str.lower() == 'yes').astype(np.int32)
    X = train.drop(columns=[ID, TARGET])
    X_test = test.drop(columns=[ID])
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    category_map = {}
    X, combo_names = preset['features'](X, num_cols, category_map, fit=True)
    X_test, _ = preset['features'](X_test, num_cols, category_map, fit=False)
    print('X shape:', X.shape, '| X_test shape:', X_test.shape, flush=True)

    def cross_validate(n_folds):
        seed_everything(SEED)
        fold_ids = kfold(y.to_numpy(), n_folds)
        oof = np.zeros(len(X))
        test_pred = np.zeros(len(X_test))
        for fold in range(n_folds):
            tr_idx, val_idx = split(fold_ids, fold)
            X_tr = X.iloc[tr_idx].copy()
            X_val = X.iloc[val_idx].copy()
            X_tst = X_test.copy()

            te = TargetEncoder(cv=n_folds, smooth='auto', shuffle=True, random_state=SEED)
            te_names = [f'_{col}TE' for col in combo_names]
            X_tr[te_names] = te.fit_transform(X_tr[combo_names], y.iloc[tr_idx])
            X_val[te_names] = te.transform(X_val[combo_names])
            X_tst[te_names] = te.transform(X_tst[combo_names])

            with working_directory(work_dir(FOLDER)):
                model = RealMLP_TD_Classifier(**preset['params'])
                model.fit(X_tr, y.iloc[tr_idx], X_val, y.iloc[val_idx])
                val_preds = model.predict_proba(X_val)[:, 1]
                fold_test_preds = model.predict_proba(X_tst)[:, 1]
            oof[val_idx] = val_preds
            test_pred += fold_test_preds / n_folds
            print(f'fold {fold}: AUC {roc_auc_score(y.iloc[val_idx], val_preds):.6f}', flush=True)
            del model
            torch.cuda.empty_cache()
        return oof, test_pred

    run_all_folds(FOLDER, pair, cross_validate, test_folds=preset['test_folds'], y=y.to_numpy())


if __name__ == '__main__':
    main()
