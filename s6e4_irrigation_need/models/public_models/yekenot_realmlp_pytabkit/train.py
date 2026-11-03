#!/usr/bin/env python
"""Public notebook "PS|S6|E4: RealMLP · PyTabKit" run on our 5 folds (the split the notebook uses).
https://www.kaggle.com/code/yekenot/ps-s6-e4-realmlp-pytabkit

Features (label-free part fitted once on all competition training rows, as in the notebook): every numeric column
floored and factorized as a category; the decimal part (rounded to 0.01) of Temperature_C, Organic_Carbon,
Soil_Moisture, Soil_pH and Sunlight_Hours; for Organic_Carbon, Soil_Moisture, Rainfall_mm and Previous_Irrigation_mm the
mean target of the original rows with the same value (the overall original mean where there is none) and 300 quantile
bins of it as a category; 6 combo categories (Soil_Type x Soil_pH, Crop_Type x Electrical_Conductivity,
Crop_Type x Humidity, Crop_Growth_Stage x Rainfall_mm, Season x Organic_Carbon, Water_Source x Organic_Carbon).
Per fold: scikit-learn TargetEncoder (cv=5, smooth 'auto', shuffle, seed 42) of the 6 combos, fitted on the fold's
training rows (3 class columns per combo). No original rows are appended.
Model: pytabkit RealMLP_TD_Classifier with the notebook's parameters: 512-256-128 SiLU layers, PLR embeddings (16 / 8,
sigma 2.33), one-hot up to 18 levels, 8 ensemble members, 4 epochs, batch 256, lr 0.05, cos_anneal schedule, label
smoothing 0.01, early stopping on the validation fold's balanced accuracy, seed 42. The global random generators are
seeded with 42 once and the 5 folds run in order in one process. Test = mean of the 5 fold models.
Usage: python train.py"""
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, TARGET_MAP, read_original, read_test, read_train  # noqa: E402
from lib.runner import run_all_folds, work_dir, working_directory  # noqa: E402

warnings.filterwarnings('ignore')

FOLDER = 'yekenot_realmlp_pytabkit'
FOLDS, SEED = 5, 42

IMPORTANT_COMBOS = [
    ('Soil_Type', 'Soil_pH'),
    ('Crop_Type', 'Electrical_Conductivity'),
    ('Crop_Type', 'Humidity'),
    ('Crop_Growth_Stage', 'Rainfall_mm'),
    ('Season', 'Organic_Carbon'),
    ('Water_Source', 'Organic_Carbon'),
]

PARAMS = {
    'random_state': 42,
    'verbosity': 2,
    'val_metric_name': '1-balanced_accuracy',

    'n_ens': 8,
    'n_epochs': 4,
    'batch_size': 256,
    'use_early_stopping': True,
    'early_stopping_additive_patience': 10,
    'early_stopping_multiplicative_patience': 1,

    'lr': 0.05,
    'wd': 0.0236,
    'sq_mom': 0.988,
    'lr_sched': 'cos_anneal',
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


def feature_engineering(df, num_cols, orig, category_map, fit=False):
    """fit=True learns the category maps / bins from df (the competition training rows) into category_map."""
    # numerics floored and factorized as categories
    for col in num_cols:
        cat_name = f'{col}_cat_'
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            code_map = {cat: i for i, cat in enumerate(category_map[col])}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype('category')

    # decimal parts
    for col in ['Temperature_C', 'Organic_Carbon', 'Soil_Moisture', 'Soil_pH', 'Sunlight_Hours']:
        df[f'_{col}_decimal'] = (df[col] % 1).round(2).astype('float32')

    # target means of the original data per value, and 300 quantile bins of them
    for col in ['Organic_Carbon', 'Soil_Moisture', 'Rainfall_mm', 'Previous_Irrigation_mm']:
        orig_enc_name = f'_{col}_mean_target_orig'
        df[orig_enc_name] = (
            df[col]
            .map(orig.groupby(col)[TARGET].mean())
            .fillna(orig[TARGET].mean())
            .astype('float32')
        )
        te_bin_name = f'{orig_enc_name[1:]}_bin_'
        if fit:
            kb = KBinsDiscretizer(n_bins=300, encode='ordinal', strategy='quantile', subsample=None)
            binned = kb.fit_transform(df[[orig_enc_name]]).ravel().astype('int32')
            category_map[te_bin_name] = kb
        else:
            binned = category_map[te_bin_name].transform(df[[orig_enc_name]]).ravel().astype('int32')
        df[te_bin_name] = binned
        df[te_bin_name] = df[te_bin_name].astype('category')

    # combo categories
    combo_names = []
    for col1, col2 in IMPORTANT_COMBOS:
        combo_name = f'{col1}_{col2}_'
        combo_names.append(combo_name)
        combo_series = df[col1].astype(str) + '_' + df[col2].astype(str)
        if fit:
            codes, uniques = combo_series.factorize()
            category_map[combo_name] = uniques
        else:
            code_map = {cat: i for i, cat in enumerate(category_map[combo_name])}
            codes = combo_series.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype('category')

    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]
    return df, new_cat_cols, new_num_cols, combo_names


def main():
    seed_everything(SEED)
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = train[TARGET].map(TARGET_MAP)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP)
    x = train.drop([ID, TARGET], axis=1)
    y = train[TARGET]
    x_test = test.drop([ID], axis=1)
    del train, test

    cat_cols = x.select_dtypes(include=['object']).columns.tolist()
    num_cols = x.select_dtypes(exclude=['object']).columns.tolist()
    category_map = {}
    x, new_cat_cols, new_num_cols, combo_names = feature_engineering(x, num_cols, orig, category_map, fit=True)
    x_test, _, _, _ = feature_engineering(x_test, num_cols, orig, category_map, fit=False)
    print(f'After FE: X={x.shape}, X_test={x_test.shape}; cat_cols: {len(cat_cols + new_cat_cols)}, '
          f'num_cols: {len(num_cols + new_num_cols)}', flush=True)
    n_classes = y.nunique()

    def fit_fold(fold, tr_idx, val_idx):
        x_tr, x_val, x_tst = x.iloc[tr_idx].copy(), x.iloc[val_idx].copy(), x_test.copy()

        # target encoding of the combo categories, fitted on the fold's training rows
        te = TargetEncoder(cv=FOLDS, smooth='auto', shuffle=True, random_state=SEED)
        tr_enc = te.fit_transform(x_tr[combo_names], y.iloc[tr_idx])
        val_enc = te.transform(x_val[combo_names])
        tst_enc = te.transform(x_tst[combo_names])
        te_names = [f'_{col}TE_class{cls}' for col in combo_names for cls in range(n_classes)]
        x_tr[te_names] = tr_enc
        x_val[te_names] = val_enc
        x_tst[te_names] = tst_enc

        with working_directory(work_dir(FOLDER)):  # pytabkit / lightning may write logs to the working directory
            model = RealMLP_TD_Classifier(**PARAMS)
            model.fit(x_tr, y.iloc[tr_idx], x_val, y.iloc[val_idx])
            val_preds = model.predict_proba(x_val)
            fold_test_preds = model.predict_proba(x_tst)
        del model
        torch.cuda.empty_cache()
        return val_preds, fold_test_preds

    run_all_folds(FOLDER, fit_fold, y=y.to_numpy())


if __name__ == '__main__':
    main()
