#!/usr/bin/env python
"""Public notebook "PS|S6|E5: RealMLP · PyTabKit" (versions 11 and 16) run on our 5 folds.
https://www.kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytabkit

Features (the notebook's; our run fits them on each fold's competition training rows and applies them to the fold's
validation, test and original rows): every numeric column floored and factorized as a string category, LapNumber /
RaceProgress and TyreLife / LapNumber, 200 quantile bins of RaceProgress, Race x Compound and Race x Year combo
categories. Version 16 adds LapTime x Cumulative_Degradation (signed and absolute) and
LapTime / |Cumulative_Degradation|, the two ratios floored as categories, counts of Driver / Compound / Race / Year /
PitStop and 7 quantile bins of LapTime.
Original rows: the fold-matched 4/5 of the original dataset (same StratifiedKFold on its own target) is appended to each
fold's training rows.
Per fold: scikit-learn TargetEncoder (cv=5, smooth 'auto', shuffle, seed 42) of the two combo columns, fitted on the
training rows (competition + original).
Model: pytabkit RealMLP_TD_Classifier with the notebook's parameters (v11: 8 members, 4 epochs, lr 0.03; v16: 24
members, 6 epochs, lr 0.01), seed 42; each fold re-seeds the global random generators with 43 + fold.
Test = mean of the 5 fold models.
Usage: python train.py --version 11|16 [--folds 0,1,2,3,4]"""
import argparse
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
from lib.runner import fold_args, orig_train_index, run_folds, work_dir, working_directory  # noqa: E402

FOLDER = 'yekenot_realmlp_pytabkit'
FOLDS, SEED = 5, 42

COMMON_PARAMS = {
    'random_state': 42,
    'verbosity': 2,
    'val_metric_name': '1-auc_ovr',
    'batch_size': 256,
    'use_early_stopping': False,
    'early_stopping_additive_patience': 10,
    'early_stopping_multiplicative_patience': 1,
    'lr_sched': 'lin_cos_log_15',
    'first_layer_lr_factor': 0.25,
    'embedding_size': 6,
    'max_one_hot_cat_size': 18,
    'hidden_sizes': [512, 256, 128],
    'act': 'silu',
    'p_drop': 0.05,
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
PARAMS = {
    11: {**COMMON_PARAMS, 'n_ens': 8, 'n_epochs': 4, 'lr': 0.03, 'wd': 0.018, 'sq_mom': 0.98, 'p_drop_sched': 'expm4t'},
    16: {**COMMON_PARAMS, 'n_ens': 24, 'n_epochs': 6, 'lr': 0.01, 'wd': 0.016, 'sq_mom': 0.99,
         'p_drop_sched': 'invsqrtp1e-3'},
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeaturesV11:
    """Version 11 feature engineering: fit() on the fold's competition training rows, transform() any frame."""

    def __init__(self, cat_cols, num_cols):
        self.num_cols = list(num_cols)
        self.category_map = {}
        self.important_combos = [('Race', 'Compound'), ('Race', 'Year')]
        self.bin_config = {'RaceProgress': [200]}
        self.combo_names = [f'{c1}_{c2}_' for c1, c2 in self.important_combos]

    def fit(self, df):
        df = df.copy()
        for col in self.num_cols:
            _, uniques = np.floor(df[col]).factorize()
            self.category_map[col] = uniques
        for col, bins_list in self.bin_config.items():
            for n_bins in bins_list:
                bin_name = f'{col}_{n_bins}_quantile_bin_'
                kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
                kb.fit(df[[col]])
                self.category_map[bin_name] = kb
        for col1, col2 in self.important_combos:
            combo_series = df[col1].astype(str) + '_' + df[col2].astype(str)
            _, uniques = combo_series.factorize()
            self.category_map[f'{col1}_{col2}_'] = uniques
        return self

    def transform(self, df):
        df = df.copy()
        for col in self.num_cols:
            code_map = {cat: i for i, cat in enumerate(self.category_map[col])}
            df[f'{col}_cat_'] = np.floor(df[col]).map(code_map).fillna(-1).astype('int32').astype(str)
        df['_LapNumber_/_RaceProgress'] = (df['LapNumber'] / (df['RaceProgress'] + 1e-6)).astype('float32')
        df['_TyreLife_/_LapNumber'] = (df['TyreLife'] / df['LapNumber'].clip(lower=1)).astype('float32')
        for col, bins_list in self.bin_config.items():
            for n_bins in bins_list:
                bin_name = f'{col}_{n_bins}_quantile_bin_'
                binned = self.category_map[bin_name].transform(df[[col]]).ravel().astype('int32')
                df[bin_name] = binned.astype(str)
        for col1, col2 in self.important_combos:
            combo_name = f'{col1}_{col2}_'
            combo_series = df[col1].astype(str) + '_' + df[col2].astype(str)
            code_map = {cat: i for i, cat in enumerate(self.category_map[combo_name])}
            df[combo_name] = combo_series.map(code_map).fillna(-1).astype('int32').astype(str)
        return df


class FeaturesV16:
    """Version 16 feature engineering: fit() on the fold's competition training rows, transform() any frame."""

    def __init__(self, cat_cols, num_cols):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.category_map = {}
        self.important_combos = [('Race', 'Compound'), ('Race', 'Year')]
        self.bin_config = {'RaceProgress': [200], 'LapTime (s)': [7]}
        self.categorized_arithmetic_cols = ['_LapNumber_/_RaceProgress', '_TyreLife_/_LapNumber']
        self.count_source_cols = self.cat_cols + ['Year_cat_', 'PitStop_cat_']
        self.combo_names = ['_'.join(cols) + '_' for cols in self.important_combos]

    @staticmethod
    def _add_arithmetic(df):
        df['_LapNumber_/_RaceProgress'] = (df['LapNumber'] / (df['RaceProgress'] + 1e-6)).astype('float32')
        df['_TyreLife_/_LapNumber'] = (df['TyreLife'] / df['LapNumber'].clip(lower=1)).astype('float32')
        df['_LapTime (s)_*_Cumulative_Degradation'] = (
            df['LapTime (s)'] * df['Cumulative_Degradation']).astype('float32')
        df['_LapTime (s)_*_Cumulative_Degradation_abs'] = (
            df['LapTime (s)'] * df['Cumulative_Degradation'].abs()).astype('float32')
        df['_LapTime (s)_/_Cumulative_Degradation_abs'] = (
            df['LapTime (s)'] / (df['Cumulative_Degradation'].abs() + 1e-6)).astype('float32')
        return df

    def _cat_name(self, col):
        return f'{col}_cat_' if col in self.num_cols else f'{col[1:]}_cat_'

    def _count_name(self, col):
        return f'_{col}_count' if col in self.cat_cols else f'_{col[:-1]}_count'

    def fit(self, df):
        df = self._add_arithmetic(df.copy())
        for col in self.num_cols + self.categorized_arithmetic_cols:
            _, uniques = np.floor(df[col]).factorize()
            self.category_map[col] = uniques
        for col in self.num_cols + self.categorized_arithmetic_cols:
            code_map = {cat: i for i, cat in enumerate(self.category_map[col])}
            df[self._cat_name(col)] = np.floor(df[col]).map(code_map).fillna(-1).astype('int32').astype(str)
        for col in self.count_source_cols:
            self.category_map[self._count_name(col)] = df[col].value_counts()
        for col, bins_list in self.bin_config.items():
            for n_bins in bins_list:
                kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
                kb.fit(df[[col]])
                self.category_map[f'{col}_{n_bins}_quantile_bin_'] = kb
        for cols in self.important_combos:
            combo_series = df[cols[0]].astype(str)
            for col in cols[1:]:
                combo_series = combo_series + '_' + df[col].astype(str)
            _, uniques = pd.factorize(combo_series, sort=False)
            self.category_map['_'.join(cols) + '_'] = uniques
        return self

    def transform(self, df):
        df = self._add_arithmetic(df.copy())
        for col in self.num_cols + self.categorized_arithmetic_cols:
            code_map = {cat: i for i, cat in enumerate(self.category_map[col])}
            df[self._cat_name(col)] = np.floor(df[col]).map(code_map).fillna(-1).astype('int32').astype(str)
        for col in self.count_source_cols:
            count_name = self._count_name(col)
            df[count_name] = df[col].map(self.category_map[count_name]).fillna(0).astype('int32')
        for col, bins_list in self.bin_config.items():
            for n_bins in bins_list:
                bin_name = f'{col}_{n_bins}_quantile_bin_'
                binned = self.category_map[bin_name].transform(df[[col]]).ravel().astype('int32')
                df[bin_name] = binned.astype(str)
        for cols in self.important_combos:
            combo_name = '_'.join(cols) + '_'
            combo_series = df[cols[0]].astype(str)
            for col in cols[1:]:
                combo_series = combo_series + '_' + df[col].astype(str)
            code_map = {cat: i for i, cat in enumerate(self.category_map[combo_name])}
            df[combo_name] = combo_series.map(code_map).fillna(-1).astype('int32').astype(str)
        return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, choices=[11, 16], required=True, help='notebook version')
    args = fold_args(ap)
    pair, params = f'{FOLDER}_v{args.version}', PARAMS[args.version]
    features = FeaturesV11 if args.version == 11 else FeaturesV16

    train, test, orig = read_train(), read_test(), read_original()
    y_orig = orig[TARGET].astype(int)
    orig_base = orig.drop(columns=[TARGET])
    x_base = train.drop(columns=[ID, TARGET])
    y = train[TARGET].astype(int)
    x_test_base = test.drop(columns=[ID])
    base_cat_cols = x_base.select_dtypes(include=['object']).columns.tolist()
    base_num_cols = x_base.select_dtypes(exclude=['object']).columns.tolist()

    def fit_fold(fold, tr_idx, val_idx):
        seed_everything(SEED + fold + 1)
        or_tr_idx = orig_train_index(fold, y_orig.to_numpy())
        x_tr_base = x_base.iloc[tr_idx].copy()
        fe = features(base_cat_cols, base_num_cols).fit(x_tr_base)
        x_tr = fe.transform(x_tr_base)
        x_val = fe.transform(x_base.iloc[val_idx].copy())
        x_tst = fe.transform(x_test_base)
        x_orig_tr = fe.transform(orig_base.iloc[or_tr_idx].copy())
        x_tr = pd.concat([x_tr, x_orig_tr], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y.iloc[tr_idx].reset_index(drop=True), y_orig.iloc[or_tr_idx].reset_index(drop=True)],
                         axis=0).reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        te_cols = fe.combo_names
        encoder = TargetEncoder(cv=FOLDS, smooth='auto', shuffle=True, random_state=SEED)
        te_names = [f'_{col}TE' for col in te_cols]
        x_tr[te_names] = encoder.fit_transform(x_tr[te_cols], y_tr)
        x_val[te_names] = encoder.transform(x_val[te_cols])
        x_tst[te_names] = encoder.transform(x_tst[te_cols])
        print(f'fold {fold}: train {x_tr.shape}, valid {x_val.shape}, test {x_tst.shape}', flush=True)

        with working_directory(work_dir(FOLDER)):  # pytabkit / lightning may write logs to the working directory
            model = RealMLP_TD_Classifier(**params)
            model.fit(x_tr, y_tr, x_val, y_val)
            valid_pred = model.predict_proba(x_val)[:, 1].astype(np.float32)
            test_pred = model.predict_proba(x_tst)[:, 1].astype(np.float32)
        del model
        torch.cuda.empty_cache()
        return valid_pred, test_pred

    run_folds(FOLDER, pair, fit_fold, args.folds, y=y.to_numpy())


if __name__ == '__main__':
    main()
