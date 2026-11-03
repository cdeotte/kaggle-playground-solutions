#!/usr/bin/env python
"""Public notebook "PS|S6|E5: RealMLP · PyTorch" (version 2) run on our 5 folds.
https://www.kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytorch

realmlp.py is the notebook's model code (unchanged); this script is its feature engineering and fold loop.
Features (the notebook's, fitted once on all competition training rows and applied to test and original rows): Driver /
Compound / Race factorized; LapNumber / RaceProgress, TyreLife / LapNumber, LapTime x Cumulative_Degradation (signed and
absolute) and LapTime / |Cumulative_Degradation|; every numeric column and the two ratios floored and factorized as
categories; counts of Driver / Compound / Race / Year / PitStop; 200 quantile bins of RaceProgress and 7 of LapTime;
Race x Compound and Race x Year combo categories. All categoricals are integer codes.
Original rows: the fold-matched 4/5 of the original dataset (same StratifiedKFold on its own target) is appended to each
fold's training rows.
Per fold: scikit-learn TargetEncoder (cv=5, smooth 'auto', shuffle, seed 42) of the two combo columns, fitted on the
training rows (competition + original).
Model: the notebook's RealMLP (16 members, 512-64-128 SiLU, 4 epochs, batch 256, AdamW lr 0.008 flat-cos); OOF = the
validation probabilities of the best epoch, test = the restored best checkpoint. Each fold re-seeds the global random
generators with 43 + fold. Test = mean of the 5 fold models; OOF and test are clipped to [1e-7, 1 - 1e-7].
Usage: python train.py [--folds 0,1,2,3,4]"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, read_original, read_test, read_train  # noqa: E402
from lib.runner import fold_args, orig_train_index, run_folds, work_dir  # noqa: E402
from realmlp import CONFIG, RealMLP_TD_Classifier  # noqa: E402

FOLDER = 'yekenot_realmlp_pytorch'
FOLDS, SEED = 5, 42
IMPORTANT_COMBOS = [('Race', 'Compound'), ('Race', 'Year')]


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def feature_engineering(df, cat_cols, num_cols, category_map, fit=False):
    # Arithmetic interaction
    df['_LapNumber_/_RaceProgress'] = (df['LapNumber'] / (df['RaceProgress'] + 1e-6)).astype('float32')
    df['_TyreLife_/_LapNumber'] = (df['TyreLife'] / df['LapNumber'].clip(lower=1)).astype('float32')
    df['_LapTime (s)_*_Cumulative_Degradation'] = (df['LapTime (s)'] * df['Cumulative_Degradation']).astype('float32')
    df['_LapTime (s)_*_Cumulative_Degradation_abs'] = (
        df['LapTime (s)'] * df['Cumulative_Degradation'].abs()).astype('float32')
    df['_LapTime (s)_/_Cumulative_Degradation_abs'] = (
        df['LapTime (s)'] / (df['Cumulative_Degradation'].abs() + 1e-6)).astype('float32')

    # Categorize string cats
    for col in cat_cols:
        if fit:
            codes, uniques = df[col].factorize()
            category_map[col] = uniques
        else:
            code_map = {cat: i for i, cat in enumerate(category_map[col])}
            codes = df[col].map(code_map).fillna(-1).astype('int32')
        df[col] = codes
        df[col] = df[col].astype('category')

    # Categorize numericals
    for col in num_cols + ['_LapNumber_/_RaceProgress', '_TyreLife_/_LapNumber']:
        cat_name = f'{col}_cat_' if col in num_cols else f'{col[1:]}_cat_'
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            code_map = {cat: i for i, cat in enumerate(category_map[col])}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype('category')

    # Count encoding
    for col in cat_cols + ['Year_cat_', 'PitStop_cat_']:
        count_name = f'_{col}_count' if col in cat_cols else f'_{col[:-1]}_count'
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].astype(object).map(count_map).fillna(0).astype('int32')

    # Discretize numericals
    bin_config = {'RaceProgress': [200], 'LapTime (s)': [7]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            bin_name = f'{col}_{n_bins}_quantile_bin_'
            if fit:
                kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
                binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                category_map[bin_name] = kb
            else:
                binned = category_map[bin_name].transform(df[[col]]).ravel().astype('int32')
            df[bin_name] = binned
            df[bin_name] = df[bin_name].astype('category')

    # Create interaction categories
    combo_names = []
    for cols in IMPORTANT_COMBOS:
        combo_name = '_'.join(cols) + '_'
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + '_' + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            code_map = {cat: i for i, cat in enumerate(category_map[combo_name])}
            codes = combo_series.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype('category')

    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    return df, new_cat_cols, combo_names


def main():
    args = fold_args()
    train, test, orig = read_train(), read_test(), read_original()
    y_orig = orig[TARGET]
    orig = orig.drop(columns=[TARGET])
    x = train.drop(columns=[ID, TARGET])
    y = train[TARGET]
    x_test = test.drop(columns=[ID])

    cat_cols = x.select_dtypes(include=['object']).columns.tolist()
    num_cols = x.select_dtypes(exclude=['object']).columns.tolist()
    category_map = {}
    x, _, combo_names = feature_engineering(x, cat_cols, num_cols, category_map, fit=True)
    x_test, _, _ = feature_engineering(x_test, cat_cols, num_cols, category_map)
    orig, new_cat_cols, _ = feature_engineering(orig, cat_cols, num_cols, category_map)
    cat_cols = cat_cols + new_cat_cols
    print(f'features: {x.shape[1]}, categorical: {len(cat_cols)}', flush=True)

    def fit_fold(fold, tr_idx, val_idx):
        seed_everything(SEED + fold + 1)
        or_tr_idx = orig_train_index(fold, y_orig.to_numpy())
        x_tr = pd.concat([x.iloc[tr_idx].copy(), orig.iloc[or_tr_idx].copy()], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        x_val = x.iloc[val_idx].copy()
        y_val = y.iloc[val_idx].reset_index(drop=True)
        x_tst = x_test.copy()

        te = TargetEncoder(cv=FOLDS, smooth='auto', shuffle=True, random_state=SEED)
        te_names = [f'_{col}TE' for col in combo_names]
        x_tr[te_names] = te.fit_transform(x_tr[combo_names], y_tr).astype('float32')
        x_val[te_names] = te.transform(x_val[combo_names]).astype('float32')
        x_tst[te_names] = te.transform(x_tst[combo_names]).astype('float32')

        ckpt_path = work_dir(FOLDER) / f'model_fold{fold}.pth'
        model = RealMLP_TD_Classifier(**CONFIG)
        model.fit(x_tr, y_tr, x_val, y_val, cat_col_names=cat_cols, ckpt_path=str(ckpt_path))
        valid_pred = model.best_val_probs_[:, 1].astype('float32')
        test_pred = model.predict_proba(x_tst)[:, 1].astype('float32')
        ckpt_path.unlink(missing_ok=True)
        del model
        torch.cuda.empty_cache()
        return valid_pred, test_pred

    run_folds(FOLDER, FOLDER, fit_fold, args.folds, y=y.to_numpy(), clip=(1e-7, 1 - 1e-7))


if __name__ == '__main__':
    main()
