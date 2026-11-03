#!/usr/bin/env python
"""YDF gradient boosted trees on the raw columns (pair ydf_gbt_depth2_raw).

Inputs: the 14 competition columns, no feature engineering and no original rows. String columns and numeric columns with
fewer than 100 distinct values over train + test (Year, PitStop, LapNumber, Stint, TyreLife, Position, Position_Change)
are cast to strings (YDF categoricals); LapTime (s), LapTime_Delta, Cumulative_Degradation and RaceProgress stay float32.
Model: ydf.GradientBoostedTreesLearner, shrinkage 0.1, max_depth 2, growing_strategy BEST_FIRST_GLOBAL,
categorical_algorithm RANDOM, up to 10,000 trees with early stopping (look-ahead 300) on the fold's validation rows;
YDF's default seed and thread count (at most 32). Test = mean of the 5 fold models. CPU only.
Usage: python train_ydf.py [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import gc
import os
import sys
from pathlib import Path

N_THREADS = 32
for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[var] = str(N_THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ydf  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, model_dir, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

PAIR = 'ydf_gbt_depth2_raw'
FOLDER = 'classic'
YDF_PARAMS = dict(
    label=TARGET,
    task=ydf.Task.CLASSIFICATION,
    shrinkage=0.1,
    early_stopping_num_trees_look_ahead=300,
    max_depth=2,
    growing_strategy='BEST_FIRST_GLOBAL',
    categorical_algorithm='RANDOM',
    num_trees=10000,
)


def safe_name(col):
    out = str(col)
    for a, b in [(' ', '_'), ('(', ''), (')', ''), ('/', '_'), ('-', '_'), ('.', '_'), ('%', 'pct')]:
        out = out.replace(a, b)
    return out


def sanitize_feature_columns(df, target=None):
    mapping, used = {}, set()
    for c in df.columns:
        if c == target:
            mapping[c] = c
            used.add(c)
            continue
        base = safe_name(c)
        name, k = base, 1
        while name in used:
            k += 1
            name = f'{base}__{k}'
        mapping[c] = name
        used.add(name)
    return df.rename(columns=mapping), mapping


def infer_cat_num(train, test):
    feature_cols = [c for c in train.columns if c != TARGET]
    both = pd.concat([train[feature_cols], test[feature_cols]], axis=0, ignore_index=True)
    cats, nums = [], []
    for c in feature_cols:
        if pd.api.types.is_object_dtype(both[c]) or pd.api.types.is_string_dtype(both[c]):
            cats.append(c)
        elif both[c].nunique(dropna=False) < 100:
            cats.append(c)
        else:
            nums.append(c)
    return cats, nums


def load_frames():
    train_raw, test_raw = read_train(), read_test()
    y = train_raw[TARGET].astype('int32').to_numpy()
    train = train_raw.drop(columns=[ID]).copy()
    test = test_raw.drop(columns=[ID]).copy()
    train[TARGET] = y
    cats, nums = infer_cat_num(train, test)
    for df in [train, test]:
        for c in cats:
            df[c] = df[c].astype(str)
        for c in nums:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    train, col_map = sanitize_feature_columns(train, target=TARGET)
    test = test.rename(columns={c: col_map.get(c, safe_name(c)) for c in test.columns})
    print('categorical columns:', [col_map[c] for c in cats], flush=True)
    print('numeric columns:', [col_map[c] for c in nums], flush=True)
    return train, test, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    ydf.verbose(0)
    out = model_dir(FOLDER)
    train, test, y = load_frames()
    fold_id = folds(y)
    for k in fold_list(args.folds):
        if have_part(out, PAIR, k):
            print(f'{PAIR} fold {k}: done', flush=True)
            continue
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        trn, val = train.iloc[tr_idx].copy(), train.iloc[va_idx].copy()
        model = ydf.GradientBoostedTreesLearner(**YDF_PARAMS).train(trn, val)
        p_val = model.predict(val).reshape(-1).astype(np.float32)
        p_test = model.predict(test).reshape(-1).astype(np.float32)
        save_part(out, PAIR, k, p_val, p_test)
        print(f'{PAIR} fold {k}: {model.num_trees()} trees, AUC {roc_auc_score(y[va_idx], p_val):.6f}', flush=True)
        del model, trn, val
        gc.collect()
    assemble(out, PAIR, fold_id, y)


if __name__ == '__main__':
    main()
