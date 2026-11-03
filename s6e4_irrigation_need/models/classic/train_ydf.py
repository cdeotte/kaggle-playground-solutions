#!/usr/bin/env python
"""YDF gradient boosted trees on the raw columns (pair ydf_gbt_depth2_raw_a, stack A).

Inputs: the 19 competition columns without id, no feature engineering and no original rows; string columns are YDF
categoricals, the 11 numerics stay numerical. Label: Low / Medium / High as 0 / 1 / 2.
Model: ydf.GradientBoostedTreesLearner (classification), shrinkage 0.1, max_depth 2, growing_strategy
BEST_FIRST_GLOBAL, categorical_algorithm RANDOM, up to 10,000 trees with early stopping (look-ahead 300) on the fold's
validation rows; YDF's default seed and thread count. Test = mean of the 5 fold models. CPU only.
Usage: python train_ydf.py [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import ydf
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, model_dir, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'ydf_gbt_depth2_raw_a'
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


def prepare_features(df):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).fillna('missing')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    ydf.verbose(0)
    out = model_dir(FOLDER)

    train, test = read_train(), read_test()
    train[TARGET] = train[TARGET].map(TARGET_MAP).astype(np.int32)
    y = train[TARGET].values
    train_df = prepare_features(train.drop(columns=[ID, TARGET]))
    train_df[TARGET] = y
    X_test = prepare_features(test.drop(columns=[ID]))

    fold_id = folds(y)
    for k in fold_list(args.folds):
        if have_part(out, PAIR, k):
            print(f'{PAIR} fold {k}: done', flush=True)
            continue
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        trn, val = train_df.iloc[tr_idx].copy(), train_df.iloc[va_idx].copy()
        model = ydf.GradientBoostedTreesLearner(**YDF_PARAMS).train(trn, val)
        p_val = np.asarray(model.predict(val.drop(columns=[TARGET])), dtype=np.float32)
        p_test = np.asarray(model.predict(X_test), dtype=np.float32)
        if p_val.ndim != 2 or p_val.shape[1] != 3:
            raise ValueError(f'expected 3-class probabilities, got {p_val.shape}')
        save_part(out, PAIR, k, p_val, p_test)
        print(f'{PAIR} fold {k}: {model.num_trees()} trees, balanced accuracy '
              f'{balanced_accuracy_score(y[va_idx], p_val.argmax(axis=1)):.6f}', flush=True)
        del model, trn, val
        gc.collect()
    assemble(out, PAIR, fold_id, y)


if __name__ == '__main__':
    main()
