#!/usr/bin/env python
"""Public notebook "RealMLP --> 0.97014" (version 3) run on our 5 folds.
https://www.kaggle.com/code/nawfeelrahman1124444/realmlp-0-97014

realmlp.py is the notebook's code up to its global target encoding: label-free features of train and test and its
RealMLP. The notebook then target / frequency encoded every column of X_raw once on all training rows (its
apply_encodings_cv: 5-fold out-of-fold, smoothing 10, seed 63) before its CV loop. Here the same function runs inside
each of our folds: on the fold's training rows (5-fold out-of-fold, seed 63) with the validation and test rows encoded
by the mean of the 5 inner maps. Features are sorted by name as in the notebook; the categorical columns are the
factorized raw predictors and the derived category features.
Model (the notebook's): one RealMLP with 16 vectorized ensemble members, 3 x 512 GELU layers, PBLD periodic embeddings
of the numeric features, one-hot / embedding of the categoricals, AdamW, flat-cos schedule, EMA 0.997875, label
smoothing, class weights, 4 epochs, batch 256. OOF = the validation probabilities of its best epoch
(best_val_probs_), test = the restored model's probabilities; test = mean of the 5 fold models.
Every fold starts from the notebook's seed 63 (in our run each fold was a separate process).
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from lib.runner import fold_args, run_folds  # noqa: E402

NAME = 'nawfeelrahman1124444_realmlp'


def main():
    fold_ids = fold_args()
    import realmlp as nb  # seeds and builds the notebook's features on import

    x_base, x_test_base = nb.X, nb.X_test
    raw, raw_test = nb.X_raw, nb.X_test_raw
    y = nb.y.reset_index(drop=True)
    target_columns = list(nb.target_enc_cols)
    categorical_columns = sorted(c for c in nb.cat_cols if c in x_base.columns)

    def fit_fold(fold, train_index, valid_index):
        nb.seed_everything(nb.SEED)
        x_train = x_base.iloc[train_index].copy().reset_index(drop=True)
        x_valid = x_base.iloc[valid_index].copy().reset_index(drop=True)
        x_test = x_test_base.copy().reset_index(drop=True)
        y_train = y.iloc[train_index].reset_index(drop=True)
        y_valid = y.iloc[valid_index].reset_index(drop=True)

        # the notebook's encoder on the fold's training rows; validation + test rows are its "test" part
        raw_train = raw.iloc[train_index].copy().reset_index(drop=True)
        raw_held = pd.concat([raw.iloc[valid_index].copy().reset_index(drop=True),
                              raw_test.copy().reset_index(drop=True)], axis=0, ignore_index=True)
        encoded_train, encoded_held = nb.apply_encodings_cv(raw_train, raw_held, cat_cols=target_columns,
                                                            target_col=y_train, n_splits=5, smoothing=10,
                                                            random_state=63)
        valid_size = len(valid_index)
        for raw_column in target_columns:
            for suffix in ('te', 'freq'):
                name = f'{raw_column}_{suffix}'
                x_train[name] = encoded_train[name].to_numpy()
                x_valid[name] = encoded_held[name].iloc[:valid_size].to_numpy()
                x_test[name] = encoded_held[name].iloc[valid_size:].to_numpy()
        ordered = sorted(x_train.columns)
        x_train = x_train.reindex(ordered, axis=1)
        x_valid = x_valid.reindex(ordered, axis=1)
        x_test = x_test.reindex(ordered, axis=1)

        model = nb.RealMLP_TD_Classifier(**nb.CONFIG)
        model.fit(x_train, y_train, x_valid, y_valid, cat_col_names=categorical_columns, X_test=x_test)
        valid_pred = np.asarray(model.best_val_probs_)[:, 1]
        test_pred = model.predict_proba(x_test)[:, 1]
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return valid_pred, test_pred

    run_folds(NAME, fit_fold, fold_ids, y=y.to_numpy())


if __name__ == '__main__':
    main()
