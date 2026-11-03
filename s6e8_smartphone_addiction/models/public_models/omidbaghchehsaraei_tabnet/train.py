#!/usr/bin/env python
"""Public notebook "TabNet for Predicting Smartphone Addiction" (version 1) run on our 5 folds.
https://www.kaggle.com/code/omidbaghchehsaraei/tabnet-for-predicting-smartphone-addiction

Features (lib/ratio_te.py): the 12 raw predictors as string categories (label codes of the sorted train + test levels,
fed to TabNet's categorical embeddings), missing_count and 9 screen-time ratio features (NaN -> 0), and per fold a
smoothed target encoding (smoothing 10) and a frequency encoding of every raw predictor. The notebook computed the
encodings once on all training rows before its CV loop; here they are computed inside each fold (inner 5-fold
out-of-fold for the training rows, the fold's training-row map for the validation and test rows). The continuous columns
are standardized on the fold's training rows.
Model (the notebook's): pytorch-tabnet TabNetClassifier, n_d = n_a = 8, 3 steps, gamma 1.3, sparsemax, categorical
embedding dim 1, Adam lr 0.02 (weight decay 1e-5) with StepLR(10, 0.5), batch 1024 / virtual 128, up to 200 epochs
with patience 10 on the validation AUC, seed 42. Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from lib.ratio_te import ID, TARGET, add_fold_encodings, categorical_frames, engineer  # noqa: E402
from lib.runner import fold_args, run_folds, work_dir, working_directory  # noqa: E402

NAME = 'omidbaghchehsaraei_tabnet'


def main():
    fold_ids = fold_args()
    train, test, initial = categorical_frames()
    category_levels = {column: list(train[column].cat.categories) for column in initial}
    engineered, engineered_test = engineer(train, initial), engineer(test, initial)
    features = [column for column in engineered.columns if column not in (TARGET, ID)]
    y = pd.Series(train[TARGET].to_numpy(np.int8))

    def fit_fold(fold, train_index, valid_index):
        x_train = engineered.iloc[train_index][features].copy().reset_index(drop=True)
        x_valid = engineered.iloc[valid_index][features].copy().reset_index(drop=True)
        x_test = engineered_test[features].copy().reset_index(drop=True)
        y_train = y.iloc[train_index].reset_index(drop=True)
        y_valid = y.iloc[valid_index].reset_index(drop=True)
        add_fold_encodings(x_train, y_train, x_valid, x_test, initial)

        for column in initial:
            mapping = {value: code for code, value in enumerate(category_levels[column])}
            x_train[column] = x_train[column].astype(str).map(mapping).astype(np.int64)
            x_valid[column] = x_valid[column].astype(str).map(mapping).astype(np.int64)
            x_test[column] = x_test[column].astype(str).map(mapping).astype(np.int64)
        continuous = [column for column in x_train.columns if column not in initial]
        x_train[continuous] = x_train[continuous].fillna(0.0)
        x_valid[continuous] = x_valid[continuous].fillna(0.0)
        x_test[continuous] = x_test[continuous].fillna(0.0)
        continuous_index = [x_train.columns.get_loc(column) for column in continuous]
        categorical_index = [x_train.columns.get_loc(column) for column in initial]
        categorical_dimensions = [len(category_levels[column]) for column in initial]

        train_values = x_train.to_numpy(dtype=np.float64)
        valid_values = x_valid.to_numpy(dtype=np.float64)
        test_values = x_test.to_numpy(dtype=np.float64)
        scaler = StandardScaler()
        train_values[:, continuous_index] = scaler.fit_transform(train_values[:, continuous_index])
        valid_values[:, continuous_index] = scaler.transform(valid_values[:, continuous_index])
        test_values[:, continuous_index] = scaler.transform(test_values[:, continuous_index])

        model = TabNetClassifier(
            cat_idxs=categorical_index, cat_dims=categorical_dimensions, cat_emb_dim=1, n_d=8, n_a=8, n_steps=3,
            gamma=1.3, n_independent=2, n_shared=2, epsilon=1e-15, lambda_sparse=1e-3, optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr': 0.02, 'weight_decay': 1e-5}, scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={'step_size': 10, 'gamma': 0.5}, mask_type='sparsemax', seed=42, verbose=1,
            device_name='cuda')
        with working_directory(work_dir(NAME)):
            model.fit(X_train=train_values, y_train=y_train.to_numpy(), eval_set=[(valid_values, y_valid.to_numpy())],
                      eval_metric=['auc'], patience=10, batch_size=1024, virtual_batch_size=128, max_epochs=200,
                      drop_last=False)
            valid_pred = model.predict_proba(valid_values)[:, 1]
            test_pred = model.predict_proba(test_values)[:, 1]
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return valid_pred, test_pred

    run_folds(NAME, fit_fold, fold_ids, y=y.to_numpy())


if __name__ == '__main__':
    main()
