#!/usr/bin/env python
"""Public notebook "RealMLP from scratchCV:0.91908" run on our 5 folds.
https://www.kaggle.com/code/yunsuxiaozi/realmlp-from-scratchcv-0-91908

Features (the notebook's, label-free, built on train and test separately; bins fitted on train): tenure,
MonthlyCharges and TotalCharges rounded to integers as categories; their digits at positions 10^-2 .. 10^3 and every
pair of the digits at 10^0 .. 10^3; MonthlyCharges / TotalCharges, TotalCharges / MonthlyCharges, TotalCharges /
tenure, MonthlyCharges / (TotalCharges / tenure), tenure^2; the Contract + InternetService + PaymentMethod combo; the
service count; quantile bins of MonthlyCharges (100, 1000) and TotalCharges (500, 4000). Columns constant on test are
dropped. Every categorical (the 15 string columns, bins, rounded values, combo, service count and digit columns) is
coded by frequency rank in train (values seen fewer than 5 times share one code).
Per fold: a scikit-learn TargetEncoder (binary, cv 5) of the categoricals with more than 9 codes is appended to the
numerics, which are then robust-scaled and smoothly clipped (fitted on the training rows).
Model (realmlp.py): RealMLP with 8 vectorized members, PBLD numeric embeddings, one-hot (up to 8 levels) or 10-d
embeddings of the categoricals, a scaling layer and 512-64-128 GELU NTP layers with dropout 0.05; AdamW (lr 0.003,
x10 for the scaling layer, x0.1 for biases; weight decay 1e-2 scaled likewise; betas 0.9 / 0.98), flat-then-linear
learning-rate decay, label smoothing 0.04 with a cosine schedule, gradient clipping 1; 3 epochs, batch 256. The
weights of the epoch with the best validation AUC (above 0.915) are restored and predict the validation and test rows.
The notebook seeds once (2026) and trains its folds in one loop, so the folds run in order in one process.
Test = mean of the fold models.
Usage: python train.py"""
import copy
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, read_test, read_train  # noqa: E402
from lib.runner import kfold, run_all_folds, split  # noqa: E402
from realmlp import (DEVICE, RealMLP, RobustScaleSmoothClipTransform, cos, cross_entropy_loss,  # noqa: E402
                     flat_anneal, get_parameter_groups_3class)

warnings.filterwarnings('ignore')

FOLDER = 'yunsuxiaozi_realmlp_pytorch'
SEED = 2026
TEST_FOLDS = 20
EPOCHS = 3
TRAIN_BS = 256
EVAL_BS = 10240
EMBED_DIM = 10
LR = 0.003
N_ENS = 8
SVC = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies']


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def fe(df):
    for c in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        df[c + '_cat'] = df[c].round().astype(str)
    for c in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        for k in range(-2, 4):
            df[f'{c}_digit{k}'] = (df[c] // (10 ** k) % 10).astype('int8')
        for k1 in range(4):
            for k2 in range(k1 + 1, 4):
                df[f'{c}_digit{k1}{k2}'] = df[f'{c}_digit{k1}'].astype(str) + df[f'{c}_digit{k2}'].astype(str)
    df['x1_0'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1e-6).fillna(0)
    df['x1_1'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1e-6).fillna(0)
    df['x2'] = df['TotalCharges'] / (df['tenure'] + 1e-6).fillna(0)
    df['x3'] = df['MonthlyCharges'] / (df['x2'])
    df['x4'] = (df['tenure'] ** 2).fillna(0)
    df['com'] = df['Contract'].astype(str) + df['InternetService'].astype(str) + df['PaymentMethod'].astype(str)
    return df


def build_features():
    """The notebook's feature engineering. The quantile bins subsample with the global numpy random state, so this
    runs right after seeding."""
    train = read_train().drop([ID], axis=1)
    test = read_test().drop([ID], axis=1)
    train[TARGET] = (train[TARGET] == 'Yes').astype(int)
    cats = [c for c in test.columns if train[c].dtype == object]
    nums = [c for c in test.columns if c != 'id' and train[c].dtype != object]

    for df in [train, test]:
        fe(df)
    for df in [train, test]:
        df['service_count'] = (df[SVC] == 'Yes').sum(axis=1).astype('float32')
    nums += ['x1_0', 'x1_1', 'x2', 'x3', 'x4']

    for col in ['MonthlyCharges', 'TotalCharges']:
        bins = [100, 1000] if col == 'MonthlyCharges' else [500, 4000]
        for kkk in bins:
            kb = KBinsDiscretizer(n_bins=kkk, encode='ordinal', strategy='quantile')
            train_values = train[col].values.reshape(-1, 1)
            kb.fit(train_values)
            train[f'{col}_bin{kkk}'] = kb.transform(train_values).flatten().astype(int)
            test[f'{col}_bin{kkk}'] = kb.transform(test[col].values.reshape(-1, 1)).flatten().astype(int)

    drop = [c for c in test.columns if test[c].nunique() == 1]
    train.drop(drop, axis=1, inplace=True)
    test.drop(drop, axis=1, inplace=True)

    category = cats + ['TotalCharges_bin500', 'TotalCharges_bin4000', 'MonthlyCharges_bin100', 'MonthlyCharges_bin1000',
                       'tenure_cat', 'MonthlyCharges_cat', 'TotalCharges_cat', 'com', 'service_count'] + \
        [c for c in test.columns if 'digit' in c]
    for c in category:
        freq = train[c].value_counts()
        mapping = {val: idx for idx, (val, count) in enumerate(freq[freq >= 5].items())}
        mapping_default = len(mapping)
        train[c] = train[c].map(lambda x: mapping.get(x, mapping_default))
        test[c] = test[c].map(lambda x: mapping.get(x, mapping_default))
    return train, test, nums, category


def predict(model, x_num_t, x_cat_t, n_rows):
    model.eval()
    with torch.no_grad():
        probs = []
        for i in range(0, n_rows, EVAL_BS):
            probs.append(model(x_num_t[i:i + EVAL_BS], x_cat_t[i:i + EVAL_BS]).mean(dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)[:, 1]


def cross_validate(n_folds):
    seed_everything(SEED)
    train, test, nums, category = build_features()
    X_num = train[nums].values
    X_cat = train[category].values
    y = train[TARGET].values
    testX_num = test[nums].values
    testX_cat = test[category].values

    oof = np.zeros(len(train))
    test_preds = []
    fold_ids = kfold(y, n_folds)
    for fold in range(n_folds):
        print(f'\nfold {fold} ------------------------------', flush=True)
        train_idx, valid_idx = split(fold_ids, fold)
        X_num_train, X_cat_train, y_train = X_num[train_idx], X_cat[train_idx], y[train_idx]
        X_num_val, X_cat_val, y_val = X_num[valid_idx], X_cat[valid_idx], y[valid_idx]

        cols_to_encode = [i for i in range(X_cat_train.shape[1]) if X_cat_train[:, i].max() > 8]
        te = TargetEncoder(target_type='binary', cv=5)
        X_cat_train_encoded = te.fit_transform(X_cat_train[:, cols_to_encode], y_train)
        X_cat_val_encoded = te.transform(X_cat_val[:, cols_to_encode])
        X_cat_test_encoded = te.transform(testX_cat[:, cols_to_encode])
        X_num_train = np.concatenate([X_num_train, X_cat_train_encoded], axis=1)
        X_num_val = np.concatenate([X_num_val, X_cat_val_encoded], axis=1)
        X_num_test_fold = np.concatenate([testX_num, X_cat_test_encoded], axis=1)

        rssc = RobustScaleSmoothClipTransform()
        rssc.fit(X_num_train)
        X_num_train = rssc.transform(X_num_train)
        X_num_val = rssc.transform(X_num_val)
        X_num_test_fold = rssc.transform(X_num_test_fold)

        train_idx_iter = np.arange(len(y_train))
        realmlp = RealMLP(
            input_dim=X_num_train.shape[1] + X_cat.shape[1] * EMBED_DIM,
            output_dim=2,
            cat_dims=X_cat.max(axis=0) + 1,
            n_numerical=X_num_train.shape[1],
            n_ens=N_ENS,
        ).to(DEVICE)
        best_auc, best_state = 0.915, None

        scale_params, weights, biases = get_parameter_groups_3class(realmlp)
        optimizer = torch.optim.AdamW([
            {'params': scale_params, 'name': 'scale', 'lr': LR * 10.0, 'weight_decay': 1e-2 * 0.1},
            {'params': weights, 'name': 'weights', 'lr': LR, 'weight_decay': 1e-2},
            {'params': biases, 'name': 'biases', 'lr': LR * 0.1, 'weight_decay': 1e-2 * 0.5},
        ], betas=(0.9, 0.98))

        X_num_train_t = torch.as_tensor(X_num_train, dtype=torch.float32, device=DEVICE)
        X_cat_train_t = torch.as_tensor(X_cat_train, dtype=torch.long, device=DEVICE)
        y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=DEVICE)
        X_num_val_t = torch.as_tensor(X_num_val, dtype=torch.float32, device=DEVICE)
        X_cat_val_t = torch.as_tensor(X_cat_val, dtype=torch.long, device=DEVICE)
        X_num_test_t = torch.as_tensor(X_num_test_fold, dtype=torch.float32, device=DEVICE)
        X_cat_test_t = torch.as_tensor(testX_cat, dtype=torch.long, device=DEVICE)

        for epoch in range(EPOCHS):
            for idx in range(0, len(y_train), TRAIN_BS):
                progress = epoch / EPOCHS + idx / (len(y_train) * EPOCHS)
                bs_idx = train_idx_iter[idx:idx + TRAIN_BS]
                realmlp.train()
                optimizer.param_groups[0]['lr'] = flat_anneal(LR * 10.0, progress)
                optimizer.param_groups[1]['lr'] = flat_anneal(LR, progress)
                optimizer.param_groups[2]['lr'] = flat_anneal(LR * 0.1, progress)
                optimizer.zero_grad()
                y_pred = realmlp(X_num_train_t[bs_idx], X_cat_train_t[bs_idx])
                loss = cross_entropy_loss(y_train_t[bs_idx].repeat_interleave(N_ENS), y_pred.reshape(-1, 2),
                                          label_smoothing=cos(0.04, progress))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(realmlp.parameters(), max_norm=1.0)
                optimizer.step()

            auc = roc_auc_score(y_val, predict(realmlp, X_num_val_t, X_cat_val_t, len(y_val)))
            if auc > best_auc:
                best_auc = auc
                best_state = copy.deepcopy(realmlp.state_dict())
                print(f'fold {fold}, epoch {epoch}: best AUC {best_auc:.5f}', flush=True)
            np.random.shuffle(train_idx_iter)

        realmlp.load_state_dict(best_state)
        oof[valid_idx] = predict(realmlp, X_num_val_t, X_cat_val_t, len(y_val))
        test_preds.append(predict(realmlp, X_num_test_t, X_cat_test_t, len(testX_num)))
        print(f'fold {fold}: AUC {roc_auc_score(y_val, oof[valid_idx]):.6f}', flush=True)
        del realmlp, optimizer, best_state
        torch.cuda.empty_cache()
    return oof, np.mean(test_preds, axis=0)


def main():
    y = (read_train()[TARGET] == 'Yes').astype(int).values
    run_all_folds(FOLDER, FOLDER, cross_validate, test_folds=TEST_FOLDS, y=y)


if __name__ == '__main__':
    main()
