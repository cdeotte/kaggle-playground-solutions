#!/usr/bin/env python
"""TabNet on the raw columns + log1p / sqrt transforms, rule-score features and original-data class priors.

Inputs (68), built on train + test (the original rows are a reference only, not training rows):
- the 8 categoricals, ordinal-encoded on original + train + test (embedded by TabNet, dim 3), first;
- the 11 numerics, their log1p and sqrt (of max(x, 0));
- rule score 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10] -
  2 [stage Harvest] - 2 [stage Sowing] - [mulch Yes], is_low (score <= 0) and is_high (score > 3);
- per class k: the original-data share of class k in every categorical level (24 columns; unseen -> original prior).
Model: pytorch-tabnet TabNetClassifier, n_d = n_a = 32, 5 steps, gamma 1.5, 2 independent + 2 shared GLU layers,
lambda_sparse 1e-4, momentum 0.02, gradient clip 2, sparsemax masks; Adam lr 2e-2, weight decay 1e-5, StepLR (10
epochs, x0.9); batch 8192, virtual batch 1024; training rows sampled with class weights n / (3 n_class) (from all train
rows); up to 300 epochs with early stopping on the held-out balanced accuracy (patience 30, best weights restored);
seed 42. Test = mean of the 5 fold models.
Usage: python train_tabnet_formula.py [--folds 0,1,2,3,4]"""
import argparse
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import OrdinalEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, TARGET, TARGET_MAP, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'tabnet_formula_orig_priors_b'
MODEL_DIR = model_dir('neural_nets')
SEED = 42
MAX_EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 8192
VIRTUAL_BATCH_SIZE = 1024

NUMS = ['Soil_pH', 'Organic_Carbon', 'Electrical_Conductivity', 'Sunlight_Hours', 'Wind_Speed_kmh', 'Field_Area_hectare',
        'Temperature_C', 'Soil_Moisture', 'Humidity', 'Previous_Irrigation_mm', 'Rainfall_mm']
CATS = list(CATEGORICALS)
NUM_FEATS = (NUMS + [f'{c}_log1p' for c in NUMS] + [f'{c}_sqrt' for c in NUMS] + ['formula_score', 'is_low', 'is_high']
             + [f'ORIG_{c}_k{k}' for c in CATS for k in range(3)])


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def formula_score(df):
    return (
        (df['Soil_Moisture'] < 25) * 2
        + (df['Rainfall_mm'] < 300) * 2
        + (df['Temperature_C'] > 30) * 1
        + (df['Wind_Speed_kmh'] > 10) * 1
        - (df['Crop_Growth_Stage'] == 'Harvest') * 2
        - (df['Crop_Growth_Stage'] == 'Sowing') * 2
        - (df['Mulching_Used'] == 'Yes') * 1
    ).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(SEED)
    train, test, orig = read_train(), read_test(), read_original()
    y = train[TARGET].map(TARGET_MAP).astype(np.int64).values
    print(f'train={train.shape}, test={test.shape}, orig={orig.shape}', flush=True)

    y_orig = orig[TARGET].map(TARGET_MAP).values
    for df in [train, test]:
        df['formula_score'] = formula_score(df)
        df['is_low'] = (df['formula_score'] <= 0).astype(np.float32)
        df['is_high'] = (df['formula_score'] > 3).astype(np.float32)
        for c in NUMS:
            v = df[c].astype(np.float32)
            df[f'{c}_log1p'] = np.log1p(v.clip(lower=0))
            df[f'{c}_sqrt'] = np.sqrt(v.clip(lower=0))
        for c in CATS:
            for k in range(3):
                m = orig.groupby(c).apply(lambda g: (g[TARGET].map(TARGET_MAP) == k).mean())
                df[f'ORIG_{c}_k{k}'] = df[c].map(m).fillna((y_orig == k).mean()).astype(np.float32)

    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    oe.fit(pd.concat([orig[CATS], train[CATS], test[CATS]], axis=0, ignore_index=True).astype(str))
    cat_dims = [int(len(c)) + 1 for c in oe.categories_]  # +1 for unknown
    cat_idxs = list(range(len(CATS)))

    def build_X(df):
        cats = np.clip(oe.transform(df[CATS].astype(str)).astype(np.int64), 0, None)  # unknown -> 0
        return np.hstack([cats, df[NUM_FEATS].astype(np.float32).values]).astype(np.float32)

    X_train = build_X(train)
    X_test = build_X(test)
    print(f'X_train={X_train.shape}, X_test={X_test.shape}, cat_dims={cat_dims}', flush=True)

    cls_counts = np.bincount(y)
    cls_weights = {k: float(cls_counts.sum()) / (3 * cls_counts[k]) for k in range(3)}
    print(f'Class weights: {cls_weights}', flush=True)

    fold_ids = folds(y)
    for k in fold_list(args.folds):
        print(f'\n--- Fold {k + 1}/5 ---', flush=True)
        tr_idx, val_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        clf = TabNetClassifier(
            n_d=32, n_a=32, n_steps=5, gamma=1.5, n_independent=2, n_shared=2, lambda_sparse=1e-4, momentum=0.02,
            clip_value=2.0, cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=3,
            optimizer_fn=torch.optim.Adam, optimizer_params={'lr': 2e-2, 'weight_decay': 1e-5},
            scheduler_fn=torch.optim.lr_scheduler.StepLR, scheduler_params={'step_size': 10, 'gamma': 0.9},
            mask_type='sparsemax', device_name='cuda', seed=SEED, verbose=0,
        )
        clf.fit(X_train[tr_idx], y[tr_idx], eval_set=[(X_train[val_idx], y[val_idx])],
                eval_metric=['balanced_accuracy'], max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
                virtual_batch_size=VIRTUAL_BATCH_SIZE, weights=cls_weights, drop_last=False)
        p_val = clf.predict_proba(X_train[val_idx]).astype(np.float32)
        p_test = clf.predict_proba(X_test).astype(np.float32)
        print(f'  BAC={balanced_accuracy_score(y[val_idx], np.argmax(p_val, axis=1)):.5f}  best_epoch={clf.best_epoch}',
              flush=True)
        save_part(MODEL_DIR, PAIR, k, p_val, p_test)
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
