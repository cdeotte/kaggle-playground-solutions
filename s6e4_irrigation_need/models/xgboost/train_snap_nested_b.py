#!/usr/bin/env python
"""Nested 5x5 GPU XGBoost on the 4 formula numerics snapped to the original grid (pair xgb_snap_original_te_nested_b).

Features (label-free; the original rows only provide the grid and the encodings):
- Soil_Moisture, Temperature_C, Rainfall_mm, Wind_Speed_kmh replaced by the nearest value of the original data, and
  <col>_diff = snapped value - value;
- the other 7 numerics and the 8 categoricals (XGBoost native categoricals);
- per-class original-data target encodings of the 4 snapped values (class share of the original rows with that value).
Nested 5x5 cross-validation: each outer fold (the shared folds) is split by an inner StratifiedKFold(5, shuffle,
seed 1000 + outer fold, outer fold = 1..5). Each of the 25 models: XGBClassifier multi:softprob, depth 4, lr 0.01,
subsample / colsample_bytree 0.8, max_bin 1024, balanced class weights, up to 50,000 rounds, early stopping 500 on the
balanced accuracy of the inner validation rows (best model kept), seed 42 + inner fold.
Output: OOF (5, n_train, 3): slice k = the inner out-of-fold predictions of outer fold k's training rows + the mean of
its 5 inner models on its held-out rows; test = mean of the 25 models.
Usage: python train_snap_nested_b.py [--folds 0,1,2,3,4] [--force]
       (each outer fold can run in its own process; finished outer folds are kept in parts/)"""
import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import fold_list, save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'xgb_snap_original_te_nested_b'
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
RANDOM_STATE = 42

NUMS = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity', 'Temperature_C', 'Humidity',
        'Rainfall_mm', 'Sunlight_Hours', 'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']
CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 'Irrigation_Type', 'Water_Source', 'Mulching_Used',
        'Region']
NUMS0 = ['Soil_Moisture', 'Temperature_C', 'Rainfall_mm', 'Wind_Speed_kmh']


def snap_to_original_and_diff(train, test, original, num_cols):
    """Replace each value by the nearest value of the original data; <col>_diff = snapped - value."""
    for col in num_cols:
        ref_vals = np.sort(original[col].dropna().unique())
        for df in (train, test):
            target_vals = df[col].values
            idx = np.searchsorted(ref_vals, target_vals, side='left')
            idx = np.clip(idx, 0, len(ref_vals) - 1)
            idx_left = np.maximum(idx - 1, 0)
            val_right = ref_vals[idx]
            val_left = ref_vals[idx_left]
            closest_vals = np.where(np.abs(target_vals - val_left) <= np.abs(target_vals - val_right), val_left,
                                    val_right)
            df[f'{col}_diff'] = closest_vals - df[col]
            df[col] = closest_vals
        print(f'snapped {col}', flush=True)
    return train, test


def build_features():
    train = read_train()
    test = read_test().set_index(ID)
    orig = read_original()
    train[TARGET] = train[TARGET].map(TARGET_MAP)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP)

    train, test = snap_to_original_and_diff(train, test, orig, NUMS0)
    diffs = [f'{c}_diff' for c in NUMS0]

    # per-class original target encodings of the snapped values
    for t in [0, 1, 2]:
        orig[f'y{t}'] = (orig[TARGET] == t).astype('float32')
    orig_te = []
    for c in NUMS0:
        for t in [0, 1, 2]:
            tt = f'y{t}'
            tmp = orig.groupby(c)[tt].mean()
            n = f'TE_{tt}_orig_{c}'
            tmp.name = n
            train = train.merge(tmp, on=c, how='left')
            test = test.merge(tmp, on=c, how='left')
            orig_te.append(n)

    features = NUMS + CATS + diffs + orig_te
    for col in CATS:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
    return train[features].copy(), train[TARGET].copy(), test[features].copy()


def balanced_accuracy_eval():
    def xgb_balanced_accuracy(y_true, y_pred):
        y_pred_labels = np.argmax(y_pred.reshape(-1, 3), axis=1)
        return balanced_accuracy_score(y_true.astype(int), y_pred_labels)
    xgb_balanced_accuracy.__name__ = 'bal_ACC'
    return xgb_balanced_accuracy


def run_outer_fold(outer_fold, outer_tr_idx, outer_va_idx, X_train_full, y_train_full, X_test_full):
    n_train = len(X_train_full)
    X_outer_tr = X_train_full.iloc[outer_tr_idx].copy()
    y_outer_tr = y_train_full.iloc[outer_tr_idx].copy()
    X_outer_va = X_train_full.iloc[outer_va_idx].copy()
    y_outer_va = y_train_full.iloc[outer_va_idx].copy()
    outer_full_pred = np.zeros((n_train, 3), dtype=np.float32)
    outer_va_pred_sum = np.zeros((len(outer_va_idx), 3), dtype=np.float32)
    test_pred = np.zeros((len(X_test_full), 3), dtype=np.float32)

    skf_inner = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=1000 + outer_fold)
    for inner_fold, (inner_tr_loc, inner_va_loc) in enumerate(skf_inner.split(X_outer_tr, y_outer_tr), start=1):
        X_tr = X_outer_tr.iloc[inner_tr_loc].copy()
        y_tr = y_outer_tr.iloc[inner_tr_loc].copy()
        X_va = X_outer_tr.iloc[inner_va_loc].copy()
        y_va = y_outer_tr.iloc[inner_va_loc].copy()

        classes = np.unique(y_tr)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
        class_weight_dict = dict(zip(classes, weights))
        sample_weights = np.array([class_weight_dict[label] for label in y_tr])

        model = xgb.XGBClassifier(
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=50000,
            objective='multi:softprob',
            learning_rate=0.01,
            eval_metric=balanced_accuracy_eval(),
            callbacks=[xgb.callback.EarlyStopping(rounds=500, metric_name='bal_ACC', maximize=True, save_best=True)],
            max_bin=1024,
            random_state=RANDOM_STATE + inner_fold,
            enable_categorical=True,
            device='cuda',
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], sample_weight=sample_weights, verbose=False)

        p_inner_va = model.predict_proba(X_va).astype(np.float32)
        outer_full_pred[outer_tr_idx[inner_va_loc]] = p_inner_va
        inner_acc = balanced_accuracy_score(y_va, np.argmax(p_inner_va, axis=1))
        print(f'outer {outer_fold - 1} inner {inner_fold - 1}: balanced accuracy {inner_acc:.5f}', flush=True)
        outer_va_pred_sum += model.predict_proba(X_outer_va).astype(np.float32)
        test_pred += model.predict_proba(X_test_full).astype(np.float32) / N_INNER_SPLITS
        del X_tr, X_va, y_tr, y_va, model
        gc.collect()

    outer_va_pred = outer_va_pred_sum / N_INNER_SPLITS
    outer_full_pred[outer_va_idx] = outer_va_pred
    print(f'outer {outer_fold - 1}: held-out balanced accuracy '
          f'{balanced_accuracy_score(y_outer_va, np.argmax(outer_va_pred, axis=1)):.5f}, all-row balanced accuracy '
          f'{balanced_accuracy_score(y_train_full, np.argmax(outer_full_pred, axis=1)):.5f}', flush=True)
    return outer_full_pred, test_pred


def part_path(k, split):
    return HERE / 'parts' / f'{PAIR}_outer{k}_{split}.npy'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4', help='outer folds to run')
    ap.add_argument('--force', action='store_true', help='rerun outer folds that are already done')
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds)
            if args.force or not (part_path(k, 'oof').exists() and part_path(k, 'test').exists())]
    if todo:
        X_train_full, y_train_full, X_test_full = build_features()
        fold_id = folds(y_train_full.values)
        for k in todo:
            oof_k, test_k = run_outer_fold(k + 1, np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k),
                                           X_train_full, y_train_full, X_test_full)
            part_path(k, 'oof').parent.mkdir(parents=True, exist_ok=True)
            np.save(part_path(k, 'oof'), oof_k)
            np.save(part_path(k, 'test'), test_k)
    if not all(part_path(k, 'oof').exists() and part_path(k, 'test').exists() for k in range(N_OUTER_SPLITS)):
        print(f'{PAIR}: not all outer folds are done yet', flush=True)
        return
    oof_matrix = np.stack([np.load(part_path(k, 'oof')) for k in range(N_OUTER_SPLITS)]).astype(np.float32)
    test_pred = np.mean([np.load(part_path(k, 'test')).astype(np.float64) for k in range(N_OUTER_SPLITS)], axis=0)
    save_pair(HERE, PAIR, oof_matrix, test_pred)
    y = labels(read_train()[TARGET])
    print(f'{PAIR}: OOF {oof_matrix.shape}, balanced accuracy of the slice mean '
          f'{balanced_accuracy_score(y, oof_matrix.mean(axis=0).argmax(1)):.6f}', flush=True)


if __name__ == '__main__':
    main()
