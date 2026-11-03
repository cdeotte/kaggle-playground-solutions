#!/usr/bin/env python
"""LightGBM on the raw columns plus the churn label of the nearest original row (pair lgbm_nearest_original_churn).

Nearest original row: tenure, MonthlyCharges, TotalCharges (missing: 0) standardized on the original rows, a cKDTree on
the original rows, and the churn label (Yes = 1) of the single nearest original row of every train / test row.
Features: the 19 columns (the 15 string categoricals ordinal-encoded on the train rows and passed to LightGBM as
categorical features) and the nearest-original label: 20 features.
Model: 600 rounds, lr 0.03, 31 leaves, max_depth 6, subsample 0.8 every round, colsample_bytree 0.8, seed 42, 32 threads,
early stopping (50) on the validation fold. Test = mean of the 5 fold models.
Usage: python train_nearest_original.py"""
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "lgbm_nearest_original_churn"
N_FOLDS = 5
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "max_depth": 6,
    "num_leaves": 31,
    "learning_rate": 0.03,
    "n_estimators": 600,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": 32,
}


def main():
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = labels(train[TARGET])
    orig[TARGET] = labels(orig[TARGET])

    print("nearest original row...", flush=True)
    scaler = StandardScaler()
    orig_scaled = scaler.fit_transform(orig[NUMERIC_COLS].fillna(0))
    tree = cKDTree(orig_scaled)
    _, indices_train = tree.query(scaler.transform(train[NUMERIC_COLS].fillna(0)), k=1)
    train["NN_Target"] = orig[TARGET].iloc[indices_train].values
    _, indices_test = tree.query(scaler.transform(test[NUMERIC_COLS].fillna(0)), k=1)
    test["NN_Target"] = orig[TARGET].iloc[indices_test].values

    ignore_cols = [TARGET, ID]
    cat_cols = [col for col in train.columns if train[col].dtype == "object" and col not in ignore_cols]
    train[cat_cols] = train[cat_cols].fillna("Missing")
    test[cat_cols] = test[cat_cols].fillna("Missing")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train[cat_cols] = encoder.fit_transform(train[cat_cols])
    test[cat_cols] = encoder.transform(test[cat_cols])

    X = train.drop(columns=[c for c in ignore_cols if c in train.columns])
    y = train[TARGET]
    X_test = test[X.columns]
    cat_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
    print(f"{X.shape[1]} features", flush=True)

    fold_ids = folds(y.values)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    for k in range(N_FOLDS):
        train_idx = np.where(fold_ids != k)[0]
        val_idx = np.where(fold_ids == k)[0]
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_indices,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        val_preds = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_preds
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"fold {k + 1}/{N_FOLDS}: AUC {roc_auc_score(y_val, val_preds):.6f}", flush=True)

    print(f"{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)
    save_pair(HERE, PAIR, oof, test_preds)


if __name__ == "__main__":
    main()
