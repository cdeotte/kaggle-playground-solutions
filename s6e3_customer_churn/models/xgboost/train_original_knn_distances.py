#!/usr/bin/env python
"""GPU XGBoost on the raw columns plus nearest-original-row features (pair xgb_original_knn5_class_distances).

Features (23): the 19 raw columns (the 15 string columns ordinal-encoded, fitted on the train rows, unseen test values
-1) and 4 lookups into the 7,043 original rows in the standardized (tenure, MonthlyCharges, TotalCharges) space (scaler
fitted on the original rows, missing values 0): the mean churn label of the 5 nearest original rows (cKDTree), the
distance to the nearest original churner, the distance to the nearest original non-churner, and their ratio.
Model: depth 6, lr 0.03, 500 rounds (no early stopping), subsample / colsample_bytree 0.8, random_state 42.
Usage: python train_original_knn_distances.py"""
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

PAIR = 'xgb_original_knn5_class_distances'
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
N_FOLDS = 5
SEED = 42
EPSILON = 1e-5

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.03,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cuda',
}


def build_features():
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = labels(train[TARGET])
    orig[TARGET] = labels(orig[TARGET])

    scaler = StandardScaler()
    orig_scaled = scaler.fit_transform(orig[NUMERIC_COLS].fillna(0))
    train_scaled = scaler.transform(train[NUMERIC_COLS].fillna(0))
    test_scaled = scaler.transform(test[NUMERIC_COLS].fillna(0))
    orig_y = orig[TARGET].to_numpy()

    # mean churn label of the 5 nearest original rows
    tree_full = cKDTree(orig_scaled)
    for df, scaled in ((train, train_scaled), (test, test_scaled)):
        _, idx = tree_full.query(scaled, k=5)
        df['NN_Target_Mean_5'] = orig_y[idx.flatten()].reshape(-1, 5).mean(axis=1)

    # distance to the nearest original churner / non-churner
    tree_pos = cKDTree(orig_scaled[orig_y == 1])
    tree_neg = cKDTree(orig_scaled[orig_y == 0])
    for df, scaled in ((train, train_scaled), (test, test_scaled)):
        dist_pos, _ = tree_pos.query(scaled, k=1)
        dist_neg, _ = tree_neg.query(scaled, k=1)
        df['Dist_to_Churner'] = dist_pos
        df['Dist_to_Loyal'] = dist_neg
        df['Dist_Ratio'] = dist_pos / (dist_neg + EPSILON)

    cat_cols = [c for c in train.columns if train[c].dtype == 'object' and c not in (TARGET, ID)]
    train[cat_cols] = train[cat_cols].fillna('Missing')
    test[cat_cols] = test[cat_cols].fillna('Missing')
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train[cat_cols] = encoder.fit_transform(train[cat_cols])
    test[cat_cols] = encoder.transform(test[cat_cols])

    X = train.drop(columns=[TARGET, ID])
    y = train[TARGET]
    X_test = test[X.columns]
    return X, y, X_test


def main():
    X, y, X_test = build_features()
    print(f'{X.shape[1]} features', flush=True)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f'fold {fold}: AUC {roc_auc_score(y_va, oof[va_idx]):.6f}', flush=True)

    print(f'{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
