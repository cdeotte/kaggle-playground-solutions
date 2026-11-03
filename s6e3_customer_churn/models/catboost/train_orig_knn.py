#!/usr/bin/env python
"""GPU CatBoost on the raw columns plus nearest-original-row features (pair catboost_orig_knn_class_distances).

The numerics tenure, MonthlyCharges and TotalCharges (blanks as 0) are standardized with a StandardScaler fit on the
original rows. KD-trees (scipy cKDTree) over the original rows give, for every train / test row:
  - NN_Target_Mean_5: the churn rate of its 5 nearest original rows
  - Dist_to_Churner / Dist_to_Loyal: the distance to the nearest original churner / non-churner, and their ratio
The 15 string columns are CatBoost-native categoricals; SeniorCitizen and the numerics stay numeric (23 features).
CatBoost (GPU): depth 6, lr 0.03, up to 1000 iterations with early stopping (50) on the validation fold, seed 42.
Test = mean of the 5 fold models.

Usage: python train_orig_knn.py"""
import sys
import warnings
from pathlib import Path

from catboost import CatBoostClassifier, Pool
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402
from models.catboost.lib.cv import run_folds  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'catboost_orig_knn_class_distances'
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
EPSILON = 1e-5

CB_PARAMS = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'random_seed': 42,
    'task_type': 'GPU',
    'verbose': False,
}


def main():
    train, test, orig = read_train(), read_test(), read_original()
    target_map = {'No': 0, 'Yes': 1}
    train[TARGET] = train[TARGET].map(target_map)
    orig[TARGET] = orig[TARGET].map(target_map)

    scaler = StandardScaler()
    orig_scaled = scaler.fit_transform(orig[NUMERIC_COLS].fillna(0))
    train_scaled = scaler.transform(train[NUMERIC_COLS].fillna(0))
    test_scaled = scaler.transform(test[NUMERIC_COLS].fillna(0))

    # churn rate of the 5 nearest original rows
    tree_full = cKDTree(orig_scaled)
    for df, scaled in ((train, train_scaled), (test, test_scaled)):
        _, indices_5 = tree_full.query(scaled, k=5)
        df['NN_Target_Mean_5'] = orig[TARGET].iloc[indices_5.flatten()].values.reshape(-1, 5).mean(axis=1)

    # distance to the nearest original churner / non-churner
    tree_pos = cKDTree(orig_scaled[orig[TARGET] == 1])
    tree_neg = cKDTree(orig_scaled[orig[TARGET] == 0])
    for df, scaled in ((train, train_scaled), (test, test_scaled)):
        dist_pos, _ = tree_pos.query(scaled, k=1)
        dist_neg, _ = tree_neg.query(scaled, k=1)
        df['Dist_to_Churner'] = dist_pos
        df['Dist_to_Loyal'] = dist_neg
        df['Dist_Ratio'] = dist_pos / (dist_neg + EPSILON)

    ignore_cols = [TARGET, ID]
    cat_cols = [c for c in train.columns if train[c].dtype == 'object' and c not in ignore_cols]
    train[cat_cols] = train[cat_cols].fillna('Missing')
    test[cat_cols] = test[cat_cols].fillna('Missing')
    X = train.drop(columns=ignore_cols)
    y = train[TARGET].values
    X_test = test[X.columns]
    print(f'features: {X.shape[1]} ({len(cat_cols)} categorical)', flush=True)

    def fit_fold(fold, tr_idx, va_idx, seed):
        train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_cols)
        val_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_cols)
        test_pool = Pool(X_test, cat_features=cat_cols)
        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=100)
        return model.predict_proba(val_pool)[:, 1], model.predict_proba(test_pool)[:, 1]

    oof, test_pred = run_folds(fit_fold, y, len(test), name=PAIR)
    print(f'{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
