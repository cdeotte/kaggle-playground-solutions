#!/usr/bin/env python
"""Public notebook "S6E5 | HistBoost Baseline" (version 2) run on our 5 folds.
https://www.kaggle.com/code/tamerlanomralinov/s6e5-histboost-baseline

Features (the notebook's, label-free): the 14 competition columns plus lap_tyre = LapNumber x TyreLife; Driver, Compound
and Race label-encoded on train + test values.
Model (the notebook's): scikit-learn HistGradientBoostingClassifier, lr 0.03, max_depth 6, l2 0.1, up to 3,000
iterations with internal early stopping (10% validation split, 100 rounds without improvement), seed 42.
Test = mean of the 5 fold models; OOF and test are clipped to [1e-7, 1 - 1e-7].
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, read_test, read_train  # noqa: E402
from lib.runner import fold_args, run_folds  # noqa: E402

FOLDER = 'tamerlanomralinov_hgb'
PARAMS = dict(max_iter=3000, learning_rate=0.03, max_depth=6, l2_regularization=0.1, early_stopping=True,
              validation_fraction=0.1, n_iter_no_change=100, random_state=42)


def main():
    args = fold_args()
    train, test = read_train(), read_test()
    features = [col for col in test.columns if col != ID]
    cat_cols = test.select_dtypes(include=['object', 'category']).columns.tolist()
    x = train[features].copy()
    y = train[TARGET].astype(int).copy()
    x_test = test[features].copy()

    x['lap_tyre'] = x['LapNumber'] * x['TyreLife']
    x_test['lap_tyre'] = x_test['LapNumber'] * x_test['TyreLife']
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([x[col], x_test[col]], axis=0).astype(str))
        x[col] = le.transform(x[col].astype(str))
        x_test[col] = le.transform(x_test[col].astype(str))

    def fit_fold(fold, tr_idx, val_idx):
        model = HistGradientBoostingClassifier(**PARAMS)
        model.fit(x.iloc[tr_idx], y.iloc[tr_idx])
        print(f'fold {fold}: {model.n_iter_} iterations', flush=True)
        return model.predict_proba(x.iloc[val_idx])[:, 1], model.predict_proba(x_test)[:, 1]

    run_folds(FOLDER, FOLDER, fit_fold, args.folds, y=y.to_numpy(), clip=(1e-7, 1 - 1e-7))


if __name__ == '__main__':
    main()
