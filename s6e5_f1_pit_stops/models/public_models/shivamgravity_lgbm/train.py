#!/usr/bin/env python
"""Public notebook "pgs-s6e5-lgbm-optuna" (version 15) run on our 5 folds.
https://www.kaggle.com/code/shivamgravity/pgs-s6e5-lgbm-optuna

Features (the notebook's): the 14 competition columns; Driver, Compound and Race as pandas categoricals (LightGBM native
categorical handling).
Model: LightGBM with the best parameters of the notebook's 20-trial Optuna search (trial 15): lr 0.0188, 177 leaves,
min_child_samples 60, feature_fraction 0.636, bagging 0.896 every 3 rounds, alpha 0.000149, lambda 0.00457, balanced
class weights, up to 1,000 rounds with early stopping 50 on the validation AUC, seed 42; CPU, 32 threads.
Test = mean of the 5 fold models (best iteration); OOF and test are clipped to [1e-7, 1 - 1e-7].
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

import lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, read_test, read_train  # noqa: E402
from lib.runner import fold_args, run_folds  # noqa: E402

FOLDER = 'shivamgravity_lgbm'
THREADS = 32
# best parameters of the notebook's Optuna search (trial 15)
PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.018788634563423806,
    'num_leaves': 177,
    'min_child_samples': 60,
    'feature_fraction': 0.6363905714987185,
    'bagging_fraction': 0.896461276471979,
    'bagging_freq': 3,
    'reg_alpha': 0.00014931734212739601,
    'reg_lambda': 0.00456942005988866,
    'class_weight': 'balanced',
    'n_estimators': 1000,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': THREADS,
    'device': 'cpu',
}


def main():
    args = fold_args()
    train, test = read_train(), read_test()
    for col in train.columns:
        if train[col].dtype == 'object':
            train[col] = train[col].astype('category')
            if col in test.columns:
                test[col] = test[col].astype('category')
    train[TARGET] = train[TARGET].astype(int)
    x = train.drop(columns=[ID, TARGET])
    y = train[TARGET]
    x_test = test.drop(columns=[ID])

    def fit_fold(fold, tr_idx, val_idx):
        x_val, y_val = x.iloc[val_idx], y.iloc[val_idx]
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(x.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(x_val, y_val)], eval_metric='auc',
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        print(f'fold {fold}: best iteration {model.best_iteration_}', flush=True)
        return (model.predict_proba(x_val, num_iteration=model.best_iteration_)[:, 1],
                model.predict_proba(x_test, num_iteration=model.best_iteration_)[:, 1])

    run_folds(FOLDER, FOLDER, fit_fold, args.folds, y=y.to_numpy(), clip=(1e-7, 1 - 1e-7))


if __name__ == '__main__':
    main()
