#!/usr/bin/env python
"""Public notebook "Multi-Level Deep Univariate Spline Transformer" (version 5) run on our 5 folds.
https://www.kaggle.com/code/ern711/multi-level-deep-univariate-spline-transformer

spline_transformer.py is the notebook's code before its cross-validation loop. This script is that loop on our folds
(the notebook split with StratifiedKFold(5, shuffle=True, random_state=21)): for every fold, train_one_fold(fold, ...)
reseeds (21 + fold), builds the fold's nested target / frequency encodings (inner 5-fold out-of-fold for the training
rows), scaler and spline knots on the training rows, trains the stage-1 spline transformer (four heads + logit mixer,
early stopping on the validation AUC) and the stage-2 hypernetwork corrections, and returns the FINAL head's
predictions for the validation and test rows. Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]   (each fold takes several hours on one GPU)"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from lib.runner import fold_args, run_folds  # noqa: E402

NAME = 'ern711_multilevel_spline_transformer'


def main():
    fold_ids = fold_args()
    import spline_transformer as st  # seeds and builds the notebook's features on import

    def fit_fold(fold, train_index, valid_index):
        result = st.train_one_fold(fold, train_index, valid_index)
        print(f'fold {fold}: best epoch {result["best_epoch"]}, FINAL AUC {result["best_auc"]:.6f}, '
              f'ADD AUC {result["best_add_auc"]:.6f}', flush=True)
        return result['val_final'], result['test_final']

    run_folds(NAME, fit_fold, fold_ids, y=st.train_raw[st.TARGET].to_numpy())


if __name__ == '__main__':
    main()
