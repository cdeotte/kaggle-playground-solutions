#!/usr/bin/env python
"""Public notebook "RealMLP for Predicting Smartphone Addiction" (version 1) run on our 5 folds.
https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction

realmlp.py is the notebook's code (cells 6-22); this script is its main() (cell 24) with our folds instead of its
StratifiedKFold(5, shuffle=True, random_state=42) split (the same assignment) and the predictions kept in memory instead
of CSV files.
Features (label-free, train medians): missing-value indicators of the 9 numeric predictors, numeric NaN -> train median,
the 3 categoricals and every numeric value factorized as categories, 10-bin quantile bins of daily screen time and
social media hours. Per fold, a scikit-learn TargetEncoder (cv=5, smooth 'auto', seed 42) of all categorical columns
except the bins, fitted on the fold's training rows.
Model (the notebook's): RealMLP with 8 vectorized ensemble members, 3 x 512 SiLU layers, PBLD periodic numeric
embeddings, AdamW lr 0.01, flat-cos schedule, EMA 0.997875, label smoothing, class weights, 8 epochs, batch 256.
OOF = the validation probabilities of the best epoch, test = the restored model's probabilities; the notebook seeds
once (42), so the 5 folds run in order in one process. Test = mean of the 5 fold models.
Usage: python train.py"""
import sys
from pathlib import Path

from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import DATA_DIR  # noqa: E402
from lib.runner import run_all_folds  # noqa: E402

NAME = 'zhenruiweng_realmlp'


def main():
    import realmlp as nb  # seeds on import

    X, y, X_test, test_id, cat_cols, num_cols = nb.load_and_preprocess(DATA_DIR / 'train.csv', DATA_DIR / 'test.csv')

    def fit_fold(fold, tr_idx, val_idx):
        X_tr = X.iloc[tr_idx].copy()
        X_val = X.iloc[val_idx].copy()
        X_tst = X_test.copy()
        y_tr = y.iloc[tr_idx]
        y_val = y.iloc[val_idx]
        # target encoding (bins excluded)
        te_cols = [col for col in cat_cols if not col.endswith('_bin_')]
        if te_cols:
            encoder = TargetEncoder(cv=nb.FOLDS, smooth='auto', shuffle=True, random_state=nb.SEED)
            tr_enc = encoder.fit_transform(X_tr[te_cols], y_tr)
            val_enc = encoder.transform(X_val[te_cols])
            tst_enc = encoder.transform(X_tst[te_cols])
            te_names = [f'_{col}TE' for col in te_cols]
            X_tr[te_names] = tr_enc
            X_val[te_names] = val_enc
            X_tst[te_names] = tst_enc
        model = nb.RealMLP_TD_Classifier(**nb.CONFIG)
        model.fit(X_tr, y_tr, X_val, y_val, cat_col_names=cat_cols)
        return model.best_val_probs_[:, 1], model.predict_proba(X_tst)[:, 1]

    run_all_folds(NAME, fit_fold, y=y.to_numpy())


if __name__ == '__main__':
    main()
