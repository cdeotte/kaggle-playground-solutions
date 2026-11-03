#!/usr/bin/env python
"""Public notebook "CNN for Predicting Smartphone Addiction" (version 1) run on our 5 folds.
https://www.kaggle.com/code/omidbaghchehsaraei/cnn-for-predicting-smartphone-addiction

cnn.py is the notebook (cells 1-9). Its features: the 9 numeric predictors, missing count and 25 ratio / intensity /
log / sqrt features (NaN -> 0), and per fold a frequency encoding and a smoothed target encoding (smoothing 10) of the
12 raw predictors (the training rows get 5-fold inner out-of-fold target encodings in our run; the notebook used the map
fitted on all training rows for them as well), quantile-normalized on the training rows.
Model (the notebook's): a 1-D CNN over per-feature piecewise-linear + periodic embeddings (two conv layers, squeeze-
excitation, avg / max pooling, MLP, multi-sample-dropout head), AdamW lr 0.002, cosine warm restarts, label smoothing
0.01, mixed precision, batch 128, up to 100 epochs with patience 15 on the validation AUC; the best epoch's weights
predict. The notebook seeds once (42) and runs its folds in one loop, so the 5 folds run in order in one process.
Test = mean of the 5 fold models.
Usage: python train.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from lib.runner import run_all_folds  # noqa: E402

NAME = 'omidbaghchehsaraei_cnn'


def main():
    import cnn as nb  # seeds and builds the notebook's features on import
    run_all_folds(NAME, nb.train_fold, y=nb.train_df[nb.TARGET].to_numpy())


if __name__ == '__main__':
    main()
