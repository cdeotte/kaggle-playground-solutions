#!/usr/bin/env python
"""Public notebook "TabTransformer : Predicting Smartphone Addiction" (version 1) run on our 5 folds.
https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction

tabtransformer.py is the notebook (cells 1-9). Its features: the 9 numeric predictors, missing count and 25 ratio /
intensity / log / sqrt features (NaN -> 0), and per fold a frequency encoding and a smoothed target encoding
(smoothing 10) of the 12 raw predictors (the training rows get 5-fold inner out-of-fold target encodings in our run;
the notebook used the map fitted on all training rows for them as well), quantile-normalized on the training rows.
Model (the notebook's): per-feature piecewise-linear + periodic embeddings projected to 64-d tokens with column
embeddings, 3 transformer blocks (4 heads), MLP, multi-sample-dropout head; mixup (alpha 0.2), label smoothing 0.005,
AdamW lr 1e-3, cosine warm restarts, mixed precision, EMA 0.999 of the weights, batch 256, up to 120 epochs with
patience 18 on the EMA model's validation AUC; the best EMA weights predict. The notebook seeds once (42) and runs its
folds in one loop, so the 5 folds run in order in one process. Test = mean of the 5 fold models.
Usage: python train.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from lib.runner import run_all_folds  # noqa: E402

NAME = 'omidbaghchehsaraei_tabtransformer'


def main():
    import tabtransformer as nb  # seeds and builds the notebook's features on import
    run_all_folds(NAME, nb.train_fold, y=nb.train_df[nb.TARGET].to_numpy())


if __name__ == '__main__':
    main()
