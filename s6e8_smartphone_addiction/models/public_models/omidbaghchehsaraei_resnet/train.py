#!/usr/bin/env python
"""Public notebook "ResNet for Predicting Smartphone Addiction" (version 1) run on our 5 folds.
https://www.kaggle.com/code/omidbaghchehsaraei/resnet-for-predicting-smartphone-addiction

Features (lib/ratio_te.py): the 12 raw predictors as string categories, missing_count and 9 screen-time ratio features
(NaN -> 0), and per fold a smoothed target encoding (smoothing 10) and a frequency encoding of every raw predictor. The
notebook computed the encodings once on all training rows before its CV loop; here they are computed inside each fold
(inner 5-fold out-of-fold for the training rows, the fold's training-row map for the validation and test rows).
Model (the notebook's): pytabkit Resnet_RTDL_D_Classifier, d 128, 2 layers, hidden factor 2, embedding 8, residual
dropout 0.1, lr 1e-3, batch 128, seed 42, fitted with the fold's validation rows. Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from lib.ratio_te import run_pytabkit  # noqa: E402
from lib.runner import fold_args  # noqa: E402

NAME = 'omidbaghchehsaraei_resnet'


def make_model():
    from pytabkit import Resnet_RTDL_D_Classifier
    return Resnet_RTDL_D_Classifier(
        random_state=42, device='cuda', module_d=128, module_n_layers=2, module_d_hidden_factor=2, module_d_embedding=8,
        batch_size=128, lr=0.001, module_residual_dropout=0.1)


if __name__ == '__main__':
    run_pytabkit(NAME, make_model, fold_args())
