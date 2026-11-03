#!/usr/bin/env python
"""Public notebook "TabM for Predicting Smartphone Addiction" (version 1) run on our 5 folds.
https://www.kaggle.com/code/omidbaghchehsaraei/tabm-for-predicting-smartphone-addiction

Features (lib/ratio_te.py): the 12 raw predictors as string categories, missing_count and 9 screen-time ratio features
(NaN -> 0), and per fold a smoothed target encoding (smoothing 10) and a frequency encoding of every raw predictor. The
notebook computed the encodings once on all training rows before its CV loop; here they are computed inside each fold
(inner 5-fold out-of-fold for the training rows, the fold's training-row map for the validation and test rows).
Model (the notebook's): pytabkit TabM_D_Classifier, tabm-mini-normal with k = 24, piecewise-linear embeddings (16),
6 blocks of 256, dropout 0.11, lr 1e-3, weight decay 1e-2, batch 128, up to 100 epochs with patience 5 on the validation
rows, seed 42. Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from lib.ratio_te import run_pytabkit  # noqa: E402
from lib.runner import fold_args  # noqa: E402

NAME = 'omidbaghchehsaraei_tabm'


def make_model():
    from pytabkit import TabM_D_Classifier
    return TabM_D_Classifier(
        random_state=42, device='cuda', patience=5, arch_type='tabm-mini-normal', tabm_k=24, num_emb_type='pwl',
        d_embedding=16, batch_size=128, lr=1e-3, n_epochs=100, dropout=0.11, d_block=256, n_blocks=6,
        weight_decay=1e-2, verbosity=2)


if __name__ == '__main__':
    run_pytabkit(NAME, make_model, fold_args())
