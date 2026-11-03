#!/usr/bin/env python
"""Public notebook "RealMLP for Predicting Smartphone Addiction" (version 2) run on our 5 folds.
https://www.kaggle.com/code/omidbaghchehsaraei/realmlp-for-predicting-smartphone-addiction

Features (lib/ratio_te.py): the 12 raw predictors as string categories, missing_count and 9 screen-time ratio features
(NaN -> 0), and per fold a smoothed target encoding (smoothing 10) and a frequency encoding of every raw predictor. The
notebook computed the encodings once on all training rows before its CV loop; here they are computed inside each fold
(inner 5-fold out-of-fold for the training rows, the fold's training-row map for the validation and test rows).
Model (the notebook's tuned parameters): pytabkit RealMLP_TD_Classifier, 8 ensemble members, 4 x 384 mish layers,
lr 0.04, up to 100 epochs, batch 128, early stopping on validation 1-AUC (additive patience 20), seed 42.
Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from lib.ratio_te import run_pytabkit  # noqa: E402
from lib.runner import fold_args  # noqa: E402

NAME = 'omidbaghchehsaraei_realmlp'


def make_model():
    from pytabkit import RealMLP_TD_Classifier
    return RealMLP_TD_Classifier(
        device='cuda', random_state=42, verbosity=2, n_epochs=100, batch_size=128, n_ens=8, val_metric_name='1-auc_ovr',
        use_early_stopping=True, early_stopping_additive_patience=20, early_stopping_multiplicative_patience=1,
        act='mish', embedding_size=8, first_layer_lr_factor=0.5962121993798933, hidden_sizes='rectangular',
        hidden_width=384, lr=0.04, ls_eps=0.011498317194338772, ls_eps_sched='coslog4', max_one_hot_cat_size=18,
        n_hidden_layers=4, p_drop=0.07301419697186451, p_drop_sched='flat_cos', plr_hidden_1=16, plr_hidden_2=8,
        plr_lr_factor=0.1151437622270563, plr_sigma=2.3316811282666916, scale_lr_factor=2.244801835541429,
        sq_mom=1.0 - 0.011834054955582318, wd=0.02369230879235962)


if __name__ == '__main__':
    run_pytabkit(NAME, make_model, fold_args())
