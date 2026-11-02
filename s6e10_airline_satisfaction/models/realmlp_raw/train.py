#!/usr/bin/env python
"""RealMLP (pytabkit RealMLP_TD_Classifier) on the 21 raw features only, averaged over 8 seeds.

Inputs: 17 numeric columns (Arrival Delay NaN -> outer-train median + one missing indicator) and the 4 categoricals as
categorical columns (one-hot inside RealMLP). No engineered features.
Parameters: the S6E8 1st-place RealMLP set with 20 epochs and plr_sigma 32. plr_sigma (the scale of the periodic numeric
embeddings) is the decisive knob: a large sigma resolves fine differences in Flight Distance (a route id), the same effect as
XGBoost's max_bin 4096.
Each fold trains on 95% of outer-train; a stratified 5% inner holdout of outer-train picks the best epoch (weights restored).
Test = mean of the 5 fold models; the pair is the mean of the 8 seeds.
Usage: python train.py [--seeds 0,1,2,3,4,5,6,7]"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, raw_arrays, target  # noqa: E402
from common.pairs import save_pair  # noqa: E402

PARAMS = {'n_ens': 8, 'n_epochs': 20, 'batch_size': 256, 'lr': 0.0175, 'lr_sched': 'lin_cos_log_15', 'num_emb_type': 'pblrd',
          'plr_sigma': 32.0, 'plr_hidden_1': 16, 'plr_hidden_2': 8, 'plr_lr_factor': 0.08, 'plr_act_name': 'relu',
          'hidden_sizes': [768, 384, 192], 'act': 'mish', 'p_drop': 0.05, 'wd': 0.017, 'ls_eps': 0.005, 'add_front_scale': True,
          'scale_lr_factor': 6.0, 'first_layer_lr_factor': 0.125, 'embedding_size': 6, 'max_one_hot_cat_size': 18,
          'val_metric_name': '1-auc_ovr', 'use_early_stopping': False, 'verbosity': 0}
HOLDOUT = 0.05


def run_fold(X, Xt, y, f, k, seed_idx, meta):
    import torch
    from pytabkit import RealMLP_TD_Classifier
    from sklearn.model_selection import train_test_split
    feats, cat_idx = meta['features'], meta['cat_idx']
    tr, va = np.where(f != k)[0], np.where(f == k)[0]
    num_idx = [j for j in range(X.shape[1]) if j not in cat_idx]
    med = np.nanmedian(X[tr][:, num_idx], axis=0)
    nan_cols = [j for j in num_idx if np.isnan(X[:, j]).any() or np.isnan(Xt[:, j]).any()]

    def frame(M):
        df = pd.DataFrame({feats[j]: np.where(np.isnan(M[:, j]), med[i], M[:, j]).astype(np.float32) for i, j in enumerate(num_idx)})
        for j in nan_cols:
            df[f'{feats[j]} is missing'] = np.isnan(M[:, j]).astype(np.float32)
        for j in cat_idx:
            df[feats[j]] = pd.Categorical(M[:, j].astype(np.int64), categories=range(len(meta['cat_codes'][feats[j]])))
        return df
    Xtr, Xva, Xte = frame(X[tr]), frame(X[va]), frame(Xt)
    fit_i, ho_i = train_test_split(np.arange(len(tr)), test_size=HOLDOUT, stratify=y[tr], random_state=7000 + k + 100 * seed_idx)
    seed = 42 + k + 1000 * seed_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    m = RealMLP_TD_Classifier(**PARAMS, random_state=seed, device='cuda', n_threads=6)
    m.fit(Xtr.iloc[fit_i].reset_index(drop=True), y[tr][fit_i], Xtr.iloc[ho_i].reset_index(drop=True), y[tr][ho_i])
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7')
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    seeds = [int(s) for s in a.seeds.split(',')]
    X, Xt, meta = raw_arrays()
    y = target()
    f = folds(y)
    oof, test = np.zeros(len(y)), np.zeros(len(Xt))
    for s in seeds:
        o = np.zeros(len(y))
        for k in range(5):
            pv, pt = run_fold(X, Xt, y, f, k, s, meta)
            o[f == k] = pv
            test += pt / (5 * len(seeds))
        print(f'seed {s}: OOF AUC {roc_auc_score(y, o):.6f}', flush=True)
        oof += o / len(seeds)
    print(f'{len(seeds)}-seed OOF AUC {roc_auc_score(y, oof):.6f}')
    save_pair(HERE, 'realmlp_raw', oof, test)


if __name__ == '__main__':
    main()
