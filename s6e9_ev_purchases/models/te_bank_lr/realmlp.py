#!/usr/bin/env python
"""RealMLP (pytabkit RealMLP_TD_Classifier) on the te_bank_lr inputs: banks S,P,I,T,P5,I5,T5,P1,I1,T1 + packs num, digits,
footprint, replica + 168 acceptance columns + income EM blocks (889 numeric columns, no hinges).

Parameters: pytabkit RealMLP-TD defaults + the S6E8 1st-place settings (lr 0.0175, [768, 384, 192] mish, PBLRD numeric
embeddings with sigma 0.94, wd 0.017, label smoothing 0.005, front scale, first-layer lr factor 0.125, batch 256), 32
ensemble members, 4 epochs. Each fold trains on 90% of the outer-train rows; a stratified 10% inner holdout picks the best
epoch (1 - AUC) and pytabkit keeps that epoch's weights.
Seeds 0, 1, 2 (torch / numpy seed 42 + fold + 1000 * seed; holdout split seed 7000 + fold + 100 * seed). The pair: per fold
the logit mean of the 3 seeds; test = mean of the 5 fold vectors.
Usage: python realmlp.py [--seeds 0,1,2] [--folds 0,1,2,3,4]   (finished parts are skipped; the pair is written once all
       15 parts exist)"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import folds, target  # noqa: E402
from common.pairs import fold_list, have_part, save_pair, save_part  # noqa: E402
from design import fold_design  # noqa: E402

NAME = 'te_bank_realmlp'
VIEW = dict(banks=['S', 'P', 'I', 'T', 'P5', 'I5', 'T5', 'P1', 'I1', 'T1'], packs=['num', 'digits', 'footprint', 'replica'], acc=True,
            blocks=['income_em', 'income_info'])
PARAMS = {'batch_size': 256, 'lr_sched': 'lin_cos_log_15', 'num_emb_type': 'pblrd', 'plr_sigma': 0.94, 'plr_hidden_1': 16, 'plr_hidden_2': 8,
          'plr_lr_factor': 0.08, 'plr_act_name': 'relu', 'hidden_sizes': [768, 384, 192], 'act': 'mish', 'p_drop': 0.05, 'wd': 0.017,
          'ls_eps': 0.005, 'add_front_scale': True, 'scale_lr_factor': 6.0, 'first_layer_lr_factor': 0.125, 'embedding_size': 6,
          'max_one_hot_cat_size': 18, 'lr': 0.0175, 'n_ens': 32, 'n_epochs': 4,
          'val_metric_name': '1-auc_ovr', 'use_early_stopping': False, 'verbosity': 2}
HOLDOUT = 0.10


def part_name(s):
    return f'{NAME}_s{s}'


def run_fold(k, f, y, s):
    import torch
    from pytabkit import RealMLP_TD_Classifier
    from sklearn.model_selection import train_test_split
    tr = np.flatnonzero(f != k)
    X, _ = fold_design(k, f, **VIEW)
    Xtr, Xva, Xte = (pd.DataFrame(X[r], columns=[f'n{j}' for j in range(X[r].shape[1])]) for r in ('train', 'valid', 'test'))
    del X
    ytr = y[tr].astype(np.int64)
    seed = 42 + k + 1000 * s
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    fit, ho = train_test_split(np.arange(len(tr)), test_size=HOLDOUT, stratify=ytr, random_state=7000 + k + 100 * s)
    m = RealMLP_TD_Classifier(**PARAMS, random_state=seed, device='cuda', n_threads=6)
    m.fit(Xtr.iloc[fit].reset_index(drop=True), ytr[fit], Xtr.iloc[ho].reset_index(drop=True), ytr[ho])
    pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    save_part(HERE, part_name(s), k, pv, pt)
    print(f'seed {s} fold {k}: AUC {roc_auc_score(y[f == k], pv):.6f}', flush=True)


def assemble(f, y, seeds):
    if not all(have_part(HERE, part_name(s), k) for s in seeds for k in range(5)):
        print('not all parts are done yet')
        return

    def bag(ps):  # logit mean over the seeds
        return 1 / (1 + np.exp(-np.mean([logit(np.clip(p, 1e-7, 1 - 1e-7)) for p in ps], 0)))
    oof, tests = np.zeros(len(y)), []
    for k in range(5):
        load = [[np.load(HERE / 'parts' / f'{part_name(s)}_f{k}_{r}.npy').astype(np.float64) for s in seeds] for r in ('valid', 'test')]
        oof[f == k] = bag(load[0])
        tests.append(bag(load[1]))
    save_pair(HERE, NAME, oof, np.mean(tests, 0))
    print(f'{NAME}: OOF AUC {roc_auc_score(y, oof):.6f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    y = target()
    f = folds(y)
    seeds = fold_list(a.seeds)
    for s in seeds:
        for k in fold_list(a.folds):
            if not have_part(HERE, part_name(s), k):
                run_fold(k, f, y, s)
    assemble(f, y, seeds)


if __name__ == '__main__':
    main()
