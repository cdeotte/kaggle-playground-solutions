#!/usr/bin/env python
"""Train the three student families of the distilled composite, one model per outer fold and seed (42-49).

Inputs (307 columns + margin): the 259-column fold matrices of the digits_diffs XGB (../income_mixture_xgb/xgb_inputs.py)
+ the 48-column joint bank (joint_bank.py); categoricals one-hot, standardized on the fit rows (sd floor 0.001,
clip +-10); the nested LR margin is an extra input and the output offset.
Student: 4 parallel MLP members 256-128-64 (GELU, dropout 0.05, zero-initialized output layer), input noise 0.03,
AdamW lr 3e-3 wd 0.01, cosine decay to 2%, gradient clip 1, batch 2048 fit rows + 2048 query rows, 8 epochs.
  distill_joint_w0   loss = BCE(true labels of the fit rows)
  distill_joint_w1   + 1.0 x BCE(query rows vs the same-fold teacher, teacher.py)
  multitask_w0       distill_joint_w0 student also reading a raw-field embedding encoder (multitask_core.py)
Writes runs/{family}_s{seed}_f{k}/predictions.npz (index = fold-k rows then test rows, prediction, margin, members).
Usage: python train_students.py --family distill_joint_w0 [--folds 0,1,2,3,4] [--seeds 42,...,49]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(2, str(HERE.parent/'income_mixture_xgb'))  # after this folder, before site-packages
from common.pairs import fold_list  # noqa: E402
from layout import STAGE1  # noqa: E402
from xgb_inputs import inputs as xgb_fold_inputs  # noqa: E402
import distill_core  # noqa: E402
import multitask_core  # noqa: E402
from joint_bank import CACHE, MASKED  # noqa: E402
from teacher import teacher  # noqa: E402

RUNS = HERE/'runs'
STUDENT = dict(epochs=8, batch=2048, lr=.003, wd=.01, widths=[256, 128, 64], members=4, noise=.03, dropout=.05)
FAMILIES = {
    'distill_joint_w0': dict(STUDENT, distill_weight=0.),
    'distill_joint_w1': dict(STUDENT, distill_weight=1.),
    'multitask_w0': dict(STUDENT, distill_weight=0., aux_embedding=16, aux_widths=[256, 128], aux_dropout=.1, aux_weight=0.),
}
SEEDS = list(range(42, 50))


def fold_inputs(k):
    X, Q, names, types, fw, fit, query, nv, mf, mq = xgb_fold_inputs(k, 'digits_diffs')
    joint = np.load(CACHE/'joint_completion_all.npy', mmap_mode='r')
    assert joint.shape[1] == 48
    X = np.column_stack([X, joint[fit]])
    Q = np.column_stack([Q, joint[query]])
    types = list(types)+['q']*48
    assert np.isfinite(X).all() and np.isfinite(Q).all()
    return X, Q, types, fit, query, nv, mf, mq


def run_fold(family, k, seeds):
    todo = [s for s in seeds if not (RUNS/f'{family}_s{s}_f{k}'/'predictions.npz').exists()]
    if not todo:
        return
    cfg = FAMILIES[family]
    X, Q, types, fit, query, nv, mf, mq = fold_inputs(k)
    fit_y = np.load(STAGE1/'y.npy')[fit].copy()
    if family.startswith('distill'):
        soft = teacher(k, query)
    else:
        bank = np.load(MASKED)
        cards = json.loads(MASKED.with_suffix('.json').read_text())['cards']
        raw, num = bank['X'], bank['numeric']
        original = [bank['original_'+str(j)] for j in range(len(cards))]
    for seed in todo:
        print(f'{family} fold {k} seed {seed}: {X.shape}', flush=True)
        extra = {}
        if family.startswith('distill'):
            members = distill_core.fit_predict(X, Q, types, fit_y, mf, mq, soft, cfg, seed)
            margin = members.mean(1)
            extra['members'] = members
        else:
            margin = multitask_core.fit_predict(X, Q, types, mf, mq, raw, num, fit, query, fit_y, original, cards, cfg, seed)
        run = RUNS/f'{family}_s{seed}_f{k}'
        run.mkdir(parents=True, exist_ok=True)
        np.savez(run/'predictions.npz', index=query, prediction=expit(margin.astype('float64')), margin=margin, **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True, choices=list(FAMILIES))
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    a = ap.parse_args()
    for k in fold_list(a.folds):
        run_fold(a.family, k, fold_list(a.seeds))


if __name__ == '__main__':
    main()
