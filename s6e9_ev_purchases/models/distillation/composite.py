#!/usr/bin/env python
"""Fixed same-fold composite of the three student families (train_students.py, seeds 42-49).

Per outer fold k:
  g_w0, g_w1 = Gaussian-shrunk mean of the 32 member margins (8 seeds x 4 members): mean / sqrt(1 + pi var / 8)
  d = g_w0 - g_w1;  gated = g_w1 + 0.5 d / (1 + (d / 0.25)^2)
  emb = mean over seeds of logit(clip(p_multitask_w0, 1e-7))
  p = sigmoid(clip(0.9 gated + 0.1 emb, +-30))
OOF = the fold-k rows; test = mean of the 5 fold probabilities.
Usage: python composite.py   -> preds/mlp_distilled_composite_{oof,test}.npy"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.pairs import save_pair  # noqa: E402

RUNS = HERE/'runs'
SEEDS = range(42, 50)
PAIR = 'mlp_distilled_composite'


def gaussian_shrink(z):
    z = z.astype('float64')
    return z.mean(1)/np.sqrt(1+np.pi*z.var(1)/8)


def load(family, k, seed):
    return np.load(RUNS/f'{family}_s{seed}_f{k}'/'predictions.npz')


def main():
    from sklearn.metrics import roc_auc_score
    y = target()
    f = folds(y)
    oof, tests = np.full(len(y), np.nan), []
    for k in range(5):
        valid = np.flatnonzero(f == k)
        nv = len(valid)
        gauss = []
        for family in ['distill_joint_w0', 'distill_joint_w1']:
            parts = [load(family, k, s) for s in SEEDS]
            assert all(np.array_equal(p['index'][:nv], valid) for p in parts)
            gauss.append(gaussian_shrink(np.column_stack([p['members'] for p in parts])))
        d = gauss[0]-gauss[1]
        gated = gauss[1]+.5*d/(1+(d/.25)**2)
        emb = np.mean([logit(load('multitask_w0', k, s)['prediction'].astype('float64').clip(1e-7, 1-1e-7)) for s in SEEDS],
                      axis=0)
        p = expit((.9*gated+.1*emb).clip(-30, 30))
        oof[valid] = p[:nv]
        tests.append(p[nv:])
    assert np.isfinite(oof).all()
    save_pair(HERE, PAIR, oof, np.mean(tests, axis=0))
    print(f'{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)


if __name__ == '__main__':
    main()
