#!/usr/bin/env python
"""Per-fold logit standardization of a TabM stream.

TabM keeps its selected checkpoint (no refit), so its 5 fold models come out with different logit scales and pooling
their raw out-of-fold predictions mixes 5 calibrations. Each fold model's logits are standardized with its own
statistics (fold identity and predictions only, no labels):
  OOF rows of fold k:  sigmoid((z_k - mean(z_k)) / sd(z_k)),  z_k = logit of fold model k's validation predictions
  test:                sigmoid(mean over k of (t_k - mean(t_k)) / sd(t_k)),
                       t_k = logit of fold model k's test predictions
logit(p) = log(p / (1 - p)) with float32 p clipped to [1e-7, 1 - 1e-7]; sd is the population sd.

Usage: python standardize.py [--stream tabm_density_mlm4_banks] [--pair tabm_density_mlm4_banks_std]
  reads parts/<stream>_f{k}_{valid,test}.npy (written by train.py), writes preds/<pair>_{oof,test}.npy"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.pairs import have_part, save_pair  # noqa: E402


def logit32(p):
    p = np.clip(np.asarray(p, np.float32), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).astype(np.float64)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def standardize(stream, pair):
    if not all(have_part(HERE, stream, k) for k in range(5)):
        print(f'{stream}: not all folds are done yet', flush=True)
        return None
    y = target()
    f = folds(y)
    oof, zt = np.zeros(len(y)), 0.0
    for k in range(5):
        zv = logit32(np.load(HERE / 'parts' / f'{stream}_f{k}_valid.npy'))
        oof[f == k] = sigmoid((zv - zv.mean()) / zv.std())
        tk = logit32(np.load(HERE / 'parts' / f'{stream}_f{k}_test.npy'))
        zt = zt + (tk - tk.mean()) / tk.std() / 5
    save_pair(HERE, pair, oof, sigmoid(zt))
    print(f'{pair}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    return oof


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stream', default='tabm_density_mlm4_banks')
    ap.add_argument('--pair', default='tabm_density_mlm4_banks_std')
    a = ap.parse_args()
    standardize(a.stream, a.pair)
