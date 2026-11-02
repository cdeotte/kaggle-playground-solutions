#!/usr/bin/env python
"""Label-free features from the per-order likelihoods of score_orders.py (all slices of one run joined).

  llr_lse              order-marginal LLR = logsumexp_o LL_yes - logsumexp_o LL_no
  llr_mean, llr_sd     mean and sd over orders of the per-order LLR
  ll_no_lse, ll_yes_lse  order-marginal log-likelihood of the row under each label
  ent_no, ent_yes      entropy of the order posterior under each label
  fllr_<field>         per-field LLR (field attribution averaged over orders), 13 fields
  fll_age, fll_inc, fll_com  label-averaged per-field log-likelihood of the 3 numeric fields Age, Annual_Income_USD, Daily_Commute_km
Writes features/<tag>.npy, float32 [train + test rows, 23].
Usage: python build_order_features.py [e55r12]"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
FIELDS = ['age', 'gender', 'inc', 'city', 'com', 'ncars', 'car', 'sth', 'stw', 'home', 'env', 'sub', 'anx']


def load_all(tag):
    parts = sorted((HERE / 'features' / 'scores').glob(f'{tag}__*-*.npz'), key=lambda p: int(p.stem.split('__')[1].split('-')[0]))
    chunks = [np.load(p) for p in parts]
    n = max(int(z['rows'].max()) for z in chunks) + 1
    k = chunks[0]['ll'].shape[1]
    ll = np.full((n, k, 2), np.nan, np.float32)
    fld = np.full((n, 2, 13), np.nan, np.float32)
    seen = np.zeros(n, bool)
    for z in chunks:
        ll[z['rows']] = z['ll']
        fld[z['rows']] = z['fld']
        seen[z['rows']] = True
    assert seen.all(), f'{tag}: {int((~seen).sum())} rows missing'
    return ll.astype(np.float64), fld.astype(np.float64)


def features(ll, fld):
    llr = ll[..., 1] - ll[..., 0]
    lse_no, lse_yes = logsumexp(ll[..., 0], 1), logsumexp(ll[..., 1], 1)

    def ent(x, lse):
        p = np.exp(x - lse[:, None])
        return -(p * np.log(np.clip(p, 1e-300, None))).sum(1)
    cols = [lse_yes - lse_no, llr.mean(1), llr.std(1), lse_no, lse_yes, ent(ll[..., 0], lse_no), ent(ll[..., 1], lse_yes)]
    cols += [fld[:, 1, j] - fld[:, 0, j] for j in range(13)]
    cols += [0.5 * (fld[:, 0, j] + fld[:, 1, j]) for j, f in enumerate(FIELDS) if f in ('age', 'inc', 'com')]
    return np.column_stack(cols).astype(np.float32)


def main():
    for tag in sys.argv[1:] or ['e55r12']:
        x = features(*load_all(tag))
        assert np.isfinite(x).all()
        np.save(HERE / 'features' / f'{tag}.npy', x)
        print(tag, x.shape, flush=True)


if __name__ == '__main__':
    main()
