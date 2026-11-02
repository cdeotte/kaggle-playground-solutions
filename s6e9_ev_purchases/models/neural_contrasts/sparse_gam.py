#!/usr/bin/env python
"""Sparse exact-value GAM: the other half of G in copula_gam_contrast.

Columns (all train + test rows): one-hot indicators of every value of the 13 raw fields (ridge 10 for exact income and
commute, 1 for the others), of inc1000, inc10000 and the leading income BPE piece (ridge 1) and of the (income // 1000,
piece) parent (ridge 3); cubic B-splines of income (32 uniform knots) and commute (24) with ridge 0.1, knots placed on
the predictors only; joint indicators concern x subsidy, concern x anxiety, subsidy x anxiety, piece x concern,
piece x subsidy and piece x anxiety (ridge 10); an intercept. Logistic link, fit on the outer-train rows with L-BFGS-B
(diagonal preconditioning, maxiter 4000, ftol 1e-11, gtol 2e-5).
Writes cache/contrast/additive_runs/logit_p10_cross10_converged/fold_<k>/{coef,valid,test}.npy (probabilities).
Needs lib.py of ../income_mixture_xgb.
Usage: python sparse_gam.py [--folds 0,1,2,3,4]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, ndtr
from sklearn.preprocessing import SplineTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.append(str(HERE.parent / 'income_mixture_xgb'))
from contrast_data import CONTRAST, ints  # noqa: E402
from lib import RAW, Data  # noqa: E402

NAME = 'logit_p10_cross10_converged'
LINK, PENALTY, INTERACTIONS, MAXITER = 'logit', 10., 10., 4000


def design(d):
    matrices, penalties, names = [], [], []

    def categorical(key, codes, penalty):
        codes = np.asarray(codes, np.int32)
        card = int(codes.max()) + 1
        matrices.append(sparse.csr_matrix((np.ones(d.N), codes, np.arange(d.N + 1)), shape=(d.N, card)))
        penalties.extend([penalty] * card)
        names.extend([f'{key}={j}' for j in range(card)])

    def joint(key, components, penalty):
        categorical(key, pd.factorize(pd.MultiIndex.from_arrays(components), sort=True)[0], penalty)
    for key in RAW:
        categorical(key, d.keys[key], PENALTY if key in ['Annual_Income_USD', 'Daily_Commute_km'] else 1.)
    for key in ['inc1000', 'inc10000', 'bpe', 'inc_parent']:
        categorical(key, d.keys[key], 3. if key == 'inc_parent' else 1.)
    for key, knots in [('Annual_Income_USD', 32), ('Daily_Commute_km', 24)]:
        v = d.df[key].to_numpy(np.float64)[:, None]
        spline = SplineTransformer(n_knots=knots, degree=3, include_bias=False, knots='uniform', sparse_output=True)
        block = spline.fit_transform(v).astype(np.float64)
        matrices.append(sparse.csr_matrix(block))
        penalties.extend([.1] * block.shape[1])
        names.extend([f'spline_{key}_{j}' for j in range(block.shape[1])])
    for keys in [('Environmental_Concern_Level', 'Subsidy_Available'), ('Environmental_Concern_Level', 'Range_Anxiety_Level'),
                 ('Subsidy_Available', 'Range_Anxiety_Level')]:
        joint('cross_' + '_'.join(keys), [d.keys[k] for k in keys], INTERACTIONS)
    for key in ['Environmental_Concern_Level', 'Subsidy_Available', 'Range_Anxiety_Level']:
        joint('cross_bpe_' + key, [d.keys['bpe'], d.keys[key]], INTERACTIONS)
    matrices.append(sparse.csr_matrix(np.ones((d.N, 1))))
    penalties.append(1e-6)
    names.append('intercept')
    return sparse.hstack(matrices, format='csr'), np.asarray(penalties), names


def fit(X, y, penalty, link):
    # diagonal curvature preconditioning; the ridge objective itself is unchanged
    scale = 1 / np.sqrt(.25 * np.asarray(X.power(2).sum(axis=0)).ravel() + penalty)
    A = X.multiply(scale).tocsr()
    AT = A.T.tocsr()
    ridge = penalty * scale * scale
    sign = 2 * y - 1

    def objective(theta):
        eta = A @ theta
        if link == 'logit':
            loss = np.logaddexp(0, eta) - y * eta
            residual = expit(eta) - y
        else:
            lp = log_ndtr(sign * eta)
            loss = -lp
            residual = -sign * np.exp(-.5 * eta ** 2 - .5 * np.log(2 * np.pi) - lp)
        return float(loss.sum() + .5 * np.dot(ridge, theta * theta)), AT @ residual + ridge * theta
    result = minimize(objective, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=MAXITER, ftol=1e-11, gtol=2e-5, maxcor=30))
    print(f'  L-BFGS-B: {result.message} ({result.nit} iterations)', flush=True)
    return result.x * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    d = Data()
    X, penalty, names = design(d)
    for k in ints(args.folds):
        out = CONTRAST / 'additive_runs' / NAME / f'fold_{k}'
        if (out / 'test.npy').exists():
            continue
        start = time.time()
        a, b = np.flatnonzero(d.folds != k), np.flatnonzero(d.folds == k)
        q = np.r_[b, d.test]
        coef = fit(X[a], d.y[a].astype(np.float64), penalty, LINK)
        eta = X[q] @ coef
        p = expit(eta) if LINK == 'logit' else ndtr(eta)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / 'coef.npy', coef)
        np.save(out / 'valid.npy', p[:len(b)].astype(np.float64))
        np.save(out / 'test.npy', p[len(b):].astype(np.float64))
        print(f'GAM fold {k}: {X.shape} ({time.time() - start:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
