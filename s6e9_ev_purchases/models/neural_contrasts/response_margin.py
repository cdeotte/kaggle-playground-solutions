#!/usr/bin/env python
"""Doubly nested response GLM: the offset R of the copula XGB and of the GPT-2 residual (logit link), and the probit half
of G in copula_gam_contrast.

Per outer fold k: pool 0 fits on the outer-train rows and scores query = [fold k rows, test rows]; pools 1-5 fit on the inner
folds of the outer-train rows (StratifiedKFold(5, shuffle=True, random_state=42)) and score their held-out inner rows, so the
train-row margins are out-of-fold as well. Design, built inside each fit pool: nested target encodings (alpha 1 and 20 and
their difference) of the 13 raw fields, inc100, inc1000, the leading income BPE piece and the (income // 1000, piece) parent;
the raw one-hot predictors; the source-rule score / probability / logit. Ridge GLM (C = 0.1, intercept free) on columns
standardized with the fit rows (clip +-15), L-BFGS-B in float64 on the GPU (maxiter 1200, ftol 1e-12, gtol 1e-7).
Probit margins are stored as logits of the probit probability.
Writes cache/contrast/margins/<name>/fold_<k>/{train,query}.npy. Needs lib.py and compact.py of ../income_mixture_xgb.
Usage: python response_margin.py --link logit --name response_logit
       python response_margin.py --link probit --name response_probit"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.append(str(HERE.parent / 'income_mixture_xgb'))
from contrast_data import CONTRAST, ints, z  # noqa: E402
from compact import source_rule  # noqa: E402
from lib import RAW, Data, splits  # noqa: E402


def design(d, fit, query):
    keys = RAW + ['inc100', 'inc1000', 'bpe', 'inc_parent']
    T, Q, names = d.response(fit, query, keys)
    static = np.column_stack([d.raw, source_rule(d)])
    snames = ['raw__' + n for n in d.raw_names] + ['rule_score', 'rule_probability', 'rule_logit']
    return np.column_stack([T, static[fit]]), np.column_stack([Q, static[query]]), names + snames


def fit_glm(X, y, link, C=.1):
    import torch
    from scipy.optimize import minimize
    mean = X.mean(0, dtype=np.float64)
    sd = np.maximum(X.std(0, dtype=np.float64), 1e-5)
    T = np.clip((X - mean) / sd, -15, 15)
    T = np.column_stack([T, np.ones(len(T))])
    A = torch.as_tensor(T, dtype=torch.float64, device='cuda')
    target = torch.as_tensor(y, dtype=torch.float64, device='cuda')
    n = len(y)
    ridge = torch.full((T.shape[1],), 1 / C, dtype=torch.float64, device='cuda')
    ridge[-1] = 0

    def objective(theta):
        coef = torch.as_tensor(theta, dtype=torch.float64, device='cuda')
        eta = A @ coef
        if link == 'logit':
            loss = torch.nn.functional.softplus(eta) - target * eta
            residual = torch.sigmoid(eta) - target
        else:
            signed = (2 * target - 1) * eta
            lp = torch.special.log_ndtr(signed)
            loss = -lp
            residual = -(2 * target - 1) * torch.exp(-.5 * eta ** 2 - .5 * np.log(2 * np.pi) - lp)
        value = (loss.sum() + .5 * (ridge * coef ** 2).sum()) / n
        grad = (A.T @ residual + ridge * coef) / n
        return float(value.item()), grad.cpu().numpy()
    r = minimize(objective, np.zeros(T.shape[1]), jac=True, method='L-BFGS-B',
                 options=dict(maxiter=1200, ftol=1e-12, gtol=1e-7, maxcor=30))
    assert r.success, r.message
    return dict(mean=mean, sd=sd, coef=r.x, link=link)


def predict_glm(X, state):
    from scipy.special import ndtr
    T = np.clip((X - state['mean']) / state['sd'], -15, 15)
    eta = T @ state['coef'][:-1] + state['coef'][-1]
    return eta if state['link'] == 'logit' else z(ndtr(eta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--link', choices=['logit', 'probit'], required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    d = Data()
    for k in ints(args.folds):
        root = CONTRAST / 'margins' / args.name / f'fold_{k}'
        if (root / 'query.npy').exists():
            continue
        start = time.time()
        a, b = np.flatnonzero(d.folds != k), np.flatnonzero(d.folds == k)
        q = np.r_[b, d.test]
        local = np.full(d.n, -1, np.int64)
        local[a] = np.arange(len(a))
        train, query, owner = np.empty(len(a)), None, np.zeros(len(a), np.int8)
        for pool, (fit, apply) in enumerate([(a, q)] + splits(a, d.y)):
            X, Q, names = design(d, fit, apply)
            p = predict_glm(Q, fit_glm(X, d.y[fit], args.link))
            if pool == 0:
                query = p
            else:
                train[local[apply]] = p
                owner[local[apply]] += 1
        assert np.all(owner == 1) and np.isfinite(train).all() and np.isfinite(query).all()
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / 'train.npy', train)
        np.save(root / 'query.npy', query)
        print(f'{args.name} fold {k} ({time.time() - start:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
