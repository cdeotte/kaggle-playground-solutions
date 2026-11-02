"""Stage-1 feature builder: fold-local target encodings and source-conditional corrections.

Builder.build(fit, query) returns the base features of cache/stage1/base.parquet plus target-encoding columns: for
the query rows they are fitted on all `fit` rows; for the fit rows they are cross-fitted (StratifiedKFold(5, seed 42)
inside `fit`) and get Gaussian noise (sd 0.04). Encodings: smoothed log-odds (alpha 1 and 20) of each raw field and of
income / commute bins and the leading income BPE piece; with score='intercept', ridge logistic intercepts per key on
top of the known source-law offset (lambda 1 and 10)."""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold

from layout import STAGE1 as CACHE

BASE_KEYS = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
             'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level',
             'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level',
             'inc10', 'inc100', 'inc1000', 'inc5000', 'com1', 'com5', 'com10', 'inc_bpe0']
SCORE_KEYS = ['Annual_Income_USD', 'Daily_Commute_km', 'Age', 'inc100', 'inc1000', 'inc_bpe0']
CONT_KEYS = ['inc_bpe01', 'inc_bpe1', 'inc_bpelast', 'inc_bpepair', 'com_bpe0', 'com_bpe01']


def split_indices(k):
    """Outer fold k: fit = outer-train rows, valid = fold-k rows, query = fold-k rows then all test rows."""
    folds = np.load(CACHE/'fold_id.npy')
    n = len(folds)
    nt = json.loads((CACHE/'manifest.json').read_text())['n_test']
    fit, valid = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
    return fit, valid, np.r_[valid, np.arange(n, n+nt)]


class Builder:
    def __init__(self, y_override=None):
        self.meta = json.loads((CACHE/'manifest.json').read_text())
        self.base = pd.read_parquet(CACHE/'base.parquet')
        self.keys = pd.read_parquet(CACHE/'keys.parquet')
        self.y = np.load(CACHE/'y.npy') if y_override is None else y_override
        self.offset = np.load(CACHE/'source_base.npy').astype('float64')
        self.t = np.clip(self.offset, -6, 6)/3
        self.n = len(self.y)

    def response(self, fit, query, continuation=False, score='none'):
        y = self.y[fit].astype(float)
        pi = y.mean()
        res = y-expit(self.offset[fit])
        names, values = [], []
        for c in BASE_KEYS+(CONT_KEYS if continuation else []):
            key = self.keys[c].to_numpy()
            a, b, sz = key[fit], key[query], self.meta['key_sizes'][c]
            count = np.bincount(a, minlength=sz)
            pos = np.bincount(a, weights=y, minlength=sz)
            qs = []
            for alpha in (1., 20.):
                q = (pos+alpha*pi)/(count+alpha)
                v = logit(q.clip(1e-5, 1-1e-5))[b]-logit(pi)
                names.append(f'te_{c}_{alpha:g}')
                values.append(v)
                qs.append(v)
            if c in ['Annual_Income_USD', 'inc10', 'inc100', 'inc1000', 'inc_bpe0']:
                names.append('te_diff_'+c)
                values.append(qs[0]-qs[1])
        if score != 'none':
            if score == 'joint':
                out, nm = self.joint_response(fit, query)
                return np.column_stack(values+[out[:, j] for j in range(out.shape[1])]).astype('float32'), names+nm
            sk = SCORE_KEYS+(CONT_KEYS[:4] if continuation else [])
            for c in sk:
                key = self.keys[c].to_numpy()
                a, b, sz = key[fit], key[query], self.meta['key_sizes'][c]
                if score == 'residual':
                    count = np.bincount(a, minlength=sz)
                    rr = np.bincount(a, weights=res, minlength=sz)
                    for alpha in (1., 20.):
                        names.append(f'resid_{c}_{alpha:g}')
                        values.append((rr/(count+alpha))[b])
                    continue
                for lam in (1., 10.):
                    theta = np.zeros((sz, 2))
                    t = self.t[fit]
                    varying = score == 'slope'
                    for _ in range(7):
                        p = expit(self.offset[fit]+theta[a, 0]+(theta[a, 1]*t if varying else 0))
                        r = y-p
                        w = p*(1-p)
                        g0 = np.bincount(a, weights=r, minlength=sz)-lam*theta[:, 0]
                        h00 = np.bincount(a, weights=w, minlength=sz)+lam
                        if varying:
                            g1 = np.bincount(a, weights=r*t, minlength=sz)-4*lam*theta[:, 1]
                            h01 = np.bincount(a, weights=w*t, minlength=sz)
                            h11 = np.bincount(a, weights=w*t*t, minlength=sz)+4*lam
                            det = h00*h11-h01*h01
                            d0 = (h11*g0-h01*g1)/det
                            d1 = (h00*g1-h01*g0)/det
                            theta[:, 1] += np.clip(d1, -1, 1)
                        else:
                            d0 = g0/h00
                        theta[:, 0] += np.clip(d0, -2, 2)
                    names.append(f'cond_intercept_{c}_{lam:g}')
                    values.append(theta[b, 0])
                    if varying:
                        names.extend([f'cond_slope_{c}_{lam:g}', f'cond_eval_{c}_{lam:g}'])
                        values.extend([theta[b, 1], theta[b, 0]+theta[b, 1]*self.t[query]])
        return np.column_stack(values).astype('float32'), names

    def joint_response(self, fit, query):
        """Ridge logistic additive backfitting anchored on the known source law."""
        fields = BASE_KEYS[:13]+['inc_bpe0', 'inc1000']
        yp = self.y[fit].astype(float)
        b = self.offset[fit]
        bq = self.offset[query]
        keys = [self.keys[c].to_numpy()[fit] for c in fields]
        qkeys = [self.keys[c].to_numpy()[query] for c in fields]
        outs, names = [], []
        for penalty in [1., 10.]:
            coeff = [np.zeros(self.meta['key_sizes'][c]) for c in fields]
            eta = b.copy()
            beta = np.array([0., 1.])
            D = np.column_stack([np.ones(len(fit)), b])
            for sweep in range(16):
                prob = expit(eta)
                w = prob*(1-prob)
                H = D.T@(D*w[:, None])+np.diag([1., 10.])
                g = D.T@(yp-prob)-np.array([beta[0], 10*(beta[1]-1)])
                step = np.linalg.solve(H, g).clip(-.5, .5)
                beta += step
                eta += D@step
                for j, c in enumerate(fields):
                    reg = penalty if c in ['Annual_Income_USD', 'inc_bpe0'] else 10.
                    key = keys[j]
                    sz = len(coeff[j])
                    prob = expit(eta)
                    w = prob*(1-prob)
                    g = np.bincount(key, weights=yp-prob, minlength=sz)-reg*coeff[j]
                    h = np.bincount(key, weights=w, minlength=sz)+reg
                    step = np.clip(g/h, -1., 1.)
                    coeff[j] += step
                    eta += step[key]
            total = beta[0]+beta[1]*bq
            for j, c in enumerate(fields):
                v = coeff[j][qkeys[j]]
                total += v
                if c in ['Annual_Income_USD', 'Daily_Commute_km', 'Age', 'inc_bpe0', 'inc1000']:
                    outs.append(v)
                    names.append(f'joint_{c}_{penalty:g}')
            outs.extend([total, total-bq])
            names.extend([f'joint_logit_{penalty:g}', f'joint_delta_{penalty:g}'])
        return np.column_stack(outs).astype('float32'), names

    def build(self, fit, query, continuation=False, score='none', noise=.04, noise_mode='legacy'):
        fit = np.asarray(fit)
        query = np.asarray(query)
        assert not np.intersect1d(fit, query).size
        a = None
        for trainloc, valloc in StratifiedKFold(5, shuffle=True, random_state=42).split(fit, self.y[fit]):
            x, names = self.response(fit[trainloc], fit[valloc], continuation, score)
            if a is None:
                a = np.empty((len(fit), x.shape[1]), dtype='float32')
            a[valloc] = x
        b, names = self.response(fit, query, continuation, score)
        if noise:
            if noise_mode == 'legacy':
                rng = np.random.default_rng(42)
                a += rng.normal(0, noise, size=a.shape).astype('float32')
            else:
                # one noise stream per column, seeded by the column name
                for j, name in enumerate(names):
                    seed = int.from_bytes(hashlib.sha256(('42:'+name).encode()).digest()[:8], 'little')
                    a[:, j] += np.random.default_rng(seed).normal(0, noise, size=len(a)).astype('float32')
        X = np.column_stack([self.base.iloc[fit].to_numpy(), a]).astype('float32')
        V = np.column_stack([self.base.iloc[query].to_numpy(), b]).astype('float32')
        assert np.isfinite(X).all() and np.isfinite(V).all()
        return X, V, list(self.base.columns)+names, self.meta['feature_types']+['q']*len(names)
