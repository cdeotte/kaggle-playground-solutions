"""Label-free feature codes for the feature-token transformers (Q-FT and Q-FT v2).

Each of the 21 features (data.csv column order) gets an integer code (its own vocabulary); 4 also get a standardized number:
  categoricals (Gender, Customer Type, Type of Travel, Class): sorted levels
  13 service ratings: the value 0-5 itself
  Age: one code per distinct age + numeric (standardized)
  Flight Distance: a route-id code for every distance seen >= 3 times in train+test (rarer -> code 0 'rare') + numeric log(FD)
  Departure / Arrival Delay: bucket code {0, 1-5, 6-15, 16-30, 31-60, 61-120, >120, NaN} + numeric log1p(delay)
Vocabularies and standardization use train+test FEATURES only (no labels). Original rows (Q-FT v2 pretraining) are coded with
the same train+test vocabularies; an original Age not seen in train+test maps to the nearest one."""
import numpy as np
import pandas as pd

from common.data import CATEGORICALS, DATA_DIR, TARGET, read_original, read_test, read_train

EDGES = np.array([0.5, 5.5, 15.5, 30.5, 60.5, 120.5])


def feature_codes(include_original=False):
    """Returns codes int64 [N, 21], nums float32 [N, 21] for rows train, test (, original) and meta."""
    cols = [c for c in pd.read_csv(DATA_DIR / 'data.csv', nrows=2).columns if c != TARGET]
    tr, te = read_train(), read_test()
    fit = pd.concat([tr[cols], te[cols]], ignore_index=True)
    parts = [fit] + ([read_original()[cols]] if include_original else [])
    app = pd.concat(parts, ignore_index=True)
    N = len(app)
    codes, nums = np.zeros((N, len(cols)), np.int64), np.zeros((N, len(cols)), np.float32)
    meta = dict(cols=cols, vocab=[], numeric=[], ntrain=len(tr), ntest=len(te), norig=N - len(fit))
    for j, c in enumerate(cols):
        v, o = fit[c], app[c]
        num = False
        if c in CATEGORICALS:
            lev = sorted(v.unique())
            cc = pd.Categorical(o, categories=lev).codes
            assert (cc >= 0).all(), c
            codes[:, j] = cc
            meta['vocab'].append(len(lev))
        elif c == 'Age':
            lev = np.sort(v.unique())
            pos = np.clip(np.searchsorted(lev, o.to_numpy()), 0, len(lev) - 1)
            prev = np.clip(pos - 1, 0, len(lev) - 1)
            codes[:, j] = np.where(np.abs(lev[prev] - o.to_numpy()) < np.abs(lev[pos] - o.to_numpy()), prev, pos)
            x = v.to_numpy(np.float64)
            nums[:, j] = (o.to_numpy(np.float64) - x.mean()) / x.std()
            num = True
            meta['vocab'].append(len(lev))
        elif c == 'Flight Distance':
            vc = v.value_counts()
            keep = np.sort(vc[vc >= 3].index.to_numpy())
            pos = np.searchsorted(keep, o.to_numpy())
            hit = (pos < len(keep)) & (keep[np.minimum(pos, len(keep) - 1)] == o.to_numpy())
            codes[:, j] = np.where(hit, pos + 1, 0)
            x = np.log(v.to_numpy(np.float64))
            nums[:, j] = (np.log(o.to_numpy(np.float64)) - x.mean()) / x.std()
            num = True
            meta['vocab'].append(len(keep) + 1)
        elif 'Delay' in c:
            x = v.to_numpy(np.float64)
            nan = np.isnan(x)
            lx = np.log1p(np.nan_to_num(x, nan=0.0))
            mu, sd = lx[~nan].mean(), lx[~nan].std()
            xo = o.to_numpy(np.float64)
            nano = np.isnan(xo)
            codes[:, j] = np.where(nano, 7, np.searchsorted(EDGES, np.nan_to_num(xo, nan=0.0)))
            nums[:, j] = np.where(nano, 0.0, (np.log1p(np.nan_to_num(xo, nan=0.0)) - mu) / sd)
            num = True
            meta['vocab'].append(8)
        else:                                                    # the 13 service ratings, values 0-5
            codes[:, j] = o.to_numpy()
            meta['vocab'].append(6)
        meta['numeric'].append(num)
    return codes, nums, meta
