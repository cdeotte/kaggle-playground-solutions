"""Same-fold soft labels: sigmoid of the mean clipped logit of the 16 digits_diffs XGB seeds (42-57) of
../income_mixture_xgb for the query rows (fold-k rows, then all test rows) of outer fold k.
Cached in cache/teacher16_f{k}.npz (query, probability, margin)."""
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
CACHE = HERE/'cache'
TEACHER_RUNS = HERE.parent/'income_mixture_xgb'/'runs'/'stage8'


def teacher(k, query):
    out = CACHE/f'teacher16_f{k}.npz'
    if out.exists():
        d = np.load(out)
        assert np.array_equal(d['query'], query)
        return d['probability']
    pieces = []
    for s in range(42, 58):
        with np.load(TEACHER_RUNS/f'digits_diffs_s{s}_f{k}'/'predictions.npz') as d:
            assert np.array_equal(d['index'], query)
            pieces.append(logit(d['prediction'].astype('float64').clip(1e-7, 1-1e-7)))
    z = np.mean(pieces, axis=0)
    pp = expit(z)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(out, query=query, probability=pp, margin=z)
    return pp
