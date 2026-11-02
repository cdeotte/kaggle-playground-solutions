"""Paths, folds and shared inputs of the neural contrasts.

Inputs from ../income_mixture_xgb (run it first):
  cache/stage1/   base.parquet, keys.parquet, predictors.parquet, manifest.json, source_base.npy (known-source offset),
                  matched_f<k>/ (nested response features, + manifest.json with their names / types), semi_f<k>.npz (EM)
  cache/stage2/teachers/income_f<k>_{outer,inner<j>}.npz   income-excluding context teachers
  cache/em_info/  base/ and features_gpu/ fold packs (copula XGB)
Written here:
  cache/stage3 ... cache/stage6   label-free banks and the fold predictions each stage hands to the next one
  cache/contrast/                 response GLM margins, sparse GAM, copula density experts and copula XGB
  cache/gpt_residual/             GPT-2 residual runs
  runs/<config>_s<seed>_f<k>/predictions.npz   one MLP / generator fit: index = query rows, prediction (and features)
Query rows of fold k = [the fold-k train rows, all test rows]; test rows are numbered n_train, n_train + 1, ..."""
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, read_test, target  # noqa: E402
from common.pairs import save_pair  # noqa: E402

UPSTREAM = HERE.parent / 'income_mixture_xgb' / 'cache'
STAGE1 = UPSTREAM / 'stage1'
TEACHERS = UPSTREAM / 'stage2' / 'teachers'
EM_INFO = UPSTREAM / 'em_info'
CACHE = HERE / 'cache'
STAGE = {s: CACHE / f'stage{s}' for s in (3, 4, 5, 6)}
CONTRAST = CACHE / 'contrast'
GPT_RESIDUAL = CACHE / 'gpt_residual'
RUNS = HERE / 'runs'
ALL16 = list(range(42, 58))
MIXTURE = ['mixture_prior_1', 'mixture_prior_20', 'mixture_delta_1', 'mixture_delta_20']
EMB8 = [f'embed_real_income_{j}' for j in range(8)]
GEN_DELTA = [f'generator_context_delta_{v}' for v in ('llr', 'h0', 'h1', 'hdiff')]


@lru_cache(None)
def labels():
    return target()


@lru_cache(None)
def fold_ids():
    return folds(labels())


@lru_cache(None)
def n_test():
    return len(read_test())


def split_indices(k):
    """fit = outer-train rows, valid = fold-k rows, query = valid rows then all test rows."""
    f = fold_ids()
    n = len(f)
    fit, valid = np.flatnonzero(f != k), np.flatnonzero(f == k)
    return fit, valid, np.r_[valid, np.arange(n, n + n_test())]


def ints(arg):
    """'0,1,2' -> [0, 1, 2]; '42-57' -> [42, ..., 57]"""
    out = []
    for part in str(arg).split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


class Stage1:
    """Prepared tables of all train + test rows (train first): base features, integer keys, key sizes."""
    def __init__(self):
        self.meta = json.loads((STAGE1 / 'manifest.json').read_text())
        self.base = pd.read_parquet(STAGE1 / 'base.parquet')
        self.keys = pd.read_parquet(STAGE1 / 'keys.parquet')
        self.y = labels()


class IncomeParents:
    """Exact income values (key) grouped into parents = (income // 1000, leading BPE piece), and the 4 contexts made of
    Subsidy_Available, Range_Anxiety_Level and Environmental_Concern_Level."""
    def __init__(self, b):
        d = b.base
        self.key = b.keys.Annual_Income_USD.to_numpy()
        self.ng = b.meta['key_sizes']['Annual_Income_USD']
        self.parent = pd.factorize(pd.MultiIndex.from_arrays([b.keys.inc1000, b.keys.inc_bpe0]), sort=True)[0]
        self.npar = int(self.parent.max()) + 1
        self.pv = np.zeros(self.ng, dtype='int32')
        self.pv[self.key] = self.parent
        sub = d.Subsidy_Available.to_numpy('int64')
        anx = d.Range_Anxiety_Level.to_numpy('int64')
        concern = pd.factorize(d.Environmental_Concern_Level.to_numpy('int64'), sort=True)[0]
        self.ctx = dict(sa=sub * 3 + anx, sc=sub * 5 + concern, ac=anx * 5 + concern, joint=(sub * 3 + anx) * 5 + concern)


def within_parent_permutation(parent, n):
    """The fixed within-parent row shuffle of the null (shuffled-context) arms of the banks."""
    perm = np.arange(n)
    rng = np.random.default_rng(421337)
    for ix in pd.Series(perm).groupby(parent).indices.values():
        perm[ix] = rng.permutation(ix)
    return perm


def source_margin():
    return np.load(STAGE1 / 'source_base.npy')


def fold_bank_dir(k):
    """Per-fold banks (embedding_income_f<k>, generative_f<k>): folds 0 and 4 under stage3, folds 1-3 under stage4."""
    return STAGE[3] if k in (0, 4) else STAGE[4]


def fold_bank(name, k):
    path = STAGE[3] / f'{name}_f{k}.npz'
    if not path.exists():
        path = STAGE[4] / f'{name}_f{k}.npz'
    return np.load(path), json.loads(path.with_suffix('.json').read_text())['names']


def all_bank(name, stage):
    path = STAGE[stage] / f'{name}_all.npy'
    return np.load(path, mmap_mode='r'), json.loads(path.with_suffix('.json').read_text())['names']


def write_names(path, names, **extra):
    Path(path).with_suffix('.json').write_text(json.dumps(dict(names=names, **extra), indent=2))


def baseline(k):
    """Fold-k matrices of ../income_mixture_xgb (raw features, frequency / source features, nested target encodings and
    source-conditional intercepts) + the 4 EM income-mixture columns (prior and EM delta, alpha 1 and 20)."""
    fit, valid, query = split_indices(k)
    p = STAGE1 / f'matched_f{k}'
    m = json.loads((p / 'manifest.json').read_text())
    d = np.load(STAGE1 / f'semi_f{k}.npz')

    def mix(a):
        r = a[:, [0, 4, 2, 6]].copy()
        r[:, 2:] -= r[:, :2]
        return r
    X = np.column_stack([np.load(p / 'train.npy'), mix(d['train_features'])])
    Q = np.column_stack([np.load(p / 'query.npy'), mix(d['query_features'])])
    return X, Q, m['feature_names'] + MIXTURE, m['feature_types'] + ['q'] * 4, fit, query


def rich_baseline(k):
    """baseline + the 8 real income-context embedding columns + the 4 generator context deltas."""
    X, Q, names, types, fit, query = baseline(k)
    for bank, columns in [('embedding_income', EMB8), ('generative', GEN_DELTA)]:
        z, nm = fold_bank(bank, k)
        ids = [nm.index(c) for c in columns]
        X = np.column_stack([X, z['train_features'][:, ids]])
        Q = np.column_stack([Q, z['query_features'][:, ids]])
        names, types = names + columns, types + ['q'] * len(columns)
    return X, Q, names, types, fit, query


def coarse_baseline(k):
    """rich_baseline + the 27 coarse parent-entropy columns."""
    X, Q, names, types, fit, query = rich_baseline(k)
    A, nm = all_bank('coarse_entropy', 5)
    X = np.column_stack([X, A[fit]])
    Q = np.column_stack([Q, A[query]])
    return X, Q, names + [f'coarse_{c}' for c in nm], types + ['q'] * len(nm), fit, query


def run_dir(config, seed, k):
    return RUNS / f'{config}_s{seed}_f{k}'


def mean_logit(config, k, seeds):
    """Mean over seeds of logit(prediction clipped to [1e-7, 1 - 1e-7]) on the query rows of fold k."""
    query = split_indices(k)[2]
    zs = []
    for s in seeds:
        d = np.load(run_dir(config, s, k) / 'predictions.npz')
        assert np.array_equal(d['index'], query), (config, s, k)
        zs.append(logit(d['prediction'].astype('float64').clip(1e-7, 1 - 1e-7)))
    return np.mean(zs, axis=0)


def z(p):
    """logit(clip(p, 1e-15, 1 - 1e-15)) clipped to +-30 (float64)."""
    p = np.clip(np.asarray(p, np.float64), 1e-15, 1 - 1e-15)
    return np.clip(np.log(p) - np.log1p(-p), -30, 30)


def save_logits(name, oof, test):
    """Writes the pair as logits z(p): these carriers come within 1e-10 of 0 and 1, beyond float32 probabilities."""
    from sklearn.metrics import roc_auc_score
    save_pair(HERE, name, z(oof), z(test))
    print(f'{name}: OOF AUC {roc_auc_score(labels(), oof):.6f}', flush=True)
