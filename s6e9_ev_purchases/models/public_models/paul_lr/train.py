#!/usr/bin/env python
"""Paul Bryan Elefante's (@heuljax) public notebook "KPS6E09 Logistic Regression Sample", run on our 5 folds.
https://www.kaggle.com/code/heuljax/kps6e09-logistic-regression-sample?scriptVersionId=353829637

His model: 229 features (300 model columns after one-hot coding) and sklearn LogisticRegression (lbfgs, C 0.1, max_iter
250, tol 2e-4) on standardized inputs. Features: label-free static columns (raw values, digits, the purchase-score formula,
train + test income / commute counts and income-group composition means, original-data income priors, a frozen ordinal
range-anxiety map) and a DonorState of label-derived maps (income / commute TEs and local rates, hierarchical income
rates, numeric-prefix hierarchy, cross TEs, gate supports, pair deviations, a fast gate mixture and an additive logistic
approximation). Outer-train rows get their DonorState features from 5 inner stratified folds (donor = the other 4/5);
valid / test rows from the state fitted on all outer-train rows.
The notebook code below runs as published except: the outer folds are our StratifiedKFold(5, shuffle=True,
random_state=42) instead of his 10 folds; plain CPU sklearn instead of cuml.accel; local paths; no output files other
than the pair. Test = mean of the 5 fold models.
Usage: python train.py   -> preds/paul_lr_{oof,test}.npy"""
# ===== notebook cell 2 =====
from __future__ import annotations

import gc
import os
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.special import expit, ndtr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from common.data import folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

EXP_NAME = MODEL_NAME = 'paul_lr'
N_FOLDS, INNER_FOLDS, RANDOM_STATE = 5, 5, 42
N_THREADS = max(1, min(32, os.cpu_count() or 1))
BATCH_ROWS = 32768
MATRIX_RAM_MB = 1536
ADDITIVE_SWEEPS = 10
MIXTURE_STEPS = 8
LOGREG_PARAMS = dict(C=0.1, solver='lbfgs', max_iter=250, tol=2e-4, random_state=RANDOM_STATE)

TARGET, ID_COL = 'Will_Buy_EV', 'id'
FLOAT_COLS = ['Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level']
INT_COLS = ['Age', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
            'Charging_Stations_Near_Work']
CAT_COLS = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
            'Subsidy_Available', 'Range_Anxiety_Level']
FEATURE_COLS = FLOAT_COLS + INT_COLS + CAT_COLS
CAT_LEVELS = {
    'Gender': ['Female', 'Male', 'Other'],
    'City_Type': ['Rural', 'Suburban', 'Urban'],
    'Current_Car_Type': ['Hatchback', 'SUV', 'Sedan', 'Truck'],
    'Home_Charging_Possible': ['No', 'Yes'],
    'Subsidy_Available': ['No', 'Yes'],
    'Range_Anxiety_Level': ['Low', 'Medium', 'High'],
}
numba.set_num_threads(min(N_THREADS, numba.config.NUMBA_NUM_THREADS))
np.random.seed(RANDOM_STATE)
TIMINGS = []


def as_host_numpy(value, dtype=None):
    if type(value) is not np.ndarray:
        unwrap = getattr(value, 'as_cpu_object', None)
        if callable(unwrap):
            value = unwrap()
        convert = getattr(value, 'to_numpy', None)
        if callable(convert):
            value = convert()
        if hasattr(value, '__cuda_array_interface__'):
            transfer = getattr(value, 'get', None)
            if not callable(transfer):
                transfer = getattr(value, 'copy_to_host', None)
            if not callable(transfer):
                raise TypeError('Array requires an explicit host-transfer method.')
            value = transfer()
    return np.asarray(value, dtype=dtype)


# ===== notebook cell 3 =====
DATA_TRAIN = read_train()
DATA_TEST = read_test()
DATA_ORIGINAL = read_original()
for label, frame, required in (
    ('train', DATA_TRAIN, [ID_COL, TARGET] + FEATURE_COLS),
    ('test', DATA_TEST, [ID_COL] + FEATURE_COLS),
    ('original', DATA_ORIGINAL, [TARGET] + FEATURE_COLS),
):
    missing = set(required) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f'{label}: empty data or missing columns {sorted(missing)}')
for label, frame in (('train', DATA_TRAIN), ('test', DATA_TEST)):
    if frame[ID_COL].isna().any() or not frame[ID_COL].is_unique:
        raise ValueError(f'{label}: IDs must be unique and nonmissing.')
    if frame[FEATURE_COLS].isna().any().any():
        raise ValueError(f'{label}: predictors must be nonmissing.')
    if not np.isfinite(as_host_numpy(frame[FLOAT_COLS + INT_COLS].to_numpy(np.float64))).all():
        raise ValueError(f'{label}: numeric predictors must be finite.')
    if (frame[['Annual_Income_USD', 'Daily_Commute_km']] < 0).any().any():
        raise ValueError(f'{label}: income and commute must be nonnegative.')
    for column, levels in CAT_LEVELS.items():
        if not frame[column].isin(levels).all():
            raise ValueError(f'{label}: unknown category in {column}.')
if TARGET in DATA_TEST or set(as_host_numpy(DATA_TRAIN[TARGET].unique())) != {'Yes', 'No'}:
    raise ValueError('Train labels must be Yes/No; test must not contain labels.')
if np.intersect1d(as_host_numpy(DATA_TRAIN[ID_COL]), as_host_numpy(DATA_TEST[ID_COL])).size:
    raise ValueError('Train and test IDs overlap.')
if DATA_ORIGINAL.empty or not DATA_ORIGINAL[TARGET].isin(['Yes', 'No']).all():
    raise ValueError('Original data must contain nonmissing Yes/No labels.')


def feature_hashes(frame):
    values = frame[FEATURE_COLS].copy()
    for column in FLOAT_COLS + INT_COLS:
        values[column] = pd.to_numeric(values[column], errors='raise').astype(np.float64)
    for column in CAT_COLS:
        values[column] = values[column].astype('string').fillna('__MISSING__')
    return as_host_numpy(pd.util.hash_pandas_object(values, index=False).to_numpy(np.uint64))


source_hashes = feature_hashes(DATA_ORIGINAL)
for label, frame in (('train', DATA_TRAIN), ('test', DATA_TEST)):
    if np.isin(feature_hashes(frame), source_hashes).any():
        raise ValueError(f'{label}: exact predictor overlap with original data.')
del source_hashes
Y = as_host_numpy((DATA_TRAIN[TARGET] == 'Yes').to_numpy(np.int8))
FOLD_ID = folds(Y)  # our outer folds: StratifiedKFold(5, shuffle=True, random_state=42)
SPLITS = [(np.flatnonzero(FOLD_ID != k), np.flatnonzero(FOLD_ID == k)) for k in range(N_FOLDS)]
print('Train:', DATA_TRAIN.shape, '| test:', DATA_TEST.shape, '| original:', DATA_ORIGINAL.shape)


class Data:

    def __init__(self):
        tr = DATA_TRAIN.copy(deep=False)
        te = DATA_TEST.copy(deep=False)
        og = DATA_ORIGINAL.copy(deep=False)
        for fr, need, nm in ((tr, [ID_COL] + FEATURE_COLS + [TARGET], 'train'), (te, [ID_COL] + FEATURE_COLS, 'test'), (og, FEATURE_COLS + [TARGET], 'original')):
            miss = set(need) - set(fr.columns)
            if miss:
                raise AssertionError(f'{nm} missing {miss}')
        assert tr[ID_COL].is_unique and te[ID_COL].is_unique
        self.train_ids = as_host_numpy(tr[ID_COL].to_numpy(np.int64))
        self.test_ids = as_host_numpy(te[ID_COL].to_numpy(np.int64))
        self.n_train = len(tr)
        self.n_test = len(te)
        self.y = as_host_numpy((tr[TARGET] == 'Yes').to_numpy(np.int8))
        al = pd.concat([tr[FEATURE_COLS], te[FEATURE_COLS]], ignore_index=True)
        self.n_all = len(al)
        self.inc = np.rint(as_host_numpy(al['Annual_Income_USD'].to_numpy(np.float64))).astype(np.int64)
        assert np.allclose(self.inc, as_host_numpy(al['Annual_Income_USD'].to_numpy()))
        self.cmt10 = np.rint(10 * as_host_numpy(al['Daily_Commute_km'].to_numpy(np.float64))).astype(np.int64)
        self.conc = np.rint(as_host_numpy(al['Environmental_Concern_Level'].to_numpy(np.float64))).astype(np.int64)
        self.age = as_host_numpy(al['Age'].to_numpy(np.int64))
        self.cars = as_host_numpy(al['Number_of_Cars_Owned'].to_numpy(np.int64))
        self.sth = as_host_numpy(al['Charging_Stations_Near_Home'].to_numpy(np.int64))
        self.stw = as_host_numpy(al['Charging_Stations_Near_Work'].to_numpy(np.int64))
        self.cat = {}
        for c, lev in CAT_LEVELS.items():
            codes = as_host_numpy(pd.Categorical(al[c], categories=lev).codes, np.int64)
            assert (codes >= 0).all(), c
            self.cat[c] = codes
        self.sub = self.cat['Subsidy_Available']
        self.anx = self.cat['Range_Anxiety_Level']
        self.home = self.cat['Home_Charging_Possible']
        self.city = self.cat['City_Type']
        self.gate = (self.conc - 1) * 6 + self.sub * 3 + self.anx
        self.og = og
        self.yo = as_host_numpy((og[TARGET] == 'Yes').to_numpy(np.int8))

    def folds(self) -> np.ndarray:
        fold = np.full(self.n_train, -1, np.int8)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for k, (_, va) in enumerate(skf.split(np.zeros(self.n_train), self.y)):
            fold[va] = k
        assert (fold >= 0).all()
        return fold

def grid_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid, code = np.unique(values, return_inverse=True)
    return (grid, code.astype(np.int64))



# ===== notebook cell 4 =====
def safe_logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def box_sum(values, radius):
    values = np.asarray(values, np.float64)
    prefix = np.empty((len(values) + 1,) + values.shape[1:], np.float64)
    prefix[0] = 0
    np.cumsum(values, axis=0, out=prefix[1:])
    index = np.arange(len(values))
    return prefix[np.minimum(index + int(radius) + 1, len(values))] - prefix[np.maximum(index - int(radius), 0)]


def smooth_counts(values, width):
    # Three box filters approximate a Gaussian in O(number of groups).
    radius = max(1, int(round((np.sqrt(1.0 + 4.0 * width * width) - 1.0) / 2.0)))
    out = np.asarray(values, np.float64)
    for _ in range(3):
        out = box_sum(out, radius) / (2 * radius + 1)
    return np.maximum(out * (np.sqrt(2.0 * np.pi) * width) - values, 0.0)


@njit(nogil=True, parallel=True)
def fit_mixture(all_counts, q0, q1, prior, steps):
    ng, nk = all_counts.shape
    result = np.empty((ng, 3), np.float64)
    for i in prange(ng):
        mu = min(max(prior[i], 1e-4), 1 - 1e-4)
        p = mu
        for _ in range(steps):
            grad = 10.0 * (mu / p - (1 - mu) / (1 - p))
            hess = 10.0 * (mu / (p * p) + (1 - mu) / ((1 - p) ** 2))
            for k in range(nk):
                delta = q1[i, k] - q0[i, k]
                ratio = delta / max(q0[i, k] + p * delta, 1e-15)
                grad += all_counts[i, k] * ratio
                hess += all_counts[i, k] * ratio * ratio
            step = min(max(grad / max(hess, 1e-12), -0.15), 0.15)
            new_p = min(max(p + step, 1e-4), 1 - 1e-4)
            if abs(new_p - p) < 1e-7:
                p = new_p
                break
            p = new_p
        info, gain, count = 0.0, 0.0, 0.0
        for k in range(nk):
            delta = q1[i, k] - q0[i, k]
            mix = max(q0[i, k] + p * delta, 1e-15)
            base = max(q0[i, k] + mu * delta, 1e-15)
            info += all_counts[i, k] * (delta / mix) ** 2
            gain += all_counts[i, k] * np.log(mix / base)
            count += all_counts[i, k]
        result[i, 0] = p
        result[i, 1] = np.log1p(info)
        result[i, 2] = gain / max(count, 1.0)
    return result


@njit(nogil=True)
def fit_additive(codes, labels, offsets, penalties, sweeps):
    # Penalized logistic regression on fixed, sparse one-hot bin coordinates.
    nt, nr = codes.shape
    prior = (labels.sum() + 0.5) / (nr + 1.0)
    intercept = np.log(prior / (1.0 - prior))
    eta = np.full(nr, intercept, np.float64)
    beta = np.zeros(offsets[-1], np.float64)
    largest = 0
    for j in range(nt):
        largest = max(largest, offsets[j + 1] - offsets[j])
    grad = np.empty(largest, np.float64)
    hess = np.empty(largest, np.float64)
    step = np.empty(largest, np.float64)
    for sweep in range(sweeps):
        g0, h0 = 0.0, 0.0
        for i in range(nr):
            p = 1.0 / (1.0 + np.exp(-min(max(eta[i], -35.0), 35.0)))
            g0 += labels[i] - p
            h0 += p * (1.0 - p)
        bstep = min(max(g0 / max(h0, 1e-12), -1.0), 1.0)
        intercept += bstep
        eta += bstep
        largest_step = abs(bstep)
        for j in range(nt):
            offset = offsets[j]
            size = offsets[j + 1] - offset
            penalty = penalties[j]
            for k in range(size):
                grad[k] = -penalty * beta[offset + k]
                hess[k] = penalty
            for i in range(nr):
                p = 1.0 / (1.0 + np.exp(-min(max(eta[i], -35.0), 35.0)))
                k = codes[j, i]
                grad[k] += labels[i] - p
                hess[k] += p * (1.0 - p)
            for k in range(size):
                step[k] = 0.8 * min(max(grad[k] / max(hess[k], 1e-12), -1.0), 1.0)
                beta[offset + k] += step[k]
                largest_step = max(largest_step, abs(step[k]))
            for i in range(nr):
                eta[i] += step[codes[j, i]]
        if largest_step < 1e-4:
            break
    return intercept, beta, eta, sweep + 1


@njit(nogil=True, parallel=True)
def gather_additive(codes, rows, offsets, components, beta, intercept):
    out = np.zeros((len(rows), 5), np.float64)
    for i in prange(len(rows)):
        out[i, 0] = intercept
        for j in range(codes.shape[0]):
            out[i, components[j]] += beta[offsets[j] + codes[j, rows[i]]]
    return out


class CodeBook:
    def __init__(self):
        self.codes, self.sizes = {}, {}

    def add(self, name, values):
        if name not in self.codes:
            _, inverse = np.unique(values, return_inverse=True)
            self.codes[name] = inverse.astype(np.int32)
            self.sizes[name] = int(inverse.max()) + 1
        return name

    def pair(self, name, a, b):
        return self.add(name, self.codes[a].astype(np.int64) * self.sizes[b] + self.codes[b])

    def stats(self, rows, labels):
        target = np.asarray(labels[rows], np.float64)
        return {name: (np.bincount(code[rows], minlength=self.sizes[name]).astype(np.float64),
                       np.bincount(code[rows], weights=target, minlength=self.sizes[name]))
                for name, code in self.codes.items()}


def subtract_stats(total, held):
    result = {}
    for name, (n, s) in total.items():
        hn, hs = held[name]
        dn, ds = n - hn, s - hs
        if np.any(dn < 0) or np.any(ds < 0) or np.any(ds > dn):
            raise AssertionError('Invalid donor sufficient statistics.')
        result[name] = dn, ds
    return result


# ===== notebook cell 5 =====
def build_labelfree(d: Data) -> tuple[np.ndarray, list[str], list[str], dict]:
    t0 = time.time()
    cols, names, scopes = ([], [], [])

    def add(name, arr, scope):
        cols.append(np.asarray(arr, dtype=np.float32))
        names.append(name)
        scopes.append(scope)
    add('Age', d.age, 'row-local')
    add('Annual_Income_USD', d.inc, 'row-local')
    add('Daily_Commute_km', d.cmt10 / 10.0, 'row-local')
    add('Number_of_Cars_Owned', d.cars, 'row-local')
    add('Charging_Stations_Near_Home', d.sth, 'row-local')
    add('Charging_Stations_Near_Work', d.stw, 'row-local')
    add('Environmental_Concern_Level', d.conc, 'row-local')
    for c in CAT_LEVELS:
        add(c, d.cat[c], 'row-local')
    for p in range(6):
        add(f'DIG_INC_10E{p}', d.inc // 10 ** p % 10, 'row-local')
    for p in range(3):
        add(f'DIG_CMT10_10E{p}', d.cmt10 // 10 ** p % 10, 'row-local')
    penalty = (d.anx == 1) * 1.0 + (d.anx == 2) * 3.0
    score = 1.2 * d.inc / 100000.0 + 0.6 * d.conc + 2.0 * d.sub - penalty
    add('BUY_SCORE', score, 'row-local')
    inc_grid, inc_code = grid_codes(d.inc)
    cmt_grid, cmt_code = grid_codes(d.cmt10)
    n_inc = np.bincount(inc_code).astype(np.float64)
    n_cmt = np.bincount(cmt_code).astype(np.float64)
    add('CNT_INC_LOG', np.log1p(n_inc[inc_code]), 'train+test transductive')
    add('CNT_CMT_LOG', np.log1p(n_cmt[cmt_code]), 'train+test transductive')
    for div in (10, 100, 1000):
        g, c = grid_codes(d.inc // div)
        add(f'CNT_INC{div}_LOG', np.log1p(np.bincount(c)[c]), 'train+test transductive')
    add('INC_RANK', inc_code / max(1, len(inc_grid) - 1), 'train+test transductive')
    gl = np.r_[np.inf, np.diff(inc_grid)]
    gr = np.r_[np.diff(inc_grid), np.inf]
    add('INC_GAP_MIN_LOG', np.log1p(np.nan_to_num(np.minimum(gl, gr)[inc_code], posinf=0)), 'train+test transductive')
    og = d.og
    oi = as_host_numpy(og['Annual_Income_USD'].dropna().to_numpy(np.float64)).astype(np.int64)
    o_grid, o_cnt = np.unique(oi, return_counts=True)
    pos = np.searchsorted(o_grid, d.inc)
    posc = np.clip(pos, 0, len(o_grid) - 1)
    seen = o_grid[posc] == d.inc
    ocount = np.where(seen, o_cnt[posc], 0)
    add('ORIG_INC_COUNT_LOG', np.log1p(ocount), 'auxiliary-source covariates')
    add('ORIG_INC_SEEN', seen, 'auxiliary-source covariates')
    lo = np.clip(pos - 1, 0, len(o_grid) - 1)
    dist = np.minimum(np.abs(d.inc - o_grid[lo]), np.abs(o_grid[posc] - d.inc))
    add('ORIG_INC_DIST_LOG', np.log1p(dist), 'auxiliary-source covariates')
    comp_share = n_inc[inc_code] / d.n_all
    orig_share = (ocount + 0.5) / (len(oi) + 0.5)
    add('INC_LIFT_LOG', np.log(comp_share) - np.log(orig_share), 'train+test transductive + source covariates')
    ogd = og.dropna(subset=['Annual_Income_USD'])
    oy = as_host_numpy((ogd[TARGET] == 'Yes').to_numpy(np.float64))
    oii = np.rint(as_host_numpy(ogd['Annual_Income_USD'].to_numpy())).astype(np.int64)
    oc = np.searchsorted(o_grid, oii)
    osum = np.bincount(oc, weights=oy, minlength=len(o_grid))
    oprior = float((og[TARGET] == 'Yes').mean())
    orate = osum / o_cnt
    add('ORIG_INC_Y', np.where(seen, orate[posc], oprior), 'auxiliary-source-label')
    loc = (box_sum(osum, 25) + 20 * oprior) / (box_sum(o_cnt.astype(float), 25) + 20)
    near = np.where(np.abs(d.inc - o_grid[lo]) <= np.abs(o_grid[posc] - d.inc), lo, posc)
    add('ORIG_INC_LOCAL_Y', loc[near], 'auxiliary-source-label')
    uniq = seen & (ocount == 1)
    ogi = ogd.copy()
    ogi['_c'] = oc
    first = ogi.drop_duplicates('_c').set_index('_c')
    donor_city = np.full(d.n_all, np.nan)
    city_map = as_host_numpy(pd.Categorical(first['City_Type'], categories=CAT_LEVELS['City_Type']).codes, float)
    donor_city_tab = np.full(len(o_grid), -1.0)
    donor_city_tab[as_host_numpy(first.index.to_numpy())] = city_map
    donor_city[uniq] = donor_city_tab[posc[uniq]]
    add('DONOR_CITY', donor_city, 'auxiliary-source covariates')
    add('DONOR_CITY_MATCH', np.where(uniq, (donor_city == d.city) * 1.0, np.nan), 'auxiliary-source covariates')
    ng = len(inc_grid)
    comp_vals = {'CONCERN': d.conc.astype(float), 'SUBSIDY': d.sub.astype(float), 'ANXPEN': penalty, 'SCORE': score, 'HOME': d.home.astype(float), 'URBAN': (d.city == 2).astype(float), 'CMT': d.cmt10 / 10.0, 'STATIONS': (d.sth + d.stw).astype(float)}
    for nm, v in comp_vals.items():
        tot = np.bincount(inc_code, weights=v, minlength=ng)
        prior = float(v.mean())
        for a in (5.0, 20.0) if nm in ('CONCERN', 'SUBSIDY', 'ANXPEN', 'SCORE') else (20.0,):
            post = (tot + a * prior) / (n_inc + a)
            add(f'LAT_INC_{nm}_A{a:g}', post[inc_code], 'train+test transductive')
            if a == 20.0:
                add(f'LAT_INC_{nm}_DEV', v - post[inc_code], 'train+test transductive')
    gcnt = np.bincount(inc_code * 30 + d.gate, minlength=ng * 30).reshape(ng, 30).astype(float)
    gmarg = gcnt.sum(0) / gcnt.sum()
    gshare = (gcnt + 10 * gmarg) / (gcnt.sum(1, keepdims=True) + 10)
    hi_cells = [(c - 1) * 6 + s * 3 + a for c in (4, 5) for s in (1,) for a in (0, 1)]
    lo_cells = [(c - 1) * 6 + s * 3 + a for c in (1, 2) for s in (0, 1) for a in (0, 1, 2)]
    add('LAT_INC_GATE_HI', gshare[:, hi_cells].sum(1)[inc_code], 'train+test transductive')
    add('LAT_INC_GATE_LO', gshare[:, lo_cells].sum(1)[inc_code], 'train+test transductive')
    X = np.nan_to_num(np.column_stack(cols).astype(np.float32), nan=-1.0)
    cats = {n: i for i, n in enumerate(names) if n in CAT_LEVELS or n.startswith('DIG_') or n == 'DONOR_CITY'}
    meta = {'seconds': time.time() - t0, 'n_features': len(names)}
    return (X, names, scopes, meta)


def ordinal_features(d):
    theta = np.array([0.6225255587437528, 1.494699077420564,
        -1.019945286550291, -1.1168814422664504, -5.505538789883744,
        -0.48577588889297657, -0.1872527353331304, 0.9736132942024454,
        0.4627264362166528, -0.015571769264975915, 0.025596073371908553])
    x = np.column_stack([np.ones(d.n_all), d.cmt10 / 500.0,
                         d.sth / 10.0, d.stw / 10.0, d.home,
                         d.city == 2, d.city == 1])
    # Same frozen label-free ordinal map, evaluated only once.
    scale_x = np.column_stack([d.home, d.city == 2, d.city == 1])
    eta = x @ theta[:7]
    sig = np.exp(scale_x @ theta[8:])
    gap = np.exp(theta[7])
    a, b = -eta / sig, (gap - eta) / sig
    pa = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
    pb = np.exp(-0.5 * b * b) / np.sqrt(2 * np.pi)
    p = np.maximum(np.column_stack([ndtr(a), ndtr(b) - ndtr(a), ndtr(-b)]), 1e-14)
    derivative = np.column_stack([-pa, pa - pb, pb])
    old_eta = 0.5 + x[:, 1] - x[:, 2] - x[:, 3] - 3.0 * d.home
    oldp = np.maximum(np.column_stack([ndtr(-old_eta),
        ndtr(2.0 - old_eta) - ndtr(-old_eta), ndtr(old_eta - 2.0)]), 1e-14)
    index = np.arange(d.n_all)
    observed = p[index, d.anx]
    names = ['ANX_CAL_LOCATION', 'ANX_CAL_OBS_SURPRISE', 'ANX_CAL_TRUNC_NOISE',
             'ANX_CAL_ENTROPY', 'ANX_CAL_LOG_DENSITY_RATIO', 'ANX_CAL_SCALE']
    matrix = np.column_stack([eta / sig, -np.log(observed),
        derivative[index, d.anx] / observed, -(p * np.log(p)).sum(1),
        np.log(observed / oldp[index, d.anx]), sig]).astype(np.float32)
    return matrix, names


static_started = time.perf_counter()
d = Data()
STATIC, STATIC_NAMES, STATIC_SCOPES, _ = build_labelfree(d)
ordinal, ordinal_names = ordinal_features(d)
STATIC = np.column_stack([STATIC, ordinal]).astype(np.float32)
STATIC_NAMES += ordinal_names
STATIC_SCOPES += ['row-local; frozen predictor-only ordinal map'] * len(ordinal_names)
del ordinal
BOOK = CodeBook()
base_values = {
    'INC': d.inc, 'INC10': d.inc // 10, 'INC100': d.inc // 100,
    'INC1000': d.inc // 1000, 'CMT': d.cmt10, 'CMT_INT': d.cmt10 // 10,
    'GATE': d.gate, 'AGE': d.age, 'CARS': d.cars, 'STH': d.sth,
    'STW': d.stw, 'CITY': d.city, 'HOME': d.home, 'SUB': d.sub,
    'ANX': d.anx, 'CONC': d.conc, 'GENDER': d.cat['Gender'],
    'CARTYPE': d.cat['Current_Car_Type'],
    'PREFIX1': d.inc // 1000, 'PREFIX2': d.inc // 100,
    'SUFFIX': d.inc % 100, 'AGEDEC': d.age // 10,
    'CMT5': d.cmt10 // 50, 'STHB': d.sth // 3, 'STWB': d.stw // 3,
    'BALANCE': d.stw - d.sth, 'HALF_CMT5': (d.cmt10 + 25) // 50,
}
for name, values in base_values.items():
    BOOK.add(name, values)
CROSS_SPECS = []
for prefix, context_list, alphas in (
    ('PREFIX1', ['CITY', 'HOME', 'SUB', 'CONC', 'ANX', 'GENDER', 'CARTYPE', 'CARS'], (20.0,)),
    ('PREFIX2', ['CITY', 'SUB', 'HOME', 'CONC'], (20.0,)),
    ('PREFIX1', ['AGEDEC', 'CMT5', 'STHB', 'STWB'], (5.0, 50.0)),
    ('PREFIX2', ['ANX', 'GENDER', 'CARTYPE', 'CARS'], (5.0, 50.0)),
    ('SUFFIX', ['CITY', 'SUB', 'HOME', 'CONC'], (5.0, 50.0)),
    ('CMT_INT', ['CITY', 'HOME', 'SUB', 'ANX', 'CONC'], (20.0,)),
    ('CMT', ['HOME', 'CITY', 'ANX'], (20.0,)),
):
    for context in context_list:
        key = BOOK.pair(prefix + '_' + context, prefix, context)
        CROSS_SPECS.append((key, alphas))
PAIR_SPECS = [('STH_CMT5', 'STH', 'CMT5'), ('CARS_BALANCE', 'CARS', 'BALANCE'),
              ('STH_HALF_CMT5', 'STH', 'HALF_CMT5')]
PAIR_MAPS = {}
for name, a, b in PAIR_SPECS:
    BOOK.pair(name, a, b)
    _, first = np.unique(BOOK.codes[name], return_index=True)
    PAIR_MAPS[name] = BOOK.codes[a][first], BOOK.codes[b][first]
for name, a, b in [('GATE_CITY', 'GATE', 'CITY'), ('GATE_HOME', 'GATE', 'HOME'),
                   ('INC_HOME', 'INC', 'HOME'), ('INC_CITY', 'INC', 'CITY')]:
    BOOK.pair(name, a, b)

# Only the mixture table uses a dense, fixed 30-cell gate axis.
INC_GATE = BOOK.codes['INC'].astype(np.int64) * 30 + d.gate
ALL_GATE_COUNTS = np.bincount(INC_GATE, minlength=BOOK.sizes['INC'] * 30).reshape(-1, 30).astype(np.float64)
ADDITIVE_TERMS = [('GATE', 1, 5.0)]
for width in (1, 8, 64):
    key = BOOK.add(f'INC_RANK{width}', BOOK.codes['INC'] // width)
    ADDITIVE_TERMS.append((key, 2, 25.0))
for width in (1, 5):
    key = BOOK.add(f'CMT_RANK{width}', BOOK.codes['CMT'] // width)
    ADDITIVE_TERMS.append((key, 3, 25.0))
for key in ['AGE', 'CARS', 'STH', 'STW', 'GENDER', 'CITY', 'CARTYPE', 'HOME']:
    ADDITIVE_TERMS.append((key, 4, 5.0))
ADDITIVE_CODES = np.ascontiguousarray(np.vstack([BOOK.codes[k] for k, _, _ in ADDITIVE_TERMS]))
ADDITIVE_OFFSETS = np.r_[0, np.cumsum([BOOK.sizes[k] for k, _, _ in ADDITIVE_TERMS])].astype(np.int64)
ADDITIVE_COMPONENTS = np.array([c for _, c, _ in ADDITIVE_TERMS], np.int64)
ADDITIVE_PENALTIES = np.array([p for _, _, p in ADDITIVE_TERMS], np.float64)
CAT_FEATURES = {n for n in STATIC_NAMES if n in CAT_LEVELS or n.startswith('DIG_') or n == 'DONOR_CITY'}
TEST_ROWS = np.arange(d.n_train, d.n_all, dtype=np.int64)
STATIC_MANIFEST = [{'feature': n, 'family': 'static', 'fit_scope': scope, 'dtype': 'float32'}
                   for n, scope in zip(STATIC_NAMES, STATIC_SCOPES)]
assert np.isfinite(STATIC).all() and len(set(STATIC_NAMES)) == len(STATIC_NAMES)
TIMINGS.append(dict(stage='static', seconds=time.perf_counter() - static_started, rows=d.n_all))
print(f'Static features: {STATIC.shape} | seconds={TIMINGS[-1]["seconds"]:.2f}')


# ===== notebook cell 6 =====
class DonorState:
    def __init__(self, donors, labels, statistics=None):
        donors = np.asarray(donors, np.int64)
        if len(donors) == 0 or donors[0] < 0 or np.any(np.diff(donors) <= 0) or donors[-1] >= len(labels):
            raise ValueError('Donor rows must be unique, increasing training indices.')
        yfit = np.asarray(labels[donors], np.float64)
        if not np.isin(yfit, [0.0, 1.0]).all():
            raise ValueError('Donor labels must be binary.')
        self.mask = np.zeros(d.n_all, bool)
        self.mask[donors] = True
        self.blocks, self.names, self.manifest = [], [], []
        stats = BOOK.stats(donors, labels) if statistics is None else statistics
        prior = float((yfit.sum() + 0.5) / (len(yfit) + 1.0))
        for key, alphas in [('INC', (5., 20., 100.)),
                            ('INC10', (1., 2., 5., 10., 50., 200.)),
                            ('INC100', (1., 2., 5., 10., 50., 200.)),
                            ('INC1000', (1., 2., 5., 10., 50., 200.)),
                            ('CMT_INT', (20., 100.)), ('CMT', (20., 100.))]:
            n, s = stats[key]
            rates = np.column_stack([(s + a * prior) / (n + a) for a in alphas])
            self.add(key, safe_logit(rates), [f'TE_{key}_A{a:g}_LOGIT' for a in alphas], 'target encoding')
            self.add(key, np.log1p(n), [f'TE_{key}_N_LOG'], 'target encoding')

        ninc, sinc = stats['INC']
        for key, widths in [('INC', (1, 5, 25, 100, 250)), ('CMT', (1, 5, 25, 50))]:
            n, s = stats[key]
            for width in widths:
                nn, ss = box_sum(n, width) - n, box_sum(s, width) - s
                local = (ss + 20.0 * prior) / (nn + 20.0)
                self.add(key, safe_logit(local), [f'LOCAL_{key}_W{width}_LOGIT'], 'local target rates')
                if key == 'INC' and width in (5, 25, 100):
                    alphas = (2., 5., 10., 20., 50.)
                    rates = np.column_stack([(s + a * local) / (n + a) for a in alphas])
                    self.add(key, safe_logit(rates), [f'HIER_INC_W{width}_A{a:g}_LOGIT' for a in alphas], 'hierarchical rates')
                    surprise = (s - n * local) / np.sqrt(n * local * (1.0 - local) + 1.0)
                    self.add(key, surprise, [f'INC_SURPRISE_W{width}'], 'hierarchical rates')

        for key in ('PREFIX1', 'PREFIX2', 'SUFFIX'):
            n, s = stats[key]
            rates = np.column_stack([(s + a * prior) / (n + a) for a in (5., 50.)])
            self.add(key, safe_logit(rates), [f'{key}_A5_LOGIT', f'{key}_A50_LOGIT'], 'numeric prefix rates')
            self.add(key, np.log1p(n), [f'{key}_N_LOG'], 'numeric prefix rates')
        for strength in (5., 20., 80.):
            parent = np.array([prior])
            previous = None
            for key in ('PREFIX1', 'PREFIX2', 'INC'):
                n, s = stats[key]
                if previous is None:
                    local = prior
                else:
                    local = parent[PARENT_MAPS[(key, previous)]]
                parent = (s + strength * local) / (n + strength)
                self.add(key, safe_logit(parent), [f'PREFIX_HIER_{key}_S{strength:g}_LOGIT'], 'numeric prefix hierarchy')
                previous = key
        for key in ('PREFIX2', 'INC'):
            n, s = stats[key]
            sd = np.sqrt(1.0 / (s + 20.0 * prior) + 1.0 / (n - s + 20.0 * (1.0 - prior)))
            self.add(key, sd, [f'{key}_POSTERIOR_LOGIT_SD'], 'posterior uncertainty')
        for key, alphas in CROSS_SPECS:
            n, s = stats[key]
            rates = np.column_stack([(s + a * prior) / (n + a) for a in alphas])
            self.add(key, safe_logit(rates), [f'CROSS_{key}_A{a:g}_LOGIT' for a in alphas], 'cross target encoding')
        for key in ('GATE', 'GATE_CITY', 'GATE_HOME'):
            n, s = stats[key]
            self.add(key, np.column_stack([safe_logit((s + 10.0 * prior) / (n + 10.0)), np.log1p(n)]),
                     [f'SUPPORT_{key}_LOGIT', f'SUPPORT_{key}_N_LOG'], 'gate support approximation')

        for name, a, b in PAIR_SPECS:
            na, sa = stats[a]
            nb, sb = stats[b]
            n, s = stats[name]
            ca, cb = PAIR_MAPS[name]
            eta = safe_logit((sa + 100 * prior) / (na + 100))[ca]
            eta += safe_logit((sb + 100 * prior) / (nb + 100))[cb] - safe_logit(prior)
            rate = (s + 200 * expit(eta)) / (n + 200)
            delta = np.clip(safe_logit(rate) - eta, -1., 1.)
            if name != 'CARS_BALANCE':
                error = n / (n + 200) * np.sqrt(1 / (s + 0.5) + 1 / (n - s + 0.5))
                delta = np.sign(delta) * np.maximum(np.abs(delta) - error, 0)
            good = (n >= 100) & (s >= 2) & (n - s >= 2) & (na[ca] >= n + 100) & (nb[cb] >= n + 100)
            self.add(name, np.where(good, delta, 0), [f'PAIR_{name}_RESIDUAL'], 'pair deviations')

        c1 = np.bincount(INC_GATE[donors], weights=yfit,
                         minlength=BOOK.sizes['INC'] * 30).reshape(-1, 30)
        c0 = np.bincount(INC_GATE[donors], weights=1.0 - yfit,
                         minlength=BOOK.sizes['INC'] * 30).reshape(-1, 30)
        marginal0 = (c0.sum(0) + 0.5) / (c0.sum() + 15.0)
        marginal1 = (c1.sum(0) + 0.5) / (c1.sum() + 15.0)
        for width in (16, 64, 256):
            s0, s1 = smooth_counts(c0, width), smooth_counts(c1, width)
            q0 = (s0 + 30 * marginal0) / (s0.sum(1, keepdims=True) + 30)
            q1 = (s1 + 30 * marginal1) / (s1.sum(1, keepdims=True) + 30)
            mu = (s1.sum(1) + 20 * prior) / (s0.sum(1) + s1.sum(1) + 20)
            result = fit_mixture(ALL_GATE_COUNTS, q0, q1, mu, MIXTURE_STEPS)
            pi = result[:, 0]
            rates = np.column_stack([pi, (sinc + 20 * pi) / (ninc + 20),
                                    (sinc + 100 * pi) / (ninc + 100)])
            names = [f'FAST_MIX_W{width}_{part}_LOGIT' for part in ('PRIOR', 'POST20', 'POST100')]
            self.add('INC', np.column_stack([safe_logit(rates), result[:, 1:]]),
                     names + [f'FAST_MIX_W{width}_INFO', f'FAST_MIX_W{width}_GAIN'], 'fast gate mixture')
            row_rate = pi[:, None] * q1 / np.maximum(q0 + pi[:, None] * (q1 - q0), 1e-15)
            self.add('INC_GATE', safe_logit(row_rate.ravel()),
                     [f'FAST_MIX_W{width}_ROW_LOGIT'], 'fast gate mixture')
        del c0, c1, s0, s1, q0, q1, row_rate, result

        fitted = fit_additive(np.ascontiguousarray(ADDITIVE_CODES[:, donors]), yfit,
                             ADDITIVE_OFFSETS, ADDITIVE_PENALTIES, ADDITIVE_SWEEPS)
        self.intercept, self.beta, eta_fit, self.sweeps = fitted
        pfit = expit(eta_fit)
        residual, curvature = yfit - pfit, np.maximum(pfit * (1 - pfit), 1e-6)
        for key, widths in [('INC', (0, 5, 25, 100)), ('CMT', (0, 5)),
                            ('INC_HOME', (0,)), ('INC_CITY', (0,))]:
            code, size = BOOK.codes[key], BOOK.sizes[key]
            r = np.bincount(code[donors], weights=residual, minlength=size)
            h = np.bincount(code[donors], weights=curvature, minlength=size)
            for width in widths:
                nr = r if width == 0 else box_sum(r, width) - r
                nh = h if width == 0 else box_sum(h, width) - h
                self.add(key, np.clip(nr / (nh + 10.0), -4.0, 4.0),
                         [f'ADJUSTED_{key}_W{width}'], 'additive residual approximation')
        self.additive_names = ['ADDITIVE_GATE_LOGIT', 'ADDITIVE_INC', 'ADDITIVE_CMT',
                               'ADDITIVE_OTHER', 'ADDITIVE_LOGIT', 'ADDITIVE_PARTIAL_LOGIT']
        self.names += self.additive_names
        self.manifest += [{'feature': n, 'family': 'additive logistic approximation',
                           'fit_scope': 'inner donor labels; outer-training refit for validation/test',
                           'dtype': 'float32'} for n in self.additive_names]
        if len(set(self.names)) != len(self.names) or not np.isfinite(self.beta).all():
            raise AssertionError('Invalid feature schema or additive coefficients.')

    def add(self, key, values, names, family):
        values = np.asarray(values, np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[1] != len(names) or not np.isfinite(values).all():
            raise ValueError('Invalid lookup table: ' + str(names))
        code = INC_GATE if key == 'INC_GATE' else BOOK.codes[key]
        self.blocks.append((code, np.ascontiguousarray(values)))
        self.names.extend(names)
        self.manifest.extend({'feature': n, 'family': family,
            'fit_scope': 'inner donor labels; outer-training refit for validation/test',
            'dtype': 'float32'} for n in names)

    def transform(self, rows):
        rows = np.asarray(rows, np.int64)
        if np.any(rows < 0) or np.any(rows >= d.n_all):
            raise IndexError('Receiving row index is outside the data.')
        if self.mask[rows].any():
            raise AssertionError('Receiving rows overlap donor rows.')
        out = np.empty((len(rows), len(self.names)), np.float32)
        offset = 0
        for code, table in self.blocks:
            width = table.shape[1]
            out[:, offset:offset + width] = table[code[rows]]
            offset += width
        parts = gather_additive(ADDITIVE_CODES, rows, ADDITIVE_OFFSETS,
                                ADDITIVE_COMPONENTS, self.beta, self.intercept)
        out[:, offset] = parts[:, 0] + parts[:, 1]
        out[:, offset + 1:offset + 4] = parts[:, 2:5]
        out[:, offset + 4] = parts.sum(1)
        out[:, offset + 5] = parts[:, 0] + parts[:, 1] + parts[:, 4] + 0.25 * (parts[:, 2] + parts[:, 3])
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite supervised features.')
        return out


PARENT_MAPS = {}
for child, parent in [('PREFIX2', 'PREFIX1'), ('INC', 'PREFIX2')]:
    _, first = np.unique(BOOK.codes[child], return_index=True)
    PARENT_MAPS[(child, parent)] = BOOK.codes[parent][first]


# ===== notebook cell 7 =====
@contextmanager
def matrix_buffer(shape, dtype, directory, name):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    matrix = (np.lib.format.open_memmap(Path(directory) / (name + '.npy'),
              mode='w+', dtype=dtype, shape=shape)
              if nbytes > MATRIX_RAM_MB * 1024**2 else np.empty(shape, dtype=dtype))
    try:
        yield matrix
    finally:
        if isinstance(matrix, np.memmap):
            matrix.flush()
            matrix._mmap.close()


def fill_features(matrix, positions, rows, state):
    ns = len(STATIC_NAMES)
    for lo in range(0, len(rows), BATCH_ROWS):
        hi = min(lo + BATCH_ROWS, len(rows))
        rr = rows[lo:hi]
        block = np.empty((len(rr), matrix.shape[1]), np.float32)
        block[:, :ns] = STATIC[rr]
        block[:, ns:] = state.transform(rr)
        matrix[positions[lo:hi]] = block


FEATURE_NAMES, FEATURE_MANIFEST = None, None


@contextmanager
def fold_features(fold):
    global FEATURE_NAMES, FEATURE_MANIFEST
    started = time.perf_counter()
    tr, va = SPLITS[fold]
    apply = np.r_[va, TEST_ROWS]
    total = BOOK.stats(tr, Y)
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=1000 + fold)
    received = np.zeros(len(tr), np.uint8)
    with ExitStack() as stack:
        directory = stack.enter_context(tempfile.TemporaryDirectory(prefix=f'{EXP_NAME}_f{fold}_'))
        xtr, xap = None, None
        for inner_fold, (donor_pos, receiving_pos) in enumerate(inner.split(np.zeros(len(tr)), Y[tr])):
            donors, receiving = tr[donor_pos], tr[receiving_pos]
            state_started = time.perf_counter()
            stats = subtract_stats(total, BOOK.stats(receiving, Y))
            state = DonorState(donors, Y, stats)
            names = STATIC_NAMES + state.names
            if FEATURE_NAMES is None:
                FEATURE_NAMES = names
                FEATURE_MANIFEST = STATIC_MANIFEST + state.manifest
            if names != FEATURE_NAMES:
                raise AssertionError('Feature order changed between donor partitions.')
            if xtr is None:
                xtr = stack.enter_context(matrix_buffer((len(tr), len(names)), np.float32, directory, 'raw_train'))
                xap = stack.enter_context(matrix_buffer((len(apply), len(names)), np.float32, directory, 'raw_apply'))
            fill_features(xtr, receiving_pos, receiving, state)
            received[receiving_pos] += 1
            TIMINGS.append(dict(stage='inner_features', fold=fold, inner=inner_fold,
                                seconds=time.perf_counter() - state_started,
                                donor_rows=len(donors), receiving_rows=len(receiving), sweeps=state.sweeps))
            del state, stats
        if not np.all(received == 1):
            raise AssertionError('Inner receiving coverage must equal one.')
        state_started = time.perf_counter()
        state = DonorState(tr, Y, total)
        if STATIC_NAMES + state.names != FEATURE_NAMES:
            raise AssertionError('Outer feature schema changed.')
        fill_features(xap, np.arange(len(apply)), apply, state)
        TIMINGS.append(dict(stage='outer_features', fold=fold,
                            seconds=time.perf_counter() - state_started,
                            donor_rows=len(tr), receiving_rows=len(apply), sweeps=state.sweeps))
        del state, total
        TIMINGS.append(dict(stage='fold_features_total', fold=fold,
                            seconds=time.perf_counter() - started,
                            features=len(FEATURE_NAMES), donor_states=INNER_FOLDS + 1))
        print(f'FOLD {fold + 1}/{N_FOLDS} | features={len(FEATURE_NAMES)} | '
              f'preprocessing={TIMINGS[-1]["seconds"]:.2f}s', flush=True)
        yield xtr, xap[:len(va)], xap[len(va):]
    gc.collect()


class FoldPreprocessor:
    def fit(self, raw):
        self.cat_indices = [i for i, n in enumerate(FEATURE_NAMES) if n in CAT_FEATURES]
        self.num_indices = [i for i in range(len(FEATURE_NAMES)) if i not in self.cat_indices]
        self.levels = {i: np.unique(raw[:, i])[1:] for i in self.cat_indices}
        self.all_names = [FEATURE_NAMES[i] for i in self.num_indices]
        self.all_parents = list(self.all_names)
        for i, levels in self.levels.items():
            self.all_names += [f'{FEATURE_NAMES[i]}=={v:g}' for v in levels]
            self.all_parents += [FEATURE_NAMES[i]] * len(levels)
        scaler = StandardScaler()
        for lo in range(0, len(raw), BATCH_ROWS):
            scaler.partial_fit(self.encode(raw[lo:lo + BATCH_ROWS]))
        self.keep = as_host_numpy(scaler.var_, np.float64) > 1e-12
        self.mean = as_host_numpy(scaler.mean_, np.float64)[self.keep].copy()
        self.scale = as_host_numpy(scaler.scale_, np.float64)[self.keep].copy()
        self.names = [n for n, k in zip(self.all_names, self.keep) if k]
        self.parents = [n for n, k in zip(self.all_parents, self.keep) if k]
        if not len(self.names):
            raise ValueError('No nonconstant model features.')
        return self

    def encode(self, raw):
        out = np.empty((len(raw), len(self.all_names)), np.float64)
        out[:, :len(self.num_indices)] = raw[:, self.num_indices]
        offset = len(self.num_indices)
        for i, levels in self.levels.items():
            for value in levels:
                out[:, offset] = raw[:, i] == value
                offset += 1
        return out

    def transform(self, raw):
        result = np.ascontiguousarray(self.encode(raw)[:, self.keep])
        result -= self.mean
        result /= self.scale
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite scaled features.')
        return result


@contextmanager
def training_matrix(raw, preprocessor):
    with tempfile.TemporaryDirectory(prefix=f'{EXP_NAME}_scaled_') as directory:
        with matrix_buffer((len(raw), len(preprocessor.names)), np.float64, directory, 'scaled') as matrix:
            for lo in range(0, len(raw), BATCH_ROWS):
                matrix[lo:lo + BATCH_ROWS] = preprocessor.transform(raw[lo:lo + BATCH_ROWS])
            yield matrix


def predict_positive(model, raw, preprocessor):
    if not np.array_equal(as_host_numpy(model.classes_), [0, 1]):
        raise ValueError('Unexpected probability class order.')
    result = np.empty(len(raw), np.float64)
    for lo in range(0, len(raw), BATCH_ROWS):
        result[lo:lo + BATCH_ROWS] = as_host_numpy(
            model.predict_proba(preprocessor.transform(raw[lo:lo + BATCH_ROWS])), np.float64)[:, 1]
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError('Invalid positive-class probabilities.')
    return result


# ===== notebook cell 10 (CV loop) =====
def main():
    oof = np.full(len(Y), np.nan, np.float64)
    test_sum = np.zeros(len(DATA_TEST), np.float64)
    with threadpool_limits(N_THREADS):
        for fold, (tr, va) in enumerate(SPLITS):
            fold_started = time.perf_counter()
            with fold_features(fold) as (raw_train, raw_valid, raw_test):
                model = preprocessor = None
                try:
                    preprocessor = FoldPreprocessor().fit(raw_train)
                    model = LogisticRegression(**LOGREG_PARAMS)
                    with training_matrix(raw_train, preprocessor) as matrix:
                        model.fit(matrix, Y[tr])
                    iterations = int(as_host_numpy(model.n_iter_).max())
                    oof[va] = predict_positive(model, raw_valid, preprocessor)
                    test_sum += predict_positive(model, raw_test, preprocessor)
                    auc = float(roc_auc_score(Y[va], oof[va]))
                    print(f'FOLD {fold + 1}/{N_FOLDS} | AUC={auc:.9f} | iterations={iterations} | '
                          f'seconds={time.perf_counter() - fold_started:.2f}', flush=True)
                finally:
                    del model, preprocessor
                    gc.collect()
            del raw_train, raw_valid, raw_test
    test_pred = test_sum / N_FOLDS
    for values in (oof, test_pred):
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise AssertionError('Invalid final probabilities.')
    save_pair(HERE, MODEL_NAME, oof, test_pred)
    print(f'{MODEL_NAME}: OOF AUC {roc_auc_score(Y, oof):.6f}')


if __name__ == '__main__':
    main()
