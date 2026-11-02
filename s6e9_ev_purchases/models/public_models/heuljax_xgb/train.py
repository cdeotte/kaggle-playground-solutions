#!/usr/bin/env python
"""Paul Bryan Elefante's (@heuljax) public notebook "KPS6E09 XGB Sample", run on our 5 folds.
https://www.kaggle.com/code/heuljax/kps6e09-xgb-sample

Model (his): one XGBoost booster per fold. hist, binary:logistic, lr 0.015, max_depth 5, max_bin 256, colsample_bynode 0.8,
subsample 1, min_child_weight 1, reg_alpha 10, reg_lambda 15, base_score 0.5, +1 monotone constraints on the 78
target-rate features, categorical splits on the 6 categoricals + 10 digit columns, base_margin = the donor GAM margin,
seed = seed + fold. Up to 6000 rounds with patience 750 on AUC.

Features (his): 173 columns. Raw values, original-data income priors, income / commute digits, and a DonorState of
label-derived maps: exact / local / hierarchical income and commute TEs, multi-scale floor TEs (MSTE), worry-key TE,
ridge-Newton GAM coordinates + base margin, commute GAM residual, income-group XGB prior (GXP) + posteriors / uncertainty,
donor latent income moments, Gaussian-smoothed gate mixtures, and composition-curve inversion. The outer-train rows get
their features from 5 inner stratified folds (donor = the other 4/5); valid / test rows from a DonorState fitted on all
outer-train rows.

How it runs here:
  * outer folds: our StratifiedKFold(5, shuffle=True, random_state=42) (the notebook used its own 10 folds);
  * round count: a stratified 10% of the outer-train rows (seed --seed) is held out, the features are rebuilt on the
    other 90% (inner 5-fold cross-fit for those rows, donor = all of them for the holdout), early stopping (patience 750,
    at most 6000 rounds) runs on the holdout; then the features are rebuilt on all outer-train rows and the model is refit
    on them for round(1.1 * best rounds) rounds;
  * the original data is deduplicated against the competition predictors (label-free) once in prepare(); its rows are
    only used for the income priors;
  * every XGBoost model, including the small income-group regressor, trains on --device with --threads; the group
    regressor predicts its ~13k income groups on CPU.
Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4] [--seed 42] [--threads 16] [--device cuda]
       -> preds/heuljax_xgb_{oof,test}.npy (finished folds are kept in parts/ and skipped)"""
from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import numba
import numpy as np
import pandas as pd
import xgboost as xgb
from numba import njit, prange
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from common.data import folds, read_original, read_test, read_train, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

NAME = 'heuljax_xgb'

# ---------------------------------------------------------------- notebook cell 1 (constants)
TARGET = 'Will_Buy_EV'
FLOAT_COLS = ['Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level']
INT_COLS = ['Age', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
            'Charging_Stations_Near_Work']
CAT_COLS = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
            'Subsidy_Available', 'Range_Anxiety_Level']
RAW_COLS = FLOAT_COLS + INT_COLS + CAT_COLS
NUM_COLS = FLOAT_COLS + INT_COLS
CATEGORY_VOCAB = {
    'Gender': ['Male', 'Female', 'Other'],
    'City_Type': ['Suburban', 'Rural', 'Urban'],
    'Current_Car_Type': ['Sedan', 'SUV', 'Hatchback', 'Truck'],
    'Home_Charging_Possible': ['Yes', 'No'],
    'Subsidy_Available': ['No', 'Yes'],
    'Range_Anxiety_Level': ['Low', 'Medium', 'High'],
}
INNER_FOLDS = 5
N_ESTIMATORS, EARLY_STOPPING_ROUNDS = 6000, 750
GROUP_ESTIMATORS = 600
GAM_ITERS = 12
GAM_STEP = 0.5
CHUNK_ROWS = 32768
ES_FRACTION = 0.10
REFIT_MULT = 1.1
USE_RATE_MONOTONICITY = True
INCOME_ALPHAS, COMMUTE_ALPHAS = (5., 20., 50.), (20., 100.)
INCOME_WIDTHS, COMMUTE_WIDTHS = (1, 2, 5, 10, 25, 50, 100, 250), (1, 2, 5, 10, 25, 50)
HIER_WIDTHS, HIER_ALPHAS = (5, 25, 100), (2., 5., 10., 20., 50.)
DETAIL_PAIRS = ((1, 5), (5, 25), (25, 100), (100, 250))
LATENT_ALPHAS = (2., 5., 10., 20., 50., 100.)
MSTE_ALPHAS = (1., 2., 5., 10., 50., 200.)
MIX_WIDTHS = (16, 64, 256)
SHIFT_GRID = np.linspace(-16., 16., 129)
OTHER_COLS = ['Age', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
              'Charging_Stations_Near_Work', 'Gender', 'City_Type',
              'Current_Car_Type', 'Home_Charging_Possible']

XGB_PARAMS = dict(objective='binary:logistic', eval_metric='auc', base_score=0.5,
                  tree_method='hist', learning_rate=0.015,
                  max_depth=5, max_bin=256, colsample_bynode=0.8,
                  colsample_bytree=1.0, subsample=1.0, min_child_weight=1.0,
                  reg_alpha=10.0, reg_lambda=15.0)                  # + device / nthread / seed per fold
GROUP_PARAMS = dict(objective='reg:logistic', tree_method='hist',
                    learning_rate=0.03, max_depth=3, min_child_weight=20.0,
                    subsample=0.8, colsample_bynode=0.8, reg_alpha=2.0,
                    reg_lambda=100.0, n_estimators=GROUP_ESTIMATORS)  # + device / n_jobs / random_state

# ---------------------------------------------------------------- notebook cell 2 (feature layout)
SOURCE_COLS = ['ORIG_INC_Y', 'ORIG_INC_COUNT_LOG', 'ORIG_INC_SEEN', 'ORIG_INC_LOCAL_Y']
BASE_FOLD_COLS = (
    [f'TE_INC_A{a:g}' for a in INCOME_ALPHAS]
    + [f'TE_CMT_A{a:g}' for a in COMMUTE_ALPHAS]
    + ['SUPPORT_INC_LOG', 'SUPPORT_CMT_LOG']
    + [f'NBR_INC_W{w}' for w in INCOME_WIDTHS]
    + [f'NBR_DETAIL_W{a}_{b}' for a, b in DETAIL_PAIRS]
    + [f'HIER_INC_W{w}_A{a:g}' for w in HIER_WIDTHS for a in HIER_ALPHAS]
    + [f'INC_SURPRISE_W{w}' for w in HIER_WIDTHS]
    + [f'NBR_CMT_W{w}' for w in COMMUTE_WIDTHS]
    + ['DCMT_LOCAL_W5', 'P_DCMT_LOCAL_W5'])
GXP_COLS = ['GXP_D3_R100'] + [f'GXP_D3_R100_A{a:g}' for a in (2., 5., 10., 20., 50.)]
LATENT_NAMES = ['CONCERN', 'SUBSIDY', 'ANXIETY_PENALTY', 'BUY_SCORE']
LAT_COLS = [f'LAT_INC_{n}_A{a:g}' for n in LATENT_NAMES for a in LATENT_ALPHAS]
LAT_COLS += ['LAT_INC_CONCERN_DEV', 'LAT_INC_SUBSIDY_DEV', 'LAT_INC_ANXIETY_DEV', 'LAT_INC_SCORE_DEV']
DIGIT_COLS = [f'DIG_INC_10E{k}' for k in range(6)] + [f'DIG_CMT10_10E{k}' for k in range(4)]
WORRY_COLS = ['WTE05_A5', 'WHIER05_W25_A20']
UNCERTAINTY_COLS = ['GXP_POST_SD_A20', 'GXP_POST_LO_A20', 'GXP_POST_HI_A20']
MIX_COLS = [f'MIX_W{w}_{s}' for w in MIX_WIDTHS
            for s in ('PRIOR', 'POST20', 'POST100', 'ROW_PROB', 'INFO', 'GAIN')]
CHANNELS = ['GXP_D3_R100', 'GXP_D3_R100_A20', 'GXP_D3_R100_A50']
CHANNELS += [f'MIX_W{w}_POST100' for w in MIX_WIDTHS]
COMPOSITION_COLS = [f'CSHIFT_{c}' for c in CHANNELS] + [f'CCORR_{c}' for c in CHANNELS]
COORD_COLS = ['GAM_GATE_COORD', 'GAM_INCOME_COORD', 'GAM_COMMUTE_COORD', 'GAM_OTHER_COORD']
MSTE_KEYS = ['INC10', 'INC100', 'INC1K', 'CMTINT']
MSTE_COLS = [c for key in MSTE_KEYS
             for c in ([f'MSTE_{key}_A{a:g}' for a in MSTE_ALPHAS] + [f'MSTE_{key}_LOGN'])]
FEATURE_COLS = (RAW_COLS + SOURCE_COLS + BASE_FOLD_COLS + GXP_COLS + LAT_COLS + DIGIT_COLS
                + WORRY_COLS + UNCERTAINTY_COLS + MIX_COLS + COMPOSITION_COLS + COORD_COLS + MSTE_COLS)
assert len(FEATURE_COLS) == len(set(FEATURE_COLS)) == 173
assert not {TARGET, 'id', 'Buyer_ID'}.intersection(FEATURE_COLS)
COL = {name: i for i, name in enumerate(FEATURE_COLS)}
FEATURE_TYPES = ['c' if c in CAT_COLS + DIGIT_COLS else 'q' for c in FEATURE_COLS]
RATE_COLS = [c for c in FEATURE_COLS if
             c in ('ORIG_INC_Y', 'ORIG_INC_LOCAL_Y', 'WTE05_A5', 'WHIER05_W25_A20',
                   'GXP_POST_LO_A20', 'GXP_POST_HI_A20')
             or c.startswith(('TE_INC_', 'TE_CMT_', 'NBR_INC_', 'NBR_CMT_', 'HIER_INC_', 'GXP_D3_R100_A'))
             or (c.startswith('MIX_') and any(s in c for s in ('_PRIOR', '_POST')))
             or (c.startswith('MSTE_') and '_A' in c)]
if USE_RATE_MONOTONICITY:
    XGB_PARAMS['monotone_constraints'] = '(' + ','.join('1' if c in RATE_COLS else '0' for c in FEATURE_COLS) + ')'


# ---------------------------------------------------------------- notebook cell 3 (kernels / tables), verbatim
def box_sum(values, width):
    values = np.asarray(values, dtype=np.float64)
    prefix = np.concatenate([np.zeros((1,) + values.shape[1:]), np.cumsum(values, axis=0)], axis=0)
    center = np.arange(len(values))
    return prefix[np.minimum(center + width + 1, len(values))] - prefix[np.maximum(center - width, 0)]


GAUSSIAN_BACKENDS = {16: 'direct', 64: 'fft', 256: 'fft'}


def gaussian_smooth(values, width, backend=None):
    backend = GAUSSIAN_BACKENDS[int(width)] if backend is None else backend
    if backend == 'direct':
        return gaussian_filter1d(values, width, axis=0, mode='constant')
    radius = int(4.0 * width + 0.5)
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (positions / width)**2)
    kernel /= kernel.sum()
    return fftconvolve(values, kernel[:, None], mode='same', axes=0)


@njit(cache=False)
def gam_kernel(codes, sizes, ridges, widths, targets, iterations, step):
    n, nt = codes.shape
    prior = min(max(np.mean(targets), 1e-6), 1.0 - 1e-6)
    intercept = math.log(prior / (1.0 - prior))
    eta = np.full(n, intercept, dtype=np.float64)
    tables = np.zeros((nt, np.max(sizes)), dtype=np.float64)
    for epoch in range(iterations):
        for term in range(nt):
            length = sizes[term]
            grad = np.zeros(length, dtype=np.float64)
            hess = np.zeros(length, dtype=np.float64)
            for i in range(n):
                e = max(min(eta[i], 700.0), -700.0)
                p = 1.0 / (1.0 + math.exp(-e))
                code = codes[i, term]
                grad[code] += targets[i] - p
                hess[code] += p * (1.0 - p)
            width = widths[term]
            if width:
                pg = np.zeros(length + 1, dtype=np.float64)
                ph = np.zeros(length + 1, dtype=np.float64)
                for j in range(length):
                    pg[j + 1] = pg[j] + grad[j]
                    ph[j + 1] = ph[j] + hess[j]
                for j in range(length):
                    lo, hi = max(0, j - width), min(length, j + width + 1)
                    grad[j] = pg[hi] - pg[lo]
                    hess[j] = ph[hi] - ph[lo]
            delta = step * grad / (hess + ridges[term])
            for j in range(length):
                tables[term, j] += delta[j]
            for i in range(n):
                eta[i] += delta[codes[i, term]]
    return intercept, tables, eta


@njit(cache=False, parallel=True)
def mixture_kernel(counts, q0, q1, mu):
    ng, nk = counts.shape
    pi = np.empty(ng, dtype=np.float64)
    information = np.empty(ng, dtype=np.float64)
    gain = np.empty(ng, dtype=np.float64)
    for i in prange(ng):
        p = mu[i]
        for iteration in range(24):
            g = 10.0 * (mu[i] / p - (1.0 - mu[i]) / (1.0 - p))
            h = 10.0 * (mu[i] / (p * p) + (1.0 - mu[i]) / ((1.0 - p) * (1.0 - p)))
            for j in range(nk):
                d = q1[i, j] - q0[i, j]
                m = max(q0[i, j] + p * d, 1e-300)
                r = d / m
                g += counts[i, j] * r
                h += counts[i, j] * r * r
            step = max(-0.15, min(0.15, g / max(h, 1e-8)))
            p = max(1e-4, min(1.0 - 1e-4, p + step))
        pi[i] = p
        ii, gg, total = 0.0, 0.0, 0.0
        for j in range(nk):
            d = q1[i, j] - q0[i, j]
            m = max(q0[i, j] + p * d, 1e-300)
            old = max(q0[i, j] + mu[i] * d, 1e-300)
            ii += counts[i, j] * (d / m)**2
            gg += counts[i, j] * math.log(m / old)
            total += counts[i, j]
        information[i] = math.log1p(ii)
        gain[i] = gg / max(total, 1.0)
    return pi, information, gain


@njit(cache=False, parallel=True)
def composition_curve_kernel(nuisance_sorted, boundaries, grid_start, grid_step, n_grid):
    ng = len(boundaries) - 1
    curve = np.zeros((ng, n_grid), dtype=np.float64)
    ratio = math.exp(-grid_step)
    for group in prange(ng):
        lo, hi = boundaries[group], boundaries[group + 1]
        for i in range(lo, hi):
            v = math.exp(min(700.0, max(-700.0, -nuisance_sorted[i] - grid_start)))
            for k in range(n_grid):
                curve[group, k] += 1.0 / (1.0 + v)
                v *= ratio
        for k in range(n_grid):
            curve[group, k] /= max(hi - lo, 1)
    return curve


@njit(cache=False, parallel=True)
def invert_curve_kernel(curves, indices, probabilities, grid):
    n, nc = probabilities.shape
    out = np.empty((n, nc), dtype=np.float64)
    for i in prange(n):
        idx = indices[i]
        for j in range(nc):
            p = probabilities[i, j]
            if p <= curves[idx, 0]:
                out[i, j] = grid[0]
            elif p >= curves[idx, -1]:
                out[i, j] = grid[-1]
            else:
                lo, hi = 0, len(grid) - 1
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if curves[idx, mid] < p:
                        lo = mid
                    else:
                        hi = mid
                denom = curves[idx, hi] - curves[idx, lo]
                frac = (p - curves[idx, lo]) / denom if denom > 1e-14 else 0.0
                out[i, j] = grid[lo] + frac * (grid[hi] - grid[lo])
    return out


class ValueAxis:
    def __init__(self, values):
        self.values, codes = np.unique(np.asarray(values), return_inverse=True)
        self.codes = codes.astype(np.int32)
        self.size = len(self.values)
        if self.size == 0:
            raise ValueError('Empty donor value axis')

    def locate(self, query):
        query = np.asarray(query)
        pos = np.searchsorted(self.values, query)
        hi = np.minimum(pos, self.size - 1)
        lo = np.maximum(pos - 1, 0)
        exact = (pos < self.size) & (self.values[hi] == query)
        nearest = np.where(np.abs(query - self.values[lo]) <= np.abs(query - self.values[hi]), lo, hi)
        return np.where(exact, hi, nearest).astype(np.int32), exact


class TargetTable:
    def __init__(self, values, targets):
        self.axis = ValueAxis(values)
        self.count = np.bincount(self.axis.codes, minlength=self.axis.size).astype(np.float64)
        self.total = np.bincount(self.axis.codes, weights=targets, minlength=self.axis.size)
        self.prior = float(np.mean(targets))
        self.local_cache = {}

    def lookup(self, query):
        idx, exact = self.axis.locate(query)
        return idx, exact, self.count[idx] * exact, self.total[idx] * exact

    def local(self, width, alpha=20.0):
        key = (width, float(alpha))
        if key not in self.local_cache:
            count = np.maximum(box_sum(self.count, width) - self.count, 0.0)
            total = box_sum(self.total, width) - self.total
            self.local_cache[key] = (total + alpha * self.prior) / (count + alpha)
        return self.local_cache[key]

    def rate(self, query, alpha):
        _, _, count, total = self.lookup(query)
        return (total + alpha * self.prior) / (count + alpha)


# ---------------------------------------------------------------- notebook cell 4 (raw packing, keys, original priors)
def feature_hashes(frame):
    normalized = frame[RAW_COLS].copy()
    for c in NUM_COLS:
        normalized[c] = pd.to_numeric(normalized[c], errors='coerce').astype(np.float64)
    for c in CAT_COLS:
        normalized[c] = normalized[c].astype('string').fillna('__MISSING__')
    return pd.util.hash_pandas_object(normalized, index=False).to_numpy(np.uint64)


def pack_raw(frame):
    out = {}
    for name in NUM_COLS:
        out[name] = pd.to_numeric(frame[name], errors='coerce').to_numpy(np.float64)
    for name in CAT_COLS:
        categories = CATEGORY_VOCAB[name]
        codes = pd.Categorical(frame[name].astype('string'), categories=categories).codes.astype(np.int16)
        out[name] = np.where(codes < 0, len(categories), codes).astype(np.int16)
    return out


def take_raw(raw, rows):
    return {k: np.ascontiguousarray(v[rows]) for k, v in raw.items()}


def fit_medians(raw):
    medians = {c: float(np.nanmedian(np.where(np.isfinite(raw[c]), raw[c], np.nan))) for c in NUM_COLS}
    if not all(np.isfinite(v) for v in medians.values()):
        raise ValueError('Entirely missing numeric donor column')
    return medians


def prepare_raw(raw, medians):
    result = {c: np.where(np.isfinite(raw[c]), raw[c], medians[c]).astype(np.float64, copy=False)
              for c in NUM_COLS}
    result.update({c: raw[c] for c in CAT_COLS})
    return result


def latent_values(raw):
    income = raw['Annual_Income_USD']
    concern = raw['Environmental_Concern_Level']
    subsidy = (raw['Subsidy_Available'] == CATEGORY_VOCAB['Subsidy_Available'].index('Yes')).astype(np.float64)
    anxiety = raw['Range_Anxiety_Level']
    penalty = np.where(anxiety == 1, 1.0, np.where(anxiety == 2, 3.0, 0.0))
    score = 1.2 * income / 100000.0 + 0.6 * concern + 2.0 * subsidy - penalty
    return np.column_stack([concern, subsidy, penalty, score])


def gate_codes(raw):
    concern = np.rint(raw['Environmental_Concern_Level']).astype(np.int32) - 1
    anxiety = raw['Range_Anxiety_Level'].astype(np.int32)
    subsidy = (raw['Subsidy_Available'] == 1).astype(np.int32)
    valid = (concern >= 0) & (concern < 5) & (anxiety < 3) & (raw['Subsidy_Available'] < 2)
    return np.where(valid, concern * 6 + subsidy * 3 + anxiety, 30).astype(np.int32)


def worry_key(raw):
    home = (raw['Home_Charging_Possible'] == 0).astype(np.float64)
    value = (raw['Daily_Commute_km'] - 5.0 * raw['Charging_Stations_Near_Home']
             - 5.0 * raw['Charging_Stations_Near_Work'] - 150.0 * home)
    return np.rint(value / 0.5).astype(np.int64)


def mste_keys(raw):
    income, commute = raw['Annual_Income_USD'], raw['Daily_Commute_km']
    return dict(INC10=np.floor(income / 10), INC100=np.floor(income / 100),
                INC1K=np.floor(income / 1000), CMTINT=np.floor(commute))


class SourceIncome:
    """Original-dataset (external, labelled) income priors: exact-income mean, log count, seen flag, +-25 local mean."""

    def __init__(self, source):
        if source is None:
            self.enabled = False
            self.prior = 0.5
            return
        self.enabled = True
        yy = source[TARGET].eq('Yes').to_numpy(np.float64)
        self.prior = float(yy.mean())
        vals = pd.to_numeric(source['Annual_Income_USD'], errors='coerce').to_numpy(np.float64)
        good = np.isfinite(vals)
        if not good.any():
            raise ValueError('Original income is entirely missing')
        self.table = TargetTable(vals[good], yy[good])
        self.mean = self.table.total / self.table.count
        self.local = ((box_sum(self.table.total, 25) + 20.0 * self.prior)
                      / (box_sum(self.table.count, 25) + 20.0))

    def transform(self, income):
        if not self.enabled:
            return np.column_stack([np.full(len(income), 0.5), np.zeros(len(income)),
                                    np.zeros(len(income)), np.full(len(income), 0.5)])
        idx, exact, count, total = self.table.lookup(income)
        mean = np.where(exact, self.mean[idx], self.prior)
        return np.column_stack([mean, np.log1p(count), exact.astype(np.float64), self.local[idx]])


# ---------------------------------------------------------------- notebook cell 5 (donor state), verbatim except that
# GAM_ITERS / GROUP_PARAMS come from `opts` (so device / threads follow the command line)
class AdditiveState:
    def __init__(self, raw, targets, axes, gam_iters, main_only=False):
        self.axes = axes
        self.terms = [('gate', 0, 5.0, 0)]
        if not main_only:
            self.terms += [('income', w, 25.0, 1) for w in (0, 2, 8, 32, 128)]
            self.terms += [('commute', w, 25.0, 2) for w in (0, 5)]
        self.terms += [(c, 0, 5.0, 3) for c in OTHER_COLS]
        arrays, sizes = [], []
        for key, width, ridge, component in self.terms:
            if key == 'gate':
                arrays.append(gate_codes(raw))
                sizes.append(31)
            else:
                arrays.append(axes[key].codes)
                sizes.append(axes[key].size)
        codes = np.ascontiguousarray(np.column_stack(arrays), dtype=np.int32)
        self.intercept, self.tables, self.fitted_eta = gam_kernel(
            codes, np.asarray(sizes, np.int32),
            np.asarray([t[2] for t in self.terms], np.float64),
            np.asarray([t[1] for t in self.terms], np.int32),
            np.asarray(targets, np.float64), gam_iters, GAM_STEP)

    def components(self, raw):
        n = len(raw['Age'])
        result = np.zeros((n, 4), dtype=np.float64)
        result[:, 3] = self.intercept
        located = {}
        for t, (key, width, ridge, component) in enumerate(self.terms):
            if key not in located:
                if key == 'gate':
                    located[key] = (gate_codes(raw), np.ones(n, dtype=bool))
                else:
                    column = {'income': 'Annual_Income_USD', 'commute': 'Daily_Commute_km'}.get(key, key)
                    located[key] = self.axes[key].locate(raw[column])
            idx, exact = located[key]
            effect = self.tables[t, idx]
            result[:, component] += effect if width else effect * exact
        return result


class IncomeGroupPrior:
    def __init__(self, raw, targets, income_table, source, seed, group_params):
        axis, counts, total = income_table.axis, income_table.count, income_table.total
        latent = latent_values(raw)
        score = latent[:, 3]
        commute = raw['Daily_Commute_km']
        values = [latent[:, 0], latent[:, 1], latent[:, 2], score, score**2,
                  expit(score - 5.5), expit(2.0 * (score - 5.5)), expit(3.0 * (score - 5.5)),
                  (raw['Home_Charging_Possible'] == 0).astype(np.float64), commute, commute**2,
                  raw['Charging_Stations_Near_Home'], raw['Charging_Stations_Near_Work'],
                  raw['Age'], raw['Number_of_Cars_Owned']]
        values += [(score >= threshold).astype(np.float64) for threshold in (4.5, 5.0, 5.5, 6.0, 6.5)]
        for column in ('City_Type', 'Current_Car_Type', 'Gender'):
            values += [(raw[column] == CATEGORY_VOCAB[column].index(category)).astype(np.float64)
                       for category in sorted(CATEGORY_VOCAB[column])]
        base = [axis.values / 100000.0, np.log1p(counts)]
        base += [np.bincount(axis.codes, weights=v, minlength=axis.size) / counts for v in values]
        base += list(source.transform(axis.values).T)
        matrix = np.column_stack(base).astype(np.float64)
        self.template = matrix.mean(axis=0)
        self.mean = self.template.copy()
        self.scale = matrix.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        matrix = np.ascontiguousarray((matrix - self.mean) / self.scale)
        parameters = dict(group_params, random_state=seed)
        self.model = xgb.XGBRegressor(**parameters)
        self.model.fit(matrix, total / counts, sample_weight=np.sqrt(counts), verbose=False)
        if parameters.get('device', 'cpu') != 'cpu':
            self.model.set_params(device='cpu')   # tiny group-level predictions on CPU (avoids device-mismatch copies)
        self.prediction = np.clip(self.model.predict(matrix), 1e-5, 1.0 - 1e-5).astype(np.float64)
        self.source = source

    def transform(self, income, idx, exact):
        prior = self.prediction[idx].copy()
        if not exact.all():
            unseen, reverse = np.unique(income[~exact], return_inverse=True)
            matrix = np.tile(self.template, (len(unseen), 1))
            matrix[:, 0] = unseen / 100000.0
            matrix[:, 1] = 0.0
            matrix[:, -4:] = self.source.transform(unseen)
            scaled = np.ascontiguousarray((matrix - self.mean) / self.scale)
            prior[~exact] = np.clip(self.model.predict(scaled), 1e-5, 1.0 - 1e-5)[reverse]
        return prior


class DonorState:
    """All label-derived maps, fitted on donor rows (raw predictors + labels of the donor rows only)."""

    def __init__(self, raw, targets, source, seed, opts):
        targets = np.asarray(targets, dtype=np.float64)
        if len(raw['Age']) != len(targets) or len(targets) == 0 or len(np.unique(targets)) != 2:
            raise ValueError('Donor must contain aligned data and both target classes')
        self.medians = fit_medians(raw)
        raw = prepare_raw(raw, self.medians)
        self.source = source
        self.prior = float(targets.mean())
        self.income = TargetTable(raw['Annual_Income_USD'], targets)
        self.commute = TargetTable(raw['Daily_Commute_km'], targets)
        self.worry = TargetTable(worry_key(raw), targets)
        self.mste = {key: TargetTable(value, targets) for key, value in mste_keys(raw).items()}
        axes = {'income': self.income.axis, 'commute': self.commute.axis}
        axes.update({c: ValueAxis(raw[c]) for c in OTHER_COLS})
        self.gam = AdditiveState(raw, targets, axes, opts['gam_iters'])
        main = AdditiveState(raw, targets, axes, opts['gam_iters'], main_only=True)
        main_probability = expit(main.fitted_eta)
        cg = self.commute.axis.codes
        gradient = np.bincount(cg, weights=targets - main_probability, minlength=self.commute.axis.size)
        hessian = np.bincount(cg, weights=main_probability * (1.0 - main_probability), minlength=self.commute.axis.size)
        self.commute_effect = box_sum(gradient, 5) / (box_sum(hessian, 5) + 20.0)
        self.main = main
        self.gxp = IncomeGroupPrior(raw, targets, self.income, source, seed, opts['group_params'])
        self.latent_prior = latent_values(raw).mean(axis=0)
        self.latent_total = np.column_stack([
            np.bincount(self.income.axis.codes, weights=v, minlength=self.income.axis.size)
            for v in latent_values(raw).T])
        income_code = self.income.axis.codes
        gate = gate_codes(raw)
        self.n_gate = max(30, int(gate.max()) + 1)
        size = self.income.axis.size * self.n_gate
        joint_code = income_code.astype(np.int64) * self.n_gate + gate
        c1 = np.bincount(joint_code, weights=targets, minlength=size).reshape(-1, self.n_gate)
        allcounts = np.bincount(joint_code, minlength=size).reshape(-1, self.n_gate).astype(np.float64)
        c0 = allcounts - c1
        marg0 = (c0.sum(axis=0) + 0.5) / (c0.sum() + 0.5 * self.n_gate)
        marg1 = (c1.sum(axis=0) + 0.5) / (c1.sum() + 0.5 * self.n_gate)
        self.marg0, self.marg1 = marg0, marg1
        self.mixtures = {}
        combined = np.column_stack([c0, c1])
        for width in MIX_WIDTHS:
            smooth = gaussian_smooth(combined, width) * np.sqrt(2.0 * np.pi) * width
            s0 = np.maximum(smooth[:, :self.n_gate] - c0, 0.0)
            s1 = np.maximum(smooth[:, self.n_gate:] - c1, 0.0)
            q0 = (s0 + 30.0 * marg0) / (s0.sum(axis=1, keepdims=True) + 30.0)
            q1 = (s1 + 30.0 * marg1) / (s1.sum(axis=1, keepdims=True) + 30.0)
            mu = np.clip((s1.sum(axis=1) + 20.0 * self.prior)
                         / (s0.sum(axis=1) + s1.sum(axis=1) + 20.0), 1e-4, 1.0 - 1e-4)
            pi, information, gain = mixture_kernel(allcounts, q0, q1, mu)
            denominator = q0 + pi[:, None] * (q1 - q0)
            row_probability = pi[:, None] * q1 / np.maximum(denominator, 1e-300)
            self.mixtures[width] = dict(pi=pi, row_probability=row_probability,
                                        information=information, gain=gain)
        donor_components = self.gam.components(raw)
        nuisance = donor_components[:, 0] + donor_components[:, 3] + 0.25 * donor_components[:, 2]
        order = np.argsort(income_code, kind='stable')
        boundaries = np.r_[0, np.cumsum(self.income.count.astype(np.int64))]
        curve = composition_curve_kernel(nuisance[order], boundaries, SHIFT_GRID[0],
                                         SHIFT_GRID[1] - SHIFT_GRID[0], len(SHIFT_GRID))
        global_curve = np.average(curve, weights=self.income.count, axis=0)
        self.curves = np.vstack([curve, global_curve])
        if not np.isfinite(self.curves).all() or np.any(np.diff(self.curves, axis=1) < -1e-12):
            raise AssertionError('Invalid composition curves')
        self.gam.fitted_eta = None
        self.main.fitted_eta = None
        for axis in list(axes.values()) + [self.worry.axis] + [t.axis for t in self.mste.values()]:
            axis.codes = None
        for width in INCOME_WIDTHS:
            self.income.local(width)
        for width in COMMUTE_WIDTHS:
            self.commute.local(width)
        self.worry.local(25)

    def transform(self, query):
        raw = prepare_raw(query, self.medians)
        n = len(raw['Age'])
        result = np.empty((n, len(FEATURE_COLS)), dtype=np.float32)
        for c in RAW_COLS:
            result[:, COL[c]] = raw[c]
        income, commute = raw['Annual_Income_USD'], raw['Daily_Commute_km']
        result[:, [COL[c] for c in SOURCE_COLS]] = self.source.transform(income)
        ii, ie, count, total = self.income.lookup(income)
        ci, ce, ccount, ctotal = self.commute.lookup(commute)
        for alpha in INCOME_ALPHAS:
            result[:, COL[f'TE_INC_A{alpha:g}']] = (total + alpha * self.prior) / (count + alpha)
        for alpha in COMMUTE_ALPHAS:
            result[:, COL[f'TE_CMT_A{alpha:g}']] = (ctotal + alpha * self.prior) / (ccount + alpha)
        result[:, COL['SUPPORT_INC_LOG']] = np.log1p(count)
        result[:, COL['SUPPORT_CMT_LOG']] = np.log1p(ccount)
        local = {w: self.income.local(w)[ii] for w in INCOME_WIDTHS}
        for width in INCOME_WIDTHS:
            result[:, COL[f'NBR_INC_W{width}']] = local[width]
        for a, b in DETAIL_PAIRS:
            result[:, COL[f'NBR_DETAIL_W{a}_{b}']] = local[a] - local[b]
        for width in HIER_WIDTHS:
            for alpha in HIER_ALPHAS:
                result[:, COL[f'HIER_INC_W{width}_A{alpha:g}']] = (total + alpha * local[width]) / (count + alpha)
            result[:, COL[f'INC_SURPRISE_W{width}']] = result[:, COL['TE_INC_A5']] - local[width]
        for width in COMMUTE_WIDTHS:
            result[:, COL[f'NBR_CMT_W{width}']] = self.commute.local(width)[ci]
        main = self.main.components(raw).sum(axis=1)
        effect = self.commute_effect[ci]
        result[:, COL['DCMT_LOCAL_W5']] = effect
        result[:, COL['P_DCMT_LOCAL_W5']] = expit(main + effect)
        prior = self.gxp.transform(income, ii, ie)
        result[:, COL['GXP_D3_R100']] = prior
        for alpha in (2., 5., 10., 20., 50.):
            result[:, COL[f'GXP_D3_R100_A{alpha:g}']] = (total + alpha * prior) / (count + alpha)
        latent = latent_values(raw)
        grouped = self.latent_total[ii] * ie[:, None]
        for j, name in enumerate(LATENT_NAMES):
            for alpha in LATENT_ALPHAS:
                result[:, COL[f'LAT_INC_{name}_A{alpha:g}']] = (grouped[:, j] + alpha * self.latent_prior[j]) / (count + alpha)
        for j, name in enumerate(('CONCERN', 'SUBSIDY', 'ANXIETY', 'SCORE')):
            expectation = (grouped[:, j] + 20.0 * self.latent_prior[j]) / (count + 20.0)
            result[:, COL[f'LAT_INC_{name}_DEV']] = latent[:, j] - expectation
        integer_income, integer_commute = np.rint(income).astype(np.int64), np.rint(10.0 * commute).astype(np.int64)
        for k in range(6):
            result[:, COL[f'DIG_INC_10E{k}']] = (integer_income // 10**k) % 10
        for k in range(4):
            result[:, COL[f'DIG_CMT10_10E{k}']] = (integer_commute // 10**k) % 10
        wi, we, wc, wt = self.worry.lookup(worry_key(raw))
        result[:, COL['WTE05_A5']] = (wt + 5.0 * self.prior) / (wc + 5.0)
        result[:, COL['WHIER05_W25_A20']] = (wt + 20.0 * self.worry.local(25)[wi]) / (wc + 20.0)
        p20 = (total + 20.0 * prior) / (count + 20.0)
        sd = np.sqrt(np.maximum(p20 * (1.0 - p20) / (count + 21.0), 1e-12))
        for c, value in zip(UNCERTAINTY_COLS, (sd, p20 - sd, p20 + sd)):
            result[:, COL[c]] = value
        gg = gate_codes(raw)
        safe_gate = np.minimum(gg, self.n_gate - 1)
        known_gate = gg < self.n_gate
        unknown_row_probability = self.prior * self.marg1[safe_gate] / np.maximum(
            self.marg0[safe_gate] + self.prior * (self.marg1[safe_gate] - self.marg0[safe_gate]), 1e-300)
        for width in MIX_WIDTHS:
            state = self.mixtures[width]
            pi = np.where(ie, state['pi'][ii], self.prior)
            row_p = np.where(ie, state['row_probability'][ii, safe_gate], unknown_row_probability)
            row_p = np.where(known_gate, row_p, pi)
            values = (pi, (total + 20.0 * pi) / (count + 20.0), (total + 100.0 * pi) / (count + 100.0),
                      row_p, np.where(ie, state['information'][ii], 0.0), np.where(ie, state['gain'][ii], 0.0))
            for suffix, value in zip(('PRIOR', 'POST20', 'POST100', 'ROW_PROB', 'INFO', 'GAIN'), values):
                result[:, COL[f'MIX_W{width}_{suffix}']] = value
        components = self.gam.components(raw)
        result[:, [COL[c] for c in COORD_COLS]] = components
        channel_probabilities = np.asarray(result[:, [COL[c] for c in CHANNELS]], dtype=np.float64)
        curve_index = np.where(ie, ii, self.income.axis.size).astype(np.int32)
        shift = invert_curve_kernel(self.curves, curve_index, channel_probabilities, SHIFT_GRID)
        result[:, [COL[c] for c in COMPOSITION_COLS[:6]]] = shift
        result[:, [COL[c] for c in COMPOSITION_COLS[6:]]] = shift - 0.25 * components[:, 1, None]
        for key, values in mste_keys(raw).items():
            _, _, cn, sy = self.mste[key].lookup(values)
            for alpha in MSTE_ALPHAS:
                result[:, COL[f'MSTE_{key}_A{alpha:g}']] = (sy + alpha * self.prior) / (cn + alpha)
            result[:, COL[f'MSTE_{key}_LOGN']] = np.log1p(cn)
        margin = (components[:, 0] + components[:, 3] + 0.25 * (components[:, 1] + components[:, 2])).astype(np.float32)
        if not np.isfinite(result).all() or not np.isfinite(margin).all():
            raise AssertionError('Nonfinite features or initial margin')
        return result, margin

    def transform_into(self, raw, rows, feature_buffer, margin_buffer, positions=None):
        rows = np.asarray(rows, dtype=np.int64)
        if positions is None:
            positions = np.arange(len(rows))
        for start in range(0, len(rows), CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, len(rows))
            values, margins = self.transform(take_raw(raw, rows[start:stop]))
            feature_buffer[positions[start:stop]] = values
            margin_buffer[positions[start:stop]] = margins


# ---------------------------------------------------------------- notebook cell 6 (fold bundle), generalised
def build_fold_bundle(raw, source, fit_rows, y_fit, query_sets, fold, seed, opts):
    """Fold feature bundle.
    fit_rows (global X_all rows) + y_fit: the only labels used. Their own features come from 5 inner stratified folds
    (donor = the other 4/5 of fit_rows). Each query row set gets features from a DonorState on all fit_rows.
    Returns (fit_x, fit_margin, [(query_x, query_margin), ...])."""
    fit_rows = np.asarray(fit_rows, np.int64)
    y_fit = np.asarray(y_fit)
    for rows in query_sets:
        if np.intersect1d(fit_rows, rows).size:
            raise AssertionError('Query rows overlap the donor rows')
    nf = len(FEATURE_COLS)
    fit_x = np.empty((len(fit_rows), nf), np.float32)
    fit_m = np.empty(len(fit_rows), np.float32)
    visits = np.zeros(len(fit_rows), np.uint8)
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed + 3000 + fold)
    for inner_fold, (fit_local, receive_local) in enumerate(inner_cv.split(np.zeros(len(fit_rows)), y_fit)):
        state = DonorState(take_raw(raw, fit_rows[fit_local]), y_fit[fit_local], source, 7000 + 10 * fold + inner_fold, opts)
        state.transform_into(raw, fit_rows[receive_local], fit_x, fit_m, receive_local)
        visits[receive_local] += 1
        del state
    if not np.all(visits == 1):
        raise AssertionError('Incomplete or duplicated inner OOF coverage')
    state = DonorState(take_raw(raw, fit_rows), y_fit, source, 8000 + fold, opts)
    outs = []
    for rows in query_sets:
        qx = np.empty((len(rows), nf), np.float32)
        qm = np.empty(len(rows), np.float32)
        state.transform_into(raw, rows, qx, qm)
        outs.append((qx, qm))
    del state
    for m in [fit_x, fit_m] + [a for pair in outs for a in pair]:
        if not np.isfinite(m).all():
            raise AssertionError('Nonfinite features / margins')
    return fit_x, fit_m, outs


def _qdm(x, margin, label, ref, threads):
    return xgb.QuantileDMatrix(x, label=label, base_margin=margin, feature_names=FEATURE_COLS,
                               feature_types=FEATURE_TYPES, enable_categorical=True,
                               max_bin=XGB_PARAMS['max_bin'], ref=ref, nthread=threads)


# ---------------------------------------------------------------- our 5 folds
def prepare(X_all, orig):
    """Label-free prep: pack the competition predictors (train rows then test rows); original-data income priors after the
    notebook's dedup (drop original rows whose 13 predictors equal a competition train/test row; keep Yes/No labels only)."""
    keep = (~np.isin(feature_hashes(orig), feature_hashes(X_all))
            & orig[TARGET].isin(['No', 'Yes']).to_numpy())
    original = orig.loc[keep].reset_index(drop=True)
    if not len(original):
        raise ValueError('No eligible original rows remain')
    raw = pack_raw(X_all[RAW_COLS])
    for c in CAT_COLS:
        assert (raw[c] < len(CATEGORY_VOCAB[c])).all(), f'unknown category in {c}'
    print(f'[{NAME}] original rows kept {len(original)}/{len(orig)}', flush=True)
    return dict(raw=raw, source=SourceIncome(original))


def run_fold(state, tr_idx, y_tr, va_idx, te_idx, cfg):
    """tr_idx / va_idx: train rows of X_all; te_idx: the test rows of X_all; y_tr: labels of tr_idx only."""
    fold, seed = int(cfg['fold']), int(cfg['seed'])
    device, threads = cfg['device'], int(cfg['threads'])
    numba.set_num_threads(max(1, min(threads, numba.config.NUMBA_NUM_THREADS)))
    opts = dict(gam_iters=GAM_ITERS, group_params=dict(GROUP_PARAMS, device=device, n_jobs=threads))
    max_rounds, patience = N_ESTIMATORS, EARLY_STOPPING_ROUNDS
    params = dict(XGB_PARAMS, device=device, nthread=threads, seed=seed + fold)
    raw, source = state['raw'], state['source']
    tr_idx, va_idx, te_idx = (np.asarray(a, np.int64) for a in (tr_idx, va_idx, te_idx))
    y_tr = np.asarray(y_tr).astype(np.int8)
    assert len(y_tr) == len(tr_idx)
    assert not np.intersect1d(tr_idx, va_idx).size and not np.intersect1d(tr_idx, te_idx).size

    with threadpool_limits(limits=threads):
        # ---- phase 1: round count from a stratified 10% holdout of tr_idx (features rebuilt on the 90% only)
        t0 = time.time()
        fit_loc, es_loc = train_test_split(np.arange(len(tr_idx)), test_size=ES_FRACTION, stratify=y_tr,
                                           random_state=seed)
        fit_loc, es_loc = np.sort(fit_loc), np.sort(es_loc)
        fx, fm, [(ex, em)] = build_fold_bundle(raw, source, tr_idx[fit_loc], y_tr[fit_loc], [tr_idx[es_loc]],
                                               fold, seed, opts)
        t1 = time.time()
        dfit = _qdm(fx, fm, y_tr[fit_loc], None, threads)
        des = _qdm(ex, em, y_tr[es_loc], dfit, threads)
        del fx, fm, ex, em
        stopper = xgb.callback.EarlyStopping(rounds=patience, metric_name='auc', data_name='es',
                                             maximize=True, save_best=False)
        booster = xgb.train(params, dfit, num_boost_round=max_rounds, evals=[(des, 'es')],
                            callbacks=[stopper], verbose_eval=500)
        best_rounds = int(booster.best_iteration) + 1
        best_auc = float(booster.best_score)
        n_refit = max(1, int(round(REFIT_MULT * best_rounds)))
        del booster, dfit, des
        gc.collect()
        t2 = time.time()
        print(f'[{NAME}] fold {fold}: ES phase feats {t1 - t0:.0f}s, xgb {t2 - t1:.0f}s | best {best_rounds} rounds '
              f'(holdout AUC {best_auc:.6f}) -> refit {n_refit}', flush=True)

        # ---- phase 2: features on all of tr_idx, refit on all of tr_idx with the fixed round count
        tx, tm, [(vx, vm), (qx, qm)] = build_fold_bundle(raw, source, tr_idx, y_tr, [va_idx, te_idx], fold, seed, opts)
        t3 = time.time()
        dtr = _qdm(tx, tm, y_tr, None, threads)
        dva = _qdm(vx, vm, None, dtr, threads)
        dte = _qdm(qx, qm, None, dtr, threads)
        del tx, tm, vx, vm, qx, qm
        booster = xgb.train(params, dtr, num_boost_round=n_refit)
        p_va = booster.predict(dva).astype(np.float64)
        p_te = booster.predict(dte).astype(np.float64)
        del booster, dtr, dva, dte
        gc.collect()
        print(f'[{NAME}] fold {fold}: refit feats {t3 - t2:.0f}s, xgb {time.time() - t3:.0f}s', flush=True)
    return p_va, p_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    tr, te = read_train(), read_test()
    y = target()
    f = folds(y)
    X_all = pd.concat([tr[RAW_COLS], te[RAW_COLS]], ignore_index=True)   # predictors only
    state = prepare(X_all, read_original())
    for k in fold_list(a.folds):
        if have_part(HERE, NAME, k):
            continue
        tr_idx, va_idx = np.where(f != k)[0], np.where(f == k)[0]
        te_idx = len(tr) + np.arange(len(te))
        cfg = dict(fold=k, seed=a.seed, device=a.device, threads=a.threads)
        p_va, p_te = run_fold(state, tr_idx, y[tr_idx].copy(), va_idx, te_idx, cfg)
        save_part(HERE, NAME, k, p_va, p_te)
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
