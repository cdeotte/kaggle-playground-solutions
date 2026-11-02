#!/usr/bin/env python
"""Fold-local multiresolution response families: cache/multires/fold_<k>/<family>_{train,valid,test}.npy + manifest.json.

Each column is a smoothed target encoding of one grouping of the rows (alpha 1 and alpha 20), as a logit difference to the
prior: logit(clip((positives + alpha p) / (count + alpha), 1e-6)) - logit(p), p = outer-train positive rate.
Groupings:
  exact   exact values of age, income, commute and the two station counts                                     (5)
  core    floor(value / width): income 1000-10000, commute 0.5-10, age 2-10, stations 2-4                      (16)
  wide    17 more income widths (50-30000), 11 commute widths (0.1-7), 5 age widths, and floor(log1p(income) /
          log(r)) for r in 1.02-1.4                                                                           (38)
Families (training rows = outer-train rows, encoded by a 5-fold inner cross-fit, StratifiedKFold with seed 777 ("single") or
the mean over seeds 777-784 ("avg8"); held-out and test rows: one encoding fitted on all outer-train labels):
  exact_single, exact_avg8 (10 columns each), core_single, core_avg8 (32), wide_avg8 (76), and response_summaries (24:
  mean / std / min / max of the alpha-20 columns of each family and 4 differences of family means).
Usage: python multires_cache.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds, labels, read_test, read_train  # noqa: E402
from common.pairs import fold_list  # noqa: E402
from features import CACHE, FEATURES, save_arrays, update_manifest  # noqa: E402

EXACT_FIELDS = ('Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work')
INNER_SEEDS = tuple(range(777, 785))
CORE_WIDTHS = {
    'Annual_Income_USD': (1000.0, 2500.0, 5000.0, 10000.0),
    'Daily_Commute_km': (0.5, 1.0, 2.0, 5.0, 10.0),
    'Age': (2.0, 5.0, 10.0),
    'Charging_Stations_Near_Home': (2.0, 4.0),
    'Charging_Stations_Near_Work': (2.0, 4.0),
}
WIDE_WIDTHS = {
    'Annual_Income_USD': (50.0, 100.0, 200.0, 300.0, 500.0, 750.0, 1250.0, 1500.0, 2000.0, 3000.0, 4000.0, 6000.0, 7500.0,
                          12500.0, 15000.0, 20000.0, 30000.0),
    'Daily_Commute_km': (0.1, 0.2, 0.25, 0.4, 0.75, 1.5, 2.5, 3.0, 4.0, 6.0, 7.0),
    'Age': (1.0, 3.0, 4.0, 6.0, 8.0),
}
LOG_INCOME_RATIOS = (1.02, 1.05, 1.10, 1.20, 1.40)
FAMILIES = ('exact_single', 'exact_avg8', 'core_single', 'core_avg8', 'wide_avg8')


def factor_codes(values):
    series = pd.Series(values, copy=False)
    if pd.api.types.is_numeric_dtype(series.dtype):
        array = pd.to_numeric(series, errors='coerce').to_numpy(np.float64)
        missing = ~np.isfinite(array)
        if missing.any():
            tokens = array.astype(object)
            tokens[missing] = '__MISSING__'
            code, _ = pd.factorize(tokens, sort=True)
        else:
            code, _ = pd.factorize(array, sort=True)
    else:
        code, _ = pd.factorize(series.astype('string').fillna('__MISSING__'), sort=True)
    return code.astype(np.int32)


def code_inventory(combined):
    exact = {field: factor_codes(combined[field]) for field in EXACT_FIELDS}
    core, wide = {}, {}
    for widths, out in ((CORE_WIDTHS, core), (WIDE_WIDTHS, wide)):
        for field, field_widths in widths.items():
            values = pd.to_numeric(combined[field], errors='coerce').to_numpy(np.float64)
            for width in field_widths:
                out[f'{field}__w{width:g}'] = factor_codes(np.floor(values / width))
    income = np.maximum(pd.to_numeric(combined['Annual_Income_USD'], errors='coerce').to_numpy(np.float64), 0.0)
    for ratio in LOG_INCOME_RATIOS:
        wide[f'Annual_Income_USD__r{ratio:g}'] = factor_codes(np.floor(np.log1p(income) / float(np.log(ratio))))
    return exact, core, wide


def fit_apply_delta(fit_codes, fit_y, apply_codes, alpha):
    size = int(max(fit_codes.max(initial=0), apply_codes.max(initial=0))) + 1
    count = np.bincount(fit_codes, minlength=size).astype(np.float64)
    positive = np.bincount(fit_codes, weights=fit_y, minlength=size).astype(np.float64)
    prior = float(np.mean(fit_y))
    probability = (positive[apply_codes] + alpha * prior) / (count[apply_codes] + alpha)
    return (logit(np.clip(probability, 1e-6, 1.0 - 1e-6)) - logit(prior)).astype(np.float32)


def crossfit_delta(codes, y, outer_fit, outer_apply, alpha, seeds):
    fit_codes, fit_y = codes[outer_fit], y[outer_fit]
    train = np.zeros(len(outer_fit), dtype=np.float64)
    for seed in seeds:
        one = np.empty(len(outer_fit), dtype=np.float32)
        for inner_fit, inner_apply in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(outer_fit)), fit_y):
            one[inner_apply] = fit_apply_delta(fit_codes[inner_fit], fit_y[inner_fit], fit_codes[inner_apply], alpha)
        train += one
    train /= np.float64(len(seeds))
    return train.astype(np.float32), fit_apply_delta(fit_codes, fit_y, codes[outer_apply], alpha)


def build_family(family, specs, seeds, y, outer_fit, outer_valid, test_rows):
    columns = {'train': [], 'valid': [], 'test': []}
    names = []
    for token, codes in specs.items():
        for alpha in (1.0, 20.0):
            train_value, valid_value = crossfit_delta(codes, y, outer_fit, outer_valid, alpha, seeds)
            columns['train'].append(train_value)
            columns['valid'].append(valid_value)
            columns['test'].append(fit_apply_delta(codes[outer_fit], y[outer_fit], codes[test_rows], alpha))
            names.append(f'te__{family}__{token}__a{alpha:g}')
    return {role: np.column_stack(c).astype(np.float32) for role, c in columns.items()}, names


def build_summaries(matrices):
    columns = {'train': [], 'valid': [], 'test': []}
    means = {'train': {}, 'valid': {}, 'test': {}}
    names = []
    for family in FAMILIES:
        values, family_names = matrices[family]
        selected = [i for i, name in enumerate(family_names) if name.endswith('__a20')]
        for role in columns:
            block = values[role][:, selected]
            means[role][family] = block.mean(axis=1)
            for reducer in (np.mean, np.std, np.min, np.max):
                columns[role].append(reducer(block, axis=1).astype(np.float32))
        names += [f'response_summary__{family}__{token}__a20' for token in ('mean', 'std', 'min', 'max')]
    for left, right in (('exact_single', 'exact_avg8'), ('core_single', 'core_avg8'), ('exact_avg8', 'core_avg8'),
                        ('core_avg8', 'wide_avg8')):
        for role in columns:
            columns[role].append((means[role][left] - means[role][right]).astype(np.float32))
        names.append(f'response_summary__mean_{left}_minus_{right}__a20')
    return {role: np.column_stack(c).astype(np.float32) for role, c in columns.items()}, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    train, test = read_train(), read_test()
    y = labels(train[TARGET])
    f = folds(y)
    combined = pd.concat([train[list(FEATURES)], test[list(FEATURES)]], ignore_index=True)
    test_rows = np.arange(len(train), len(combined), dtype=np.int64)
    exact, core, wide = code_inventory(combined)
    definitions = (('exact_single', exact, (777,)), ('exact_avg8', exact, INNER_SEEDS), ('core_single', core, (777,)),
                   ('core_avg8', core, INNER_SEEDS), ('wide_avg8', wide, INNER_SEEDS))
    for k in fold_list(a.folds):
        outer_fit, outer_valid = np.flatnonzero(f != k), np.flatnonzero(f == k)
        folder = CACHE / 'multires' / f'fold_{k}'
        save_arrays(folder, {'train_indices': outer_fit.astype(np.int64), 'valid_indices': outer_valid.astype(np.int64)})
        matrices = {}
        for family, specs, seeds in definitions:
            matrices[family] = build_family(family, specs, seeds, y, outer_fit, outer_valid, test_rows)
        matrices['response_summaries'] = build_summaries(matrices)
        for family, (values, _) in matrices.items():
            save_arrays(folder, {f'{family}_{role}': value for role, value in values.items()})
        update_manifest(folder, {'families': {family: {'names': names} for family, (_, names) in matrices.items()}})
        print(f'fold {k}: ' + ', '.join(f'{family} {len(names)}' for family, (_, names) in matrices.items()), flush=True)


if __name__ == '__main__':
    main()
