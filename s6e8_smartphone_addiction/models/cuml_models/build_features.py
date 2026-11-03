#!/usr/bin/env python
"""Shared inputs of the six cuML pipelines -> cache/.

base (label-free, 240 columns, one matrix for train and one for test): cache/base_train.npy, base_test.npy and
base_columns.json (column names and groups)
  raw (9)            the 9 numerics
  mask (41)          9 missing flags + 32 one-hots of the 5-bit missingness mask of the screen fields (daily, social,
                     gaming, work/study, weekend)
  artifact (30)      for the 6 hour fields (not age, notifications, app opens): round(100 x) mod 100 and the distances
                     to the 0.05 and 0.10 grids; log1p of the train + test count of each field's value (numerics
                     formatted to 2 decimals)
  category (11)      one-hots of the 3 categoricals' levels (train + test + original, missing as a level)
  derived (18)       observed / partial / strict sums of social + gaming + work, daily minus those sums, weekend minus
                     and over daily, social / gaming / work shares of daily, mean / std / max / min of the 3 components,
                     log1p notifications and app opens, notifications per (app opens + 1), missing screen fields
  source (27)        per numeric: CDF in the original data, distance to the nearest original value, log1p of the number
                     of original rows with exactly this value
  source_knn (17), source_impute (15), observed_joint (72)   from ../xgb_forensic (lib/inputs.py)
response (37 columns per outer fold; rows: outer-train rows, held-out rows, test rows):
cache/response/fold_<k>/{train,valid,test}.npy
  12 target logits of the 12 raw fields (tokens: numerics formatted to 2 decimals, missing as a token), smoothed
  towards the outer-train prior with m 20, clipped to [1e-5, 1 - 1e-5]; outer-train rows use exact leave-one-row-out
  sums (and a row-excluded prior), held-out / test rows the full outer-train map (unseen -> prior)
  12 log1p token counts (outer-train rows: count - 1)
  3 aggregates: sum of (logit - prior logit) over the 9 numerics, the same over the 5 screen fields, mean of the 12
  logits
  10 pair-lift columns of ../pairlift_models
Usage: python build_features.py [--part base|response|all]"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import CATEGORICALS, NUMERICS, TARGET, cache_dir, folds, labels, read_original, read_test, read_train  # noqa: E402
from lib.inputs import observed_joint, pairlift, source_knn, source_knn_impute  # noqa: E402

MISSING_TOKEN = '__NA__'
SMOOTHING = 20.0
SCREEN = ('daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'weekend_screen_time')


def safe_ratio(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        value = left / right
    value[~np.isfinite(value)] = np.nan
    return value.astype(np.float32)


def tokenise(series):
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors='coerce').to_numpy(np.float64)
        return np.asarray([f'{value:.2f}' if np.isfinite(value) else MISSING_TOKEN for value in values], dtype=str)
    return series.astype('string').fillna(MISSING_TOKEN).astype(str).to_numpy()


def build_handcrafted(frame, combined, source):
    values, names, groups = [], [], []

    def add(name, value, group):
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if np.isinf(array).any():
            raise RuntimeError(f'infinite values in {name}')
        values.append(array)
        names.append(name)
        groups.append(group)

    numeric = {name: pd.to_numeric(frame[name], errors='coerce').to_numpy(np.float64) for name in NUMERICS}
    for name in NUMERICS:
        x = numeric[name]
        add(f'raw__{name}', x, 'raw')
        add(f'missing__{name}', ~np.isfinite(x), 'mask')
        if name not in {'age', 'notifications_per_day', 'app_opens_per_day'}:
            add(f'grid__{name}__fractional_cent', np.mod(np.rint(x * 100.0), 100.0), 'artifact')
            add(f'grid__{name}__distance_005', np.abs(x - np.rint(x / 0.05) * 0.05), 'artifact')
            add(f'grid__{name}__distance_010', np.abs(x - np.rint(x / 0.10) * 0.10), 'artifact')

    for name in CATEGORICALS:
        raw = frame[name].astype('string').fillna(MISSING_TOKEN).astype(str)
        levels = sorted(set(combined[name].astype('string').fillna(MISSING_TOKEN).astype(str))
                        | set(source[name].astype('string').fillna(MISSING_TOKEN).astype(str)))
        for level in levels:
            add(f'category__{name}__{level}', (raw == level).to_numpy(), 'category')

    daily, social = numeric['daily_screen_time_hours'], numeric['social_media_hours']
    gaming, work = numeric['gaming_hours'], numeric['work_study_hours']
    weekend = numeric['weekend_screen_time']
    components = np.column_stack([social, gaming, work])
    partial_sum = np.nansum(components, axis=1)
    strict_sum = social + gaming + work
    derived = {
        'component_observed_count': np.isfinite(components).sum(axis=1),
        'component_partial_sum': partial_sum,
        'component_strict_sum': strict_sum,
        'daily_minus_partial_sum': daily - partial_sum,
        'daily_minus_strict_sum': daily - strict_sum,
        'weekend_minus_daily': weekend - daily,
        'weekend_over_daily': safe_ratio(weekend, daily),
        'social_share': safe_ratio(social, daily),
        'gaming_share': safe_ratio(gaming, daily),
        'work_share': safe_ratio(work, daily),
        'component_mean': np.nanmean(components, axis=1),
        'component_std': np.nanstd(components, axis=1),
        'component_max': np.nanmax(components, axis=1),
        'component_min': np.nanmin(components, axis=1),
        'notification_log1p': np.log1p(numeric['notifications_per_day']),
        'app_open_log1p': np.log1p(numeric['app_opens_per_day']),
        'notification_per_open': safe_ratio(numeric['notifications_per_day'], numeric['app_opens_per_day'] + 1.0),
        'screen_missing_count': np.column_stack([~np.isfinite(numeric[name]) for name in SCREEN]).sum(axis=1),
    }
    for name, value in derived.items():
        add(f'derived__{name}', value, 'derived')

    screen_mask = np.zeros(len(frame), np.int16)
    for bit, name in enumerate(SCREEN):
        screen_mask |= (~np.isfinite(numeric[name])).astype(np.int16) << bit
    for mask in range(32):
        add(f'screen_mask__{mask:02d}', screen_mask == mask, 'mask')

    for name in (*NUMERICS, *CATEGORICALS):
        count = pd.Series(tokenise(combined[name]), copy=False).value_counts(sort=False)
        own = pd.Series(tokenise(frame[name]), copy=False).map(count).fillna(0).to_numpy()
        add(f'frequency__{name}', np.log1p(own), 'artifact')

    source_numeric = source[list(NUMERICS)].to_numpy(np.float64)
    for column, name in enumerate(NUMERICS):
        src = np.sort(source_numeric[:, column])
        x = numeric[name]
        finite = np.isfinite(x)
        cdf = np.full(len(frame), np.nan, np.float64)
        nearest = np.full(len(frame), np.nan, np.float64)
        exact = np.zeros(len(frame), np.float64)
        if finite.any():
            q = x[finite]
            pos = np.searchsorted(src, q, side='left')
            lower = src[np.clip(pos - 1, 0, len(src) - 1)]
            upper = src[np.clip(pos, 0, len(src) - 1)]
            nearest[finite] = np.minimum(np.abs(q - lower), np.abs(q - upper))
            cdf[finite] = np.searchsorted(src, q, side='right') / len(src)
            exact[finite] = np.searchsorted(src, q, side='right') - np.searchsorted(src, q, side='left')
        add(f'source_cdf__{name}', cdf, 'source')
        add(f'source_nearest_distance__{name}', nearest, 'source')
        add(f'source_exact_count__{name}', np.log1p(exact), 'source')

    return np.column_stack(values).astype(np.float32), names, groups


def build_base():
    out = cache_dir('cuml_models')
    train, test, source = read_train(), read_test(), read_original()
    predictors = [column for column in test.columns if column != 'id']
    combined = pd.concat([train[predictors], test[predictors]], ignore_index=True)
    train_parts, names, groups = build_handcrafted(train, combined, source)
    test_parts, _, _ = build_handcrafted(test, combined, source)
    train_parts, test_parts = [train_parts], [test_parts]
    for loader, prefix, group in ((source_knn, 'oknn__', 'source_knn'), (source_knn_impute, 'oimp__', 'source_impute'),
                                  (observed_joint, 'ojoint__', 'observed_joint')):
        block_train, block_test, block_names = loader()
        if len(block_train) != len(train) or len(block_test) != len(test):
            raise RuntimeError(f'{group}: row count mismatch')
        train_parts.append(np.asarray(block_train, np.float32))
        test_parts.append(np.asarray(block_test, np.float32))
        names += [prefix + name for name in block_names]
        groups += [group] * len(block_names)
    train_matrix = np.column_stack(train_parts).astype(np.float32)
    test_matrix = np.column_stack(test_parts).astype(np.float32)
    if len(names) != 240 or len(set(names)) != len(names) or np.isinf(train_matrix).any() or np.isinf(test_matrix).any():
        raise RuntimeError(f'invalid base matrix ({len(names)} columns)')
    np.save(out / 'base_train.npy', train_matrix)
    np.save(out / 'base_test.npy', test_matrix)
    (out / 'base_columns.json').write_text(json.dumps({'names': names, 'groups': groups}, indent=1))
    print(f'base: {train_matrix.shape[1]} columns -> {out}', flush=True)


def response_one_column(train_token, train_target, apply_token):
    """Smoothed target logit + log count of a token: exact leave-one-row-out for the outer-train rows, full outer-train
    map for the applied rows."""
    y = np.asarray(train_target, np.float64)
    prior = float(y.mean())
    stats = pd.DataFrame({'token': train_token, 'target': y}).groupby('token', sort=True)['target'].agg(['sum', 'count'])
    token_series = pd.Series(train_token, copy=False)
    group_sum = token_series.map(stats['sum']).to_numpy(np.float64)
    group_count = token_series.map(stats['count']).to_numpy(np.float64)
    prior_loo = (y.sum() - y) / (len(y) - 1.0)
    probability = (group_sum - y + SMOOTHING * prior_loo) / (group_count - 1.0 + SMOOTHING)
    full_rate = (stats['sum'] + SMOOTHING * prior) / (stats['count'] + SMOOTHING)
    apply_series = pd.Series(apply_token, copy=False)
    apply_probability = apply_series.map(full_rate).fillna(prior).to_numpy(np.float64)
    apply_count = apply_series.map(stats['count']).fillna(0).to_numpy(np.float64)
    reliability = np.log1p(np.maximum(group_count - 1.0, 0.0))
    apply_reliability = np.log1p(apply_count)
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    apply_probability = np.clip(apply_probability, 1e-5, 1.0 - 1e-5)
    return (np.log(probability / (1.0 - probability)).astype(np.float32),
            np.log(apply_probability / (1.0 - apply_probability)).astype(np.float32),
            reliability.astype(np.float32), apply_reliability.astype(np.float32))


def build_response():
    train, test = read_train(), read_test()
    y = labels(train[TARGET])
    f = folds(y)
    fields = [*NUMERICS, *CATEGORICALS]
    train_tokens = {name: tokenise(train[name]) for name in fields}
    test_tokens = {name: tokenise(test[name]) for name in fields}
    raw_indices = list(range(len(NUMERICS)))
    screen_indices = [fields.index(name) for name in SCREEN]
    for k in range(5):
        tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
        logits = {'train': [], 'valid': [], 'test': []}
        counts = {'train': [], 'valid': [], 'test': []}
        for name in fields:
            train_logit, valid_logit, train_count, valid_count = response_one_column(
                train_tokens[name][tr], y[tr], train_tokens[name][va])
            _, test_logit, _, test_count = response_one_column(train_tokens[name][tr], y[tr], test_tokens[name])
            for role, logit, count in (('train', train_logit, train_count), ('valid', valid_logit, valid_count),
                                       ('test', test_logit, test_count)):
                logits[role].append(logit)
                counts[role].append(count)
        prior_logit = math.log(y[tr].mean() / (1 - y[tr].mean()))
        d = cache_dir('cuml_models') / 'response' / f'fold_{k}'
        d.mkdir(parents=True, exist_ok=True)
        for role in ('train', 'valid', 'test'):
            role_logits = np.column_stack(logits[role])
            aggregate = np.column_stack([(role_logits[:, raw_indices] - prior_logit).sum(axis=1),
                                         (role_logits[:, screen_indices] - prior_logit).sum(axis=1),
                                         role_logits.mean(axis=1)])
            matrix = np.column_stack([role_logits, np.column_stack(counts[role]), aggregate,
                                      pairlift(k, role)]).astype(np.float32)
            if matrix.shape[1] != 37 or not np.isfinite(matrix).all():
                raise RuntimeError(f'response fold {k} {role}: invalid matrix')
            np.save(d / f'{role}.npy', matrix)
        print(f'response fold {k}: {len(tr)} / {len(va)} rows -> {d}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all', choices=['base', 'response', 'all'])
    a = ap.parse_args()
    if a.part in ('base', 'all'):
        build_base()
    if a.part in ('response', 'all'):
        build_response()


if __name__ == '__main__':
    main()
