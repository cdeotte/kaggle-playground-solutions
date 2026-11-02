#!/usr/bin/env python
"""Fold-local base cache: cache/base/fold_<k>/<block>_{train,valid,test}.npy + manifest.json (column names).

For outer fold k only the labels of the outer-train rows (folds != k) are used. "train" = outer-train rows (train.csv order),
"valid" = the rows of fold k, "test" = all test rows; valid and test rows are always encoded with all outer-train labels.
  source_full_global           original-data source bank (features.build_source_bank) with value counts over train + test
                               (label-free; the same values in every fold), 113 columns
  response_single_loo          per-field target rate (alpha 20) as logit + log count for the 13 fields, and mean / std / min /
                               max of the 13 rates (30 columns); training rows: exact leave-one-out over outer-train
  response_local_loo           income window residuals (window +-10, alpha 5; window +-25, alpha 5, gated by 50 / (exact income
                               count + 50)) and the commute window residual (+-0.1 km, alpha 100); residual = window logit
                               minus exact-value logit; training rows: exact leave-one-out
  response_income_multiscale   the two income residuals; training rows: 5-fold cross-fit (KFold, seed 950000 + k)
  response_commute_residual    the commute residual; training rows: 5-fold cross-fit (KFold, seed 950000 + k)
  response_target_gender_home  Gender x Home_Charging_Possible logit lift over the additive per-field logits (alpha 20);
                               training rows: exact leave-one-out
manifest.json: <block>_names for every block, and source_focus_names (the source-bank columns of income, environmental
concern, home charging, subsidy and range anxiety).
Usage: python base_cache.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import KFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds, labels, read_test, read_train  # noqa: E402
from common.pairs import fold_list  # noqa: E402
from features import (CACHE, FEATURES, build_source_bank, exact_codes, normalize_source, read_original_text,  # noqa: E402
                      save_arrays, source_labels, update_manifest)

FOCUS_FIELDS = ('Annual_Income_USD', 'Environmental_Concern_Level', 'Home_Charging_Possible', 'Subsidy_Available',
                'Range_Anxiety_Level')


def fit_rate(code, y, fit, alpha, prior):
    size = int(code.max()) + 1
    count = np.bincount(code[fit], minlength=size).astype(np.float64)
    positive = np.bincount(code[fit], weights=y[fit], minlength=size).astype(np.float64)
    denominator = count + alpha
    rate = np.full(size, prior, dtype=np.float64)
    np.divide(positive + alpha * prior, denominator, out=rate, where=denominator > 0.0)
    return rate, count


def single_response(exact, y, fit, apply):
    """Per-field rates (alpha 20) fitted on `fit`, applied to `apply`."""
    prior = float(y[fit].mean())
    columns, names, probabilities = [], [], []
    for name in FEATURES:
        rate, count = fit_rate(exact[name], y, fit, 20.0, prior)
        probability = rate[exact[name][apply]]
        columns += [logit(np.clip(probability, 1e-5, 1 - 1e-5)).astype(np.float32),
                    np.log1p(count[exact[name][apply]]).astype(np.float32)]
        names += [f'resp_logit__{name}', f'resp_logcount__{name}']
        probabilities.append(probability)
    stack = np.column_stack(probabilities)
    columns += [x.astype(np.float32) for x in (stack.mean(1), stack.std(1), stack.min(1), stack.max(1))]
    names += ['resp_p_mean', 'resp_p_std', 'resp_p_min', 'resp_p_max']
    return np.column_stack(columns).astype(np.float32), names


def leave_one_out_single_response(exact, y, outer_fit):
    """single_response for the outer-train rows, each encoded from all other outer-train labels."""
    labels_ = y[outer_fit].astype(np.float64)
    prior = (labels_.sum() - labels_) / (len(labels_) - 1.0)
    columns, probabilities = [], []
    for name in FEATURES:
        code = exact[name]
        size = int(code.max()) + 1
        count = np.bincount(code[outer_fit], minlength=size).astype(np.float64)
        positive = np.bincount(code[outer_fit], weights=labels_, minlength=size).astype(np.float64)
        own = code[outer_fit]
        own_count = count[own] - 1.0
        probability = (positive[own] - labels_ + 20.0 * prior) / (own_count + 20.0)
        columns += [logit(np.clip(probability, 1e-5, 1 - 1e-5)).astype(np.float32), np.log1p(own_count).astype(np.float32)]
        probabilities.append(probability)
    stack = np.column_stack(probabilities)
    columns += [x.astype(np.float32) for x in (stack.mean(1), stack.std(1), stack.min(1), stack.max(1))]
    return np.column_stack(columns).astype(np.float32)


def crossfit_income_count(code, y, outer_fit, fold):
    """float32 log1p exact-income count of the training rows from a 5-fold inner cross-fit (StratifiedKFold, seed
    930000 + fold); it gates the +-25 income window of response_income_multiscale."""
    size = int(code.max()) + 1
    out = np.empty(len(outer_fit), dtype=np.float32)
    splitter = StratifiedKFold(5, shuffle=True, random_state=930000 + fold)
    for inner_fit, inner_apply in splitter.split(np.zeros(len(outer_fit)), y[outer_fit]):
        count = np.bincount(code[outer_fit[inner_fit]], minlength=size).astype(np.float64)
        out[inner_apply] = np.log1p(count[code[outer_fit[inner_apply]]]).astype(np.float32)
    return out


def neighborhood_apply(values, y, fit, apply, bandwidth, alpha):
    """Window rate (|value - query| <= bandwidth) and exact-value rate fitted on `fit`; returns [window logit, window logit
    minus exact logit, log1p window count]."""
    fit_values = np.asarray(values[fit], dtype=np.float64)
    query = np.asarray(values[apply], dtype=np.float64)
    prior = float(y[fit].mean())
    unique, inverse = np.unique(fit_values, return_inverse=True)
    count = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    positive = np.bincount(inverse, weights=y[fit], minlength=len(unique)).astype(np.float64)
    prefix_count = np.concatenate(([0.0], np.cumsum(count)))
    prefix_positive = np.concatenate(([0.0], np.cumsum(positive)))
    left = np.searchsorted(unique, query - bandwidth, side='left')
    right = np.searchsorted(unique, query + bandwidth, side='right')
    local_count = prefix_count[right] - prefix_count[left]
    local_positive = prefix_positive[right] - prefix_positive[left]
    smooth = (local_positive + alpha * prior) / (local_count + alpha)
    position = np.searchsorted(unique, query)
    inside = position < len(unique)
    matched = np.zeros(len(query), dtype=bool)
    matched[inside] = unique[position[inside]] == query[inside]
    exact_count = np.zeros(len(query))
    exact_positive = np.zeros(len(query))
    exact_count[matched] = count[position[matched]]
    exact_positive[matched] = positive[position[matched]]
    exact = (exact_positive + alpha * prior) / (exact_count + alpha)
    smooth_logit = logit(np.clip(smooth, 1e-5, 1 - 1e-5))
    exact_logit = logit(np.clip(exact, 1e-5, 1 - 1e-5))
    return np.column_stack([smooth_logit, smooth_logit - exact_logit, np.log1p(local_count)]).astype(np.float32)


def crossfit_neighborhood(values, y, outer_fit, outer_apply, fold, bandwidth, alpha):
    """Training rows: 5-fold inner cross-fit with label-independent routing (KFold, seed 950000 + fold)."""
    fit_matrix = np.empty((len(outer_fit), 3), dtype=np.float32)
    for inner_fit, inner_apply in KFold(5, shuffle=True, random_state=950000 + fold).split(np.zeros(len(outer_fit))):
        fit_matrix[inner_apply] = neighborhood_apply(values, y, outer_fit[inner_fit], outer_fit[inner_apply], bandwidth, alpha)
    return fit_matrix, neighborhood_apply(values, y, outer_fit, outer_apply, bandwidth, alpha)


def leave_one_out_neighborhood(values, y, outer_fit, bandwidth, alpha):
    """neighborhood_apply for the outer-train rows with each row's own value and label removed. Also returns the
    leave-one-out exact-value count."""
    fit_values = np.asarray(values[outer_fit], dtype=np.float64)
    fit_target = y[outer_fit].astype(np.float64)
    unique, inverse = np.unique(fit_values, return_inverse=True)
    count = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    positive = np.bincount(inverse, weights=fit_target, minlength=len(unique)).astype(np.float64)
    prefix_count = np.concatenate(([0.0], np.cumsum(count)))
    prefix_positive = np.concatenate(([0.0], np.cumsum(positive)))
    left = np.searchsorted(unique, fit_values - bandwidth, side='left')
    right = np.searchsorted(unique, fit_values + bandwidth, side='right')
    local_count = prefix_count[right] - prefix_count[left] - 1.0
    local_positive = prefix_positive[right] - prefix_positive[left] - fit_target
    exact_count = count[inverse] - 1.0
    exact_positive = positive[inverse] - fit_target
    prior = (fit_target.sum() - fit_target) / (len(fit_target) - 1.0)
    smooth = (local_positive + alpha * prior) / (local_count + alpha)
    exact = (exact_positive + alpha * prior) / (exact_count + alpha)
    smooth_logit = logit(np.clip(smooth, 1e-5, 1 - 1e-5))
    exact_logit = logit(np.clip(exact, 1e-5, 1 - 1e-5))
    return np.column_stack([smooth_logit, smooth_logit - exact_logit, np.log1p(local_count)]).astype(np.float32), exact_count


def _counts(code, labels_, fit):
    size = int(code.max()) + 1
    return (np.bincount(code[fit], minlength=size).astype(np.float64),
            np.bincount(code[fit], weights=labels_[fit], minlength=size).astype(np.float64))


def _joint(codes):
    out = codes[0].astype(np.int64)
    for code in codes[1:]:
        out = out * (int(code.max()) + 1) + code.astype(np.int64)
    return out


def interaction_apply(codes, labels_, fit, apply, alpha=20.0):
    """Logit of the joint rate, shrunk toward the additive per-field model, minus the additive logit."""
    prior = float(labels_[fit].mean())
    additive = np.full(len(apply), -(len(codes) - 1) * float(logit(prior)), dtype=np.float64)
    for code in codes:
        count, positive = _counts(code, labels_, fit)
        additive += logit(np.clip((positive[code[apply]] + 20.0 * prior) / (count[code[apply]] + 20.0), 1e-5, 1 - 1e-5))
    joint = _joint(codes)
    count, positive = _counts(joint, labels_, fit)
    probability = (positive[joint[apply]] + alpha * expit(additive)) / (count[joint[apply]] + alpha)
    return (logit(np.clip(probability, 1e-5, 1 - 1e-5)) - additive).astype(np.float32)


def interaction_loo(codes, labels_, fit, alpha=20.0):
    fit_labels = labels_[fit].astype(np.float64)
    prior = (fit_labels.sum() - fit_labels) / (len(fit_labels) - 1.0)
    additive = -(len(codes) - 1) * logit(prior)
    for code in codes:
        count, positive = _counts(code, labels_, fit)
        own = code[fit]
        additive += logit(np.clip((positive[own] - fit_labels + 20.0 * prior) / (count[own] - 1.0 + 20.0), 1e-5, 1 - 1e-5))
    joint = _joint(codes)
    count, positive = _counts(joint, labels_, fit)
    own = joint[fit]
    probability = (positive[own] - fit_labels + alpha * expit(additive)) / (count[own] - 1.0 + alpha)
    return (logit(np.clip(probability, 1e-5, 1 - 1e-5)) - additive).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    train, test = read_train(), read_test()
    source = normalize_source(read_original_text())
    y, source_y = labels(train[TARGET]), source_labels(source)
    f = folds(y)
    n_train = len(train)
    test_rows = np.arange(n_train, n_train + len(test))

    # Value counts come from all train + test rows, so the bank does not depend on the fold.
    every_row = pd.concat([train[list(FEATURES)], test[list(FEATURES)]], ignore_index=True)
    bank, bank_names = build_source_bank(every_row, every_row, source, source_y)
    focus_names = [name for name in bank_names if name.rsplit('__', 1)[-1] in FOCUS_FIELDS]
    exact = exact_codes(train, test)
    income = every_row['Annual_Income_USD'].to_numpy(np.float64)
    commute = every_row['Daily_Commute_km'].to_numpy(np.float64)
    gender_home = [exact['Gender'], exact['Home_Charging_Possible']]

    for k in fold_list(a.folds):
        fit, valid = np.flatnonzero(f != k), np.flatnonzero(f == k)
        apply = np.concatenate([valid, test_rows])
        nv = len(valid)
        blocks = {'source_full_global': (bank[fit], bank[valid], bank[n_train:])}

        single_apply, single_names = single_response(exact, y, fit, apply)
        blocks['response_single_loo'] = (leave_one_out_single_response(exact, y, fit), single_apply[:nv], single_apply[nv:])

        # exact-income support (count) of every row, used to gate the +-25 window
        income_index = single_names.index('resp_logcount__Annual_Income_USD')
        count_fit = np.maximum(np.expm1(crossfit_income_count(exact['Annual_Income_USD'], y, fit, k).astype(np.float64)), 0.0)
        count_apply = np.maximum(np.expm1(single_apply[:, income_index].astype(np.float64)), 0.0)
        income10_fit, income10_apply = crossfit_neighborhood(income, y, fit, apply, k, 10.0, 5.0)
        income25_fit, income25_apply = crossfit_neighborhood(income, y, fit, apply, k, 25.0, 5.0)
        commute_fit, commute_apply = crossfit_neighborhood(commute, y, fit, apply, k, 0.1, 100.0)
        gated25_fit = (income25_fit[:, 1].astype(np.float64) * 50.0 / (count_fit + 50.0)).astype(np.float32)
        gated25_apply = (income25_apply[:, 1].astype(np.float64) * 50.0 / (count_apply + 50.0)).astype(np.float32)
        multiscale_fit = np.column_stack([income10_fit[:, 1], gated25_fit]).astype(np.float32)
        multiscale_apply = np.column_stack([income10_apply[:, 1], gated25_apply]).astype(np.float32)
        blocks['response_income_multiscale'] = (multiscale_fit, multiscale_apply[:nv], multiscale_apply[nv:])
        blocks['response_commute_residual'] = (commute_fit[:, 1:2], commute_apply[:nv, 1:2], commute_apply[nv:, 1:2])

        income10_loo, loo_count = leave_one_out_neighborhood(income, y, fit, 10.0, 5.0)
        income25_loo, _ = leave_one_out_neighborhood(income, y, fit, 25.0, 5.0)
        commute_loo, _ = leave_one_out_neighborhood(commute, y, fit, 0.1, 100.0)
        gated25_loo = (income25_loo[:, 1].astype(np.float64) * 50.0 / (loo_count + 50.0)).astype(np.float32)
        local_fit = np.column_stack([income10_loo[:, 1], gated25_loo, commute_loo[:, 1]]).astype(np.float32)
        local_apply = np.column_stack([multiscale_apply[:, 0], multiscale_apply[:, 1], commute_apply[:, 1]]).astype(np.float32)
        blocks['response_local_loo'] = (local_fit, local_apply[:nv], local_apply[nv:])

        lift_apply = interaction_apply(gender_home, y, fit, apply)[:, None]
        blocks['response_target_gender_home'] = (interaction_loo(gender_home, y, fit)[:, None], lift_apply[:nv], lift_apply[nv:])

        folder = CACHE / 'base' / f'fold_{k}'
        save_arrays(folder, {'train_indices': fit.astype(np.int64), 'valid_indices': valid.astype(np.int64)})
        for block, (a_tr, a_va, a_te) in blocks.items():
            save_arrays(folder, {f'{block}_train': a_tr, f'{block}_valid': a_va, f'{block}_test': a_te})
        update_manifest(folder, {
            'source_full_global_names': bank_names,
            'source_focus_names': focus_names,
            'response_single_loo_names': single_names,
            'response_local_loo_names': ['resp_delta__income_window10_alpha5_local_loo',
                                         'resp_delta__income_window25_alpha5_exactsupport50_local_loo',
                                         'resp_delta__commute_window0.1_alpha100_local_loo'],
            'response_income_multiscale_names': ['resp_delta__income_window10_alpha5_vs_exact5',
                                                 'resp_delta__income_window25_alpha5_exactsupport50'],
            'response_commute_residual_names': ['resp_delta__commute_window0.1_alpha100_vs_exact100'],
            'response_target_gender_home_names': ['pair_lift__targeted__Gender__Home_Charging_Possible'],
        })
        print(f'fold {k}: {", ".join(blocks)} -> {folder}', flush=True)


if __name__ == '__main__':
    main()
