#!/usr/bin/env python
"""Inner-cross-fitted response channels on the screen-allocation geometry, one cache per outer fold.

Row-local geometry of the screen fields (18 values): daily, component sum (social + gaming + work), strict slack
(daily - sum, NaN unless all 3 components are observed), partial slack, slack / daily, sum / daily, the 3 component
shares of daily and of the sum, Herfindahl index, dominance, spread, weekend - daily, weekend / daily, daily / sleep.
Keys: each geometry value cut at its q16 and q64 quantile edges (edges from the fold's outer-training rows, NaN = own
key), 5 q16 x q16 pairs and the 6-bit screen missing mask (42 keys).
Per key: smoothed target rate (m = 20) -> logit, and log1p(count). The fold's outer-training rows are encoded with an
inner KFold(5, shuffle, random_state 742 + fold); the fold's rows and the test rows with all outer-training rows.
Writes cache/screen_response/fold_{k}_{train,valid,test}.npy (84 columns), fold_{k}_{train,valid}_indices.npy and
fold_{k}_manifest.json.
Usage: python build_screen_response.py [--folds 0,1,2,3,4]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import folds as fold_ids, labels, read_test, read_train, TARGET  # noqa: E402
from common.pairs import fold_list  # noqa: E402
from models.forensic_features.lib.blocks import cache, safe_ratio  # noqa: E402

EPSILON = 1e-6
PAIR_KEYS = (("strict_slack", "daily"), ("strict_slack", "weekend_minus_daily"), ("component_share_daily", "daily"),
             ("component_hhi", "daily"), ("daily", "weekend_minus_daily"))


def geometry(frame):
    columns = ["daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "sleep_hours",
               "weekend_screen_time"]
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    daily, social, gaming, work, sleep, weekend = values.T
    components = np.column_stack([social, gaming, work])
    observed = np.isfinite(components)
    filled = np.where(observed, components, 0.0)
    component_sum = filled.sum(axis=1)
    complete = observed.all(axis=1)
    strict_sum = component_sum.copy()
    strict_sum[~complete] = np.nan
    strict_slack = daily - strict_sum
    partial_slack = daily - component_sum
    component_share = filled / np.where(component_sum[:, None] > 0, component_sum[:, None], 1.0)
    daily_share = filled / np.where(daily[:, None] > 0, daily[:, None], np.nan)
    output = np.column_stack([
        daily, component_sum, strict_slack, partial_slack, safe_ratio(strict_slack, daily),
        safe_ratio(component_sum, daily), daily_share, component_share, np.sum(component_share ** 2, axis=1),
        np.max(component_share, axis=1), np.max(filled, axis=1) - np.min(filled, axis=1), weekend - daily,
        safe_ratio(weekend, daily), safe_ratio(daily, sleep)])
    names = ["daily", "component_sum", "strict_slack", "partial_slack", "strict_slack_share_daily",
             "component_share_daily", "social_share_daily", "gaming_share_daily", "work_share_daily",
             "social_share_components", "gaming_share_components", "work_share_components", "component_hhi",
             "component_dominance", "component_spread", "weekend_minus_daily", "weekend_over_daily", "daily_over_sleep"]
    missing_mask = (frame[columns].isna().to_numpy(np.int32) * (1 << np.arange(6))).sum(axis=1)
    return output, names, missing_mask.astype(np.int32)


def quantile_codes(fit_values, apply_values, bins):
    finite = fit_values[np.isfinite(fit_values)]
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    output = np.full(len(apply_values), -1, dtype=np.int32)
    observed = np.isfinite(apply_values)
    output[observed] = np.searchsorted(edges[1:-1], apply_values[observed], side="right").astype(np.int32)
    return output


def pair_codes(left, right, bins):
    output = np.full(len(left), -1, dtype=np.int32)
    observed = (left >= 0) & (right >= 0)
    output[observed] = left[observed] * bins + right[observed]
    return output


def encode(fit_key, fit_target, apply_key):
    maximum = max(int(fit_key.max()), int(apply_key.max())) + 2
    fit_shifted, apply_shifted = fit_key + 1, apply_key + 1
    count = np.bincount(fit_shifted, minlength=maximum).astype(np.float64)
    total = np.bincount(fit_shifted, weights=fit_target, minlength=maximum).astype(np.float64)
    prior = float(fit_target.mean())
    probability = (total[apply_shifted] + 20.0 * prior) / (count[apply_shifted] + 20.0)
    probability = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return np.column_stack([np.log(probability) - np.log1p(-probability),
                            np.log1p(count[apply_shifted])]).astype(np.float32)


def build_fold(fold, train, test, target, folds):
    train_geometry, geometry_names, train_missing = geometry(train)
    test_geometry, _, test_missing = geometry(test)
    train_indices = np.flatnonzero(folds != fold)
    valid_indices = np.flatnonzero(folds == fold)
    inner = KFold(n_splits=5, shuffle=True, random_state=742 + fold)
    inner_fold = np.full(len(train_indices), -1, dtype=np.int8)
    for inner_number, (_, held) in enumerate(inner.split(train_indices)):
        inner_fold[held] = inner_number
    key_data, code_cache = [], {}
    for bins in (16, 64):
        for index, name in enumerate(geometry_names):
            fit_values = train_geometry[train_indices, index]
            train_key = quantile_codes(fit_values, train_geometry[:, index], bins)
            test_key = quantile_codes(fit_values, test_geometry[:, index], bins)
            key_data.append((f"{name}__q{bins}", train_key, test_key))
            code_cache[(name, bins)] = (train_key, test_key)
    for left, right in PAIR_KEYS:
        train_left, test_left = code_cache[(left, 16)]
        train_right, test_right = code_cache[(right, 16)]
        key_data.append((f"pair__{left}__{right}__q16", pair_codes(train_left, train_right, 16),
                         pair_codes(test_left, test_right, 16)))
    key_data.append(("screen_missing_mask", train_missing, test_missing))

    train_parts, valid_parts, test_parts, feature_names = [], [], [], []
    outer_target = target[train_indices]
    for key_name, global_train_key, test_key in key_data:
        outer_key = global_train_key[train_indices]
        encoded_train = np.empty((len(train_indices), 2), dtype=np.float32)
        for inner_number in range(5):
            fit_mask, held_mask = inner_fold != inner_number, inner_fold == inner_number
            encoded_train[held_mask] = encode(outer_key[fit_mask], outer_target[fit_mask], outer_key[held_mask])
        train_parts.append(encoded_train)
        valid_parts.append(encode(outer_key, outer_target, global_train_key[valid_indices]))
        test_parts.append(encode(outer_key, outer_target, test_key))
        feature_names.extend([f"screen_response_logit__{key_name}", f"screen_response_count__{key_name}"])

    out = cache() / "screen_response"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"fold_{fold}_train.npy", np.column_stack(train_parts).astype(np.float32))
    np.save(out / f"fold_{fold}_valid.npy", np.column_stack(valid_parts).astype(np.float32))
    np.save(out / f"fold_{fold}_test.npy", np.column_stack(test_parts).astype(np.float32))
    np.save(out / f"fold_{fold}_train_indices.npy", train_indices)
    np.save(out / f"fold_{fold}_valid_indices.npy", valid_indices)
    (out / f"fold_{fold}_manifest.json").write_text(
        json.dumps({"fold": fold, "feature_count": len(feature_names), "feature_names": feature_names}, indent=1) + "\n")
    print(f"screen response fold {fold}: {len(key_data)} keys, {len(feature_names)} columns", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    train, test = read_train(), read_test()
    target = labels(train[TARGET])
    folds = fold_ids(target)
    for fold in fold_list(args.folds):
        build_fold(fold, train, test, target, folds)


if __name__ == "__main__":
    main()
