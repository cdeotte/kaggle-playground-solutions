#!/usr/bin/env python
"""Label-free source-fragment cache for all train + test rows (CatBoost-native categories and numeric companions).

Fitted from the original's predictors / labels and the unlabeled train + test predictors (the competition target is
not read). Values are tokenized exactly (age, notifications, opens x1; hours x100).
numeric.npy (float32)
  raw / observed            the 9 numerics and their observed flags
  source_cat_prior          smoothed (m 5) original target rate of each categorical level
  source_alignment          per numeric: signed / absolute distance to the nearest original value, exact-support flag,
                            log1p(original exact count), original rate of the exact value / of the nearest value, rank
  domain                    20 screen-budget / ratio / threshold coordinates
  fragment_summary          pair-recurrence summaries: mean / max log pair count, # recurrent pairs (>= 2, >= 10),
                            # original-supported pairs, # unsupported recurrent pairs, unique-original-row label votes,
                            consensus original row votes and label
categorical.npy (int32)
  raw_cat, mask (5 missing-pattern bitmasks + missing count), exact_value (exact tokens), semantic_state (original
  rule region, screen allocation state), source_nearest_value, source_residual_tick (clipped tick distance),
  recurrence_c2 / recurrence_c10 (pair id when the exact pair recurs >= 2 / >= 10 times in train + test),
  source_pair_row (0 missing, 1 not in original, 2 ambiguous, 3 + row of the unique matching original row),
  fragment_summary (consensus original row, its vote count)
Writes cache/cat_source_fragments_v1/{numeric.npy, categorical.npy, manifest.json} (train rows first, then test).
Usage: python build_source_fragments.py"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import cache_dir, read_original, read_test, read_train  # noqa: E402

NUMERIC = ("age", "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "sleep_hours",
           "notifications_per_day", "app_opens_per_day", "weekend_screen_time")
CATEGORICAL = ("gender", "stress_level", "academic_work_impact")
SCREEN = ("daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "weekend_screen_time")
SCALES = {"age": 1, "daily_screen_time_hours": 100, "social_media_hours": 100, "gaming_hours": 100,
          "work_study_hours": 100, "sleep_hours": 100, "notifications_per_day": 1, "app_opens_per_day": 1,
          "weekend_screen_time": 100}
MISSING_TOKEN = np.iinfo(np.int32).min


def safe_ratio(numerator, denominator):
    output = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0.0)
    output[valid] = numerator[valid] / denominator[valid]
    return output


def scaled_tokens(values, scale):
    output = np.full(len(values), MISSING_TOKEN, dtype=np.int64)
    observed = np.isfinite(values)
    output[observed] = np.rint(values[observed] * scale).astype(np.int64)
    return output


def categorical_codes(combined, source):
    combined_values = combined.astype("string").fillna("<NA>").astype(str).to_numpy()
    source_values = source.astype("string").fillna("<NA>").astype(str).to_numpy()
    levels = sorted(set(combined_values) | set(source_values))
    lookup = {value: index for index, value in enumerate(levels)}
    return (np.fromiter((lookup[value] for value in combined_values), np.int32),
            np.fromiter((lookup[value] for value in source_values), np.int32), levels)


def consensus_id(matrix):
    """An original row id only when at least two supported pair votes agree."""
    ordered = np.sort(np.where(matrix >= 3, matrix, 0), axis=1)
    rows = np.arange(len(matrix))
    winner = np.zeros(len(matrix), dtype=np.int32)
    strength = np.zeros(len(matrix), dtype=np.int16)
    for run_length in range(2, 16):
        matches = (ordered[:, run_length - 1:] == ordered[:, :-(run_length - 1)]) & (ordered[:, run_length - 1:] >= 3)
        any_match = matches.any(axis=1)
        if not any_match.any():
            continue
        first = np.argmax(matches, axis=1)
        winner[any_match] = ordered[rows[any_match], first[any_match]]
        strength[any_match] = run_length
    return winner, strength


def main():
    columns = ["id", *NUMERIC, *CATEGORICAL]
    train, test = read_train()[columns], read_test()[columns]
    source = read_original()[[*NUMERIC, *CATEGORICAL, "addicted_label"]]
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    values = {name: pd.to_numeric(combined[name], errors="coerce").to_numpy(np.float64) for name in NUMERIC}
    source_values = {name: pd.to_numeric(source[name], errors="raise").to_numpy(np.float64) for name in NUMERIC}
    tokens = {name: scaled_tokens(values[name], SCALES[name]) for name in NUMERIC}
    source_tokens = {name: scaled_tokens(source_values[name], SCALES[name]) for name in NUMERIC}
    source_target = source["addicted_label"].to_numpy(np.float64)
    source_prior = float(source_target.mean())

    numeric_parts, numeric_names, categorical_parts, categorical_names, groups = [], [], [], [], {}

    def add_numeric(prefix, matrix, names):
        local = np.asarray(matrix, dtype=np.float32)
        local = local[:, None] if local.ndim == 1 else local
        qualified = [f"{prefix}__{name}" for name in names]
        numeric_parts.append(local)
        numeric_names.extend(qualified)
        groups.setdefault(prefix, []).extend(qualified)

    def add_categorical(prefix, matrix, names):
        local = np.asarray(matrix, dtype=np.int32)
        local = local[:, None] if local.ndim == 1 else local
        qualified = [f"{prefix}__{name}" for name in names]
        categorical_parts.append(local)
        categorical_names.extend(qualified)
        groups.setdefault(prefix, []).extend(qualified)

    raw_numeric = np.column_stack([values[name] for name in NUMERIC])
    observed_numeric = np.isfinite(raw_numeric).astype(np.float32)
    add_numeric("raw", raw_numeric, list(NUMERIC))
    add_numeric("observed", observed_numeric, list(NUMERIC))

    raw_cat_codes, raw_cat_rates, cat_levels = {}, [], {}
    for name in CATEGORICAL:
        code, source_code, levels = categorical_codes(combined[name], source[name])
        raw_cat_codes[name] = code
        cat_levels[name] = levels
        sums = np.bincount(source_code, weights=source_target, minlength=len(levels))
        counts = np.bincount(source_code, minlength=len(levels)).astype(np.float64)
        raw_cat_rates.append(((sums + 5.0 * source_prior) / (counts + 5.0))[code])
    add_categorical("raw_cat", np.column_stack([raw_cat_codes[name] for name in CATEGORICAL]), list(CATEGORICAL))
    add_numeric("source_cat_prior", np.column_stack(raw_cat_rates), list(CATEGORICAL))

    missing = np.column_stack([combined[name].isna().to_numpy() for name in [*NUMERIC, *CATEGORICAL]])
    mask_views = {"all12": tuple(range(12)), "numeric9": tuple(range(9)),
                  "screen5": tuple(NUMERIC.index(name) for name in SCREEN),
                  "behavior2": (NUMERIC.index("notifications_per_day"), NUMERIC.index("app_opens_per_day")),
                  "categorical3": (9, 10, 11)}
    mask_codes = []
    for indices in mask_views.values():
        weights = (1 << np.arange(len(indices), dtype=np.int64))[None, :]
        mask_codes.append(np.sum(missing[:, indices] * weights, axis=1, dtype=np.int64))
    add_categorical("mask", np.column_stack(mask_codes), list(mask_views))
    add_categorical("mask", missing.sum(axis=1).astype(np.int32), ["missing_count"])
    add_categorical("exact_value", np.column_stack(
        [np.where(tokens[name] == MISSING_TOKEN, 0, tokens[name] + 1) for name in NUMERIC]), list(NUMERIC))

    nearest_codes, residual_codes, alignment, alignment_names = [], [], [], []
    for name in NUMERIC:
        token, source_token = tokens[name], source_tokens[name]
        unique, inverse, counts = np.unique(source_token, return_inverse=True, return_counts=True)
        sums = np.bincount(inverse, weights=source_target, minlength=len(unique))
        rates = (sums + 5.0 * source_prior) / (counts + 5.0)
        observed = token != MISSING_TOKEN
        query = token[observed]
        upper = np.clip(np.searchsorted(unique, query, side="left"), 0, len(unique) - 1)
        lower = np.clip(upper - 1, 0, len(unique) - 1)
        nearest_index = np.where(np.abs(unique[upper] - query) < np.abs(unique[lower] - query), upper, lower)
        residual_tick = query - unique[nearest_index]
        exact = residual_tick == 0
        nearest_code = np.zeros(len(combined), dtype=np.int32)
        nearest_code[observed] = nearest_index.astype(np.int32) + 1
        residual_code = np.zeros(len(combined), dtype=np.int32)
        residual_code[observed] = np.clip(residual_tick, -20, 20).astype(np.int32) + 22
        exact_count = np.zeros(len(combined), dtype=np.float64)
        exact_rate = np.full(len(combined), source_prior, dtype=np.float64)
        nearest_rate = np.full(len(combined), source_prior, dtype=np.float64)
        exact_count[observed] = np.where(exact, counts[nearest_index], 0.0)
        exact_rate[observed] = np.where(exact, rates[nearest_index], source_prior)
        nearest_rate[observed] = rates[nearest_index]
        residual = np.full(len(combined), np.nan, dtype=np.float64)
        residual[observed] = residual_tick.astype(np.float64) / float(SCALES[name])
        rank = np.full(len(combined), np.nan, dtype=np.float64)
        rank[observed] = (nearest_index + 0.5) / len(unique)
        nearest_codes.append(nearest_code)
        residual_codes.append(residual_code)
        alignment += [residual, np.abs(residual), (exact_count > 0).astype(np.float64), np.log1p(exact_count),
                      exact_rate, nearest_rate, rank]
        alignment_names += [f"{name}__{suffix}" for suffix in ("signed_residual", "abs_residual", "exact_support",
                                                                "exact_logcount", "exact_label_prior",
                                                                "nearest_label_prior", "nearest_rank")]
    add_categorical("source_nearest_value", np.column_stack(nearest_codes), list(NUMERIC))
    add_categorical("source_residual_tick", np.column_stack(residual_codes), list(NUMERIC))
    add_numeric("source_alignment", np.column_stack(alignment), alignment_names)

    daily, social, gaming, work = (values[n] for n in ("daily_screen_time_hours", "social_media_hours", "gaming_hours",
                                                        "work_study_hours"))
    sleep, notifications, opens, weekend = (values[n] for n in ("sleep_hours", "notifications_per_day",
                                                                 "app_opens_per_day", "weekend_screen_time"))
    components = np.column_stack([social, gaming, work])
    component_observed = np.isfinite(components)
    component_sum = np.where(component_observed, components, 0.0).sum(axis=1)
    component_count = component_observed.sum(axis=1)
    component_complete = component_count == 3
    strict_sum = component_sum.copy()
    strict_sum[~component_complete] = np.nan
    domain_values = np.column_stack([
        component_sum, component_count, daily - component_sum, daily - strict_sum, daily - social, daily - gaming,
        daily - work, safe_ratio(social, daily), safe_ratio(gaming, daily), safe_ratio(work, daily), weekend - daily,
        safe_ratio(weekend, daily), safe_ratio(daily, sleep), safe_ratio(notifications, opens),
        np.log1p(np.clip(notifications, 0.0, None)), np.log1p(np.clip(opens, 0.0, None)), daily - 6.0, daily - 8.0,
        social - 4.0, observed_numeric.sum(axis=1)])
    add_numeric("domain", domain_values, [
        "component_sum", "component_count", "partial_slack", "strict_slack", "daily_minus_social", "daily_minus_gaming",
        "daily_minus_work", "social_share_daily", "gaming_share_daily", "work_share_daily", "weekend_minus_daily",
        "weekend_over_daily", "daily_over_sleep", "notifications_per_open", "log_notifications", "log_opens",
        "daily_minus_6", "daily_minus_8", "social_minus_4", "observed_numeric_count"])
    raw_rule = np.zeros(len(combined), dtype=np.int32)
    rule_observed = np.isfinite(daily) & np.isfinite(social)
    raw_rule[rule_observed] = 1
    raw_rule[rule_observed & (daily <= 6.0) & (social <= 4.0)] = 2
    raw_rule[rule_observed & ((daily > 8.0) | (social > 4.0))] = 3
    allocation_state = np.zeros(len(combined), dtype=np.int32)
    allocation_state[np.isfinite(daily) & component_complete] = 1
    allocation_state[np.isfinite(daily) & component_complete & (daily >= component_sum)] = 2
    add_categorical("semantic_state", np.column_stack([raw_rule, allocation_state]),
                    ["source_rule_region", "screen_allocation_state"])

    pair_names, recurrence2, recurrence10, source_pair_rows = [], [], [], []
    count_columns, support_columns, label_columns = [], [], []
    for left, right in combinations(NUMERIC, 2):
        pair_names.append(f"{left}__{right}")
        left_values, right_values = tokens[left], tokens[right]
        complete = (left_values != MISSING_TOKEN) & (right_values != MISSING_TOKEN)
        source_left, source_right = source_tokens[left], source_tokens[right]
        base = int(max(right_values[complete].max(initial=0), source_right.max()) + 1)
        keys = left_values[complete] * base + right_values[complete]
        unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        local_counts = counts[inverse]
        c2 = np.zeros(len(combined), dtype=np.int32)
        c10 = np.zeros(len(combined), dtype=np.int32)
        c2[complete] = np.where(local_counts >= 2, inverse + 2, 1).astype(np.int32)
        c10[complete] = np.where(local_counts >= 10, inverse + 2, 1).astype(np.int32)
        recurrence2.append(c2)
        recurrence10.append(c10)
        count_column = np.zeros(len(combined), dtype=np.float32)
        count_column[complete] = local_counts
        count_columns.append(count_column)

        source_unique, source_first, source_counts = np.unique(source_left * base + source_right, return_index=True,
                                                               return_counts=True)
        positions = np.searchsorted(source_unique, keys)
        supported_local = (positions < len(source_unique)) & (
            source_unique[np.clip(positions, 0, len(source_unique) - 1)] == keys)
        supported = np.zeros(len(combined), dtype=np.int32)
        supported[complete] = supported_local.astype(np.int32)
        support_columns.append(supported)
        row_code = np.zeros(len(combined), dtype=np.int32)
        row_code[complete] = 1
        local_source_count = np.zeros(len(keys), dtype=np.int64)
        local_source_count[supported_local] = source_counts[positions[supported_local]]
        ambiguous = supported_local & (local_source_count > 1)
        unique_support = supported_local & (local_source_count == 1)
        complete_rows = np.flatnonzero(complete)
        row_code[complete_rows[ambiguous]] = 2
        matched_source_row = source_first[positions[unique_support]]
        row_code[complete_rows[unique_support]] = matched_source_row.astype(np.int32) + 3
        source_pair_rows.append(row_code)
        label_column = np.full(len(combined), np.nan, dtype=np.float32)
        label_column[complete_rows[unique_support]] = source_target[matched_source_row].astype(np.float32)
        label_columns.append(label_column)

    source_pair_matrix = np.column_stack(source_pair_rows)
    count_matrix = np.column_stack(count_columns).astype(np.float32)
    support_matrix = np.column_stack(support_columns).astype(np.float32)
    pair_label_matrix = np.column_stack(label_columns).astype(np.float32)
    add_categorical("recurrence_c2", np.column_stack(recurrence2), pair_names)
    add_categorical("recurrence_c10", np.column_stack(recurrence10), pair_names)
    add_categorical("source_pair_row", source_pair_matrix, pair_names)

    consensus, consensus_strength = consensus_id(source_pair_matrix)
    pair_label_count = np.isfinite(pair_label_matrix).sum(axis=1)
    pair_label_rate = np.full(len(combined), source_prior, dtype=np.float64)
    has_pair_label = pair_label_count > 0
    pair_label_rate[has_pair_label] = (np.nansum(pair_label_matrix, axis=1)[has_pair_label] + 2.0 * source_prior) / (
        pair_label_count[has_pair_label] + 2.0)
    consensus_label = np.full(len(combined), source_prior, dtype=np.float64)
    consensus_rows = consensus >= 3
    consensus_label[consensus_rows] = source_target[consensus[consensus_rows] - 3]
    add_numeric("fragment_summary", np.column_stack([
        np.log1p(count_matrix).mean(axis=1), np.log1p(count_matrix).max(axis=1), (count_matrix >= 2).sum(axis=1),
        (count_matrix >= 10).sum(axis=1), support_matrix.sum(axis=1),
        ((count_matrix >= 2) & (support_matrix == 0)).sum(axis=1),
        ((count_matrix >= 10) & (support_matrix == 0)).sum(axis=1), pair_label_count, pair_label_rate,
        consensus_strength, consensus_label]), [
        "mean_log_pair_count", "max_log_pair_count", "recurrent2_pair_count", "recurrent10_pair_count",
        "source_supported_pair_count", "unsupported_recurrent2_count", "unsupported_recurrent10_count",
        "unique_source_pair_label_count", "source_pair_label_rate", "consensus_pair_votes", "consensus_source_label"])
    add_categorical("fragment_summary", np.column_stack([consensus, consensus_strength.astype(np.int32)]),
                    ["consensus_source_row", "consensus_pair_votes"])

    out = cache_dir("catboost_forensic") / "cat_source_fragments_v1"
    out.mkdir(parents=True, exist_ok=True)
    numeric = np.column_stack(numeric_parts).astype(np.float32)
    categorical = np.column_stack(categorical_parts).astype(np.int32)
    np.save(out / "numeric.npy", numeric)
    np.save(out / "categorical.npy", categorical)
    manifest = {"rows": len(combined), "train_rows": len(train), "test_rows": len(test),
                "numeric_names": numeric_names, "categorical_names": categorical_names, "groups": groups,
                "source_prior": source_prior, "categorical_levels": cat_levels}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"source fragments: numeric {numeric.shape}, categorical {categorical.shape}", flush=True)


if __name__ == "__main__":
    main()
