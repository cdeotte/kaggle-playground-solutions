#!/usr/bin/env python
"""Fixed-original likelihood channels (583 columns) for every train and test row.

All fitted state comes from the 7,500-row original dataset (its predictors, its binary label and its 4-level
addiction_level); the competition labels are not used.
  univariate         per key: binary log-likelihood ratio (alpha 20), smoothed posterior, log1p(original count);
                     keys = each numeric rounded to 2 decimals and cut at the original's q8/q16/q32/q64/q128 quantile
                     edges, each categorical's exact level
  latent_univariate  per q16 / exact key: log-ratio of the 4 addiction-level likelihoods to the key's base rate
  pair               all 66 pairs of the 12 predictors on their q8 / exact codes: LLR (alpha 50), posterior, log count
  geometry           per numeric: empirical CDF and nearest distance to the class-0 / class-1 original values, deltas,
                     observed flag, exact-2-decimal original support
  aggregate          naive-Bayes sums of the univariate LLRs per block (all / screen / core / background /
                     categorical) and resolution, the pair-LLR sum, softmax addiction-level posteriors per block
  missing            missing flags and missing counts
Writes cache/source_channel_{train,test}.npy and cache/source_channel_manifest.json.
Usage: python build_source_channel.py"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import TARGET, read_original, read_test, read_train  # noqa: E402
from models.forensic_features.lib.blocks import CATEGORICAL, NUMERIC, PREDICTORS, cache  # noqa: E402

LEVELS = ["None", "Mild", "Moderate", "Severe"]
BLOCKS = {
    "all": PREDICTORS,
    "screen": ["daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "weekend_screen_time"],
    "core": ["daily_screen_time_hours", "social_media_hours", "weekend_screen_time"],
    "background": ["age", "sleep_hours", "notifications_per_day", "app_opens_per_day"],
    "categorical": CATEGORICAL,
}
QUANTILE_BINS = (8, 16, 32, 64, 128)


def numeric_values(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def encode_source_values(source_values, train_values, test_values, decimals):
    """Exact (rounded) values that occur in the original -> code; anything else -> -1."""
    missing = np.iinfo(np.int64).min
    scale = 10 ** decimals

    def scaled(values):
        result = np.full(len(values), missing, dtype=np.int64)
        finite = np.isfinite(values)
        result[finite] = np.rint(values[finite] * scale).astype(np.int64)
        return result

    source_scaled = scaled(source_values)
    unique = np.unique(source_scaled[source_scaled != missing])

    def codes(values):
        raw = scaled(values)
        result = np.full(len(raw), -1, dtype=np.int32)
        valid = raw != missing
        position = np.searchsorted(unique, raw[valid])
        clipped = np.minimum(position, len(unique) - 1)
        matched = unique[clipped] == raw[valid]
        valid_indices = np.flatnonzero(valid)
        result[valid_indices[matched]] = clipped[matched].astype(np.int32)
        return result

    source_code = np.searchsorted(unique, source_scaled).astype(np.int32)
    return source_code, codes(train_values), codes(test_values)


def encode_quantiles(source_values, train_values, test_values, bins):
    finite_source = source_values[np.isfinite(source_values)]
    edges = np.unique(np.quantile(finite_source, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf], dtype=np.float64)
    else:
        edges[0], edges[-1] = -np.inf, np.inf

    def codes(values):
        result = np.full(len(values), -1, dtype=np.int32)
        finite = np.isfinite(values)
        result[finite] = np.searchsorted(edges[1:-1], values[finite], side="right").astype(np.int32)
        return result

    return codes(source_values), codes(train_values), codes(test_values)


def encode_category(source, train, test):
    levels = np.sort(source.astype("string").dropna().unique())

    def codes(series):
        values = series.astype("string").to_numpy()
        result = np.full(len(values), -1, dtype=np.int32)
        observed = ~pd.isna(values)
        position = np.searchsorted(levels, values[observed])
        clipped = np.minimum(position, len(levels) - 1)
        matched = levels[clipped] == values[observed]
        observed_indices = np.flatnonzero(observed)
        result[observed_indices[matched]] = clipped[matched].astype(np.int32)
        return result

    return codes(source), codes(train), codes(test)


def binary_evidence(source_key, train_key, test_key, target, alpha):
    valid = source_key >= 0
    n_keys = int(source_key[valid].max()) + 1
    count = np.bincount(source_key[valid], minlength=n_keys).astype(np.float64)
    count1 = np.bincount(source_key[valid], weights=target[valid], minlength=n_keys)
    count0 = count - count1
    n1 = float(target[valid].sum())
    n0 = float(valid.sum() - n1)
    prior = n1 / (n0 + n1)
    base = count / count.sum()
    probability1 = (count1 + alpha * base) / (n1 + alpha)
    probability0 = (count0 + alpha * base) / (n0 + alpha)
    llr_map = np.log(np.clip(probability1, 1e-12, None)) - np.log(np.clip(probability0, 1e-12, None))
    posterior_map = (count1 + alpha * prior) / (count + alpha)

    def apply(key):
        observed = key >= 0
        llr = np.zeros(len(key), dtype=np.float32)
        posterior = np.full(len(key), prior, dtype=np.float32)
        reliability = np.zeros(len(key), dtype=np.float32)
        llr[observed] = llr_map[key[observed]].astype(np.float32)
        posterior[observed] = posterior_map[key[observed]].astype(np.float32)
        reliability[observed] = np.log1p(count[key[observed]]).astype(np.float32)
        return llr, posterior, reliability

    return (*apply(train_key), *apply(test_key))


def latent_evidence(source_key, train_key, test_key, level, alpha):
    valid = source_key >= 0
    n_keys = int(source_key[valid].max()) + 1
    total = np.bincount(source_key[valid], minlength=n_keys).astype(np.float64)
    base = total / total.sum()
    maps = np.empty((len(LEVELS), n_keys), dtype=np.float64)
    for index in range(len(LEVELS)):
        mask = valid & (level == index)
        counts = np.bincount(source_key[mask], minlength=n_keys).astype(np.float64)
        probability = (counts + alpha * base) / (mask.sum() + alpha)
        maps[index] = np.log(np.clip(probability, 1e-12, None)) - np.log(np.clip(base, 1e-12, None))

    def apply(key):
        output = np.zeros((len(key), len(LEVELS)), dtype=np.float32)
        observed = key >= 0
        output[observed] = maps[:, key[observed]].T.astype(np.float32)
        return output

    return apply(train_key), apply(test_key)


def combine_keys(left, right, right_cardinality):
    result = np.full(len(left), -1, dtype=np.int32)
    valid = (left >= 0) & (right >= 0)
    result[valid] = left[valid] * right_cardinality + right[valid]
    return result


def empirical_geometry(values, reference):
    reference = np.sort(reference[np.isfinite(reference)])
    percentile = np.full(len(values), 0.5, dtype=np.float32)
    distance = np.zeros(len(values), dtype=np.float32)
    finite = np.isfinite(values)
    query = values[finite]
    position = np.searchsorted(reference, query, side="right")
    percentile[finite] = (position / len(reference)).astype(np.float32)
    lo = reference[np.clip(position - 1, 0, len(reference) - 1)]
    hi = reference[np.clip(position, 0, len(reference) - 1)]
    scale = max(float(np.std(reference, ddof=1)), 1e-8)
    distance[finite] = (np.minimum(np.abs(query - lo), np.abs(query - hi)) / scale).astype(np.float32)
    return percentile, distance


def build():
    train, test, source = read_train(), read_test(), read_original(keep_level=True)
    source_y = source[TARGET].to_numpy(np.int8)
    source_levels = source["addiction_level"].fillna("None").map(LEVELS.index).to_numpy(np.int8)
    if not np.array_equal((source_levels >= 2).astype(np.int8), source_y):
        raise RuntimeError("original addiction level does not collapse to the binary target")
    keys, base_pair_keys = {}, {}
    for column in NUMERIC:
        source_values, train_values, test_values = (numeric_values(f, column) for f in (source, train, test))
        keys[f"{column}__round2"] = encode_source_values(source_values, train_values, test_values, 2)
        for bins in QUANTILE_BINS:
            encoded = encode_quantiles(source_values, train_values, test_values, bins)
            keys[f"{column}__q{bins}"] = encoded
            if bins == 8:
                base_pair_keys[column] = encoded
    for column in CATEGORICAL:
        encoded = encode_category(source[column], train[column], test[column])
        keys[f"{column}__exact"] = encoded
        base_pair_keys[column] = encoded

    train_parts, test_parts, names, groups = [], [], [], {}

    def add(group, train_values, test_values, new_names):
        train_array = np.asarray(train_values, dtype=np.float32)
        test_array = np.asarray(test_values, dtype=np.float32)
        if train_array.ndim == 1:
            train_array, test_array = train_array[:, None], test_array[:, None]
        start = len(names)
        train_parts.append(train_array)
        test_parts.append(test_array)
        names.extend(new_names)
        groups.setdefault(group, []).extend(range(start, start + len(new_names)))

    binary_sums_train, binary_sums_test = {}, {}
    latent_sums_train = {block: np.zeros((len(train), 4), np.float64) for block in BLOCKS}
    latent_sums_test = {block: np.zeros((len(test), 4), np.float64) for block in BLOCKS}
    for key_name, (source_key, train_key, test_key) in keys.items():
        column = key_name.split("__", 1)[0]
        train_llr, train_post, train_count, test_llr, test_post, test_count = binary_evidence(
            source_key, train_key, test_key, source_y, alpha=20.0)
        add("univariate", np.column_stack([train_llr, train_post, train_count]),
            np.column_stack([test_llr, test_post, test_count]),
            [f"binary_llr__{key_name}", f"binary_posterior__{key_name}", f"source_logcount__{key_name}"])
        resolution = key_name.split("__", 1)[1]
        for block, columns in BLOCKS.items():
            if column in columns:
                binary_sums_train.setdefault((block, resolution), np.zeros(len(train), np.float64))
                binary_sums_test.setdefault((block, resolution), np.zeros(len(test), np.float64))
                binary_sums_train[(block, resolution)] += train_llr
                binary_sums_test[(block, resolution)] += test_llr
        if resolution in {"q16", "exact"}:
            train_latent, test_latent = latent_evidence(source_key, train_key, test_key, source_levels, alpha=20.0)
            add("latent_univariate", train_latent, test_latent, [f"latent_logrel__{level}__{key_name}" for level in LEVELS])
            for block, columns in BLOCKS.items():
                if column in columns:
                    latent_sums_train[block] += train_latent
                    latent_sums_test[block] += test_latent

    pair_llr_train, pair_llr_test = np.zeros(len(train), np.float64), np.zeros(len(test), np.float64)
    for left, right in combinations(PREDICTORS, 2):
        source_left, train_left, test_left = base_pair_keys[left]
        source_right, train_right, test_right = base_pair_keys[right]
        right_cardinality = int(source_right.max()) + 1
        source_key = combine_keys(source_left, source_right, right_cardinality)
        train_key = combine_keys(train_left, train_right, right_cardinality)
        test_key = combine_keys(test_left, test_right, right_cardinality)
        train_llr, train_post, train_count, test_llr, test_post, test_count = binary_evidence(
            source_key, train_key, test_key, source_y, alpha=50.0)
        add("pair", np.column_stack([train_llr, train_post, train_count]),
            np.column_stack([test_llr, test_post, test_count]),
            [f"pair_llr__{left}__{right}", f"pair_posterior__{left}__{right}", f"pair_logcount__{left}__{right}"])
        pair_llr_train += train_llr
        pair_llr_test += test_llr

    suffixes = ["cdf0", "cdf1", "cdf_delta", "distance0", "distance1", "logdistance_delta", "observed", "source_supported"]
    for column in NUMERIC:
        source_values, train_values, test_values = (numeric_values(f, column) for f in (source, train, test))
        _, train_key, test_key = keys[f"{column}__round2"]
        blocks = []
        for values, key in ((train_values, train_key), (test_values, test_key)):
            cdf0, dist0 = empirical_geometry(values, source_values[source_y == 0])
            cdf1, dist1 = empirical_geometry(values, source_values[source_y == 1])
            blocks.append(np.column_stack([cdf0, cdf1, cdf1 - cdf0, dist0, dist1, np.log1p(dist1) - np.log1p(dist0),
                                           np.isfinite(values).astype(np.float32), (key >= 0).astype(np.float32)]))
        add("geometry", blocks[0], blocks[1], [f"geometry__{column}__{suffix}" for suffix in suffixes])

    source_prior = float(source_y.mean())
    prior_log_odds = float(np.log(source_prior / (1.0 - source_prior)))
    aggregate_train, aggregate_test, aggregate_names = [], [], []
    for (block, resolution), train_sum in sorted(binary_sums_train.items()):
        test_sum = binary_sums_test[(block, resolution)]
        aggregate_train.extend([train_sum, expit(prior_log_odds + train_sum)])
        aggregate_test.extend([test_sum, expit(prior_log_odds + test_sum)])
        aggregate_names.extend([f"nb_llr__{block}__{resolution}", f"nb_probability__{block}__{resolution}"])
    aggregate_train.extend([pair_llr_train, expit(prior_log_odds + pair_llr_train)])
    aggregate_test.extend([pair_llr_test, expit(prior_log_odds + pair_llr_test)])
    aggregate_names.extend(["pair_nb_llr__all", "pair_nb_probability__all"])
    level_prior = np.bincount(source_levels, minlength=4).astype(np.float64)
    level_prior /= level_prior.sum()
    for block in BLOCKS:
        train_probability = softmax(np.log(level_prior)[None, :] + latent_sums_train[block], axis=1)
        test_probability = softmax(np.log(level_prior)[None, :] + latent_sums_test[block], axis=1)
        aggregate_train.append(train_probability[:, 2] + train_probability[:, 3])
        aggregate_test.append(test_probability[:, 2] + test_probability[:, 3])
        aggregate_names.append(f"latent_probability_addicted__{block}")
        for index, level in enumerate(LEVELS):
            aggregate_train.append(train_probability[:, index])
            aggregate_test.append(test_probability[:, index])
            aggregate_names.append(f"latent_probability__{level}__{block}")
    add("aggregate", np.column_stack(aggregate_train), np.column_stack(aggregate_test), aggregate_names)

    def missing_block(frame):
        flags = frame[PREDICTORS].isna().to_numpy(np.float32)
        return np.column_stack([flags, flags.sum(axis=1), frame[BLOCKS["screen"]].isna().sum(axis=1).to_numpy(),
                                frame[BLOCKS["background"]].isna().sum(axis=1).to_numpy()])
    add("missing", missing_block(train), missing_block(test),
        [f"missing__{column}" for column in PREDICTORS] + ["missing_count_all", "missing_count_screen",
                                                          "missing_count_background"])

    train_matrix = np.column_stack(train_parts).astype(np.float32)
    test_matrix = np.column_stack(test_parts).astype(np.float32)
    groups["compact"] = [
        index for index, name in enumerate(names)
        if name.startswith(("nb_", "latent_probability", "geometry__", "missing__", "missing_count"))
        or ("__q16" in name and not name.startswith("source_logcount"))
    ]
    return train_matrix, test_matrix, names, groups


def main():
    train_matrix, test_matrix, names, groups = build()
    out = cache()
    np.save(out / "source_channel_train.npy", train_matrix)
    np.save(out / "source_channel_test.npy", test_matrix)
    manifest = {"feature_count": len(names), "feature_names": names, "groups": groups,
                "train_shape": list(train_matrix.shape), "test_shape": list(test_matrix.shape)}
    (out / "source_channel_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"source channel: train {train_matrix.shape}, test {test_matrix.shape}, "
          f"compact {len(groups['compact'])} columns", flush=True)


if __name__ == "__main__":
    main()
