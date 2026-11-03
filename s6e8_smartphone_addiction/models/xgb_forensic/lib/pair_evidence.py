"""Pair-level evidence blocks of the forensic pipeline (all row-wise given their fitted state, so the outer-training,
validation and test rows are transformed by the same functions).

observed joint   label-free: for each of the 36 numeric pairs, over the train + test rows where both values are
                 observed: log1p(exact pair count) and PMI = log(n_pair * n / (n_left * n_right)) (0 if a value is
                 missing). The model uses the PMIs of the 18 pairs in PMI_PAIRS18, shrunk by count / (count + 5).
exact copula     fixed original: density ratio log(competition pair frequency) - log((original pair count + 0.5) /
                 (n + 0.5 (unique pairs + 1))) of the PMI_PAIRS18 pairs, kept only where the exact pair does not occur in
                 the original (else 0).
unsupported pair response
                 target: for the PMI_PAIRS18 pairs whose exact (left, right) value does not occur in the original, the
                 smoothed (m = 20) target rate of the exact pair -> logit - logit(prior); training rows are 5-fold
                 cross-fitted with the label-free split (index * 2654435761 + 2718281) mod 5. Output: reliability-weighted
                 mean over the available pairs (weights count / (count + 20)) and the fraction of available pairs.
imputed screen response
                 target: quantile codes (q16, q64; edges from the outer-training rows) of strict slack, component share
                 of daily and weekend - daily computed on the l2-imputed screen fields; smoothed (m = 20) rate -> logit;
                 training rows cross-fitted with the split (index * 2654435761 + 8111) mod 5."""
import itertools
import json

import numpy as np
import pandas as pd

from common.data import read_original

NUMERIC = ("age", "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "sleep_hours",
           "notifications_per_day", "app_opens_per_day", "weekend_screen_time")
SCREEN_FIELDS = ("daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours",
                 "weekend_screen_time")
PMI_PAIRS18 = (
    ("daily_screen_time_hours", "social_media_hours"), ("daily_screen_time_hours", "notifications_per_day"),
    ("daily_screen_time_hours", "app_opens_per_day"), ("social_media_hours", "weekend_screen_time"),
    ("app_opens_per_day", "weekend_screen_time"), ("daily_screen_time_hours", "work_study_hours"),
    ("notifications_per_day", "weekend_screen_time"), ("daily_screen_time_hours", "gaming_hours"),
    ("gaming_hours", "work_study_hours"), ("social_media_hours", "work_study_hours"),
    ("social_media_hours", "notifications_per_day"), ("social_media_hours", "app_opens_per_day"),
    ("daily_screen_time_hours", "weekend_screen_time"), ("notifications_per_day", "app_opens_per_day"),
    ("sleep_hours", "weekend_screen_time"), ("work_study_hours", "sleep_hours"), ("gaming_hours", "sleep_hours"),
    ("daily_screen_time_hours", "sleep_hours"),
)
JOINT_MODE = "conditioned_pmi_logcount"


# ---------------------------------------------------------------- observed joint (label-free, train + test)
def _numeric(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def _key(frame, columns):
    arrays = [pd.to_numeric(frame[column], errors="coerce").to_numpy() for column in columns]
    if len(arrays) == 1:
        return pd.Index(arrays[0], name=columns[0])
    return pd.MultiIndex.from_arrays(arrays, names=list(columns))


def _apply_pair(frame, left, right, pair_counts, left_counts, right_counts, reference_rows):
    complete = np.isfinite(_numeric(frame, left)) & np.isfinite(_numeric(frame, right))
    output = np.zeros((len(frame), 2), dtype=np.float32)
    if not complete.any():
        return output
    observed = frame.loc[complete]
    joint = pair_counts.reindex(_key(observed, (left, right))).to_numpy(np.float64)
    left_count = left_counts.reindex(_key(observed, (left,))).to_numpy(np.float64)
    right_count = right_counts.reindex(_key(observed, (right,))).to_numpy(np.float64)
    output[complete, 0] = np.log1p(joint).astype(np.float32)
    output[complete, 1] = np.log(joint * reference_rows / (left_count * right_count)).astype(np.float32)
    return output


def build_observed_joint(train, test):
    """(train matrix, test matrix, column names, complete pair counts); 2 columns (logcount, pmi) per numeric pair."""
    train, test = train[list(NUMERIC)], test[list(NUMERIC)]
    combined = pd.concat([train, test], ignore_index=True)
    pairs = list(itertools.combinations(NUMERIC, 2))
    train_out = np.empty((len(train), 2 * len(pairs)), dtype=np.float32)
    test_out = np.empty((len(test), 2 * len(pairs)), dtype=np.float32)
    names, complete_counts = [], {}
    for number, (left, right) in enumerate(pairs):
        complete = np.isfinite(_numeric(combined, left)) & np.isfinite(_numeric(combined, right))
        reference = combined.loc[complete]
        reference_rows = float(complete.sum())
        pair_counts = _key(reference, (left, right)).value_counts(sort=False)
        left_counts = _key(reference, (left,)).value_counts(sort=False)
        right_counts = _key(reference, (right,)).value_counts(sort=False)
        for frame, out in ((train, train_out), (test, test_out)):
            out[:, 2 * number:2 * number + 2] = _apply_pair(frame, left, right, pair_counts, left_counts, right_counts,
                                                            reference_rows)
        names += [f"observed_joint_logcount__{JOINT_MODE}__{left}__{right}",
                  f"observed_joint_pmi__{JOINT_MODE}__{left}__{right}"]
        complete_counts[f"{left}__{right}"] = int(complete.sum())
    return train_out, test_out, names, complete_counts


def pair_from_name(name):
    left, right = name.split(f"__{JOINT_MODE}__", 1)[1].split("__", 1)
    return left, right


def _column_index(names, prefix, pair):
    return next(i for i, name in enumerate(names) if name.startswith(prefix) and pair_from_name(name) == pair)


def observed_joint_pmi18(matrix_rows, names, strength=5.0):
    """'pmi_top18_shrink5': the PMIs of the PMI_PAIRS18 pairs (in cache column order) times count / (count + 5)."""
    selected = [i for i, name in enumerate(names)
                if name.startswith("observed_joint_pmi") and pair_from_name(name) in PMI_PAIRS18]
    values = np.asarray(matrix_rows[:, selected], np.float32).copy()
    out_names = []
    for column, source_index in enumerate(selected):
        pair = pair_from_name(names[source_index])
        count = np.expm1(np.asarray(matrix_rows[:, _column_index(names, "observed_joint_logcount", pair)], np.float64))
        values[:, column] = (np.asarray(values[:, column], np.float64) * count / (count + strength)).astype(np.float32)
        out_names.append(names[source_index].replace(JOINT_MODE, "pmi_top18_shrink5") + f"__count_shrink_{strength:g}")
    return pd.DataFrame(values, columns=out_names)


# ---------------------------------------------------------------- exact original copula
def _copula_state(source, pair):
    left, right = pair
    complete = source[left].notna() & source[right].notna()
    local = source.loc[complete, [left, right]]
    joint = pd.MultiIndex.from_frame(local).value_counts(sort=False).astype(np.float64)
    rows = float(len(local))
    return joint, rows, rows + 0.5 * (len(joint) + 1)


def exact_copula_unsupported(frame, matrix_rows, names, complete_counts):
    """'density_ratio18_unsupported' for the rows of `frame` (matrix_rows = their observed-joint rows)."""
    columns = sorted({column for pair in PMI_PAIRS18 for column in pair})
    source = read_original()[columns]
    output, out_names = [], []
    for pair in PMI_PAIRS18:
        left, right = pair
        joint, _, denominator = _copula_state(source, pair)
        count = np.expm1(np.asarray(matrix_rows[:, _column_index(names, "observed_joint_logcount", pair)], np.float64))
        competition_rows = float(complete_counts["__".join(pair)])
        left_value = pd.to_numeric(frame[left], errors="coerce").to_numpy(np.float64)
        right_value = pd.to_numeric(frame[right], errors="coerce").to_numpy(np.float64)
        complete = np.isfinite(left_value) & np.isfinite(right_value)
        rows = np.flatnonzero(complete)
        density = np.zeros(len(frame), np.float64)
        support = np.zeros(len(frame), np.float64)
        if len(rows):
            key = pd.MultiIndex.from_arrays([left_value[complete], right_value[complete]])
            joint_values = joint.reindex(key, fill_value=0.0).to_numpy(np.float64)
            support[rows] = joint_values > 0.0
            competition_probability = np.maximum(count[rows], 1.0) / competition_rows
            source_probability = (joint_values + 0.5) / denominator
            density[rows] = np.log(competition_probability) - np.log(source_probability)
        output.append((density * (1.0 - support)).astype(np.float32))
        out_names.append(f"exact_copula_density_ratio18_unsupported__{left}__{right}")
    return pd.DataFrame(np.column_stack(output), columns=out_names)


# ---------------------------------------------------------------- unsupported pair response (target)
def _logit(probability, eps):
    value = np.clip(np.asarray(probability, np.float64), eps, 1.0 - eps)
    return np.log(value / (1.0 - value))


def _codes(train, apply):
    """Exact-value codes from the outer-training vocabulary; missing / unseen -> one extra code."""
    train_value = pd.to_numeric(train, errors="coerce").to_numpy(np.float64)
    apply_value = pd.to_numeric(apply, errors="coerce").to_numpy(np.float64)
    vocabulary = np.unique(train_value[np.isfinite(train_value)])
    unknown = len(vocabulary)

    def encode(values):
        observed = np.isfinite(values)
        output = np.full(len(values), unknown, np.int32)
        positions = np.searchsorted(vocabulary, values[observed])
        bounded = np.minimum(positions, max(len(vocabulary) - 1, 0))
        found = (positions < len(vocabulary)) & (vocabulary[bounded] == values[observed])
        rows = np.flatnonzero(observed)
        output[rows[found]] = positions[found].astype(np.int32)
        return output

    return encode(train_value), encode(apply_value), unknown + 1


def _unsupported(frame, pair, source):
    left, right = pair
    left_value = pd.to_numeric(frame[left], errors="coerce").to_numpy(np.float64)
    right_value = pd.to_numeric(frame[right], errors="coerce").to_numpy(np.float64)
    complete = np.isfinite(left_value) & np.isfinite(right_value)
    output = np.zeros(len(frame), bool)
    if complete.any():
        source_index = pd.MultiIndex.from_frame(source[[left, right]])
        keys = pd.MultiIndex.from_arrays([left_value[complete], right_value[complete]])
        output[np.flatnonzero(complete)] = ~keys.isin(source_index)
    return output


def _encode_pair(train_key, apply_key, train_available, apply_available, y, inner_fold, groups, smoothing):
    train_value, train_support = np.zeros(len(y), np.float64), np.zeros(len(y), np.float64)
    for fold in range(5):
        fit_all = inner_fold != fold
        fit = fit_all & train_available
        apply = (inner_fold == fold) & train_available
        prior = float(y[fit_all].mean())
        count = np.bincount(train_key[fit], minlength=groups).astype(np.float64)
        total = np.bincount(train_key[fit], weights=y[fit], minlength=groups)
        probability = (total + smoothing * prior) / (count + smoothing)
        train_value[apply] = _logit(probability[train_key[apply]], 1e-6) - float(_logit(prior, 1e-6))
        train_support[apply] = count[train_key[apply]]
    prior = float(y.mean())
    count = np.bincount(train_key[train_available], minlength=groups).astype(np.float64)
    total = np.bincount(train_key[train_available], weights=y[train_available], minlength=groups)
    probability = (total + smoothing * prior) / (count + smoothing)
    apply_value, apply_support = np.zeros(len(apply_key), np.float64), np.zeros(len(apply_key), np.float64)
    apply_value[apply_available] = _logit(probability[apply_key[apply_available]], 1e-6) - float(_logit(prior, 1e-6))
    apply_support[apply_available] = count[apply_key[apply_available]]
    return train_value, apply_value, train_support, apply_support


def unsupported_pair_response(raw_train, raw_apply, target, train_idx, smoothing=20.0, mode="mean_s20"):
    """'mean_s20' for the outer-training rows (cross-fitted) and the application rows (validation + test)."""
    source = read_original()[list(NUMERIC)]
    y = np.asarray(target, np.int8).astype(np.float64)
    indices = np.asarray(train_idx, np.uint64)
    inner_fold = ((indices * np.uint64(2654435761) + np.uint64(2718281)) % np.uint64(5)).astype(np.int8)
    codes = {column: _codes(raw_train[column], raw_apply[column]) for column in NUMERIC}
    values, supports = ([], []), ([], [])
    for left, right in PMI_PAIRS18:
        right_groups = codes[right][2]
        train_value, apply_value, train_support, apply_support = _encode_pair(
            codes[left][0] * right_groups + codes[right][0], codes[left][1] * right_groups + codes[right][1],
            _unsupported(raw_train, (left, right), source), _unsupported(raw_apply, (left, right), source),
            y, inner_fold, codes[left][2] * right_groups, smoothing)
        values[0].append(train_value)
        values[1].append(apply_value)
        supports[0].append(train_support)
        supports[1].append(apply_support)
    frames = []
    for value_list, support_list in zip(values, supports):
        matrix, support = np.column_stack(value_list), np.column_stack(support_list)
        available = support > 0
        reliability = support / (support + smoothing)
        weighted_mean = (matrix * reliability).sum(axis=1) / np.maximum(reliability.sum(axis=1), 1e-12)
        weighted_mean[~available.any(axis=1)] = 0.0
        frames.append(pd.DataFrame({
            f"unsupported_pair_response__{mode}__weighted_mean": weighted_mean.astype(np.float32),
            f"unsupported_pair_response__{mode}__available_fraction": available.mean(axis=1).astype(np.float32)}))
    return frames[0], frames[1]


# ---------------------------------------------------------------- imputed screen response (target)
def _imputed_geometry(model):
    imputed = {name: f"imputer_v2_l2_imputed__{name}" for name in SCREEN_FIELDS}
    daily = model[imputed["daily_screen_time_hours"]].to_numpy(np.float64)
    social = model[imputed["social_media_hours"]].to_numpy(np.float64)
    gaming = model[imputed["gaming_hours"]].to_numpy(np.float64)
    work = model[imputed["work_study_hours"]].to_numpy(np.float64)
    weekend = model[imputed["weekend_screen_time"]].to_numpy(np.float64)
    component_sum = social + gaming + work
    return {"strict_slack": daily - component_sum, "component_share_daily": component_sum / np.maximum(daily, 1e-3),
            "weekend_minus_daily": weekend - daily}


def _quantile_keys(train_value, apply_value, bins):
    edges = np.unique(np.quantile(train_value, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    return (np.searchsorted(edges, train_value).astype(np.int16), np.searchsorted(edges, apply_value).astype(np.int16),
            len(edges) + 1)


def _encode_bins(train_key, apply_key, y, inner_fold, groups):
    train_output = np.empty(len(y), np.float64)
    for fold in range(5):
        fit = inner_fold != fold
        prior = float(y[fit].mean())
        count = np.bincount(train_key[fit], minlength=groups).astype(np.float64)
        total = np.bincount(train_key[fit], weights=y[fit], minlength=groups)
        train_output[~fit] = ((total + 20.0 * prior) / (count + 20.0))[train_key[~fit]]
    prior = float(y.mean())
    count = np.bincount(train_key, minlength=groups).astype(np.float64)
    total = np.bincount(train_key, weights=y, minlength=groups)
    return train_output, ((total + 20.0 * prior) / (count + 20.0))[apply_key]


def imputed_screen_response(train_model, apply_model, target, train_idx):
    """'v2_l2_compact': 6 columns for the outer-training rows (cross-fitted) and the application rows."""
    y = np.asarray(target, np.float64)
    indices = np.asarray(train_idx, np.uint64)
    inner_fold = ((indices * np.uint64(2654435761) + np.uint64(8111)) % np.uint64(5)).astype(np.int8)
    train_geometry, apply_geometry = _imputed_geometry(train_model), _imputed_geometry(apply_model)
    train_columns, apply_columns = {}, {}
    for name in ("strict_slack", "component_share_daily", "weekend_minus_daily"):
        for bins in (16, 64):
            train_key, apply_key, groups = _quantile_keys(train_geometry[name], apply_geometry[name], bins)
            train_probability, apply_probability = _encode_bins(train_key, apply_key, y, inner_fold, groups)
            feature = f"imputed_screen_response_logit__v2_l2__{name}__q{bins}"
            train_columns[feature] = _logit(train_probability, 1e-5).astype(np.float32)
            apply_columns[feature] = _logit(apply_probability, 1e-5).astype(np.float32)
    return pd.DataFrame(train_columns), pd.DataFrame(apply_columns)


def load_observed_joint(cache):
    """The cached observed-joint matrices of build_observed_joint.py: train, test, names, complete pair counts."""
    root = cache / "observed_joint_cache_v1"
    manifest = json.loads((root / "manifest.json").read_text())
    return (np.load(root / "train_features.npy", mmap_mode="r"), np.load(root / "test_features.npy", mmap_mode="r"),
            list(manifest["columns"]), manifest["feature_audit"]["complete_pair_counts"])
