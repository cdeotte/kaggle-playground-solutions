"""Label-free row-local feature blocks for TabM (no fitted statistics; safe to build on all rows).

raw                     the 9 numerics (float32) and 3 categoricals (strings, missing = '__MISSING__')
exact_grid_onehots      exact-value tokens of the 9 numerics (hours x100, others x1) as categoricals
smooth_latent_geometry  14 smooth coordinates from fixed anchors: screen load, leisure vs productive, weekend escalation,
                        recovery pressure, interruption load, notification / open imbalance, log balances, z products,
                        known component mean, budget slack / balance
missing_gated_views     11 numeric and 3 categorical views of the missing pattern and an equal-slack completion of the
                        missing screen components
fixed_bins              fixed-origin bins of every numeric (2 or 3 widths) as categoricals
build_rowwise_features(frame, blocks) returns the concatenated blocks (block order as given) and the categorical names;
the stateful block 'sparse_pair_contrast_12' is skipped here (lib/fold_features.py)."""
import numpy as np
import pandas as pd

ID_COLUMN, TARGET = "id", "addicted_label"
MISSING_TOKEN = "__MISSING__"
NUMERIC_COLUMNS = ("age", "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours",
                   "sleep_hours", "notifications_per_day", "app_opens_per_day", "weekend_screen_time")
CATEGORICAL_COLUMNS = ("gender", "stress_level", "academic_work_impact")
TIME_COLUMNS = ("daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "sleep_hours",
                "weekend_screen_time")
COMPONENT_COLUMNS = ("social_media_hours", "gaming_hours", "work_study_hours")
STATEFUL_BLOCKS = ("sparse_pair_contrast_12",)
LATENT_ANCHORS = {
    "age": (25.0, 6.0), "daily_screen_time_hours": (7.5, 2.5), "social_media_hours": (3.25, 1.75),
    "gaming_hours": (2.0, 1.25), "work_study_hours": (3.25, 1.75), "sleep_hours": (7.0, 1.5),
    "notifications_per_day": (125.0, 75.0), "app_opens_per_day": (85.0, 50.0), "weekend_screen_time": (9.0, 3.0),
}
SMOOTH_LATENT_COLUMNS = (
    "latent__screen_load", "latent__leisure_vs_productive", "latent__weekend_escalation", "latent__recovery_pressure",
    "latent__interruption_load", "latent__notification_open_imbalance", "latent__social_gaming_balance",
    "latent__screen_sleep_balance", "latent__daily_social_synergy", "latent__daily_weekend_synergy",
    "latent__notification_open_synergy", "latent__known_component_mean", "latent__known_budget_slack",
    "latent__known_budget_balance")
MISSING_GATED_NUMERIC_COLUMNS = (
    "gate__numeric_missing_count", "gate__screen_observed_count", "gate__screen_missing_count",
    "gate__known_component_sum", "gate__daily_completed_lower_bound", "gate__social_completed_equal_slack",
    "gate__gaming_completed_equal_slack", "gate__work_completed_equal_slack", "gate__completion_slack",
    "gate__completion_mass", "gate__complete_weekly_screen")
MISSING_GATED_CATEGORICAL_COLUMNS = ("gate__numeric_missing_mask", "gate__screen_observation_mask",
                                     "gate__completion_mode")


def _num(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _f32(values, index):
    return pd.Series(values, index=index).replace([np.inf, -np.inf], np.nan).astype("float32")


def _cat(values, index):
    return pd.Series(values, index=index, dtype="string").fillna(MISSING_TOKEN).astype(object)


def _signed_log1p(values):
    return np.sign(values) * np.log1p(values.abs())


def _mask(frame, columns):
    bits = frame[list(columns)].notna().to_numpy(dtype=np.int64)
    weights = 1 << np.arange(len(columns), dtype=np.int64)
    return _cat("m:" + (bits @ weights).astype(str), frame.index)


def _raw_block(frame):
    out = pd.DataFrame(index=frame.index)
    for name in NUMERIC_COLUMNS:
        out[name] = _num(frame, name).astype("float32")
    for name in CATEGORICAL_COLUMNS:
        out[name] = _cat(frame[name], frame.index)
    return out, list(CATEGORICAL_COLUMNS)


def _bin_token(values, width, prefix):
    numeric = pd.to_numeric(values, errors="coerce")
    token = np.floor(numeric / width + 1e-10).astype("Int64")
    return _cat(prefix + token.astype("string"), values.index)


def _fixed_bins_block(frame):
    widths = {
        "age": (2.0, 4.0), "daily_screen_time_hours": (0.2, 0.5, 1.0), "social_media_hours": (0.2, 0.5),
        "gaming_hours": (0.2, 0.5), "work_study_hours": (0.2, 0.5), "sleep_hours": (0.2, 0.5),
        "notifications_per_day": (5.0, 20.0), "app_opens_per_day": (5.0, 20.0),
        "weekend_screen_time": (0.2, 0.5, 1.0),
    }
    out = pd.DataFrame(index=frame.index)
    for name, column_widths in widths.items():
        for width in column_widths:
            suffix = str(width).replace(".", "p")
            out[f"bin{suffix}__{name}"] = _bin_token(frame[name], width, f"{name}:b{suffix}:")
    return out, list(out.columns)


def _exact_grid(frame):
    out = pd.DataFrame(index=frame.index)
    for column in NUMERIC_COLUMNS:
        scale = 100 if column in TIME_COLUMNS else 1
        integer = (_num(frame, column) * scale).round().astype("Int64")
        out[f"grid_exact__{column}"] = _cat("v:" + integer.astype("string"), frame.index)
    return out, list(out.columns)


def _smooth_latent(frame):
    z = {column: (_num(frame, column) - LATENT_ANCHORS[column][0]) / LATENT_ANCHORS[column][1]
         for column in NUMERIC_COLUMNS}
    daily, social, gaming, work, sleep = (_num(frame, c) for c in (
        "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "sleep_hours"))
    parts = pd.concat([social, gaming, work], axis=1)
    known_sum = parts.sum(axis=1, min_count=1)
    values = {
        "latent__screen_load": pd.concat([z["daily_screen_time_hours"], z["social_media_hours"], z["gaming_hours"],
                                          z["weekend_screen_time"]], axis=1).mean(axis=1),
        "latent__leisure_vs_productive": (z["social_media_hours"] + z["gaming_hours"] - z["work_study_hours"])
        / np.sqrt(3.0),
        "latent__weekend_escalation": z["weekend_screen_time"] - z["daily_screen_time_hours"],
        "latent__recovery_pressure": z["daily_screen_time_hours"] - z["sleep_hours"],
        "latent__interruption_load": pd.concat([z["notifications_per_day"], z["app_opens_per_day"]],
                                               axis=1).mean(axis=1),
        "latent__notification_open_imbalance": z["notifications_per_day"] - z["app_opens_per_day"],
        "latent__social_gaming_balance": _signed_log1p(social) - _signed_log1p(gaming),
        "latent__screen_sleep_balance": _signed_log1p(daily) - _signed_log1p(sleep),
        "latent__daily_social_synergy": z["daily_screen_time_hours"] * z["social_media_hours"],
        "latent__daily_weekend_synergy": z["daily_screen_time_hours"] * z["weekend_screen_time"],
        "latent__notification_open_synergy": z["notifications_per_day"] * z["app_opens_per_day"],
        "latent__known_component_mean": parts.mean(axis=1),
        "latent__known_budget_slack": _signed_log1p(daily - known_sum),
        "latent__known_budget_balance": (daily - known_sum) / (1.0 + daily.abs() + known_sum.abs()),
    }
    out = pd.DataFrame(index=frame.index)
    for name, column_values in values.items():
        out[name] = _f32(column_values, frame.index)
    return out[list(SMOOTH_LATENT_COLUMNS)], []


def _completion(frame):
    """Missing screen components share the non-negative slack daily - known sum equally."""
    daily = _num(frame, "daily_screen_time_hours")
    parts = pd.DataFrame({c: _num(frame, c) for c in COMPONENT_COLUMNS}, index=frame.index)
    observed = parts.notna()
    observed_count = observed.sum(axis=1)
    missing_count = len(COMPONENT_COLUMNS) - observed_count
    known_sum = parts.fillna(0.0).sum(axis=1)
    daily_completed = daily.fillna(known_sum)
    available_slack = (daily_completed - known_sum).clip(lower=0.0)
    share = available_slack / missing_count.where(missing_count > 0, 1)
    completed = parts.copy()
    for column in COMPONENT_COLUMNS:
        completed[column] = parts[column].where(observed[column], share)
    return dict(daily=daily, observed_count=observed_count, missing_count=missing_count, known_sum=known_sum,
                daily_completed=daily_completed, completed=completed,
                completion_slack=daily_completed - completed.sum(axis=1),
                completion_mass=(completed - parts.fillna(0.0)).sum(axis=1))


def _missing_gated(frame):
    v = _completion(frame)
    weekend_completed = _num(frame, "weekend_screen_time").fillna(v["daily_completed"])
    mode = np.select(
        [v["daily"].isna(), v["missing_count"].eq(0), v["missing_count"].eq(1), v["missing_count"].eq(2)],
        ["daily_lower_bound", "observed_complete", "equal_slack_1", "equal_slack_2"], default="equal_slack_3")
    numeric = {
        "gate__numeric_missing_count": frame[list(NUMERIC_COLUMNS)].isna().sum(axis=1),
        "gate__screen_observed_count": v["observed_count"],
        "gate__screen_missing_count": v["missing_count"],
        "gate__known_component_sum": v["known_sum"],
        "gate__daily_completed_lower_bound": v["daily_completed"],
        "gate__social_completed_equal_slack": v["completed"]["social_media_hours"],
        "gate__gaming_completed_equal_slack": v["completed"]["gaming_hours"],
        "gate__work_completed_equal_slack": v["completed"]["work_study_hours"],
        "gate__completion_slack": v["completion_slack"],
        "gate__completion_mass": v["completion_mass"],
        "gate__complete_weekly_screen": (5 * v["daily_completed"] + 2 * weekend_completed) / 7,
    }
    out = pd.DataFrame(index=frame.index)
    for name, values in numeric.items():
        out[name] = _f32(values, frame.index)
    out["gate__numeric_missing_mask"] = _mask(frame, NUMERIC_COLUMNS)
    out["gate__screen_observation_mask"] = _mask(frame, ("daily_screen_time_hours", *COMPONENT_COLUMNS))
    out["gate__completion_mode"] = _cat(mode, frame.index)
    return out[list(MISSING_GATED_NUMERIC_COLUMNS + MISSING_GATED_CATEGORICAL_COLUMNS)], \
        list(MISSING_GATED_CATEGORICAL_COLUMNS)


ROW_BUILDERS = {"raw": _raw_block, "fixed_bins": _fixed_bins_block}
PREDICTOR_BUILDERS = {"exact_grid_onehots": _exact_grid, "smooth_latent_geometry": _smooth_latent,
                      "missing_gated_views": _missing_gated}


def build_rowwise_features(frame, blocks):
    """Row-local blocks in the given order; returns (features, categorical names)."""
    frame = frame.reset_index(drop=True)
    predictor_frame = frame.drop(columns=[ID_COLUMN, TARGET], errors="ignore")
    pieces, categorical = [], []
    for block in blocks:
        if block in STATEFUL_BLOCKS:
            continue
        if block in PREDICTOR_BUILDERS:
            piece, cats = PREDICTOR_BUILDERS[block](predictor_frame)
            piece = piece.reset_index(drop=True)
        else:
            piece, cats = ROW_BUILDERS[block](frame)
        pieces.append(piece)
        categorical.extend(cats)
    return pd.concat(pieces, axis=1).reset_index(drop=True), categorical
