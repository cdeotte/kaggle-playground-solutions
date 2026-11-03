"""Fold-fitted preprocessing of the TabM frames (state from the outer-training rows only).

sparse_pair_contrast_12  12 contrasts z(left) - z(right) of selected numeric pairs, z = (x - outer-training median) /
                         outer-training std (ddof 1), NaN when either value is missing
prepare_fold_frames      adds the contrasts, fills every numeric NaN / inf with its outer-training median (0 if none)
                         and casts the categoricals to strings ('__MISSING__' for missing)"""
import numpy as np
import pandas as pd

MISSING_TOKEN = "__MISSING__"
PAIR_SPECS = (
    ("social_media_hours", "work_study_hours"), ("app_opens_per_day", "weekend_screen_time"),
    ("daily_screen_time_hours", "app_opens_per_day"), ("notifications_per_day", "weekend_screen_time"),
    ("daily_screen_time_hours", "social_media_hours"), ("gaming_hours", "weekend_screen_time"),
    ("daily_screen_time_hours", "gaming_hours"), ("daily_screen_time_hours", "work_study_hours"),
    ("work_study_hours", "notifications_per_day"), ("work_study_hours", "sleep_hours"),
    ("gaming_hours", "sleep_hours"), ("social_media_hours", "weekend_screen_time"),
)
SOURCE_COLUMNS = tuple(dict.fromkeys(column for pair in PAIR_SPECS for column in pair))
OUTPUT_COLUMNS = tuple(f"contrast12__z_{left}__minus__z_{right}" for left, right in PAIR_SPECS)


def _numeric_sources(frame):
    return pd.DataFrame({column: pd.to_numeric(frame[column], errors="coerce").astype("float64")
                         for column in SOURCE_COLUMNS}, index=frame.index)


def fit_sparse_pair_contrast(outer_train):
    numeric = _numeric_sources(outer_train)
    return {column: (float(numeric[column].median()), float(numeric[column].std(ddof=1))) for column in SOURCE_COLUMNS}


def apply_sparse_pair_contrast(frame, state):
    numeric = _numeric_sources(frame)
    standardized = {column: (numeric[column] - center) / scale for column, (center, scale) in state.items()}
    output = pd.DataFrame(index=frame.index)
    for output_name, (left, right) in zip(OUTPUT_COLUMNS, PAIR_SPECS):
        output[output_name] = (standardized[left] - standardized[right]).astype("float32")
    return output.reset_index(drop=True)


def prepare_fold_frames(X_train, X_valid, X_test, categorical_names, blocks):
    train = X_train.reset_index(drop=True).copy()
    valid = X_valid.reset_index(drop=True).copy()
    test = X_test.reset_index(drop=True).copy() if X_test is not None else None
    cats = list(categorical_names)
    if "sparse_pair_contrast_12" in blocks:
        state = fit_sparse_pair_contrast(train)
        train = pd.concat([train, apply_sparse_pair_contrast(train, state)], axis=1, copy=False)
        valid = pd.concat([valid, apply_sparse_pair_contrast(valid, state)], axis=1, copy=False)
        if test is not None:
            test = pd.concat([test, apply_sparse_pair_contrast(test, state)], axis=1, copy=False)
    frames = [train, valid] + ([test] if test is not None else [])

    numeric_names = [name for name in train.columns if name not in set(cats)]
    for name in numeric_names:
        median = float(pd.to_numeric(train[name], errors="coerce").replace([np.inf, -np.inf], np.nan).median())
        if not np.isfinite(median):
            median = 0.0
        for frame in frames:
            values = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
            frame[name] = values.fillna(median).astype("float32")
    for name in cats:
        for frame in frames:
            frame[name] = frame[name].astype("string").fillna(MISSING_TOKEN).astype(object)
    return train, valid, test, cats
