"""Row-local RealMLP blocks (no statistics are learned from the frame).

raw                       the 9 numerics (float32, NaN kept) and the 3 categoricals as string categories
exact_numeric_embeddings  every numeric as a categorical token of its exact value (hours x100, counts x1), so RealMLP
                          learns an embedding per exact value
smooth_screen_geometry    19 screen-budget shapes: sums of the observed parts, budget remaining / balance, shares,
                          leisure share, HHI, dominant share, daily-social and weekend-daily gaps and log ratios, weekly mean
Numeric NaNs stay in the frame; the runner imputes them with outer-training medians."""
import numpy as np
import pandas as pd

MISSING_TOKEN = '__MISSING__'
NUMERIC_COLUMNS = ('age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours',
                   'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time')
CATEGORICAL_COLUMNS = ('gender', 'stress_level', 'academic_work_impact')
TIME_COLUMNS = ('daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours',
                'weekend_screen_time')
SCREEN_COMPONENT_COLUMNS = ('social_media_hours', 'gaming_hours', 'work_study_hours')
# hour-valued columns have two decimals, age / counts are integers
EXACT_TOKEN_SCALES = {c: (100 if c in TIME_COLUMNS else 1) for c in NUMERIC_COLUMNS}
BLOCKS = ('raw', 'exact_numeric_embeddings', 'smooth_screen_geometry')


def _numeric(frame, column):
    return pd.to_numeric(frame[column], errors='coerce').astype('float64')


def _as_float32(series):
    return series.replace([np.inf, -np.inf], np.nan).astype('float32')


def _as_category(values, index):
    return pd.Series(pd.Categorical(values), index=index)


def _integer_token(series, scale, prefix):
    numeric = pd.to_numeric(series, errors='coerce').astype('float64')
    integer = (numeric * float(scale)).round().astype('Int64')
    return _as_category((prefix + integer.astype('string')).fillna(MISSING_TOKEN), series.index)


def _bounded_share(numerator, reference):
    return numerator / (1.0 + numerator.abs() + reference.abs())


def _signed_log1p(series):
    return np.sign(series) * np.log1p(series.abs())


def _log_ratio(left, right):
    return _signed_log1p(left) - _signed_log1p(right)


def _raw_block(frame):
    out = pd.DataFrame(index=frame.index)
    for c in NUMERIC_COLUMNS:
        out[c] = _numeric(frame, c).astype('float32')
    for c in CATEGORICAL_COLUMNS:
        out[c] = _as_category(frame[c].astype('string').fillna(MISSING_TOKEN), frame.index)
    return out, list(CATEGORICAL_COLUMNS)


def _exact_numeric_embeddings_block(frame):
    out = pd.DataFrame(index=frame.index)
    for c in NUMERIC_COLUMNS:
        out[f'emb_exact__{c}'] = _integer_token(frame[c], EXACT_TOKEN_SCALES[c], 'v:')
    return out, list(out.columns)


def _smooth_screen_geometry_block(frame):
    daily = _numeric(frame, 'daily_screen_time_hours')
    weekend = _numeric(frame, 'weekend_screen_time')
    social = _numeric(frame, 'social_media_hours')
    gaming = _numeric(frame, 'gaming_hours')
    work = _numeric(frame, 'work_study_hours')
    components = pd.concat([social, gaming, work], axis=1)
    components.columns = list(SCREEN_COMPONENT_COLUMNS)

    observed_count = components.notna().sum(axis=1)
    known_sum = components.sum(axis=1, min_count=1)
    strict_sum = social + gaming + work
    partial_sum = components.fillna(0.0).sum(axis=1)
    partial_remaining = daily - partial_sum
    strict_remaining = daily - strict_sum
    complete = daily.notna() & (observed_count == len(SCREEN_COMPONENT_COLUMNS))
    leisure = social + gaming
    hhi = components.pow(2).sum(axis=1, min_count=1) / (1.0 + known_sum.abs().pow(2))
    dominant = components.max(axis=1, skipna=True)

    out = pd.DataFrame(index=frame.index)
    out['screen__known_parts_sum'] = _as_float32(known_sum)
    out['screen__strict_parts_sum'] = _as_float32(strict_sum)
    out['screen__parts_observed_count'] = observed_count.astype('float32')
    out['screen__budget_complete'] = complete.astype('float32')
    out['screen__partial_budget_remaining'] = _as_float32(partial_remaining)
    out['screen__strict_budget_remaining'] = _as_float32(strict_remaining)
    out['screen__partial_budget_balance'] = _as_float32(_bounded_share(partial_remaining, daily + partial_sum))
    out['screen__strict_budget_balance'] = _as_float32(_bounded_share(strict_remaining, daily + strict_sum))
    out['screen__known_parts_share'] = _as_float32(_bounded_share(known_sum, daily))
    out['screen__strict_parts_share'] = _as_float32(_bounded_share(strict_sum, daily))
    out['screen__leisure_sum'] = _as_float32(leisure)
    out['screen__leisure_share'] = _as_float32(_bounded_share(leisure, daily))
    out['screen__component_hhi'] = _as_float32(hhi)
    out['screen__dominant_component_share'] = _as_float32(_bounded_share(dominant, known_sum))
    out['screen__daily_social_gap'] = _as_float32(daily - social)
    out['screen__daily_social_log_ratio'] = _as_float32(_log_ratio(daily, social))
    out['screen__weekend_daily_gap'] = _as_float32(weekend - daily)
    out['screen__weekend_daily_log_ratio'] = _as_float32(_log_ratio(weekend, daily))
    out['screen__weekly_mean'] = _as_float32((5.0 * daily + 2.0 * weekend) / 7.0)
    return out, []


_BUILDERS = {'raw': _raw_block, 'exact_numeric_embeddings': _exact_numeric_embeddings_block,
             'smooth_screen_geometry': _smooth_screen_geometry_block}


def build_rowwise_features(frame, blocks=BLOCKS):
    """-> (DataFrame with the blocks in order, list of categorical column names)."""
    parts, categorical = [], []
    for block in blocks:
        part, cats = _BUILDERS[block](frame)
        parts.append(part.reset_index(drop=True))
        categorical.extend(cats)
    return pd.concat(parts, axis=1), categorical
