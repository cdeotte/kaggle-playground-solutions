"""Row-local TabM feature blocks and the fold preprocessing of the TabM parent.

Row-local blocks (pure functions of one predictor row; categoricals are strings, missing -> __MISSING__):
  raw                     9 numerics (float32) + 3 categoricals
  exact_grid_onehots      exact value token of every numeric (hours x 100, counts x 1) as a categorical
  smooth_latent_geometry  14 smooth oblique coordinates: z-scores against fixed anchors combined into screen load,
                          leisure vs productive, weekend escalation, recovery pressure, interruption load, ...,
                          log balances and the known screen-budget slack
  missing_gated_views     missing counts, observation masks and an equal-slack completion of missing screen components
  fixed_bins              fixed-origin bin tokens (2-3 widths per numeric) as categoricals
Fold state (prepare_fold_frames, fit on the outer-training rows only):
  sparse_pair_contrast_12 12 differences of standardized pairs, z = (x - median) / sample std of the outer-training rows
                          (NaN if an operand is missing)
  every numeric NaN -> outer-training median (float32); categoricals -> strings"""
import numpy as np
import pandas as pd

MISSING_TOKEN = '__MISSING__'
NUMERIC_COLUMNS = ('age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours',
                   'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time')
CATEGORICAL_COLUMNS = ('gender', 'stress_level', 'academic_work_impact')
TIME_COLUMNS = ('daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours',
                'weekend_screen_time')
COMPONENT_COLUMNS = ('social_media_hours', 'gaming_hours', 'work_study_hours')
EXACT_SCALES = {c: (100 if c in TIME_COLUMNS else 1) for c in NUMERIC_COLUMNS}
LATENT_ANCHORS = {
    'age': (25.0, 6.0), 'daily_screen_time_hours': (7.5, 2.5), 'social_media_hours': (3.25, 1.75),
    'gaming_hours': (2.0, 1.25), 'work_study_hours': (3.25, 1.75), 'sleep_hours': (7.0, 1.5),
    'notifications_per_day': (125.0, 75.0), 'app_opens_per_day': (85.0, 50.0), 'weekend_screen_time': (9.0, 3.0),
}
BIN_WIDTHS = {
    'age': (2.0, 4.0), 'daily_screen_time_hours': (0.2, 0.5, 1.0), 'social_media_hours': (0.2, 0.5),
    'gaming_hours': (0.2, 0.5), 'work_study_hours': (0.2, 0.5), 'sleep_hours': (0.2, 0.5),
    'notifications_per_day': (5.0, 20.0), 'app_opens_per_day': (5.0, 20.0), 'weekend_screen_time': (0.2, 0.5, 1.0),
}
CONTRAST_PAIRS = (
    ('social_media_hours', 'work_study_hours'), ('app_opens_per_day', 'weekend_screen_time'),
    ('daily_screen_time_hours', 'app_opens_per_day'), ('notifications_per_day', 'weekend_screen_time'),
    ('daily_screen_time_hours', 'social_media_hours'), ('gaming_hours', 'weekend_screen_time'),
    ('daily_screen_time_hours', 'gaming_hours'), ('daily_screen_time_hours', 'work_study_hours'),
    ('work_study_hours', 'notifications_per_day'), ('work_study_hours', 'sleep_hours'), ('gaming_hours', 'sleep_hours'),
    ('social_media_hours', 'weekend_screen_time'),
)
CONTRAST_SOURCES = tuple(dict.fromkeys(column for pair in CONTRAST_PAIRS for column in pair))
CONTRAST_COLUMNS = tuple(f'contrast12__z_{left}__minus__z_{right}' for left, right in CONTRAST_PAIRS)
STATEFUL_BLOCKS = ('sparse_pair_contrast_12',)


def _num(frame, column):
    return pd.to_numeric(frame[column], errors='coerce').astype('float64')


def _f32(values, index):
    return pd.Series(values, index=index).replace([np.inf, -np.inf], np.nan).astype('float32')


def _cat(values, index):
    return pd.Series(values, index=index, dtype='string').fillna(MISSING_TOKEN).astype(object)


def _signed_log1p(values):
    return np.sign(values) * np.log1p(values.abs())


def _mask(frame, columns):
    bits = frame[list(columns)].notna().to_numpy(dtype=np.int64)
    return _cat('m:' + (bits @ (1 << np.arange(len(columns), dtype=np.int64))).astype(str), frame.index)


def _raw(frame):
    out = pd.DataFrame(index=frame.index)
    for name in NUMERIC_COLUMNS:
        out[name] = _num(frame, name).astype('float32')
    for name in CATEGORICAL_COLUMNS:
        out[name] = _cat(frame[name], frame.index)
    return out, list(CATEGORICAL_COLUMNS)


def _exact_grid(frame):
    out = pd.DataFrame(index=frame.index)
    for column in NUMERIC_COLUMNS:
        integer = (_num(frame, column) * EXACT_SCALES[column]).round().astype('Int64')
        out[f'grid_exact__{column}'] = _cat('v:' + integer.astype('string'), frame.index)
    return out, list(out.columns)


def _smooth_latent(frame):
    z = {c: (_num(frame, c) - LATENT_ANCHORS[c][0]) / LATENT_ANCHORS[c][1] for c in NUMERIC_COLUMNS}
    daily, social, gaming, work, sleep = (_num(frame, c) for c in (
        'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours'))
    parts = pd.concat([social, gaming, work], axis=1)
    known_sum = parts.sum(axis=1, min_count=1)
    values = {
        'latent__screen_load': pd.concat([z['daily_screen_time_hours'], z['social_media_hours'], z['gaming_hours'],
                                          z['weekend_screen_time']], axis=1).mean(axis=1),
        'latent__leisure_vs_productive': (z['social_media_hours'] + z['gaming_hours'] - z['work_study_hours'])
        / np.sqrt(3.0),
        'latent__weekend_escalation': z['weekend_screen_time'] - z['daily_screen_time_hours'],
        'latent__recovery_pressure': z['daily_screen_time_hours'] - z['sleep_hours'],
        'latent__interruption_load': pd.concat([z['notifications_per_day'], z['app_opens_per_day']],
                                               axis=1).mean(axis=1),
        'latent__notification_open_imbalance': z['notifications_per_day'] - z['app_opens_per_day'],
        'latent__social_gaming_balance': _signed_log1p(social) - _signed_log1p(gaming),
        'latent__screen_sleep_balance': _signed_log1p(daily) - _signed_log1p(sleep),
        'latent__daily_social_synergy': z['daily_screen_time_hours'] * z['social_media_hours'],
        'latent__daily_weekend_synergy': z['daily_screen_time_hours'] * z['weekend_screen_time'],
        'latent__notification_open_synergy': z['notifications_per_day'] * z['app_opens_per_day'],
        'latent__known_component_mean': parts.mean(axis=1),
        'latent__known_budget_slack': _signed_log1p(daily - known_sum),
        'latent__known_budget_balance': (daily - known_sum) / (1.0 + daily.abs() + known_sum.abs()),
    }
    out = pd.DataFrame(index=frame.index)
    for name, series in values.items():
        out[name] = _f32(series, frame.index)
    return out, []


def _missing_gated(frame):
    daily = _num(frame, 'daily_screen_time_hours')
    parts = pd.DataFrame({c: _num(frame, c) for c in COMPONENT_COLUMNS}, index=frame.index)
    observed = parts.notna()
    observed_count = observed.sum(axis=1)
    missing_count = len(COMPONENT_COLUMNS) - observed_count
    known_sum = parts.fillna(0.0).sum(axis=1)
    daily_completed = daily.fillna(known_sum)
    share = (daily_completed - known_sum).clip(lower=0.0) / missing_count.where(missing_count > 0, 1)
    completed = parts.copy()
    for column in COMPONENT_COLUMNS:
        completed[column] = parts[column].where(observed[column], share)
    weekend_completed = _num(frame, 'weekend_screen_time').fillna(daily_completed)
    mode = np.select([daily.isna(), missing_count.eq(0), missing_count.eq(1), missing_count.eq(2)],
                     ['daily_lower_bound', 'observed_complete', 'equal_slack_1', 'equal_slack_2'],
                     default='equal_slack_3')
    numeric = {
        'gate__numeric_missing_count': frame[list(NUMERIC_COLUMNS)].isna().sum(axis=1),
        'gate__screen_observed_count': observed_count,
        'gate__screen_missing_count': missing_count,
        'gate__known_component_sum': known_sum,
        'gate__daily_completed_lower_bound': daily_completed,
        'gate__social_completed_equal_slack': completed['social_media_hours'],
        'gate__gaming_completed_equal_slack': completed['gaming_hours'],
        'gate__work_completed_equal_slack': completed['work_study_hours'],
        'gate__completion_slack': daily_completed - completed.sum(axis=1),
        'gate__completion_mass': (completed - parts.fillna(0.0)).sum(axis=1),
        'gate__complete_weekly_screen': (5 * daily_completed + 2 * weekend_completed) / 7,
    }
    out = pd.DataFrame(index=frame.index)
    for name, series in numeric.items():
        out[name] = _f32(series, frame.index)
    out['gate__numeric_missing_mask'] = _mask(frame, NUMERIC_COLUMNS)
    out['gate__screen_observation_mask'] = _mask(frame, ('daily_screen_time_hours', *COMPONENT_COLUMNS))
    out['gate__completion_mode'] = _cat(mode, frame.index)
    return out, ['gate__numeric_missing_mask', 'gate__screen_observation_mask', 'gate__completion_mode']


def _fixed_bins(frame):
    out = pd.DataFrame(index=frame.index)
    for name, widths in BIN_WIDTHS.items():
        for width in widths:
            suffix = str(width).replace('.', 'p')
            token = np.floor(pd.to_numeric(frame[name], errors='coerce') / width + 1e-10).astype('Int64')
            out[f'bin{suffix}__{name}'] = _cat(f'{name}:b{suffix}:' + token.astype('string'), frame.index)
    return out, list(out.columns)


BUILDERS = {'raw': _raw, 'exact_grid_onehots': _exact_grid, 'smooth_latent_geometry': _smooth_latent,
            'missing_gated_views': _missing_gated, 'fixed_bins': _fixed_bins}


def build_rowwise_features(frame, blocks):
    """Row-local blocks in the given order (stateful blocks are added by prepare_fold_frames).
    Returns (features, categorical column names)."""
    frame = frame.reset_index(drop=True)
    pieces, categorical = [], []
    for block in blocks:
        if block in STATEFUL_BLOCKS:
            continue
        piece, cats = BUILDERS[block](frame)
        pieces.append(piece)
        categorical.extend(cats)
    return pd.concat(pieces, axis=1).reset_index(drop=True), categorical


def _pair_contrasts(frame, centers, scales):
    output = pd.DataFrame(index=frame.index)
    for name, (left, right) in zip(CONTRAST_COLUMNS, CONTRAST_PAIRS):
        z_left = (pd.to_numeric(frame[left], errors='coerce').astype('float64') - centers[left]) / scales[left]
        z_right = (pd.to_numeric(frame[right], errors='coerce').astype('float64') - centers[right]) / scales[right]
        output[name] = (z_left - z_right).astype('float32')
    return output.reset_index(drop=True)


def prepare_fold_frames(train, valid, test, categorical_names, blocks):
    """Fit the fold state on the outer-training frame and apply it to all three frames."""
    train, valid = train.reset_index(drop=True).copy(), valid.reset_index(drop=True).copy()
    test = test.reset_index(drop=True).copy() if test is not None else None
    if 'sparse_pair_contrast_12' in blocks:
        centers = {c: float(pd.to_numeric(train[c], errors='coerce').median()) for c in CONTRAST_SOURCES}
        scales = {c: float(pd.to_numeric(train[c], errors='coerce').std(ddof=1)) for c in CONTRAST_SOURCES}
        train = pd.concat([train, _pair_contrasts(train, centers, scales)], axis=1)
        valid = pd.concat([valid, _pair_contrasts(valid, centers, scales)], axis=1)
        if test is not None:
            test = pd.concat([test, _pair_contrasts(test, centers, scales)], axis=1)
    frames = [train, valid] + ([test] if test is not None else [])
    cats = list(categorical_names)
    numeric_names = [name for name in train.columns if name not in set(cats)]
    for name in numeric_names:
        median = float(pd.to_numeric(train[name], errors='coerce').replace([np.inf, -np.inf], np.nan).median())
        if not np.isfinite(median):
            median = 0.0
        for frame in frames:
            values = pd.to_numeric(frame[name], errors='coerce').replace([np.inf, -np.inf], np.nan)
            frame[name] = values.fillna(median).astype('float32')
    for name in cats:
        for frame in frames:
            frame[name] = frame[name].astype('string').fillna(MISSING_TOKEN).astype(object)
    return train, valid, test, cats
