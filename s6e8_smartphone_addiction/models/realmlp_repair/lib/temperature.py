"""Generator-temperature channels (target-free): how often each exact value appears in the synthetic outer-training
rows relative to its count in the original data.

Per numeric field: log1p(original count), log1p(outer-training count) (a training row's own occurrence removed),
raw amplification (synthetic - original log count, original-supported values only), residual from a weighted power
law log1p(synthetic) ~ a + b log1p(original) fit over the shared values (weights sqrt(original x synthetic)),
original support and observed flags; row mean / std / min / max of raw and residual and the support fraction over
the 5 screen fields ('strong5') and all 9 numerics. The RealMLP uses the 39 channels of the 5 screen fields
(6 per field) and the strong5 aggregates."""
import numpy as np
import pandas as pd

FIELDS = ('age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours',
          'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time')
STRONG = ('daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'weekend_screen_time')


def fit_power(source_count, synthetic_count):
    shared = source_count.index.intersection(synthetic_count.index)
    source = source_count.reindex(shared).to_numpy(np.float64)
    synthetic = synthetic_count.reindex(shared).to_numpy(np.float64)
    x, t = np.log1p(source), np.log1p(synthetic)
    weight = np.sqrt(source * synthetic)
    design = np.column_stack([np.ones(len(x)), x])
    coefficient = np.linalg.lstsq(design * np.sqrt(weight)[:, None], t * np.sqrt(weight), rcond=None)[0]
    return {'intercept': float(coefficient[0]), 'slope': float(coefficient[1])}


def feature_names():
    names = []
    for field in FIELDS:
        names += [f'temperature__{field}__{m}' for m in ('source_log_count', 'synthetic_log_count', 'raw_amplification',
                                                          'power_residual', 'source_support', 'observed')]
    for group in ('strong5', 'all9'):
        for metric in ('raw', 'residual'):
            names += [f'temperature__aggregate__{group}__{metric}_{s}' for s in ('mean', 'std', 'min', 'max')]
        names.append(f'temperature__aggregate__{group}__support_fraction')
    return names


STRONG39 = [i for i, n in enumerate(feature_names())
            if n.startswith(tuple(f'temperature__{f}__' for f in STRONG)) or n.startswith('temperature__aggregate__strong5__')]


def map_frame(frame, source_counts, synthetic_counts, powers, training_rows):
    per_field, blocks = {}, []
    for field in FIELDS:
        value = pd.to_numeric(frame[field], errors='coerce')
        observed = value.notna().to_numpy()
        source = value.map(source_counts[field]).fillna(0).to_numpy(np.float64)
        synthetic = value.map(synthetic_counts[field]).fillna(0).to_numpy(np.float64)
        if training_rows:  # remove the row's own occurrence from the outer-training histogram
            synthetic = np.maximum(synthetic - observed.astype(np.float64), 0.0)
        support = (source > 0) & observed
        source_log, synthetic_log = np.log1p(source), np.log1p(synthetic)
        raw = np.zeros(len(frame))
        residual = np.zeros(len(frame))
        raw[support] = synthetic_log[support] - source_log[support]
        power = powers[field]
        residual[support] = synthetic_log[support] - (power['intercept'] + power['slope'] * source_log[support])
        per_field[field] = {'raw': raw, 'residual': residual, 'support': support.astype(np.float64)}
        blocks += [source_log, synthetic_log, raw, residual, support.astype(np.float64), observed.astype(np.float64)]
    for group in (STRONG, FIELDS):
        for metric in ('raw', 'residual'):
            values = np.column_stack([per_field[f][metric] for f in group])
            support = np.column_stack([per_field[f]['support'] for f in group])
            masked = np.where(support > 0, values, np.nan)
            count = support.sum(axis=1)
            safe = np.where(np.isfinite(masked), masked, 0.0)
            mean = safe.sum(axis=1) / np.maximum(count, 1.0)
            centered = np.where(np.isfinite(masked), masked - mean[:, None], 0.0)
            std = np.sqrt(np.square(centered).sum(axis=1) / np.maximum(count, 1.0))
            minimum = np.where(count > 0, np.min(np.where(np.isfinite(masked), masked, np.inf), axis=1), 0.0)
            maximum = np.where(count > 0, np.max(np.where(np.isfinite(masked), masked, -np.inf), axis=1), 0.0)
            blocks += [mean, std, minimum, maximum]
        blocks.append(np.column_stack([per_field[f]['support'] for f in group]).mean(axis=1))
    out = np.column_stack(blocks).astype(np.float32)
    assert out.shape[1] == len(feature_names()) and np.isfinite(out).all()
    return out


def build_fold(train, original, fold_train, fold_valid, test=None):
    fit = train.iloc[fold_train]
    source_counts, synthetic_counts, powers = {}, {}, {}
    for field in FIELDS:
        source_counts[field] = pd.to_numeric(original[field], errors='coerce').value_counts(dropna=True, sort=False)
        synthetic_counts[field] = pd.to_numeric(fit[field], errors='coerce').value_counts(dropna=True, sort=False)
        powers[field] = fit_power(source_counts[field], synthetic_counts[field])
    out = {'train': map_frame(fit, source_counts, synthetic_counts, powers, True),
           'valid': map_frame(train.iloc[fold_valid], source_counts, synthetic_counts, powers, False)}
    if test is not None:
        out['test'] = map_frame(test, source_counts, synthetic_counts, powers, False)
    return out
