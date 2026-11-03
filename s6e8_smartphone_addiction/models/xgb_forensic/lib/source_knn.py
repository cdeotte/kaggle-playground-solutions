"""Mask-aware nearest-neighbour geometry against the 7,500 original rows (GPU, cupy).

Coordinates: the 9 numerics standardized with the original's mean / std and 1.5 x one-hot original levels of the
3 categoricals. A query's missing fields contribute zero distance; distances are normalized by the number of observed
fields: d = sqrt(sum over observed j of (q_j - s_j)^2 / n_observed).
knn_features():    label probabilities of the 4 / 16 / 64 nearest original rows (mean and inverse-distance weighted),
                   nearest label, nearest and mean distances, nearest negative / positive distance, their difference and
                   log ratio, entropy of the k16 probability, number of observed fields (17 columns)
knn_imputations(): per screen field, the inverse-distance weighted mean of the 64 nearest original values: imputed
                   value, prediction-if-missing and its weighted std if missing (15 columns)"""
import numpy as np
import pandas as pd

NUMERIC = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "sleep_hours",
           "notifications_per_day", "app_opens_per_day", "weekend_screen_time"]
CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]
IMPUTED = ["daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "weekend_screen_time"]
KNN_FEATURE_NAMES = [
    "source_knn_probability_mean_k4", "source_knn_probability_mean_k16", "source_knn_probability_mean_k64",
    "source_knn_probability_inverse_k4", "source_knn_probability_inverse_k16", "source_knn_probability_inverse_k64",
    "source_knn_nearest_label", "source_knn_nearest_distance", "source_knn_mean_distance_k4",
    "source_knn_mean_distance_k16", "source_knn_mean_distance_k64", "source_knn_nearest_negative_distance",
    "source_knn_nearest_positive_distance", "source_knn_signed_distance_advantage", "source_knn_log_distance_ratio",
    "source_knn_probability_entropy_k16", "source_knn_observed_field_count",
]
IMPUTE_FEATURE_NAMES = [name for feature in IMPUTED for name in (
    f"source_knn_imputed__{feature}", f"source_knn_prediction_if_missing__{feature}",
    f"source_knn_uncertainty_if_missing__{feature}")]


def encode(frame, source):
    """Query coordinates, observed-coordinate mask and observed-field count."""
    source_numeric = source[NUMERIC].apply(pd.to_numeric, errors="raise")
    mean = source_numeric.mean().to_numpy(np.float64)
    scale = source_numeric.std().to_numpy(np.float64)
    numeric = frame[NUMERIC].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    observed_numeric = np.isfinite(numeric)
    numeric = (numeric - mean) / scale
    numeric[~observed_numeric] = 0.0
    parts, masks = [numeric], [observed_numeric.astype(np.float64)]
    field_counts = observed_numeric.sum(axis=1).astype(np.float64)
    for column in CATEGORICAL:
        levels = sorted(source[column].astype(str).unique())
        raw = frame[column].astype("string")
        observed = raw.notna().to_numpy()
        values = raw.fillna("<NA>").astype(str).to_numpy()
        parts.append(1.5 * np.column_stack([(values == level) for level in levels]).astype(np.float64))
        masks.append(np.repeat(observed[:, None], len(levels), axis=1).astype(np.float64))
        field_counts += observed
    return (np.column_stack(parts).astype(np.float32), np.column_stack(masks).astype(np.float32),
            field_counts.astype(np.float32))


def _distances(cp, q, m, source_gpu, source_sq_gpu):
    # exact masked squared Euclidean distance: sum_j m_j (q_j - s_j)^2
    distance2 = cp.sum(cp.square(q), axis=1)[:, None] + m @ source_sq_gpu.T - cp.float32(2.0) * (q @ source_gpu.T)
    return cp.maximum(distance2, cp.float32(0.0))


def knn_features(frame, source, source_matrix, source_target, chunk_rows=6000):
    import cupy as cp
    query, mask, field_count = encode(frame, source)
    source_gpu = cp.asarray(source_matrix, dtype=cp.float32)
    source_sq_gpu = cp.square(source_gpu)
    target_gpu = cp.asarray(source_target, dtype=cp.float32)
    negative, positive = cp.flatnonzero(target_gpu == 0), cp.flatnonzero(target_gpu == 1)
    output = np.empty((len(frame), len(KNN_FEATURE_NAMES)), np.float32)
    for start in range(0, len(frame), chunk_rows):
        stop = min(start + chunk_rows, len(frame))
        count = cp.asarray(field_count[start:stop])
        distance2 = _distances(cp, cp.asarray(query[start:stop]), cp.asarray(mask[start:stop]), source_gpu,
                               source_sq_gpu)
        top = cp.argpartition(distance2, kth=63, axis=1)[:, :64]
        top_distance2 = cp.take_along_axis(distance2, top, axis=1)
        order = cp.argsort(top_distance2, axis=1)
        top = cp.take_along_axis(top, order, axis=1)
        top_distance2 = cp.take_along_axis(top_distance2, order, axis=1)
        normalized_distance = cp.sqrt(top_distance2 / count[:, None])
        labels = target_gpu[top]
        columns = [cp.mean(labels[:, :k], axis=1) for k in (4, 16, 64)]
        for k in (4, 16, 64):
            inverse = cp.float32(1.0) / cp.maximum(normalized_distance[:, :k], cp.float32(1e-4))
            columns.append(cp.sum(inverse * labels[:, :k], axis=1) / cp.sum(inverse, axis=1))
        columns.append(labels[:, 0])
        columns += [normalized_distance[:, 0], cp.mean(normalized_distance[:, :4], axis=1),
                    cp.mean(normalized_distance[:, :16], axis=1), cp.mean(normalized_distance[:, :64], axis=1)]
        nearest_negative = cp.sqrt(cp.min(distance2[:, negative], axis=1) / count)
        nearest_positive = cp.sqrt(cp.min(distance2[:, positive], axis=1) / count)
        columns += [nearest_negative, nearest_positive, nearest_negative - nearest_positive,
                    cp.log((nearest_negative + cp.float32(1e-5)) / (nearest_positive + cp.float32(1e-5)))]
        p16 = cp.clip(columns[1], cp.float32(1e-5), cp.float32(1.0 - 1e-5))
        columns.append(-(p16 * cp.log(p16) + (1 - p16) * cp.log(1 - p16)))
        columns.append(count)
        output[start:stop] = cp.asnumpy(cp.column_stack(columns).astype(cp.float32))
    return output


def knn_imputations(frame, source, source_matrix, source_values, chunk_rows=6000):
    import cupy as cp
    query, mask, field_count = encode(frame, source)
    source_gpu = cp.asarray(source_matrix, dtype=cp.float32)
    source_sq_gpu = cp.square(source_gpu)
    values_gpu = cp.asarray(source_values, dtype=cp.float32)
    raw = frame[IMPUTED].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    output = np.empty((len(frame), len(IMPUTE_FEATURE_NAMES)), np.float32)
    for start in range(0, len(frame), chunk_rows):
        stop = min(start + chunk_rows, len(frame))
        count = cp.asarray(field_count[start:stop])
        distance2 = _distances(cp, cp.asarray(query[start:stop]), cp.asarray(mask[start:stop]), source_gpu,
                               source_sq_gpu)
        top = cp.argpartition(distance2, kth=63, axis=1)[:, :64]
        normalized_distance = cp.sqrt(cp.take_along_axis(distance2, top, axis=1) / count[:, None])
        weight = (cp.float32(1.0) / cp.maximum(normalized_distance, cp.float32(1e-4)))[:, :, None]
        neighbor_values = values_gpu[top]
        estimate = cp.sum(weight * neighbor_values, axis=1) / cp.sum(weight, axis=1)
        variance = cp.sum(weight * cp.square(neighbor_values - estimate[:, None, :]), axis=1) / cp.sum(weight, axis=1)
        estimate_cpu = cp.asnumpy(estimate)
        uncertainty_cpu = cp.asnumpy(cp.sqrt(cp.maximum(variance, cp.float32(0.0))))
        raw_block = raw[start:stop]
        missing = ~np.isfinite(raw_block)
        columns = []
        for index in range(len(IMPUTED)):
            filled = raw_block[:, index].copy()
            filled[missing[:, index]] = estimate_cpu[missing[:, index], index]
            columns += [filled, np.where(missing[:, index], estimate_cpu[:, index], 0.0),
                        np.where(missing[:, index], uncertainty_cpu[:, index], 0.0)]
        output[start:stop] = np.column_stack(columns).astype(np.float32)
    return output
