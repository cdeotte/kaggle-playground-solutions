"""Views of the cross-fitted reconstruction cache (lib/upstream.reconstruction_names) used by the TabM and CatBoost
parents: neural completion of missing predictors, missing flags, and CatBoost reconstruction categories."""
import numpy as np
import pandas as pd

from lib.upstream import CATEGORICAL_LEVELS, NUMERIC_FIELDS, SCREEN_FIELDS


def completed_predictors(frame, reconstruction, names, fill_source='hybrid'):
    """Missing numerics <- <source>full__completed__<field>; missing categoricals <- argmax of the categorical
    probabilities (hybrid: mean of the XGBoost and autoencoder probabilities)."""
    index = {name: column for column, name in enumerate(names)}
    completed = frame.copy()
    for field in NUMERIC_FIELDS:
        values = np.asarray(reconstruction[:, index[f'{fill_source}full__completed__{field}']], dtype=np.float64)
        missing = pd.to_numeric(completed[field], errors='coerce').isna().to_numpy()
        completed.loc[missing, field] = values[missing]
    for field, levels in CATEGORICAL_LEVELS.items():
        missing = completed[field].isna().to_numpy()
        if not missing.any():
            continue
        ae = np.column_stack([reconstruction[:, index[f'ae__cat_probability_if_missing__{field}__{level}']]
                              for level in levels])
        if fill_source == 'ae':
            probabilities = ae
        else:
            xgb = np.column_stack([reconstruction[:, index[f'xgb__cat_probability_if_missing__{field}__{level}']]
                                   for level in levels])
            probabilities = 0.5 * (ae + xgb)
        labels = np.asarray(levels, dtype=object)[np.argmax(probabilities, axis=1)]
        completed.loc[missing, field] = labels[missing]
    return completed


def missing_flags(frame):
    fields = list(NUMERIC_FIELDS) + list(CATEGORICAL_LEVELS)
    return pd.DataFrame({f'nnmissing__{field}': frame[field].isna().to_numpy(dtype=np.float32) for field in fields})


def _quantile_edges(values, bins):
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    return np.unique(np.quantile(finite, np.arange(1, bins) / bins)).astype(np.float64)


def _apply_bins(values, edges):
    values = np.asarray(values, np.float64)
    output = np.zeros(len(values), dtype=np.int32)
    finite = np.isfinite(values)
    output[finite] = np.searchsorted(edges, values[finite], side='right').astype(np.int32) + 1
    return output


def _missing_codes(frame):
    fields = list(NUMERIC_FIELDS) + list(CATEGORICAL_LEVELS)
    missing = np.column_stack([frame[field].isna().to_numpy() for field in fields])
    all_code = np.sum(missing * (1 << np.arange(len(fields), dtype=np.int64))[None, :], axis=1).astype(np.int32)
    screen = [fields.index(name) for name in SCREEN_FIELDS]
    screen_code = np.sum(missing[:, screen] * (1 << np.arange(len(screen), dtype=np.int64))[None, :],
                         axis=1).astype(np.int32)
    return all_code, screen_code, missing.sum(axis=1).astype(np.int32)


def reconstruction_categories(train_raw, apply_raw, train_recon, apply_recon, names, view='core'):
    """CatBoost-native categories of the reconstruction (30 columns in the core view):
      reconcat__missing__{all12,screen5,count}        missing-pattern bit codes and count of the raw row
      reconcat__hybrid_q32__<numeric>                 32-quantile bin of the hybrid completed value
      reconcat__disagreement_q8__<numeric>            8-quantile bin of the XGBoost / autoencoder disagreement if
                                                      missing
      reconcat__{completed,hybrid_prediction,agreement}__<categorical>   completed level, predicted level if missing,
                                                      1 + (XGBoost and autoencoder levels differ) if missing
    Quantile edges are fit on the outer-training reconstruction values only ('rich' adds more bins; not used here).
    Returns (train frame, list of apply frames, names)."""
    lookup = {name: index for index, name in enumerate(names)}
    train_parts, apply_parts = {}, [{} for _ in apply_raw]

    def add(name, train_values, values):
        train_parts[name] = np.asarray(train_values, np.int32)
        for destination, local in zip(apply_parts, values):
            destination[name] = np.asarray(local, np.int32)

    train_masks = _missing_codes(train_raw)
    apply_masks = [_missing_codes(frame) for frame in apply_raw]
    for index, label in enumerate(('all12', 'screen5', 'count')):
        add(f'reconcat__missing__{label}', train_masks[index], [values[index] for values in apply_masks])

    def binned(label, source_name, bins):
        column = lookup[source_name]
        edges = _quantile_edges(train_recon[:, column], bins)
        add(label, _apply_bins(train_recon[:, column], edges), [_apply_bins(m[:, column], edges) for m in apply_recon])

    for field in NUMERIC_FIELDS:
        binned(f'reconcat__hybrid_q32__{field}', f'hybridfull__completed__{field}', 32)
        binned(f'reconcat__disagreement_q8__{field}', f'hybrid__disagreement_norm_if_missing__{field}', 8)
        if view == 'rich':
            for mechanism in ('xgb', 'ae'):
                binned(f'reconcat__{mechanism}_q16__{field}', f'{mechanism}full__completed__{field}', 16)
            for label, source_name in (('uncertainty', f'hybrid__ae_uncertainty_if_missing__{field}'),
                                       ('source_llr', f'hybrid__source_llr_if_missing__{field}'),
                                       ('xgb_residual', f'xgbfull__residual_norm_observed__{field}'),
                                       ('ae_residual', f'aefull__residual_norm_observed__{field}')):
                binned(f'reconcat__{label}_q8__{field}', source_name, 8)

    for field, levels in CATEGORICAL_LEVELS.items():
        level_lookup = {str(level): index + 1 for index, level in enumerate(levels)}
        xgb_columns = [lookup[f'xgb__cat_probability_if_missing__{field}__{level}'] for level in levels]
        ae_columns = [lookup[f'ae__cat_probability_if_missing__{field}__{level}'] for level in levels]

        def bundle(raw, matrix):
            missing = raw[field].isna().to_numpy()
            xgb = np.asarray(matrix[:, xgb_columns], np.float64)
            ae = np.asarray(matrix[:, ae_columns], np.float64)
            xgb_code = np.argmax(xgb, axis=1).astype(np.int32) + 1
            ae_code = np.argmax(ae, axis=1).astype(np.int32) + 1
            hybrid_code = np.argmax(0.5 * (xgb + ae), axis=1).astype(np.int32) + 1
            observed = raw[field].map(level_lookup).fillna(0).to_numpy(np.int32)
            return (np.where(missing, hybrid_code, observed), np.where(missing, hybrid_code, 0),
                    np.where(missing, 1 + (xgb_code != ae_code), 0), np.where(missing, xgb_code, 0))

        train_bundle = bundle(train_raw, train_recon)
        apply_bundles = [bundle(raw, matrix) for raw, matrix in zip(apply_raw, apply_recon)]
        for local_index, label in enumerate(('completed', 'hybrid_prediction', 'agreement', 'xgb_prediction')):
            if view == 'core' and label == 'xgb_prediction':
                continue
            add(f'reconcat__{label}__{field}', train_bundle[local_index], [b[local_index] for b in apply_bundles])

    names_out = list(train_parts)
    return pd.DataFrame(train_parts), [pd.DataFrame(parts) for parts in apply_parts], names_out
