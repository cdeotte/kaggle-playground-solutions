"""Exact-pair lift bank: nine two-column target residuals plus their row mean (10 columns).

Every value is tokenized to its exact two-decimal string (missing -> __NA__). For a pair (A, B):
  marginal rates   P(y=1|A), P(y=1|B)   shrunk to the prior with smoothing 20
  baseline         sigmoid(logit P(y=1|A) + logit P(y=1|B) - logit P(y=1))   (log-odds independence)
  lift             (pair_positive - pair_count * baseline) / (pair_count + s)  with s = 20 (100 for notifications x
                   app opens); an unseen pair gets exactly 0
Inside outer fold k the outer-training rows get their values from a five-way inner cross-fit
(StratifiedKFold(5, shuffle=True, random_state=777) on the outer-training labels; a row never sees its own inner part);
outer-validation and test rows use the state fitted on all outer-training rows."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from lib.paths import pairlift_dir

MARGINAL_SMOOTHING = 20.0
MISSING_TOKEN = '__NA__'
INNER_FOLDS, INNER_SEED = 5, 777
PAIR_SPECS = (
    ('app_opens_per_day', 'weekend_screen_time', 20.0),
    ('daily_screen_time_hours', 'app_opens_per_day', 20.0),
    ('notifications_per_day', 'weekend_screen_time', 20.0),
    ('daily_screen_time_hours', 'weekend_screen_time', 20.0),
    ('daily_screen_time_hours', 'notifications_per_day', 20.0),
    ('daily_screen_time_hours', 'gaming_hours', 20.0),
    ('notifications_per_day', 'app_opens_per_day', 100.0),
    ('sleep_hours', 'app_opens_per_day', 20.0),
    ('social_media_hours', 'sleep_hours', 20.0),
)
FEATURE_NAMES = tuple(f'pairlift__{left}__{right}' for left, right, _ in PAIR_SPECS) + ('pairlift__mean',)


@dataclass(frozen=True)
class PairLiftState:
    prior: float
    left_rate: pd.Series
    right_rate: pd.Series
    pair_positive: pd.Series
    pair_count: pd.Series
    smoothing: float


def tokenise(series):
    return series.map(lambda value: f'{value:.2f}' if pd.notna(value) else MISSING_TOKEN).to_numpy(dtype=str)


def tokenise_frame(frame):
    columns = sorted({field for left, right, _ in PAIR_SPECS for field in (left, right)})
    return {column: tokenise(frame[column]) for column in columns}


def _logit(probability):
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _pair_keys(left, right):
    return np.char.add(np.char.add(left.astype(str), '|'), right.astype(str))


def fit_state(left, right, target, smoothing):
    y = np.asarray(target, dtype=np.float64)
    prior = float(y.mean())
    left_group = pd.DataFrame({'token': left, 'target': y}).groupby('token', sort=True)['target'].agg(
        ['sum', 'count'])
    right_group = pd.DataFrame({'token': right, 'target': y}).groupby('token', sort=True)['target'].agg(
        ['sum', 'count'])
    left_rate = (left_group['sum'] + MARGINAL_SMOOTHING * prior) / (left_group['count'] + MARGINAL_SMOOTHING)
    right_rate = (right_group['sum'] + MARGINAL_SMOOTHING * prior) / (right_group['count'] + MARGINAL_SMOOTHING)
    pair_group = pd.DataFrame({'key': _pair_keys(left, right), 'target': y}).groupby('key', sort=True)['target'].agg(
        ['sum', 'count'])
    return PairLiftState(prior=prior, left_rate=left_rate, right_rate=right_rate, pair_positive=pair_group['sum'],
                         pair_count=pair_group['count'], smoothing=float(smoothing))


def apply_state(state, left, right):
    left_probability = pd.Series(left, copy=False).map(state.left_rate).fillna(state.prior).to_numpy()
    right_probability = pd.Series(right, copy=False).map(state.right_rate).fillna(state.prior).to_numpy()
    base_logit = _logit(left_probability) + _logit(right_probability) - float(_logit(state.prior))
    baseline = 1.0 / (1.0 + np.exp(-base_logit))
    keys = pd.Series(_pair_keys(left, right), copy=False)
    count = keys.map(state.pair_count).fillna(0.0).to_numpy(dtype=np.float64)
    positive = keys.map(state.pair_positive).fillna(0.0).to_numpy(dtype=np.float64)
    # equal to shrunk pair probability - baseline; an unseen pair gives exactly zero
    return (positive - count * baseline) / (count + state.smoothing)


def apply_bank(fit_tokens, fit_target, query_tokens):
    columns = []
    for left, right, smoothing in PAIR_SPECS:
        state = fit_state(fit_tokens[left], fit_tokens[right], fit_target, smoothing)
        columns.append(apply_state(state, query_tokens[left], query_tokens[right]))
    matrix = np.column_stack(columns)
    return np.column_stack([matrix, matrix.mean(axis=1)]).astype(np.float32)


def subset_tokens(tokens, indices):
    selected = np.asarray(indices, dtype=np.int64)
    return {name: values[selected] for name, values in tokens.items()}


def build_fold(train_tokens, test_tokens, y, train_idx, valid_idx):
    """(outer-train matrix, outer-valid matrix, test matrix) of one outer fold."""
    train_matrix = np.empty((len(train_idx), len(FEATURE_NAMES)), dtype=np.float32)
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=INNER_SEED)
    for fit_position, apply_position in inner.split(np.zeros(len(train_idx)), y[train_idx]):
        fit_indices = train_idx[fit_position]
        train_matrix[apply_position] = apply_bank(subset_tokens(train_tokens, fit_indices), y[fit_indices],
                                                  subset_tokens(train_tokens, train_idx[apply_position]))
    outer_tokens = subset_tokens(train_tokens, train_idx)
    valid_matrix = apply_bank(outer_tokens, y[train_idx], subset_tokens(train_tokens, valid_idx))
    test_matrix = apply_bank(outer_tokens, y[train_idx], test_tokens)
    return train_matrix, valid_matrix, test_matrix


def load(fold, role):
    """Cached pair-lift matrix of one fold; role in train / valid / test (build_pairlift.py writes it)."""
    return np.load(pairlift_dir(fold) / f'{role}_features.npy').astype(np.float32)


def append(frame, fold, role):
    """frame + the 10 pair-lift columns (appended after the parent's columns)."""
    values = load(fold, role)
    assert len(values) == len(frame), f'{role} pair-lift rows differ from the model frame'
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(values, columns=list(FEATURE_NAMES))], axis=1)
