"""Class-conditional original-to-synthetic transition likelihood (a generative model; no classifier is fitted).

score = base log-likelihood ratio + transition_weight * transition log-likelihood ratio, p = sigmoid(clip(score, +-35)).

Base (no original data): per-class Dirichlet-smoothed histograms (total pseudo-count 32) of the 9 raw numerics and 13
derived contrasts (32 outer-train quantile bins + a missing bin), the 3 categoricals and the 12-bit missingness pattern
(pseudo-count 256); event weights 0.8 raw, 0.35 derived, 0.5 categorical, 0.8 pattern; the weighted sum of
log-likelihood ratios is divided by the total weight and added to the prior log odds.

Transition: the original row is a latent cause. For each donor relation (lib/donors.py) the top `candidates` donors get
retrieval weights exp(-temperature (score - top score)) * (0.25 + 0.75 exact_fraction), normalized. Per donor events:
numeric residual (standardized by the original robust scale, 12 boundaries, exact match separate) x donor-value
quintile (weight 1), original category x observed category (1), retrieval score bin x exact bin (0.35), original label
x severity (0.5) and, if donor_identity, the donor row id (0.35, pseudo-count x 8). Tables are fitted per class on the
outer-train rows with the retrieval weights as sample weights (pseudo-count `alpha`). Per relation the donor
log-likelihoods (event-weighted mean) are marginalized with logsumexp over the donors under each class; the relation
log-likelihood ratios are averaged over the covered relations."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from .donors import CATEGORICAL, NUMERIC

BLOCKS = (
    ('screen_core', ('daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours'), ()),
    ('screen_weekend', ('daily_screen_time_hours', 'social_media_hours', 'weekend_screen_time'), ()),
    ('engagement', ('daily_screen_time_hours', 'weekend_screen_time', 'notifications_per_day', 'app_opens_per_day'), ()),
    ('behavior', ('social_media_hours', 'gaming_hours', 'notifications_per_day', 'app_opens_per_day'), ()),
    ('lifestyle', ('age', 'daily_screen_time_hours', 'work_study_hours', 'sleep_hours'), ()),
    ('profile', ('age', 'sleep_hours'), ('gender', 'stress_level', 'academic_work_impact')),
)
RESIDUAL_BOUNDARIES = np.asarray([-2.0, -1.0, -0.5, -0.2, -0.08, -0.02, 0.02, 0.08, 0.2, 0.5, 1.0, 2.0], dtype=np.float32)
SCORE_BOUNDARIES = np.asarray([-0.70, -0.50, -0.30, -0.10, 0.0, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0], dtype=np.float32)
EXACT_TOLERANCE = np.float32(5e-6)
BASE_ALPHA = 32.0
BASE_BINS = 32
VARIANTS = {'control': {'candidates': 0, 'temperature': 0.0, 'alpha': 0.0, 'donor_identity': False,
                        'transition_weight': 0.0}}
for _family, _candidates, _temperature, _alpha, _identity in (('k1', 1, 1.50, 24.0, False), ('k4', 4, 1.25, 48.0, True),
                                                              ('k8', 8, 0.75, 96.0, True)):
    for _label, _weight in (('w010', 0.10), ('w025', 0.25), ('w050', 0.50)):
        VARIANTS[f'{_family}_{_label}'] = {'candidates': _candidates, 'temperature': _temperature, 'alpha': _alpha,
                                          'donor_identity': _identity, 'transition_weight': _weight}


@dataclass(frozen=True)
class SourceSchema:
    numeric: np.ndarray
    numeric_scale: np.ndarray
    numeric_quintile: np.ndarray
    categorical: np.ndarray
    category_levels: dict
    source_target: np.ndarray
    source_severity: np.ndarray


@dataclass(frozen=True)
class QueryData:
    numeric: np.ndarray
    categorical: np.ndarray
    missing_pattern: np.ndarray


@dataclass(frozen=True)
class EventProbability:
    name: str
    log_p0: np.ndarray
    log_p1: np.ndarray
    weight: float


@dataclass(frozen=True)
class BaseState:
    continuous_names: tuple
    continuous_edges: tuple
    continuous_events: tuple
    categorical_events: tuple
    missing_event: EventProbability
    prior_log_odds: float


@dataclass(frozen=True)
class BlockState:
    name: str
    events: tuple


@dataclass(frozen=True)
class TransitionState:
    config: dict
    blocks: tuple


@dataclass(frozen=True)
class LikelihoodState:
    base: BaseState
    transition: object
    variant: str


def build_source_schema(source):
    numeric = source.loc[:, NUMERIC].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    if not np.isfinite(numeric).all():
        raise RuntimeError('original numerics contain missing values')
    q25 = np.quantile(numeric, 0.25, axis=0)
    q75 = np.quantile(numeric, 0.75, axis=0)
    robust = ((q75 - q25) / 1.349).astype(np.float32)
    standard = np.std(numeric, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(robust > 1e-6, robust, standard)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    quintile = np.empty_like(numeric, dtype=np.uint8)
    for column in range(numeric.shape[1]):
        edges = np.quantile(numeric[:, column], [0.2, 0.4, 0.6, 0.8])
        quintile[:, column] = np.searchsorted(edges, numeric[:, column], side='right').astype(np.uint8)
    categorical = np.empty((len(source), len(CATEGORICAL)), dtype=np.int16)
    levels = {}
    for j, column in enumerate(CATEGORICAL):
        values = source[column].astype('string').fillna('<MISSING>')
        current = tuple(sorted(values.unique().tolist()))
        levels[column] = current
        mapping = {value: code for code, value in enumerate(current)}
        categorical[:, j] = values.map(mapping).to_numpy(np.int16)
    source_target = source['addicted_label'].to_numpy(np.int8)
    severity = source['addiction_level'].map({'Mild': 0, 'Moderate': 1, 'Severe': 2}).to_numpy(np.float64)
    fallback = np.where(source_target == 1, 2, 0)
    severity = np.where(np.isfinite(severity), severity, fallback).astype(np.int8)
    return SourceSchema(numeric=numeric, numeric_scale=scale, numeric_quintile=quintile, categorical=categorical,
                        category_levels=levels, source_target=source_target, source_severity=severity)


def encode_query(frame, rows, schema):
    selected = frame.iloc[np.asarray(rows, dtype=np.int64)]
    numeric = selected.loc[:, NUMERIC].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    categorical = np.full((len(selected), len(CATEGORICAL)), -1, dtype=np.int16)
    for j, column in enumerate(CATEGORICAL):
        mapping = {value: code for code, value in enumerate(schema.category_levels[column])}
        values = selected[column].astype('string')
        observed = values.notna().to_numpy()
        mapped = values.map(mapping).fillna(-1).to_numpy(np.int16)
        categorical[observed, j] = mapped[observed]
    missing = np.concatenate((~np.isfinite(numeric), categorical < 0), axis=1)
    pattern = np.zeros(len(selected), dtype=np.uint16)
    for column in range(missing.shape[1]):
        pattern |= missing[:, column].astype(np.uint16) << np.uint16(column)
    return QueryData(numeric=numeric, categorical=categorical, missing_pattern=pattern)


def _safe_ratio(numerator, denominator):
    output = np.full(len(numerator), np.nan, dtype=np.float32)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 0.05)
    output[valid] = numerator[valid] / denominator[valid]
    return output


def continuous_base_features(query):
    """9 raw numerics (weight 0.8) + 13 derived contrasts (weight 0.35)."""
    x = query.numeric
    columns = [x[:, j] for j in range(x.shape[1])]
    names = list(NUMERIC)
    weights = [0.8] * len(NUMERIC)
    age, daily, social, gaming, work, sleep, notifications, opens, weekend = columns
    components = np.stack((social, gaming, work), axis=1)
    derived = (
        daily - social - gaming - work,
        daily - social - gaming,
        weekend - daily,
        _safe_ratio(social, daily),
        _safe_ratio(gaming, daily),
        _safe_ratio(work, daily),
        _safe_ratio(weekend, daily),
        _safe_ratio(notifications, opens + np.float32(1.0)),
        _safe_ratio(opens, daily + np.float32(0.25)),
        _safe_ratio(notifications, daily + np.float32(0.25)),
        _safe_ratio(daily, sleep),
        np.max(components, axis=1) - np.min(components, axis=1),
        social + gaming + work,
    )
    derived_names = ('screen_component_slack', 'daily_minus_social_gaming', 'weekend_minus_daily', 'social_share',
                     'gaming_share', 'work_share', 'weekend_ratio', 'notifications_per_open', 'opens_per_screen',
                     'notifications_per_screen', 'screen_sleep_ratio', 'component_spread', 'component_total')
    columns.extend(derived)
    names.extend(derived_names)
    weights.extend([0.35] * len(derived_names))
    return np.column_stack(columns).astype(np.float32), tuple(names), np.asarray(weights, dtype=np.float64)


def _fit_event(name, codes, target, n_codes, alpha, sample_weight=None, event_weight=1.0):
    """Per-class log probabilities of event codes, Dirichlet smoothing with total pseudo-count alpha."""
    codes_flat = np.asarray(codes, dtype=np.int64).reshape(-1)
    if codes_flat.size == 0 or np.min(codes_flat) < 0 or np.max(codes_flat) >= n_codes:
        raise RuntimeError(f'invalid event codes for {name}')
    if sample_weight is None:
        weights = np.ones(codes_flat.size, dtype=np.float64)
        expanded_target = np.asarray(target, dtype=np.int8).reshape(-1)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        target_array = np.asarray(target, dtype=np.int8).reshape(-1)
        expanded_target = np.repeat(target_array, codes_flat.size // target_array.size)
    pseudo = np.float64(alpha) / np.float64(n_codes)
    log_probabilities = []
    for label in (0, 1):
        selected_weights = weights * (expanded_target == label)
        counts = np.bincount(codes_flat, weights=selected_weights, minlength=n_codes).astype(np.float64)
        denominator = np.sum(selected_weights, dtype=np.float64) + np.float64(alpha)
        log_probabilities.append(np.log((counts + pseudo) / denominator))
    return EventProbability(name=name, log_p0=log_probabilities[0], log_p1=log_probabilities[1], weight=float(event_weight))


def _continuous_codes(values, edges):
    values = np.asarray(values, dtype=np.float32)
    codes = np.zeros(len(values), dtype=np.int16)
    observed = np.isfinite(values)
    codes[observed] = np.searchsorted(edges, values[observed], side='right').astype(np.int16) + 1
    return codes


def fit_base_state(query, target):
    continuous, names, weights = continuous_base_features(query)
    edges_out, events = [], []
    quantiles = np.linspace(0.0, 1.0, BASE_BINS + 1, dtype=np.float64)[1:-1]
    for j, name in enumerate(names):
        observed = continuous[np.isfinite(continuous[:, j]), j]
        edges = np.empty(0, dtype=np.float32) if observed.size == 0 else \
            np.unique(np.quantile(observed, quantiles)).astype(np.float32)
        codes = _continuous_codes(continuous[:, j], edges)
        events.append(_fit_event(f'base::{name}', codes, target, len(edges) + 2, BASE_ALPHA,
                                 event_weight=float(weights[j])))
        edges_out.append(edges)
    categorical_events = []
    for j, name in enumerate(CATEGORICAL):
        levels = max(int(np.max(query.categorical[:, j], initial=-1)) + 1, 1)
        codes = np.where(query.categorical[:, j] >= 0, query.categorical[:, j] + 1, 0)
        categorical_events.append(_fit_event(f'base::{name}', codes, target, levels + 1, BASE_ALPHA, event_weight=0.5))
    missing_event = _fit_event('base::missing_pattern', query.missing_pattern, target,
                               1 << (len(NUMERIC) + len(CATEGORICAL)), 256.0, event_weight=0.8)
    positives, negatives = float(np.sum(target == 1)), float(np.sum(target == 0))
    prior = float(np.log((positives + 1.0) / (negatives + 1.0)))
    return BaseState(continuous_names=names, continuous_edges=tuple(edges_out), continuous_events=tuple(events),
                     categorical_events=tuple(categorical_events), missing_event=missing_event, prior_log_odds=prior)


def predict_base(query, state):
    continuous, names, _ = continuous_base_features(query)
    score = np.full(len(query.numeric), state.prior_log_odds, dtype=np.float64)
    total_weight = 0.0
    for j, event in enumerate(state.continuous_events):
        codes = _continuous_codes(continuous[:, j], state.continuous_edges[j])
        score += event.weight * (event.log_p1[codes] - event.log_p0[codes])
        total_weight += event.weight
    for j, event in enumerate(state.categorical_events):
        codes = np.where(query.categorical[:, j] >= 0, query.categorical[:, j] + 1, 0)
        codes = np.minimum(codes, len(event.log_p0) - 1)
        score += event.weight * (event.log_p1[codes] - event.log_p0[codes])
        total_weight += event.weight
    event = state.missing_event
    score += event.weight * (event.log_p1[query.missing_pattern] - event.log_p0[query.missing_pattern])
    total_weight += event.weight
    return state.prior_log_odds + (score - state.prior_log_odds) / np.float64(total_weight)


def candidate_weights(view, rows, block_index, candidates, temperature):
    indices = np.asarray(view.indices[rows, block_index, :candidates], dtype=np.int32)
    score = np.asarray(view.score[rows, block_index, :candidates], dtype=np.float32)
    exact = np.asarray(view.exact[rows, block_index, :candidates], dtype=np.float32)
    valid = np.asarray(view.coverage[rows, block_index] > 0, dtype=bool)
    relative = score - score[:, :1]
    raw = np.exp(np.clip(-np.float32(temperature) * relative, -20.0, 20.0))
    raw *= np.float32(0.25) + np.float32(0.75) * exact
    denominator = np.sum(raw, axis=1, keepdims=True, dtype=np.float32)
    weights = raw / np.maximum(denominator, np.float32(1e-20))
    weights[~valid] = 0.0
    return indices, score, exact, weights


def numeric_transition_codes(query_values, donor_indices, numeric_index, schema):
    """1 + donor quintile * 14 + residual state (0 = exact match, 1..13 = residual bin); 0 = field missing."""
    source_values = schema.numeric[donor_indices, numeric_index]
    observed = np.isfinite(query_values)
    delta = (query_values[:, None] - source_values) / schema.numeric_scale[numeric_index]
    exact = np.abs(query_values[:, None] - source_values) <= EXACT_TOLERANCE
    residual_bin = np.digitize(delta, RESIDUAL_BOUNDARIES).astype(np.int16)
    local = np.where(exact, 0, residual_bin + 1).astype(np.int16)
    source_bin = schema.numeric_quintile[donor_indices, numeric_index].astype(np.int16)
    codes = 1 + source_bin * np.int16(14) + local
    codes[~observed, :] = 0
    return codes.astype(np.int16)


def categorical_transition_codes(query_values, donor_indices, categorical_index, schema):
    levels = len(schema.category_levels[CATEGORICAL[categorical_index]])
    source_values = schema.categorical[donor_indices, categorical_index]
    codes = (1 + source_values * levels + query_values[:, None]).astype(np.int16)
    codes[query_values < 0, :] = 0
    return codes, 1 + levels * levels


def retrieval_codes(score, exact):
    score_bin = np.digitize(score, SCORE_BOUNDARIES).astype(np.int16)
    exact_bin = np.clip(np.rint(exact * np.float32(8.0)), 0, 8).astype(np.int16)
    return score_bin * np.int16(9) + exact_bin


def _block_event_codes(block_index, query, donor_indices, score, exact, schema, donor_identity):
    """(event name, codes [rows, candidates], number of codes, event weight) for one relation."""
    name, numeric_names, categorical_names = BLOCKS[block_index]
    events = []
    for column in numeric_names:
        j = NUMERIC.index(column)
        events.append((f'{name}::residual::{column}',
                       numeric_transition_codes(query.numeric[:, j], donor_indices, j, schema), 71, 1.0))
    for column in categorical_names:
        j = CATEGORICAL.index(column)
        codes, n_codes = categorical_transition_codes(query.categorical[:, j], donor_indices, j, schema)
        events.append((f'{name}::categorical::{column}', codes, n_codes, 1.0))
    events.append((f'{name}::retrieval', retrieval_codes(score, exact), 117, 0.35))
    source_attribute = (schema.source_target[donor_indices].astype(np.int16) * np.int16(3)
                        + schema.source_severity[donor_indices].astype(np.int16))
    events.append((f'{name}::source_attribute', source_attribute, 6, 0.5))
    if donor_identity:
        events.append((f'{name}::source_identity', donor_indices.astype(np.int16), 7500, 0.35))
    return events


def fit_transition_state(query, target, view, rows, schema, config):
    candidates = int(config['candidates'])
    blocks = []
    for block_index, (name, _, _) in enumerate(BLOCKS):
        donor_indices, score, exact, weights = candidate_weights(view, rows, block_index, candidates,
                                                                 float(config['temperature']))
        event_states = []
        for event_name, codes, n_codes, event_weight in _block_event_codes(
                block_index, query, donor_indices, score, exact, schema, bool(config['donor_identity'])):
            event_alpha = float(config['alpha'])
            if event_name.endswith('source_identity'):
                event_alpha *= 8.0
            event_states.append(_fit_event(event_name, codes, target, n_codes, event_alpha, sample_weight=weights,
                                           event_weight=event_weight))
        blocks.append(BlockState(name=name, events=tuple(event_states)))
    return TransitionState(config=dict(config), blocks=tuple(blocks))


def predict_transition(query, view, rows, schema, state):
    config = state.config
    candidates = int(config['candidates'])
    relation_scores = np.zeros((len(rows), len(BLOCKS)), dtype=np.float64)
    relation_valid = np.zeros((len(rows), len(BLOCKS)), dtype=bool)
    for block_index, block in enumerate(state.blocks):
        donor_indices, score, exact, weights = candidate_weights(view, rows, block_index, candidates,
                                                                 float(config['temperature']))
        codes_by_name = {name: codes for name, codes, _, _ in _block_event_codes(
            block_index, query, donor_indices, score, exact, schema, bool(config['donor_identity']))}
        log_p0 = np.zeros((len(rows), candidates), dtype=np.float64)
        log_p1 = np.zeros((len(rows), candidates), dtype=np.float64)
        total_weight = 0.0
        for event in block.events:
            codes = codes_by_name[event.name]
            log_p0 += event.weight * event.log_p0[codes]
            log_p1 += event.weight * event.log_p1[codes]
            total_weight += event.weight
        log_p0 /= np.float64(total_weight)
        log_p1 /= np.float64(total_weight)
        log_weights = np.log(np.maximum(weights, np.float32(1e-30))).astype(np.float64)
        likelihood0 = logsumexp(log_weights + log_p0, axis=1)
        likelihood1 = logsumexp(log_weights + log_p1, axis=1)
        valid = np.asarray(view.coverage[rows, block_index] > 0, dtype=bool)
        relation_scores[valid, block_index] = likelihood1[valid] - likelihood0[valid]
        relation_valid[:, block_index] = valid
    valid_count = np.sum(relation_valid, axis=1)
    score = np.divide(np.sum(relation_scores, axis=1), np.maximum(valid_count, 1), out=np.zeros(len(rows), dtype=np.float64),
                      where=np.maximum(valid_count, 1) > 0)
    return score, relation_scores


def fit_likelihood_state(frame, target, view, rows, schema, variant):
    """Fits the base and transition tables on `rows` of `frame` (the outer-train rows)."""
    target_rows = np.asarray(target[np.asarray(rows, dtype=np.int64)], dtype=np.int8)
    query = encode_query(frame, rows, schema)
    base = fit_base_state(query, target_rows)
    config = VARIANTS[variant]
    transition = None
    if int(config['candidates']) > 0:
        transition = fit_transition_state(query, target_rows, view, rows, schema, config)
    return LikelihoodState(base=base, transition=transition, variant=variant)


def predict_likelihood(frame, view, rows, schema, state):
    query = encode_query(frame, rows, schema)
    base_score = predict_base(query, state.base)
    transition_score = np.zeros(len(rows), dtype=np.float64)
    if state.transition is not None:
        transition_score, _ = predict_transition(query, view, rows, schema, state.transition)
    log_odds = base_score + np.float64(VARIANTS[state.variant]['transition_weight']) * transition_score
    return expit(np.clip(log_odds, -35.0, 35.0)).astype(np.float64), base_score, transition_score
