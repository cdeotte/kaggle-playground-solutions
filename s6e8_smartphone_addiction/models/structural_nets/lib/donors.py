"""Target-free donor retrieval from the 7,500-row original dataset.

Competition rows look like mosaics of fragments of original rows, so retrieval runs separately for six semantic field
blocks. For each block and each row, the TOP_K original rows with the lowest score are kept:

    score = mean over the row's observed block fields of the robust-scaled squared distance
            (a categorical mismatch counts 0.75) - 0.75 * fraction of observed block fields that match exactly

Robust scales ((q75 - q25) / 1.349, else the std) come from the original predictors only; neither competition labels
nor original labels are read. Rows are grouped by their block missingness pattern so only observed fields enter."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.data import CATEGORICALS as CATEGORICAL, NUMERICS as NUMERIC, cache_dir

BLOCKS = {
    'screen_core': (('daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours'), ()),
    'screen_weekend': (('daily_screen_time_hours', 'social_media_hours', 'weekend_screen_time'), ()),
    'engagement': (('daily_screen_time_hours', 'weekend_screen_time', 'notifications_per_day', 'app_opens_per_day'), ()),
    'behavior': (('social_media_hours', 'gaming_hours', 'notifications_per_day', 'app_opens_per_day'), ()),
    'lifestyle': (('age', 'daily_screen_time_hours', 'work_study_hours', 'sleep_hours'), ()),
    'profile': (('age', 'sleep_hours'), ('gender', 'stress_level', 'academic_work_impact')),
}
BLOCK_NAMES = tuple(BLOCKS)
TOP_K = 16
EXACT_BONUS = np.float32(0.75)
CATEGORICAL_MISMATCH = np.float32(0.75)
SOURCE_ROWS = 7500
DONOR_FILES = ('indices', 'score', 'exact_fraction', 'coverage', 'pattern')


@dataclass(frozen=True)
class RetrievalState:
    numeric: np.ndarray
    categorical: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    category_levels: dict


def fit_retrieval_state(source):
    """Predictor-only state of the original data (no label input)."""
    numeric = source.loc[:, NUMERIC].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    if not np.isfinite(numeric).all():
        raise RuntimeError('original numeric predictors contain missing values')
    center = np.median(numeric, axis=0).astype(np.float32)
    q25 = np.quantile(numeric, 0.25, axis=0)
    q75 = np.quantile(numeric, 0.75, axis=0)
    robust = ((q75 - q25) / np.float64(1.349)).astype(np.float32)
    standard = np.std(numeric, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(robust > 1e-8, robust, standard)
    scale = np.where(scale > 1e-8, scale, 1.0).astype(np.float32)
    categorical = np.empty((len(source), len(CATEGORICAL)), dtype=np.int16)
    levels = {}
    for j, column in enumerate(CATEGORICAL):
        values = source[column].astype('string').fillna('<MISSING>')
        current = tuple(sorted(values.unique().tolist()))
        levels[column] = current
        mapping = {value: i for i, value in enumerate(current)}
        categorical[:, j] = values.map(mapping).to_numpy(np.int16)
    return RetrievalState(numeric=numeric, categorical=categorical, center=center, scale=scale, category_levels=levels)


def encode_queries(frame, state):
    """Numeric fields as float32 (NaN = missing); categoricals as codes of the original levels (-1 = missing or unseen)."""
    numeric = frame.loc[:, NUMERIC].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    categorical = np.full((len(frame), len(CATEGORICAL)), -1, dtype=np.int16)
    for j, column in enumerate(CATEGORICAL):
        mapping = {value: i for i, value in enumerate(state.category_levels[column])}
        values = frame[column].astype('string')
        observed = values.notna().to_numpy()
        mapped = values.map(mapping).fillna(-1).to_numpy(np.int16)
        categorical[observed, j] = mapped[observed]
    return numeric, categorical


def build_donor_arrays(frame, state, device='cuda', chunk_size=4096, top_k=TOP_K):
    """Donors of every row of `frame` (GPU). Returns indices [rows, 6, top_k] (uint16), score and exact_fraction
    (float16), coverage (observed fields per block, uint8) and pattern (block missingness bits, uint8)."""
    import torch

    query_numeric, query_categorical = encode_queries(frame, state)
    source_numeric = torch.as_tensor(state.numeric, dtype=torch.float32, device=device)
    source_categorical = torch.as_tensor(state.categorical, dtype=torch.int16, device=device)
    scale = torch.as_tensor(state.scale, dtype=torch.float32, device=device)
    # tiny deterministic tie-break towards lower original row ids
    tie_break = torch.arange(len(state.numeric), dtype=torch.float32, device=device)
    tie_break = tie_break * (np.float32(1e-7) / np.float32(len(state.numeric)))
    n = len(frame)
    indices_out = np.zeros((n, len(BLOCKS), top_k), dtype=np.uint16)
    score_out = np.zeros((n, len(BLOCKS), top_k), dtype=np.float16)
    exact_out = np.zeros((n, len(BLOCKS), top_k), dtype=np.float16)
    coverage_out = np.zeros((n, len(BLOCKS)), dtype=np.uint8)
    pattern_out = np.zeros((n, len(BLOCKS)), dtype=np.uint8)

    for b, (block, (numeric_names, categorical_names)) in enumerate(BLOCKS.items()):
        numeric_indices = [NUMERIC.index(name) for name in numeric_names]
        categorical_indices = [CATEGORICAL.index(name) for name in categorical_names]
        block_numeric = query_numeric[:, numeric_indices]
        block_categorical = query_categorical[:, categorical_indices]
        observed_numeric = np.isfinite(block_numeric)
        observed_categorical = block_categorical >= 0
        pattern = np.zeros(n, dtype=np.uint8)
        for j in range(len(numeric_indices)):
            pattern |= observed_numeric[:, j].astype(np.uint8) << j
        for j in range(len(categorical_indices)):
            pattern |= observed_categorical[:, j].astype(np.uint8) << (len(numeric_indices) + j)
        pattern_out[:, b] = pattern
        for pattern_value in np.unique(pattern):
            rows = np.flatnonzero(pattern == pattern_value)
            active_numeric = [j for j in range(len(numeric_indices)) if pattern_value & (1 << j)]
            active_categorical = [j for j in range(len(categorical_indices))
                                  if pattern_value & (1 << (len(numeric_indices) + j))]
            observed_count = len(active_numeric) + len(active_categorical)
            coverage_out[rows, b] = observed_count
            if observed_count == 0:
                continue
            for offset in range(0, len(rows), chunk_size):
                selected = rows[offset:offset + chunk_size]
                distance = torch.zeros((len(selected), len(state.numeric)), dtype=torch.float32, device=device)
                exact_count = torch.zeros_like(distance)
                if active_numeric:
                    global_indices = [numeric_indices[j] for j in active_numeric]
                    query = torch.as_tensor(block_numeric[selected][:, active_numeric], dtype=torch.float32, device=device)
                    source = source_numeric[:, global_indices]
                    query_scaled = query / scale[global_indices]
                    source_scaled = source / scale[global_indices]
                    distance = (torch.square(query_scaled).sum(dim=1, keepdim=True)
                                + torch.square(source_scaled).sum(dim=1)[None, :]
                                - np.float32(2.0) * query_scaled @ source_scaled.T)
                    distance = torch.clamp(distance, min=0.0)
                    for local, global_index in enumerate(global_indices):
                        exact_count += torch.isclose(query[:, local:local + 1], source_numeric[:, global_index][None, :],
                                                     rtol=0.0, atol=5e-6).to(torch.float32)
                for local in active_categorical:
                    global_index = categorical_indices[local]
                    query_code = torch.as_tensor(block_categorical[selected, local], dtype=torch.int16, device=device)
                    matches = query_code[:, None] == source_categorical[:, global_index][None, :]
                    distance += (~matches).to(torch.float32) * CATEGORICAL_MISMATCH
                    exact_count += matches.to(torch.float32)
                distance /= np.float32(observed_count)
                exact_fraction = exact_count / np.float32(observed_count)
                score = distance - EXACT_BONUS * exact_fraction + tie_break[None, :]
                values, top = torch.topk(score, k=top_k, dim=1, largest=False, sorted=True)
                selected_exact = torch.gather(exact_fraction, 1, top)
                indices_out[selected, b] = top.cpu().numpy().astype(np.uint16)
                score_out[selected, b] = values.cpu().numpy().astype(np.float16)
                exact_out[selected, b] = selected_exact.cpu().numpy().astype(np.float16)
        print(f'  block {block}: {len(np.unique(pattern))} missingness patterns', flush=True)
    return {'indices': indices_out, 'score': score_out, 'exact_fraction': exact_out, 'coverage': coverage_out,
            'pattern': pattern_out}


@dataclass(frozen=True)
class DonorView:
    indices: np.ndarray   # [rows, 6, 16] original row ids
    score: np.ndarray     # [rows, 6, 16] retrieval score (ascending)
    exact: np.ndarray     # [rows, 6, 16] exact-match fraction
    coverage: np.ndarray  # [rows, 6] observed fields per block
    pattern: np.ndarray   # [rows, 6] block missingness bits

    @property
    def rows(self):
        return int(self.indices.shape[0])


def donor_dir(split):
    return cache_dir('structural_nets') / 'donors' / split


def load_donors(split):
    """The donor arrays written by build_donors.py for 'train' or 'test' (memory-mapped)."""
    d = donor_dir(split)
    a = {name: np.load(d / f'{name}.npy', mmap_mode='r') for name in DONOR_FILES}
    if a['indices'].shape[1:] != (len(BLOCKS), TOP_K):
        raise RuntimeError(f'unexpected donor array shape {a["indices"].shape}')
    return DonorView(indices=a['indices'], score=a['score'], exact=a['exact_fraction'], coverage=a['coverage'],
                     pattern=a['pattern'])
