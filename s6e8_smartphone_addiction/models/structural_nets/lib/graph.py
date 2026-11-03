"""Source-genealogy graph features.

The 7,500 original rows are latent hub nodes; every competition row links to them through the six donor relations
(lib/donors.py). Kernels over a relation's donors:
  k1    the nearest donor only
  k4    the 4 nearest, weights exp(-1.25 (score - best score)) * (0.35 + 0.65 exact_fraction), normalized
  k16   all 16, weights exp(-0.60 (score - best score)) * (0.25 + 0.75 exact_fraction), normalized
Structural features (no competition labels): per relation the coverage, missingness pattern, top score, top exact
fraction and top donor id / 7500, and per kernel the weight entropy and the weighted original label / severity.
Label messages: competition labels of the seed rows (outer-train rows) flow through shared donors and shared donor
communities. For seed rows themselves (self_exclude=True) the row's own weight and label are removed from every
aggregate and from the prior before its feature is computed."""
import numpy as np

from .donors import BLOCK_NAMES, SOURCE_ROWS

KERNELS = ('k1', 'k4', 'k16')
DONOR_SMOOTHING = (4.0, 24.0, 128.0)
PATTERN_SMOOTHING = (12.0, 64.0)
PAIR_SMOOTHING = (16.0, 96.0)
TRIPLE_SMOOTHING = (32.0, 160.0)
BLOCK_PAIRS = tuple((left, right) for left in range(6) for right in range(left + 1, 6))
BLOCK_TRIPLES = ((0, 1, 2), (0, 3, 4), (2, 3, 5))


def kernel_weights(view, rows, block, kernel):
    if kernel == 'k1':
        width, temperature, exact_floor, exact_scale = 1, 0.0, 1.0, 0.0
    elif kernel == 'k4':
        width, temperature, exact_floor, exact_scale = 4, 1.25, 0.35, 0.65
    elif kernel == 'k16':
        width, temperature, exact_floor, exact_scale = 16, 0.60, 0.25, 0.75
    else:
        raise ValueError(kernel)
    indices = np.asarray(view.indices[rows, block, :width], dtype=np.int64)
    score = np.asarray(view.score[rows, block, :width], dtype=np.float32)
    exact = np.asarray(view.exact[rows, block, :width], dtype=np.float32)
    if width == 1:
        weight = np.ones_like(score, dtype=np.float32)
    else:
        relative = np.clip(score - score[:, :1], 0.0, 20.0)
        weight = np.exp(-np.float32(temperature) * relative).astype(np.float32)
        weight *= np.float32(exact_floor) + np.float32(exact_scale) * exact
    active = np.asarray(view.coverage[rows, block] > 0, dtype=np.float32)
    weight *= active[:, None]
    denominator = weight.sum(axis=1, keepdims=True)
    weight = np.divide(weight, denominator, out=np.zeros_like(weight), where=denominator > 0)
    return indices, weight


def build_structural_features(view, rows, source_target, source_severity):
    """84 label-free columns: 6 relations x (5 + 3 kernels x 3)."""
    columns, names = [], []
    for block, block_name in enumerate(BLOCK_NAMES):
        coverage = np.asarray(view.coverage[rows, block], dtype=np.float32)
        pattern = np.asarray(view.pattern[rows, block], dtype=np.float32)
        score = np.asarray(view.score[rows, block], dtype=np.float32)
        exact = np.asarray(view.exact[rows, block], dtype=np.float32)
        active = coverage > 0
        for name, value in (
                ('coverage', coverage),
                ('pattern', pattern),
                ('top_score', np.where(active, score[:, 0], 0.0)),
                ('top_exact', np.where(active, exact[:, 0], 0.0)),
                ('top_source_id', np.asarray(view.indices[rows, block, 0], dtype=np.float32) / SOURCE_ROWS)):
            columns.append(np.asarray(value, dtype=np.float32))
            names.append(f'struct__{block_name}__{name}')
        for kernel in KERNELS:
            indices, weight = kernel_weights(view, rows, block, kernel)
            entropy = -(weight * np.log(np.clip(weight, 1e-12, 1.0))).sum(axis=1)
            original_vote = (weight * source_target[indices]).sum(axis=1)
            severity_vote = (weight * source_severity[indices]).sum(axis=1)
            for name, value in (('entropy', entropy), ('original_target', original_vote),
                                ('original_severity', severity_vote)):
                columns.append(np.asarray(value, dtype=np.float32))
                names.append(f'struct__{block_name}__{kernel}__{name}')
    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise RuntimeError('non-finite structural graph features')
    return matrix, names


def _donor_messages(seed_view, seed_rows, seed_target, query_view, query_rows, self_exclude):
    """Per relation and kernel: smoothed label posterior of each donor (weighted counts over the seed rows), averaged
    with the query row's kernel weights, at 3 smoothings, plus the weighted log support."""
    prior = np.float64(seed_target.mean())
    if self_exclude:
        query_prior = (np.float64(seed_target.sum()) - seed_target) / np.float64(len(seed_target) - 1)
    else:
        query_prior = np.full(len(query_rows), prior, dtype=np.float64)
    columns, names = [], []
    for block, block_name in enumerate(BLOCK_NAMES):
        for kernel in KERNELS:
            seed_indices, seed_weight = kernel_weights(seed_view, seed_rows, block, kernel)
            query_indices, query_weight = kernel_weights(query_view, query_rows, block, kernel)
            denominator = np.bincount(seed_indices.ravel(), weights=seed_weight.ravel(),
                                      minlength=SOURCE_ROWS).astype(np.float64)
            numerator = np.bincount(seed_indices.ravel(), weights=(seed_weight * seed_target[:, None]).ravel(),
                                    minlength=SOURCE_ROWS).astype(np.float64)
            edge_denominator = denominator[query_indices]
            edge_numerator = numerator[query_indices]
            if self_exclude:
                if seed_view is not query_view or not np.array_equal(seed_rows, query_rows):
                    raise RuntimeError('self exclusion requires identical seed / query rows')
                edge_denominator = edge_denominator - query_weight
                edge_numerator = edge_numerator - query_weight * seed_target[:, None]
            for smoothing in DONOR_SMOOTHING:
                posterior = (edge_numerator + smoothing * query_prior[:, None]) / (edge_denominator + smoothing)
                message = (query_weight * posterior).sum(axis=1)
                inactive = np.asarray(query_view.coverage[query_rows, block] == 0)
                message[inactive] = query_prior[inactive]
                columns.append(message.astype(np.float32))
                names.append(f'message__{block_name}__{kernel}__s{smoothing:g}')
            support = (query_weight * np.log1p(np.maximum(edge_denominator, 0.0))).sum(axis=1)
            columns.append(support.astype(np.float32))
            names.append(f'message__{block_name}__{kernel}__log_support')
    return columns, names


def _group_lookup(seed_key, seed_active, seed_target, query_key, query_active, smoothing_values, self_exclude):
    """Smoothed label mean of the seed rows sharing the query row's key (+ log count)."""
    prior = np.float64(seed_target.mean())
    if self_exclude:
        query_prior = (np.float64(seed_target.sum()) - seed_target) / np.float64(len(seed_target) - 1)
    else:
        query_prior = np.full(len(query_key), prior, dtype=np.float64)
    active_key = np.asarray(seed_key[seed_active], dtype=np.uint64)
    unique, inverse = np.unique(active_key, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=seed_target[seed_active]).astype(np.float64)
    query_key_u = np.asarray(query_key, dtype=np.uint64)
    position = np.searchsorted(unique, query_key_u)
    found = query_active & (position < len(unique))
    safe_position = np.minimum(position, max(len(unique) - 1, 0))
    if len(unique):
        found &= unique[safe_position] == query_key_u
    else:
        found[:] = False
    query_count = np.zeros(len(query_key), dtype=np.float64)
    query_sum = np.zeros(len(query_key), dtype=np.float64)
    if len(unique):
        query_count[found] = counts[safe_position[found]]
        query_sum[found] = sums[safe_position[found]]
    if self_exclude:
        if len(seed_key) != len(query_key) or not np.array_equal(seed_key, query_key):
            raise RuntimeError('group self exclusion requires identical keys')
        query_count[query_active] -= 1.0
        query_sum[query_active] -= seed_target[query_active]
    outputs = [((query_sum + smoothing * query_prior) / (query_count + smoothing)).astype(np.float32)
               for smoothing in smoothing_values]
    return outputs, np.log1p(np.maximum(query_count, 0.0)).astype(np.float32)


def _community_messages(seed_view, seed_rows, seed_target, query_view, query_rows, self_exclude):
    """Communities: (top donor, block pattern) per relation, top-donor pairs over all 15 relation pairs and top-donor
    triples over 3 relation triples."""
    columns, names = [], []
    seed_top = np.asarray(seed_view.indices[seed_rows, :, 0], dtype=np.uint64)
    query_top = np.asarray(query_view.indices[query_rows, :, 0], dtype=np.uint64)
    seed_active = np.asarray(seed_view.coverage[seed_rows] > 0)
    query_active = np.asarray(query_view.coverage[query_rows] > 0)

    for block, block_name in enumerate(BLOCK_NAMES):
        seed_key = seed_top[:, block] * np.uint64(256) + np.asarray(seed_view.pattern[seed_rows, block], dtype=np.uint64)
        query_key = query_top[:, block] * np.uint64(256) + np.asarray(query_view.pattern[query_rows, block], dtype=np.uint64)
        messages, count = _group_lookup(seed_key, seed_active[:, block], seed_target, query_key, query_active[:, block],
                                        PATTERN_SMOOTHING, self_exclude)
        for smoothing, message in zip(PATTERN_SMOOTHING, messages):
            columns.append(message)
            names.append(f'community__{block_name}__donor_pattern__s{smoothing:g}')
        columns.append(count)
        names.append(f'community__{block_name}__donor_pattern__log_count')

    for left, right in BLOCK_PAIRS:
        seed_key = seed_top[:, left] * np.uint64(SOURCE_ROWS) + seed_top[:, right]
        query_key = query_top[:, left] * np.uint64(SOURCE_ROWS) + query_top[:, right]
        messages, count = _group_lookup(seed_key, seed_active[:, left] & seed_active[:, right], seed_target, query_key,
                                        query_active[:, left] & query_active[:, right], PAIR_SMOOTHING, self_exclude)
        label = f'{BLOCK_NAMES[left]}_{BLOCK_NAMES[right]}'
        for smoothing, message in zip(PAIR_SMOOTHING, messages):
            columns.append(message)
            names.append(f'community__pair__{label}__s{smoothing:g}')
        columns.append(count)
        names.append(f'community__pair__{label}__log_count')

    for left, middle, right in BLOCK_TRIPLES:
        seed_key = (seed_top[:, left] * np.uint64(SOURCE_ROWS) + seed_top[:, middle]) * np.uint64(SOURCE_ROWS) \
            + seed_top[:, right]
        query_key = (query_top[:, left] * np.uint64(SOURCE_ROWS) + query_top[:, middle]) * np.uint64(SOURCE_ROWS) \
            + query_top[:, right]
        messages, count = _group_lookup(
            seed_key, seed_active[:, left] & seed_active[:, middle] & seed_active[:, right], seed_target, query_key,
            query_active[:, left] & query_active[:, middle] & query_active[:, right], TRIPLE_SMOOTHING, self_exclude)
        label = '_'.join(BLOCK_NAMES[index] for index in (left, middle, right))
        for smoothing, message in zip(TRIPLE_SMOOTHING, messages):
            columns.append(message)
            names.append(f'community__triple__{label}__s{smoothing:g}')
        columns.append(count)
        names.append(f'community__triple__{label}__log_count')
    return columns, names


def build_label_message_features(seed_view, seed_rows, seed_target, query_view, query_rows, self_exclude):
    """144 label-message columns of the query rows from the labels of the seed rows: 72 donor messages + 72 community
    messages. self_exclude=True when the query rows are the seed rows (outer-train rows)."""
    seed_rows = np.asarray(seed_rows, dtype=np.int64)
    query_rows = np.asarray(query_rows, dtype=np.int64)
    seed_target = np.asarray(seed_target, dtype=np.float64)
    donor_columns, donor_names = _donor_messages(seed_view, seed_rows, seed_target, query_view, query_rows, self_exclude)
    community_columns, community_names = _community_messages(seed_view, seed_rows, seed_target, query_view, query_rows,
                                                             self_exclude)
    matrix = np.column_stack(donor_columns + community_columns).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError('non-finite label graph messages')
    return matrix, donor_names + community_names
