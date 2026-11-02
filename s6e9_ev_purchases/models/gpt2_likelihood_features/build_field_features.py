#!/usr/bin/env python
"""1,330 label-free features from the 26 score_fields.py maps (replicas e55, e60 x 13 held-out fields).

Per map, 19 measures: 10 Yes-vs-No contrasts (smooth sum / mean / min / first / last LLR, top-50 sum / mean LLR, support
advantage = tokens outside the top 50 under No minus under Yes, margin mean / min advantage) and 9 label-averaged densities (the
same quantities averaged over the two labels, plus the value token count). Features: every (replica, field, measure); per field
the e55/e60 mean and the e60 - e55 difference; per view (e55, e60, their mean) and measure the sum, mean, std, min, max and range
over the 13 fields.
Writes features/field_features.npy (float32 [train + test rows, 1330]), field_feature_names.json and field_feature_families.json
(contrast, compact_contrast, full: column indices for fit_field_xgb.py).
Usage: python build_field_features.py"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.gpt_rows import SOURCE_ORDER  # noqa: E402
from score_fields import COLUMNS as RAW_COLUMNS  # noqa: E402

FEATURES = HERE / 'features'
REPLICAS = ['e55', 'e60']
CONTRAST_MEASURES = ['smooth_sum_llr', 'smooth_mean_llr', 'smooth_min_llr', 'smooth_first_llr', 'smooth_last_llr', 'topk_sum_llr',
                     'topk_mean_llr', 'support_advantage', 'margin_mean_advantage', 'margin_min_advantage']
DENSITY_MEASURES = ['joint_smooth_sum', 'joint_smooth_mean', 'joint_smooth_min', 'joint_topk_sum', 'joint_topk_mean',
                    'joint_out_count', 'joint_margin_mean', 'joint_margin_min', 'value_token_count']
ALL_MEASURES = CONTRAST_MEASURES + DENSITY_MEASURES


def derive(raw):
    c = {name: raw[:, i].astype(np.float64) for i, name in enumerate(RAW_COLUMNS)}
    out = {
        'smooth_sum_llr': c['yes_smooth_sum'] - c['no_smooth_sum'],
        'smooth_mean_llr': c['yes_smooth_mean'] - c['no_smooth_mean'],
        'smooth_min_llr': c['yes_smooth_min'] - c['no_smooth_min'],
        'smooth_first_llr': c['yes_smooth_first'] - c['no_smooth_first'],
        'smooth_last_llr': c['yes_smooth_last'] - c['no_smooth_last'],
        'topk_sum_llr': c['yes_topk_sum'] - c['no_topk_sum'],
        'topk_mean_llr': c['yes_topk_mean'] - c['no_topk_mean'],
        'support_advantage': c['no_out_count'] - c['yes_out_count'],
        'margin_mean_advantage': c['yes_margin_mean'] - c['no_margin_mean'],
        'margin_min_advantage': c['yes_margin_min'] - c['no_margin_min'],
        'joint_smooth_sum': 0.5 * (c['yes_smooth_sum'] + c['no_smooth_sum']),
        'joint_smooth_mean': 0.5 * (c['yes_smooth_mean'] + c['no_smooth_mean']),
        'joint_smooth_min': 0.5 * (c['yes_smooth_min'] + c['no_smooth_min']),
        'joint_topk_sum': 0.5 * (c['yes_topk_sum'] + c['no_topk_sum']),
        'joint_topk_mean': 0.5 * (c['yes_topk_mean'] + c['no_topk_mean']),
        'joint_out_count': 0.5 * (c['yes_out_count'] + c['no_out_count']),
        'joint_margin_mean': 0.5 * (c['yes_margin_mean'] + c['no_margin_mean']),
        'joint_margin_min': 0.5 * (c['yes_margin_min'] + c['no_margin_min']),
        'value_token_count': c['value_token_count'],
    }
    assert list(out) == ALL_MEASURES
    return out


def main():
    derived = {(r, j): derive(np.load(FEATURES / f'field_{r}_c{j:02d}.npy', mmap_mode='r')) for r in REPLICAS for j in range(13)}
    n = len(derived[REPLICAS[0], 0][ALL_MEASURES[0]])
    features = {}

    def add(name, value):
        value = np.asarray(value, np.float64)
        assert value.shape == (n,) and np.isfinite(value).all() and name not in features, name
        features[name] = value

    for r in REPLICAS:
        for j, col in enumerate(SOURCE_ORDER):
            for m in ALL_MEASURES:
                add(f'field__{r}__c{j:02d}_{col}__{m}', derived[r, j][m])
    for j, col in enumerate(SOURCE_ORDER):
        for m in ALL_MEASURES:
            first, second = derived['e55', j][m], derived['e60', j][m]
            add(f'fieldavg__c{j:02d}_{col}__{m}__mean', 0.5 * (first + second))
            add(f'fieldavg__c{j:02d}_{col}__{m}__e60_minus_e55', second - first)
    for view in [*REPLICAS, 'mean']:
        for m in ALL_MEASURES:
            if view == 'mean':
                x = np.column_stack([0.5 * (derived['e55', j][m] + derived['e60', j][m]) for j in range(13)])
            else:
                x = np.column_stack([derived[view, j][m] for j in range(13)])
            prefix = f'global__{view}__{m}'
            add(prefix + '__sum', x.sum(1))
            add(prefix + '__mean', x.mean(1))
            add(prefix + '__std', x.std(1))
            add(prefix + '__min', x.min(1))
            add(prefix + '__max', x.max(1))
            add(prefix + '__range', np.ptp(x, axis=1))

    names = list(features)
    matrix = np.column_stack([features[k] for k in names]).astype(np.float32)
    assert np.isfinite(matrix).all()
    contrast = [i for i, k in enumerate(names) if any(m in k for m in CONTRAST_MEASURES)]
    compact = [i for i, k in enumerate(names) if k.startswith('fieldavg__') or k.startswith('global__')]
    families = {'contrast': contrast, 'compact_contrast': [i for i in compact if i in set(contrast)], 'full': list(range(len(names)))}
    np.save(FEATURES / 'field_features.npy', matrix)
    (FEATURES / 'field_feature_names.json').write_text(json.dumps(names, indent=1) + '\n')
    (FEATURES / 'field_feature_families.json').write_text(json.dumps(families) + '\n')
    print(f'field features: {matrix.shape}', flush=True)


if __name__ == '__main__':
    main()
