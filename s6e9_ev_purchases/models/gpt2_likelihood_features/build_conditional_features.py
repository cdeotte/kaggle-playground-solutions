#!/usr/bin/env python
"""306 label-free features from the score_conditional.py (12) and score_topk.py (6) maps.

Per (replica, order): forward LLR (sum and mean), inverse logit, joint log-likelihood (sum and mean) and token count; their mean,
std and range over orders per replica, over replicas per order, and over everything; the log-sum-exp over the 3 orders of the
per-class sums (an order-marginal LLR per replica). Top-50 maps: capped LLR (sum, mean), support advantage (tokens outside the top
50 under No minus under Yes), margin advantages (mean, min) and the joint out-of-support count, with the same summaries.
Writes features/conditional_features.npy (float32 [train + test rows, 306]), conditional_feature_names.json and
conditional_feature_families.json (forward, inverse, combined, full: column indices for fit_conditional_lr.py).
Usage: python build_conditional_features.py"""
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
FEATURES = HERE / 'features'
REPLICAS = ['e55', 'e60', 'e65', 'e70']
ORDERS = ['source', 'competition', 'perm42']
TOPK_REPLICAS = ['e60', 'e65']


def main():
    raw = {(r, o): np.load(FEATURES / f'raw_{r}_{o}.npy').astype(np.float64) for r in REPLICAS for o in ORDERS}
    topk_raw = {(r, o): np.load(FEATURES / f'topk_{r}_{o}.npy').astype(np.float64) for r in TOPK_REPLICAS for o in ORDERS}
    n = len(raw[REPLICAS[0], ORDERS[0]])
    feature = {}

    def add(name, value):
        value = np.asarray(value, dtype=np.float64)
        assert value.shape == (n,) and np.isfinite(value).all() and name not in feature, name
        feature[name] = value

    measures = {}
    for r in REPLICAS:
        for o in ORDERS:
            no_sum, yes_sum, no_mean, yes_mean, inverse = raw[r, o].T
            item = {
                'forward_sum': yes_sum - no_sum,
                'forward_mean': yes_mean - no_mean,
                'inverse': inverse,
                'joint_sum': 0.5 * (yes_sum + no_sum),
                'joint_mean': 0.5 * (yes_mean + no_mean),
                'token_count': 0.5 * (no_sum / np.minimum(no_mean, -1e-12) + yes_sum / np.minimum(yes_mean, -1e-12)),
            }
            measures[r, o] = item
            for m, v in item.items():
                add(f'raw__{r}__{o}__{m}', v)

    summary = ['forward_sum', 'forward_mean', 'inverse', 'joint_mean']
    for r in REPLICAS:
        for m in summary:
            x = np.column_stack([measures[r, o][m] for o in ORDERS])
            add(f'ckpt__{r}__{m}__mean', x.mean(1))
            add(f'ckpt__{r}__{m}__std', x.std(1))
            add(f'ckpt__{r}__{m}__range', np.ptp(x, axis=1))
        yes = np.column_stack([raw[r, o][:, 1] for o in ORDERS])
        no = np.column_stack([raw[r, o][:, 0] for o in ORDERS])
        add(f'ckpt__{r}__order_logmarginal_llr', logsumexp(yes, axis=1) - logsumexp(no, axis=1))
    for o in ORDERS:
        for m in summary:
            x = np.column_stack([measures[r, o][m] for r in REPLICAS])
            add(f'order__{o}__{m}__mean', x.mean(1))
            add(f'order__{o}__{m}__std', x.std(1))
            add(f'order__{o}__{m}__range', np.ptp(x, axis=1))
    for m in summary:
        x = np.column_stack([measures[r, o][m] for r in REPLICAS for o in ORDERS])
        add(f'global__{m}__mean', x.mean(1))
        add(f'global__{m}__std', x.std(1))
        add(f'global__{m}__min', x.min(1))
        add(f'global__{m}__max', x.max(1))
        add(f'global__{m}__range', np.ptp(x, axis=1))

    topk = {}
    for r in TOPK_REPLICAS:
        for o in ORDERS:
            v = topk_raw[r, o]
            item = {
                'capped_sum_llr': v[:, 1] - v[:, 0],
                'capped_mean_llr': v[:, 3] - v[:, 2],
                'support_advantage': v[:, 4] - v[:, 5],
                'margin_mean_advantage': v[:, 7] - v[:, 6],
                'margin_min_advantage': v[:, 9] - v[:, 8],
                'joint_out_count': 0.5 * (v[:, 4] + v[:, 5]),
            }
            topk[r, o] = item
            for m, x in item.items():
                add(f'topk_raw__{r}__{o}__{m}', x)
    topk_summary = list(next(iter(topk.values())))
    for r in TOPK_REPLICAS:
        for m in topk_summary:
            x = np.column_stack([topk[r, o][m] for o in ORDERS])
            add(f'topk_ckpt__{r}__{m}__mean', x.mean(1))
            add(f'topk_ckpt__{r}__{m}__std', x.std(1))
            add(f'topk_ckpt__{r}__{m}__range', np.ptp(x, axis=1))
    for o in ORDERS:
        for m in topk_summary:
            x = np.column_stack([topk[r, o][m] for r in TOPK_REPLICAS])
            add(f'topk_order__{o}__{m}__mean', x.mean(1))
            add(f'topk_order__{o}__{m}__difference', x[:, 1] - x[:, 0])
    for m in topk_summary:
        x = np.column_stack([topk[r, o][m] for r in TOPK_REPLICAS for o in ORDERS])
        add(f'topk_global__{m}__mean', x.mean(1))
        add(f'topk_global__{m}__std', x.std(1))
        add(f'topk_global__{m}__range', np.ptp(x, axis=1))

    names = list(feature)
    matrix = np.column_stack([feature[k] for k in names]).astype(np.float32)
    assert np.isfinite(matrix).all()
    forward = [i for i, k in enumerate(names) if 'forward_' in k or 'logmarginal_llr' in k or 'topk' in k]
    inverse = [i for i, k in enumerate(names) if 'inverse' in k]
    families = {'forward': forward, 'inverse': inverse, 'combined': sorted(set(forward + inverse)), 'full': list(range(len(names)))}
    np.save(FEATURES / 'conditional_features.npy', matrix)
    (FEATURES / 'conditional_feature_names.json').write_text(json.dumps(names, indent=1) + '\n')
    (FEATURES / 'conditional_feature_families.json').write_text(json.dumps(families) + '\n')
    print(f'conditional features: {matrix.shape}', flush=True)


if __name__ == '__main__':
    main()
