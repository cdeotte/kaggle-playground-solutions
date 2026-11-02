#!/usr/bin/env python
"""GPU XGBoost on static, response and margin features, boosted from the nested logistic margin (base_margin).

--pair xgb_nested_margin (227 columns, max_depth 4)
  cache/static: raw (24), smooth (106), source_pairs (28)
  source_full_global restricted to source_focus_names (40, cache/base), response_alpha1 and response_alpha20 (13 + 13,
  cache/static/fold_k), response_income_multiscale (2) and response_commute_residual (1, cache/base)
  base_margin: nested_response_margin (cache/static/fold_k), read as float32
--pair xgb_response_geometry (472 columns, max_depth 5): the same, plus
  multires:exact_avg8, core_avg8, wide_avg8 (118, cache/multires) and 127 income response differences (adjacent widths,
  adjacent log-ratios, exact minus each coarser grouping, alpha 1 minus alpha 20; see difference_pairs);
  Gaussian noise (sd 0.04, seed 560001 + 10003 k) added to the training rows of every response_* / multires:* block;
  column-sampling weights: static and source 1.0, response_* 0.6, multires:* 0.2, differences 0.05
--pair xgb_acceptance_entropy (526 columns): xgb_response_geometry plus
  the 52 legal-token entropy columns (family constrained_entropy) of cache/acceptance, weight 0.5, after the static blocks;
  response_bpe_prefix (2, weight 1.0, last): target encoding (alpha 1 and 20) of the leading GPT-2 BPE piece of the income
  (cache/bpe), training rows averaged over 8 inner 5-fold cross-fits (StratifiedKFold seeds 777-784)
All: 900 rounds, eta 0.03, min_child_weight 10, subsample 0.9, colsample_bytree / bynode 0.7, lambda 5, alpha 0.15, max_bin
512, num_parallel_tree 6, seed 42001 + fold; the held-out fold is only predicted. Test = mean of the 5 fold models.
Usage: python train_xgb.py --pair xgb_nested_margin|xgb_response_geometry|xgb_acceptance_entropy [--folds 0,1,2,3,4]"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, read_test, read_train, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from features import CACHE, read_manifest  # noqa: E402

MODELS = {
    'xgb_nested_margin': dict(max_depth=4, geometry=False, bpe=False, acceptance=None),
    'xgb_response_geometry': dict(max_depth=5, geometry=True, bpe=False, acceptance=None),
    'xgb_acceptance_entropy': dict(max_depth=5, geometry=True, bpe=True, acceptance=('constrained_entropy', 0.5)),
}
STATIC_BLOCKS = ('raw', 'smooth', 'source_pairs')
BASE_BLOCKS = (  # (cache folder, block, column subset)
    ('base', 'source_full_global', 'source_focus_names'),
    ('static', 'response_alpha1', None),
    ('static', 'response_alpha20', None),
    ('base', 'response_income_multiscale', None),
    ('base', 'response_commute_residual', None),
)
MULTIRES_FAMILIES = ('exact_avg8', 'core_avg8', 'wide_avg8')
NOISE_SD, NOISE_SEED = 0.04, 560001
ROUNDS = 900
THREADS = 8
PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'device': 'cuda', 'eta': 0.03,
          'min_child_weight': 10, 'subsample': 0.9, 'colsample_bytree': 0.7, 'colsample_bynode': 0.7, 'lambda': 5,
          'alpha': 0.15, 'max_bin': 512, 'num_parallel_tree': 6, 'scale_pos_weight': 1.0, 'nthread': THREADS}


def difference_pairs(family_names):
    """Income response differences between multires columns: {'name', 'left', 'right'} with left / right keys
    'multires:<family>::<column>'. family_names: {family: column names} of exact_avg8, core_avg8, wide_avg8."""
    entries = []
    for family in MULTIRES_FAMILIES:
        for name in family_names[family]:
            if '__Annual_Income_USD__' not in name:
                continue
            entries.append(dict(family=family, name=name, key=f'multires:{family}::{name}',
                                alpha=re.search(r'__(a(?:1|20))$', name).group(1), stem=re.sub(r'__a(?:1|20)$', '', name)))
    # adjacent widths (exact = width 0) and adjacent log-ratio groupings
    grouped, ratio_grouped = {}, {}
    for e in entries:
        window, ratio = re.search(r'__w([0-9.]+)__a', e['name']), re.search(r'__r([0-9.]+)__a', e['name'])
        if e['family'] == 'exact_avg8':
            grouped.setdefault(e['alpha'], []).append((0.0, e['key']))
        elif window:
            grouped.setdefault(e['alpha'], []).append((float(window.group(1)), e['key']))
        elif ratio:
            ratio_grouped.setdefault(e['alpha'], []).append((float(ratio.group(1)), e['key']))
    adjacent = []
    for prefix, groups in (('detail', grouped), ('ratio_detail', ratio_grouped)):
        for alpha, items in sorted(groups.items()):
            ordered = sorted(items)
            if prefix == 'detail':  # one column per width
                unique = {}
                for scale, key in ordered:
                    unique.setdefault(scale, key)
                ordered = sorted(unique.items())
            for (fine, fine_key), (coarse, coarse_key) in zip(ordered, ordered[1:]):
                adjacent.append(dict(name=f'{prefix}__Annual_Income_USD__{alpha}__{fine:g}_minus_{coarse:g}',
                                     left=fine_key, right=coarse_key))
    # exact minus every coarser grouping
    exact = {e['alpha']: e['key'] for e in entries if e['family'] == 'exact_avg8'}
    anchor = [dict(name=f'anchor__{e["stem"]}__{e["alpha"]}', left=exact[e['alpha']], right=e['key'])
              for e in entries if e['family'] != 'exact_avg8']
    # alpha 1 minus alpha 20 of the same grouping
    by_stem = {(e['family'], e['stem'], e['alpha']): e['key'] for e in entries}
    alpha_gap = [dict(name=f'smoothgap__{family}__{stem}', left=key, right=by_stem[(family, stem, 'a20')])
                 for (family, stem, alpha), key in sorted(by_stem.items()) if alpha == 'a1']
    pairs, seen = [], set()
    for pair in adjacent + anchor + alpha_gap:
        if (pair['left'], pair['right']) not in seen:
            seen.add((pair['left'], pair['right']))
            pairs.append(pair)
    return pairs


def response_logit(probability):
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(probability / (1.0 - probability))


def fit_apply_bpe_delta(fit_codes, fit_target, apply_codes, alpha):
    size = int(max(fit_codes.max(initial=0), apply_codes.max(initial=0))) + 1
    counts = np.bincount(fit_codes, minlength=size).astype(np.float64)
    positives = np.bincount(fit_codes, weights=fit_target, minlength=size).astype(np.float64)
    prior = float(np.mean(fit_target))
    probability = (positives[apply_codes] + alpha * prior) / (counts[apply_codes] + alpha)
    return (response_logit(probability) - response_logit(prior)).astype(np.float32)


def bpe_prefix_block(y, tr, va, alphas=(1.0, 20.0), seeds=tuple(range(777, 785))):
    from sklearn.model_selection import StratifiedKFold
    mapping = pd.read_csv(CACHE / 'bpe' / 'income_to_first_piece.csv')
    piece = mapping.set_index('income')['first_piece']
    vocabulary = {p: i for i, p in enumerate(sorted(mapping['first_piece'].astype(str).unique()))}
    codes = [df['Annual_Income_USD'].map(piece).astype(str).map(vocabulary).to_numpy(np.int32)
             for df in (read_train(), read_test())]
    fit_codes, fit_target, valid_codes = codes[0][tr], np.asarray(y[tr], dtype=np.int8), codes[0][va]
    columns = {'train': [], 'valid': [], 'test': []}
    for alpha in alphas:
        train_sum = np.zeros(len(tr), dtype=np.float64)
        for seed in seeds:
            one = np.empty(len(tr), dtype=np.float32)
            for inner_fit, inner_apply in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(tr)), fit_target):
                one[inner_apply] = fit_apply_bpe_delta(fit_codes[inner_fit], fit_target[inner_fit], fit_codes[inner_apply], alpha)
            train_sum += one
        columns['train'].append((train_sum / np.float64(len(seeds))).astype(np.float32))
        columns['valid'].append(fit_apply_bpe_delta(fit_codes, fit_target, valid_codes, alpha))
        columns['test'].append(fit_apply_bpe_delta(fit_codes, fit_target, codes[1], alpha))
    names = [f'tex_IncBpePre_s{alpha:g}' for alpha in alphas]
    return tuple(np.column_stack(columns[role]).astype(np.float32) for role in ('train', 'valid', 'test')), names


def response_weight(block):
    if block == 'response_bpe_prefix':
        return 1.0
    if block == 'response_multiscale_deltas':
        return 0.05
    if block.startswith('multires:'):
        return 0.2
    if block.startswith('response_'):
        return 0.6
    return 1.0


def build_inputs(pair, k, y, f):
    """Feature matrices (train, valid, test), column names, column-sampling weights and base margins of fold k."""
    spec = MODELS[pair]
    tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
    blocks = []  # (block, train, valid, test, names, column-sampling weight)

    static = CACHE / 'static'
    static_names = read_manifest(static)['block_names']
    for block in STATIC_BLOCKS:
        whole = np.load(static / f'{block}_train.npy', mmap_mode='r')
        blocks.append((block, whole[tr], whole[va], np.load(static / f'{block}_test.npy'), static_names[block], 1.0))
    if spec['acceptance']:
        family, weight = spec['acceptance']
        manifest = read_manifest(CACHE / 'acceptance')
        idx = sorted(set(manifest['families'][family]))
        whole = np.load(CACHE / 'acceptance' / 'train.npy', mmap_mode='r')
        test_whole = np.load(CACHE / 'acceptance' / 'test.npy', mmap_mode='r')
        blocks.append(('acceptance', whole[tr][:, idx], whole[va][:, idx], test_whole[:, idx],
                       [manifest['names'][j] for j in idx], weight))

    response = []  # (block, train, valid, test, names)
    for folder_name, block, subset in BASE_BLOCKS:
        folder = CACHE / folder_name / f'fold_{k}'
        manifest = read_manifest(folder)
        names = list(manifest[f'{block}_names'])
        values = [np.load(folder / f'{block}_{role}.npy') for role in ('train', 'valid', 'test')]
        if subset:
            lookup = {name: j for j, name in enumerate(names)}
            names = list(manifest[subset])
            values = [v[:, [lookup[name] for name in names]] for v in values]
        response.append((block, *values, names))
    if spec['geometry']:
        folder = CACHE / 'multires' / f'fold_{k}'
        families = read_manifest(folder)['families']
        for family in MULTIRES_FAMILIES:
            response.append((f'multires:{family}', *[np.load(folder / f'{family}_{role}.npy') for role in ('train', 'valid', 'test')],
                             list(families[family]['names'])))
        lookup = {}
        for block, a_tr, a_va, a_te, names in response:
            for j, name in enumerate(names):
                lookup[f'{block}::{name}'] = tuple(np.asarray(a[:, j], dtype=np.float32) for a in (a_tr, a_va, a_te))
        pairs = difference_pairs({family: families[family]['names'] for family in MULTIRES_FAMILIES})
        deltas = [np.column_stack([np.asarray(lookup[p['left']][r] - lookup[p['right']][r], dtype=np.float32) for p in pairs])
                  for r in range(3)]
        response.append(('response_multiscale_deltas', *deltas, [p['name'] for p in pairs]))
    if spec['bpe']:
        values, names = bpe_prefix_block(y, tr, va)
        response.append(('response_bpe_prefix', *values, names))

    rng = np.random.default_rng(NOISE_SEED + 10003 * k)
    for block, a_tr, a_va, a_te, names in response:
        a_tr = np.asarray(a_tr, dtype=np.float32)
        if spec['geometry'] and block.startswith(('response_', 'multires:')):
            a_tr = np.asarray(a_tr + rng.normal(0.0, NOISE_SD, size=a_tr.shape).astype(np.float32), dtype=np.float32)
        blocks.append((block, a_tr, a_va, a_te, names, response_weight(block)))

    names = [f'{b[0]}::{name}' for b in blocks for name in b[4]]
    x = [np.ascontiguousarray(np.column_stack([np.asarray(b[r], dtype=np.float32) for b in blocks]), dtype=np.float32)
         for r in (1, 2, 3)]
    weights = np.asarray([b[5] for b in blocks for _ in b[4]], dtype=np.float32) if spec['geometry'] else None
    folder = CACHE / 'static' / f'fold_{k}'
    margins = [np.load(folder / f'nested_response_margin_{role}.npy')[:, 0].astype(np.float64) for role in ('train', 'valid', 'test')]
    if not spec['geometry']:
        margins = [m.astype(np.float32).astype(np.float64) for m in margins]
    return x, names, weights, margins


def run_fold(pair, k, y, f):
    import xgboost as xgb
    x, names, weights, margins = build_inputs(pair, k, y, f)
    tr = np.flatnonzero(f != k)
    print(f'fold {k}: {len(names)} columns', flush=True)
    params = dict(PARAMS, max_depth=MODELS[pair]['max_depth'], seed=42001 + k)
    dtrain = xgb.DMatrix(x[0], label=y[tr], base_margin=margins[0], feature_names=names, feature_weights=weights, nthread=THREADS)
    model = xgb.train(params, dtrain, num_boost_round=ROUNDS, verbose_eval=False)
    preds = [model.predict(xgb.DMatrix(x[r], base_margin=margins[r], feature_names=names, feature_weights=weights,
                                       nthread=THREADS)) for r in (1, 2)]
    (HERE / 'checkpoints').mkdir(exist_ok=True)
    model.save_model(HERE / 'checkpoints' / f'{pair}_f{k}.ubj')
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True, choices=list(MODELS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    y = target()
    f = folds(y)
    for k in fold_list(a.folds):
        if have_part(HERE, a.pair, k):
            print(f'fold {k}: done', flush=True)
            continue
        valid_pred, test_pred = run_fold(a.pair, k, y, f)
        print(f'fold {k}: AUC {roc_auc_score(y[f == k], valid_pred):.6f}', flush=True)
        save_part(HERE, a.pair, k, valid_pred, test_pred)
    assemble(HERE, a.pair, f, y)


if __name__ == '__main__':
    main()
