#!/usr/bin/env python
"""Combo design: the 642-column fold-local XGB design (one per outer fold) that train.py extends to 692 columns.

Blocks, in column order:
  raw (24), smooth (106), source_pairs (28)       static, label-free                      feature weight 1.0
  acceptance (168)                                 3 families of the GPT-2 acceptance pack  feature weight 0.5
  source_full_global (40, source-focus subset)     fold cache                               feature weight 1.0
  response_alpha1, response_alpha20 (13 + 13)      fold cache                               feature weight 0.6
  response_income_multiscale (2)                   fold cache                               feature weight 0.6
  response_commute_residual (1)                    fold cache                               feature weight 0.6
  multires:exact_avg8 / core_avg8 / wide_avg8      fold cache (10 + 32 + 76)                feature weight 0.2
  response_multiscale_deltas (127)                 differences of the income TEs above      feature weight 0.05
                                                   (pairs in combo_deltas.json)
  response_bpe_prefix (2)                          TE of the income's first GPT-2 BPE piece feature weight 1.0
Label-derived blocks are fold-local: outer-train rows are encoded by inner cross-fitting, valid / test rows by the state
fitted on all outer-train rows. The BPE-prefix block is built here: centered-logit TE (alpha 1 and 20); outer-train rows
get the mean of 8 inner StratifiedKFold(5) encodings (seeds 777-784). Base margin = the nested fold-local logistic
regression margin from the response cache.
The response columns get train-only Gaussian noise in train.py (fresh per XGB seed), so none is added here.
Inputs: ../response_features/cache/{static,base,multires,acceptance}.
Writes features/combo/fold_k/{x_train,x_valid,x_test,margin_train,margin_valid,margin_test}.npy + meta.json.
Usage: python build_combo.py [--folds 0,1,2,3,4]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.bpe import income_bpe_table  # noqa: E402
from common.data import folds, read_test, read_train, target  # noqa: E402
from common.pairs import fold_list  # noqa: E402

RF = HERE.parent / 'response_features' / 'cache'
STATIC = RF / 'static'          # {block}_train.npy (all train rows), {block}_test.npy, manifest.json (block_names)
MAIN_CACHE = RF / 'base'        # fold_k/{block}_{train,valid,test}.npy + manifest.json
ALPHA_CACHE = RF / 'static'     # fold_k/response_alpha{1,20}_{train,valid,test}.npy + manifest.json
MARGIN_CACHE = RF / 'static'    # fold_k/nested_response_margin_{train,valid,test}.npy
MULTIRES_CACHE = RF / 'multires'  # fold_k/{family}_{train,valid,test}.npy + manifest.json
ACCEPTANCE = RF / 'acceptance'  # train.npy, test.npy, manifest.json
OUT = HERE / 'features' / 'combo'

STATIC_BLOCKS = ('raw', 'smooth', 'source_pairs')
ACCEPTANCE_FAMILIES = ('constrained_entropy', 'acceptance_probability', 'numeric_center')
BASE_BLOCKS = (  # (cache, block, manifest key of the selected column subset)
    (MAIN_CACHE, 'source_full_global', 'source_focus_names'),
    (ALPHA_CACHE, 'response_alpha1', None),
    (ALPHA_CACHE, 'response_alpha20', None),
    (MAIN_CACHE, 'response_income_multiscale', None),
    (MAIN_CACHE, 'response_commute_residual', None),
)
MULTIRES_FAMILIES = ('exact_avg8', 'core_avg8', 'wide_avg8')
BPE_ALPHAS, BPE_SEEDS = (1.0, 20.0), tuple(range(777, 785))
ROLES = ('train', 'valid', 'test')


def block_weight(block):
    if block == 'response_bpe_prefix':
        return 1.0
    if block == 'response_multiscale_deltas':
        return 0.05
    if block.startswith('multires:'):
        return 0.2
    if block.startswith('response_'):
        return 0.6
    return 1.0


def cache_block(cache, k, block, select=None):
    d = cache / f'fold_{k}'
    man = json.loads((d / 'manifest.json').read_text())
    names = list(man[f'{block}_names'])
    arrays = [np.load(d / f'{block}_{r}.npy', mmap_mode='r') for r in ROLES]
    if select:
        pos = {n: i for i, n in enumerate(names)}
        idx = [pos[n] for n in man[select]]
        arrays, names = [a[:, idx] for a in arrays], list(man[select])
    return [np.asarray(a, np.float32) for a in arrays], names


def multires_family(k, family):
    d = MULTIRES_CACHE / f'fold_{k}'
    names = list(json.loads((d / 'manifest.json').read_text())['families'][family]['names'])
    return [np.asarray(np.load(d / f'{family}_{r}.npy', mmap_mode='r'), np.float32) for r in ROLES], names


def response_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def fit_apply_bpe_delta(fit_codes, fit_target, apply_codes, alpha):
    """Smoothed first-piece target mean fitted on fit rows, as logit minus the fit-prior logit."""
    size = int(max(fit_codes.max(initial=0), apply_codes.max(initial=0))) + 1
    counts = np.bincount(fit_codes, minlength=size).astype(np.float64)
    positives = np.bincount(fit_codes, weights=fit_target, minlength=size).astype(np.float64)
    prior = float(np.mean(fit_target))
    p = (positives[apply_codes] + alpha * prior) / (counts[apply_codes] + alpha)
    return (response_logit(p) - response_logit(prior)).astype(np.float32)


def bpe_codes():
    """Code of the first GPT-2 BPE piece of every train / test income (label-free)."""
    inc_tr, inc_te = read_train()['Annual_Income_USD'], read_test()['Annual_Income_USD']
    table = income_bpe_table(np.concatenate([inc_tr.to_numpy(), inc_te.to_numpy()]))
    vocab = {p: i for i, p in enumerate(sorted(table['p1'].astype(str).unique()))}
    code = table['p1'].astype(str).map(vocab)
    c_tr, c_te = inc_tr.map(code), inc_te.map(code)
    assert c_tr.notna().all() and c_te.notna().all()
    return c_tr.to_numpy(np.int32), c_te.to_numpy(np.int32)


def bpe_prefix_block(y, tr, va, codes_tr, codes_te):
    fit_codes, fit_target = codes_tr[tr], np.asarray(y[tr], dtype=np.int8)
    cols = {r: [] for r in ROLES}
    for alpha in BPE_ALPHAS:
        train_sum = np.zeros(len(tr), dtype=np.float64)
        for seed in BPE_SEEDS:
            one = np.empty(len(tr), dtype=np.float32)
            for fit, app in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(tr)), fit_target):
                one[app] = fit_apply_bpe_delta(fit_codes[fit], fit_target[fit], fit_codes[app], alpha)
            train_sum += one
        cols['train'].append((train_sum / np.float64(len(BPE_SEEDS))).astype(np.float32))
        cols['valid'].append(fit_apply_bpe_delta(fit_codes, fit_target, codes_tr[va], alpha))
        cols['test'].append(fit_apply_bpe_delta(fit_codes, fit_target, codes_te, alpha))
    return [np.column_stack(cols[r]).astype(np.float32) for r in ROLES], [f'tex_IncBpePre_s{a:g}' for a in BPE_ALPHAS]


def build_fold(k, f, y, codes):
    tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
    X, names, weights = {r: [] for r in ROLES}, [], []

    def add(block, arrays, cols, weight):
        for r, a in zip(ROLES, arrays):
            X[r].append(np.asarray(a, np.float32))
        names.extend(f'{block}::{c}' for c in cols)
        weights.extend([weight] * len(cols))

    static_names = json.loads((STATIC / 'manifest.json').read_text())['block_names']
    for b in STATIC_BLOCKS:
        whole = np.load(STATIC / f'{b}_train.npy', mmap_mode='r')
        add(b, (whole[tr], whole[va], np.load(STATIC / f'{b}_test.npy', mmap_mode='r')), static_names[b], 1.0)
    man = json.loads((ACCEPTANCE / 'manifest.json').read_text())
    idx = sorted(set(i for fam in ACCEPTANCE_FAMILIES for i in man['families'][fam]))
    whole = np.load(ACCEPTANCE / 'train.npy', mmap_mode='r')
    add('acceptance', (whole[tr][:, idx], whole[va][:, idx], np.load(ACCEPTANCE / 'test.npy', mmap_mode='r')[:, idx]),
        [man['names'][i] for i in idx], 0.5)

    response = [(b,) + tuple(cache_block(cache, k, b, sel)) for cache, b, sel in BASE_BLOCKS]
    response += [(f'multires:{fam}',) + tuple(multires_family(k, fam)) for fam in MULTIRES_FAMILIES]
    lookup = {f'{b}::{n}': [a[:, j] for a in arrays] for b, arrays, cols in response for j, n in enumerate(cols)}
    deltas = json.loads((HERE / 'combo_deltas.json').read_text())
    response.append(('response_multiscale_deltas',
                     [np.column_stack([lookup[d['left']][i] - lookup[d['right']][i] for d in deltas]) for i in range(3)],
                     [d['name'] for d in deltas]))
    response.append(('response_bpe_prefix',) + tuple(bpe_prefix_block(y, tr, va, *codes)))
    for b, arrays, cols in response:
        add(b, arrays, cols, block_weight(b))

    out = OUT / f'fold_{k}'
    out.mkdir(parents=True, exist_ok=True)
    for r in ROLES:
        x = np.ascontiguousarray(np.column_stack(X[r]), dtype=np.float32)
        assert x.shape[1] == len(names) and np.isfinite(x).all(), r
        np.save(out / f'x_{r}.npy', x)
        m = np.load(MARGIN_CACHE / f'fold_{k}' / f'nested_response_margin_{r}.npy', mmap_mode='r')
        np.save(out / f'margin_{r}.npy', np.asarray(m[:, 0], dtype=np.float64))
    assert len(names) == len(set(names))
    (out / 'meta.json').write_text(json.dumps(dict(names=names, weights=np.asarray(weights, np.float32).tolist())))
    print(f'fold {k}: {len(names)} columns', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    y = target()
    f = folds(y)
    codes = bpe_codes()
    for k in fold_list(a.folds):
        build_fold(k, f, y, codes)


if __name__ == '__main__':
    main()
