#!/usr/bin/env python
"""Combo XGB: XGBoost on the 692-column fold-local design, 32 seeds, nested-LR base margin.

Columns per fold: the 642-column combo design (build_combo.py) + digits (14) + footprint (10) + digit_te (8) + income_em (4)
+ income_info (14) (build_blocks.py, footprint.py). Feature weights: the combo design's own, 1.0 for the added blocks.
Base margin: the nested fold-local logistic-regression logit of the combo design.
XGB: depth 6, eta 0.03, min_child_weight 10, subsample 0.9, colsample_bytree / bynode 0.7, lambda 5, alpha 0.15, max_bin
512, num_parallel_tree 6, 600 rounds.
Seeds: 32 per fold, in 8 parts of 4: seed = 42001 + fold + 100 + 4000 * part + 1000 * j (j = 0..3). Every seed draws fresh
train-only Gaussian noise (sd 0.04, RNG seed 7 * seed + fold) on the response-encoding columns (response_*, multires:*)
and the digit_te columns; valid / test rows are never noised.
A part = the mean logit of its 4 seeds (parts/). Per fold the 8 parts are averaged in logit space: OOF = sigmoid of the
held-out mean logit; test = mean over the 5 folds of the sigmoid of the test mean logit.
Usage: python train.py [--folds 0,1,2,3,4] [--parts 0,1,2,3,4,5,6,7]   (finished parts are skipped; the pair is written
       once all 40 parts exist)"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.pairs import fold_list, have_part, save_pair, save_part  # noqa: E402

NAME = 'combo_xgb_32seeds'
FEATURES = HERE / 'features'
PACKS = ('digits', 'footprint')
BLOCKS = ('digit_te', 'income_em', 'income_info')
N_PARTS, SEEDS_PER_PART, ROUNDS, NOISE_SD = 8, 4, 600, 0.04
PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'device': 'cuda', 'max_depth': 6, 'eta': 0.03,
          'min_child_weight': 10, 'subsample': 0.9, 'colsample_bytree': 0.7, 'colsample_bynode': 0.7, 'lambda': 5, 'alpha': 0.15,
          'max_bin': 512, 'num_parallel_tree': 6, 'nthread': 8, 'scale_pos_weight': 1.0}
ROLES = ('train', 'valid', 'test')


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def part_name(p):
    return f'{NAME}_p{p}'


def load_fold(k, f):
    d = FEATURES / 'combo' / f'fold_{k}'
    meta = json.loads((d / 'meta.json').read_text())
    X = {r: [np.load(d / f'x_{r}.npy')] for r in ROLES}
    M = {r: np.load(d / f'margin_{r}.npy') for r in ROLES}
    names, weights = list(meta['names']), list(meta['weights'])
    tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
    for pack in PACKS:  # label-free, train rows then test rows
        A = np.load(FEATURES / f'{pack}.npy', mmap_mode='r')
        cols = json.loads((FEATURES / f'{pack}_names.json').read_text())
        for r, rows in (('train', tr), ('valid', va), ('test', slice(len(f), None))):
            X[r].append(np.asarray(A[rows], np.float32))
        names += [f'pack::{pack}::{c}' for c in cols]
        weights += [1.0] * len(cols)
    for blk in BLOCKS:  # fold-local
        bd = FEATURES / 'fold_blocks' / blk / f'fold_{k}'
        cols = json.loads((bd / 'meta.json').read_text())['names']
        for r in ROLES:
            X[r].append(np.load(bd / f'{r}.npy').astype(np.float32))
        names += [f'block::{blk}::{c}' for c in cols]
        weights += [1.0] * len(cols)
    X = {r: np.ascontiguousarray(np.column_stack(v), dtype=np.float32) for r, v in X.items()}
    noisy = [i for i, n in enumerate(names) if n.split('::')[0].startswith(('response_', 'multires:'))]
    noisy += [i for i, n in enumerate(names) if n.startswith('block::digit_te::')]
    return X, M, names, np.asarray(weights, np.float32), noisy


def train_fold(k, f, y, parts):
    todo = [p for p in parts if not have_part(HERE, part_name(p), k)]
    if not todo:
        return
    X, M, names, fw, noisy = load_fold(k, f)
    tr = np.flatnonzero(f != k)
    types = ['q'] * len(names)
    clean = X['train'][:, noisy].copy()
    dva = xgb.DMatrix(X['valid'], base_margin=M['valid'], feature_names=names, feature_types=types, nthread=8)
    dte = xgb.DMatrix(X['test'], base_margin=M['test'], feature_names=names, feature_types=types, nthread=8)
    for p in todo:
        vsum, tsum = np.zeros(X['valid'].shape[0]), np.zeros(X['test'].shape[0])
        for j in range(SEEDS_PER_PART):
            seed = 42001 + k + 100 + 4000 * p + 1000 * j
            X['train'][:, noisy] = clean + np.random.default_rng(seed * 7 + k).normal(0.0, NOISE_SD, clean.shape).astype(np.float32)
            dtr = xgb.DMatrix(X['train'], label=y[tr], base_margin=M['train'], feature_names=names, feature_types=types,
                              feature_weights=fw, nthread=8)
            model = xgb.train(dict(PARAMS, seed=seed), dtr, num_boost_round=ROUNDS, verbose_eval=False)
            vsum += np.asarray(model.predict(dva, output_margin=True), np.float64)
            tsum += np.asarray(model.predict(dte, output_margin=True), np.float64)
            del model, dtr
        save_part(HERE, part_name(p), k, vsum / SEEDS_PER_PART, tsum / SEEDS_PER_PART)
        print(f'fold {k} part {p}: AUC {roc_auc_score(y[f == k], vsum):.6f}', flush=True)


def assemble(f, y):
    if not all(have_part(HERE, part_name(p), k) for p in range(N_PARTS) for k in range(5)):
        print('not all parts are done yet')
        return
    oof, test = np.zeros(len(y)), 0.0
    for k in range(5):
        logits = [[np.load(HERE / 'parts' / f'{part_name(p)}_f{k}_{s}.npy').astype(np.float64) for p in range(N_PARTS)]
                  for s in ('valid', 'test')]
        oof[f == k] = sig(np.mean(logits[0], 0))
        test = test + sig(np.mean(logits[1], 0)) / 5
    save_pair(HERE, NAME, oof, test)
    print(f'{NAME}: OOF AUC {roc_auc_score(y, oof):.6f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--parts', default=','.join(str(p) for p in range(N_PARTS)))
    a = ap.parse_args()
    y = target()
    f = folds(y)
    for k in fold_list(a.folds):
        train_fold(k, f, y, fold_list(a.parts))
    assemble(f, y)


if __name__ == '__main__':
    main()
