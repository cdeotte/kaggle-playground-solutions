"""Fold design shared by lr.py and realmlp.py: TE banks + label-free packs + GPT-2 acceptance columns + income EM blocks.

Rows: outer-train (train.csv order), fold k (train.csv order), test. Column names: '<bank>:<column>', 'pack:<pack>:<column>',
'acc:<column>', 'block:<block>:<column>'.
Inputs: cache/ (build_banks.py, replica_prior.py), ../combo_xgb/features/ (digits, footprint, income EM blocks),
../response_features/cache/acceptance (the acceptance pack)."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'cache'
COMBO = HERE.parent / 'combo_xgb' / 'features'
ACCEPTANCE = HERE.parent / 'response_features' / 'cache' / 'acceptance'
PACKS = {'num': (CACHE / 'num.npy', CACHE / 'num_names.json'),
         'digits': (COMBO / 'digits.npy', COMBO / 'digits_names.json'),
         'footprint': (COMBO / 'footprint.npy', COMBO / 'footprint_names.json'),
         'replica': (CACHE / 'replica_prior.npy', CACHE / 'replica_prior_names.json')}
ACC_FAMILIES = ('constrained_entropy', 'acceptance_probability', 'numeric_center')
ROLES = ('train', 'valid', 'test')


def acceptance():
    """168 label-free acceptance columns, train rows then test rows."""
    man = json.loads((ACCEPTANCE / 'manifest.json').read_text())
    idx = sorted(set(i for f in ACC_FAMILIES for i in man['families'][f]))
    a = np.concatenate([np.load(ACCEPTANCE / 'train.npy', mmap_mode='r')[:, idx], np.load(ACCEPTANCE / 'test.npy', mmap_mode='r')[:, idx]])
    return a, [f'acc:{man["names"][i]}' for i in idx]


def fold_design(k, folds, banks=(), packs=(), acc=False, blocks=()):
    """-> ({'train', 'valid', 'test'}: float32 arrays, column names) for outer fold k."""
    ti, vi = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
    n_tr = len(folds)
    parts, names = {r: [] for r in ROLES}, []
    for b in banks:
        for r in ROLES:
            parts[r].append(np.load(CACHE / f'fold_{k}' / f'{b}_{r}.npy', mmap_mode='r'))
        names += [f'{b}:{n}' for n in json.loads((CACHE / f'{b}_names.json').read_text())]
    sources = []
    for p in packs:
        sources.append((np.load(PACKS[p][0], mmap_mode='r'), [f'pack:{p}:{n}' for n in json.loads(PACKS[p][1].read_text())]))
    if acc:
        sources.append(acceptance())
    for a, nm in sources:
        parts['train'].append(a[ti])
        parts['valid'].append(a[vi])
        parts['test'].append(a[n_tr:])
        names += nm
    for b in blocks:
        bd = COMBO / 'fold_blocks' / b / f'fold_{k}'
        for r in ROLES:
            parts[r].append(np.load(bd / f'{r}.npy'))
        names += [f'block:{b}:{n}' for n in json.loads((bd / 'meta.json').read_text())['names']]
    X = {r: np.ascontiguousarray(np.column_stack(v), dtype=np.float32) for r, v in parts.items()}
    for r in ROLES:
        assert X[r].shape[1] == len(names) and np.isfinite(X[r]).all(), r
    return X, names
