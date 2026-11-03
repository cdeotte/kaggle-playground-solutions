#!/usr/bin/env python
"""source_graph_lr: source-genealogy graph features + one L2 logistic regression per outer fold.

Features (256 columns, lib/graph.py):
- 24 raw: the 9 numerics, the 3 categoricals as original-level codes (unseen / missing -> NaN) and 12 missing flags;
- 84 structural (label-free): per donor relation coverage, missingness pattern, top score, top exact fraction, top donor
  id / 7500, and for the k1 / k4 / k16 donor kernels the weight entropy and the weighted original label and severity
  ((Mild, Moderate, Severe) -> (0, 0.5, 1), missing -> label);
- 4 anchors from the fold forensic matrix of ../lgbm_margin: strong__raw_te_woe_sum, strong__screen_te_woe_sum,
  strong__selected_base_margin, strong__selected_base_probability;
- 144 label messages from the outer-train labels: per relation and kernel smoothed donor posteriors (m 4, 24, 128) +
  log support (72); community means of rows sharing (top donor, pattern) per relation (m 12, 64), top-donor pairs over
  all 15 relation pairs (m 16, 96) and top-donor triples over (screen_core, screen_weekend, engagement),
  (screen_core, behavior, lifestyle), (engagement, behavior, profile) (m 32, 160), each + log count (72).
  Outer-train rows get messages with their own weight and label removed (and a row-excluded prior); held-out and test
  rows get messages from all outer-train rows.
Model: median fill, standardize (outer-train mean / std), clip +-12; scikit-learn LogisticRegression(C=0.08, L2,
lbfgs, max_iter=300, tol=1e-7, random_state=1801) fitted on the outer-train rows, predicting the held-out fold and test.
Test = mean of the 5 fold models. --variant linear_control drops the 144 label messages (matched control).
Needs cache/donors (build_donors.py) and ../lgbm_margin's fold matrices.
Usage: python train_graph_lr.py [--folds 0,1,2,3,4] [--variant linear_graph]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib.donors import CATEGORICAL, NUMERIC, load_donors  # noqa: E402
from lib.graph import build_label_message_features, build_structural_features  # noqa: E402
from lib.inputs import forensic_columns, load_forensic  # noqa: E402

PAIR = {'linear_graph': 'source_graph_lr', 'linear_control': 'source_graph_lr_control'}
ANCHORS = ('strong__raw_te_woe_sum', 'strong__screen_te_woe_sum', 'strong__selected_base_margin',
           'strong__selected_base_probability')


def source_arrays(source):
    target = source[TARGET].to_numpy(np.float32)
    severity = source['addiction_level'].map({'Mild': 1.0, 'Moderate': 2.0, 'Severe': 3.0}).to_numpy(np.float32)
    fallback = np.float32(1.0) + np.float32(2.0) * target
    severity = np.where(np.isfinite(severity), severity, fallback)
    severity = ((severity - np.float32(1.0)) / np.float32(2.0)).astype(np.float32)
    return target, severity


def raw_features(frame, rows, source):
    selected = frame.iloc[rows]
    numeric = selected.loc[:, NUMERIC].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    categorical = np.full((len(rows), len(CATEGORICAL)), np.nan, dtype=np.float32)
    for j, column in enumerate(CATEGORICAL):
        levels = tuple(sorted(source[column].astype('string').fillna('<MISSING>').unique()))
        mapping = {value: position for position, value in enumerate(levels)}
        values = selected[column].astype('string')
        observed = values.notna().to_numpy()
        mapped = values.map(mapping).to_numpy(dtype=np.float32, na_value=np.nan)
        categorical[observed, j] = mapped[observed]
    raw = np.concatenate([numeric, categorical], axis=1)
    missing = (~np.isfinite(raw)).astype(np.float32)
    return np.concatenate([raw, missing], axis=1).astype(np.float32)


def role_features(frame, view, rows, source, source_target, source_severity, anchors, seed_view, seed_rows, seed_target,
                  self_exclude, graph):
    """[raw 24 | structural 84 | anchors 4] (+ [messages 144] for the graph variant) for `rows` of `frame`."""
    structural, _ = build_structural_features(view, rows, source_target, source_severity)
    parts = [raw_features(frame, rows, source), structural, np.asarray(anchors, dtype=np.float32)]
    if graph:
        messages, _ = build_label_message_features(seed_view, seed_rows, seed_target, view, rows, self_exclude)
        parts.append(messages)
    return np.concatenate(parts, axis=1).astype(np.float32)


def fit_linear_state(matrix):
    finite = np.isfinite(matrix)
    median = np.nanmedian(np.where(finite, matrix, np.nan), axis=0).astype(np.float32)
    median[~np.isfinite(median)] = 0.0
    filled = np.where(finite, matrix, median[None, :]).astype(np.float64)
    center = filled.mean(axis=0).astype(np.float32)
    scale = filled.std(axis=0).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-5)] = 1.0
    return median, center, scale


def transform_linear(matrix, median, center, scale):
    value = np.where(np.isfinite(matrix), matrix, median[None, :])
    value = (value - center[None, :]) / scale[None, :]
    return np.clip(value, -12.0, 12.0).astype(np.float32)


def run_fold(k, f, y, train, test, source, views, graph):
    source_target, source_severity = source_arrays(source)
    tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
    te = np.arange(len(test), dtype=np.int64)
    columns = forensic_columns(k, ANCHORS)
    anchors = {role: np.asarray(load_forensic(k, role)[:, columns], dtype=np.float32) for role in ('train', 'valid', 'test')}
    if len(anchors['train']) != len(tr) or len(anchors['valid']) != len(va) or len(anchors['test']) != len(te):
        raise RuntimeError(f'fold {k}: forensic matrix rows do not match the fold')
    seed_target = y[tr].astype(np.float64)
    shared = dict(source=source, source_target=source_target, source_severity=source_severity, seed_view=views['train'],
                  seed_rows=tr, seed_target=seed_target, graph=graph)
    x_train = role_features(train, views['train'], tr, anchors=anchors['train'], self_exclude=True, **shared)
    x_valid = role_features(train, views['train'], va, anchors=anchors['valid'], self_exclude=False, **shared)
    x_test = role_features(test, views['test'], te, anchors=anchors['test'], self_exclude=False, **shared)
    print(f'fold {k}: {x_train.shape[1]} features', flush=True)

    median, center, scale = fit_linear_state(x_train)
    model = LogisticRegression(C=0.08, penalty='l2', solver='lbfgs', max_iter=300, tol=1e-7, random_state=1801)
    model.fit(transform_linear(x_train, median, center, scale), y[tr])
    pv = model.predict_proba(transform_linear(x_valid, median, center, scale))[:, 1]
    pt = model.predict_proba(transform_linear(x_test, median, center, scale))[:, 1]
    return pv, pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--variant', default='linear_graph', choices=list(PAIR))
    a = ap.parse_args()
    name = PAIR[a.variant]
    train, test, source = read_train(), read_test(), read_original(keep_level=True)
    y = labels(train[TARGET])
    f = folds(y)
    views = {'train': load_donors('train'), 'test': load_donors('test')}
    for k in fold_list(a.folds):
        if have_part(HERE, name, k):
            print(f'fold {k}: done', flush=True)
            continue
        pv, pt = run_fold(k, f, y, train, test, source, views, graph=a.variant == 'linear_graph')
        save_part(HERE, name, k, pv, pt)
        print(f'fold {k}: AUC {roc_auc_score(y[f == k], pv):.6f}', flush=True)
    assemble(HERE, name, f, y)


if __name__ == '__main__':
    main()
