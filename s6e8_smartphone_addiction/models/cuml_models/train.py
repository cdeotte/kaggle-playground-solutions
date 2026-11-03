#!/usr/bin/env python
"""Six cuML pipelines on the shared base matrix and the per-fold response bank of build_features.py (lib/pipelines.py).

  model                 pair                        pipeline
  original_knn_lr       cuml_original_knn_lr        168 base columns (all but the observed-joint block) + 37 response
                                                    columns, robust-scaled -> logistic regression (C 0.12)
  residual_pca_svm      cuml_residual_pca_svm       240 base columns robust-scaled -> whitened PCA-64; + response ->
                                                    squared-hinge linear SVM (C 0.035), sigmoid of the decision function
  landmark_rbf_lr       cuml_landmark_rbf_lr        whitened PCA-40 + 256 KMeans landmarks x 3 RBF bandwidths + response
                                                    -> logistic regression (C 0.08)
  umap_lr               cuml_umap_lr                UMAP-16 embedding + response -> logistic regression (C 0.16)
  artifact_rf           cuml_artifact_rf            240 base columns, median-filled (no response bank) -> random forest
                                                    (900 trees, depth 18, max_features 0.55, min leaf 18, 256 bins,
                                                    max_samples 0.78)
  cluster_posterior_lr  cuml_cluster_posterior_lr   whitened PCA-32 + leave-one-row-out KMeans 64 / 256 / 1024 cluster
                                                    posteriors + response -> logistic regression (C 0.12)

Each fold fits on the outer-train rows and predicts the held-out fold and the test rows; test = mean of the 5 fold
models. Seeds: KMeans landmarks 2600 + fold, UMAP 3600 + fold, random forest 4600 + fold, cluster KMeans
5600 + 10 fold + resolution.
Usage: python train.py <model> [--folds 0,1,2,3,4]   (finished folds are skipped)"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import cache_dir, folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib.pipelines import PIPELINES  # noqa: E402

ALL_GROUPS = {'raw', 'mask', 'category', 'derived', 'artifact', 'source', 'source_knn', 'source_impute', 'observed_joint'}
GROUPS = {model: ALL_GROUPS for model in PIPELINES}
GROUPS['original_knn_lr'] = ALL_GROUPS - {'observed_joint'}


def pair_name(model):
    return f'cuml_{model}'


def load_fold(model, k, f, y):
    out = cache_dir('cuml_models')
    groups = json.loads((out / 'base_columns.json').read_text())['groups']
    keep = np.asarray([group in GROUPS[model] for group in groups])
    tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
    base_train = np.load(out / 'base_train.npy', mmap_mode='r')
    base_test = np.load(out / 'base_test.npy', mmap_mode='r')
    base = (np.asarray(base_train[tr][:, keep], np.float32), np.asarray(base_train[va][:, keep], np.float32),
            np.asarray(base_test[:, keep], np.float32))
    response = {role: np.load(out / 'response' / f'fold_{k}' / f'{role}.npy').astype(np.float32)
                for role in ('train', 'valid', 'test')}
    if len(response['train']) != len(tr) or len(response['valid']) != len(va):
        raise RuntimeError(f'fold {k}: response bank rows do not match the fold')
    return base, response, y[tr].astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', choices=list(PIPELINES))
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    name = pair_name(a.model)
    y = target()
    f = folds(y)
    for k in fold_list(a.folds):
        if have_part(HERE, name, k):
            print(f'fold {k}: done', flush=True)
            continue
        base, response, y_train = load_fold(a.model, k, f, y)
        print(f'{a.model} fold {k}: {base[0].shape[1]} base columns, {response["train"].shape[1]} response columns',
              flush=True)
        pv, pt = PIPELINES[a.model](base, response, y_train, k)
        if not (np.isfinite(pv).all() and np.isfinite(pt).all()):
            raise RuntimeError('non-finite predictions')
        save_part(HERE, name, k, pv, pt)
        print(f'{a.model} fold {k}: AUC {roc_auc_score(y[f == k], pv):.6f}', flush=True)
    assemble(HERE, name, f, y)


if __name__ == '__main__':
    main()
