#!/usr/bin/env python
"""TabPFN v3 with context ensembling (pair tabpfn_v3_context_ensemble).

Features (lib/tabpfn_features.py, 28 columns): the 12 raw columns (categoricals as ordinal codes, NaN kept), nested
exact-value target encodings of the 12 columns (smoothing 20) and slack, sleep + daily, utilization, missing count.
Per outer fold: K = 8 label-stratified context subsets of 50,000 outer-train rows (seed 777 + 100 x fold + k); the
pretrained TabPFN v3 classifier (n_estimators 2, CUDA) is fit on each context and predicts the fold's validation rows
and the test rows; the 8 probabilities are averaged. Test = mean of the 5 fold models.
Weights: the default TabPFN v3 classifier checkpoint (tabpfn-v3-classifier-v3_default.ckpt) of the Hugging Face model
Prior-Labs/tabpfn_3, downloaded on first use (accept its licence on the model page and log in to Hugging Face first), or
a local copy given with --model-path.
Usage: python train.py [--folds 0,1,2,3,4] [--model-path <checkpoint>]   (finished folds are kept in parts/ and skipped)"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import TARGET, folds as fold_ids, labels, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib import tabpfn_features as TF  # noqa: E402

PAIR = 'tabpfn_v3_context_ensemble'
SEED = 777
CONTEXT = 50000
K = 8
N_ESTIMATORS = 2
HF_REPO = 'Prior-Labs/tabpfn_3'
CKPT_NAME = 'tabpfn-v3-classifier-v3_default.ckpt'


def checkpoint(model_path=None):
    """Local path of the TabPFN v3 classifier checkpoint: --model-path, else the Hugging Face download (cached)."""
    if model_path:
        return model_path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, filename=CKPT_NAME)


def strat_sample(idx, y, n, seed):
    rng = np.random.default_rng(seed)
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    n_pos = min(int(n * y[idx].mean()), len(pos))
    n_neg = min(n - n_pos, len(neg))
    return np.concatenate([rng.choice(pos, n_pos, replace=False),
                           rng.choice(neg, n_neg, replace=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--model-path', default=None, help='local TabPFN v3 classifier checkpoint (default: Hugging Face)')
    a = ap.parse_args()
    todo = [f for f in fold_list(a.folds) if not have_part(HERE, PAIR, f)]

    t0 = time.time()
    train, test = read_train(), read_test()
    y = labels(train[TARGET]).astype(np.int64)
    folds = fold_ids(y)
    if todo:
        from tabpfn import TabPFNClassifier
        X_fold, X_test = TF.config_b_features(train, test, y, folds)
        print(f'built the features in {time.time() - t0:.0f}s ({X_fold[0].shape[1]} cols)', flush=True)
        clf = TabPFNClassifier(model_path=checkpoint(a.model_path), device='cuda', n_estimators=N_ESTIMATORS)

    for f in todo:
        trn = np.where(folds != f)[0]
        val = np.where(folds == f)[0]
        val_preds = np.zeros(len(val))
        te_preds = np.zeros(len(test))
        for k in range(K):
            ctx = strat_sample(trn, y, CONTEXT, seed=SEED + f * 100 + k)
            clf.fit(X_fold[f][ctx], y[ctx])
            val_preds += clf.predict_proba(X_fold[f][val])[:, 1]
            te_preds += clf.predict_proba(X_test[f])[:, 1]
        val_preds /= K
        te_preds /= K
        save_part(HERE, PAIR, f, val_preds, te_preds)
        print(f'fold {f}: val AUC={roc_auc_score(y[val], val_preds):.6f} ({time.time() - t0:.0f}s)', flush=True)
    assemble(HERE, PAIR, folds, y)


if __name__ == '__main__':
    main()
