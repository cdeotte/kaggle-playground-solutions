#!/usr/bin/env python
"""Zero-shot TabICL v2, eight-member vote (pair tabicl_v2_zero_shot_8seed).

Inputs: the 14 raw competition columns (Driver, Compound and Race stay strings; TabICL encodes them itself). No feature
engineering and no original rows. Per outer fold, eight pretrained TabICLClassifier models (n_estimators=2,
kv_cache=True, random_state 0..7; the fold's training rows are the in-context examples, nothing is trained) predict the
fold's validation rows and the test rows. The 8 probabilities are averaged and clipped to [1e-7, 1 - 1e-7].
Test = mean of the 5 fold models.
Weights: TabICLClassifier's default checkpoint tabicl-classifier-v2-20260212.ckpt, downloaded from the Hugging Face model
jingang/TabICL on first use.
Usage: python train_zero_shot.py [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import FEATURES, folds, model_dir, read_test, read_train, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

PAIR = 'tabicl_v2_zero_shot_8seed'
FOLDER = 'tabicl'
N_MODELS = 8
PARAMS = [{'device': 'cuda', 'n_estimators': 2, 'kv_cache': True, 'random_state': i} for i in range(N_MODELS)]


def cleanup_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    out = model_dir(FOLDER)
    y = target().astype(int)
    fold_id = folds(y)
    todo = [k for k in fold_list(args.folds) if not have_part(out, PAIR, k)]
    if todo:
        from tabicl import TabICLClassifier
        X = read_train()[list(FEATURES)]
        X_test = read_test()[list(FEATURES)]
    for k in todo:
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        X_tr, y_tr, X_va = X.iloc[tr_idx].copy(), y[tr_idx], X.iloc[va_idx].copy()
        p_va, p_te = [], []
        for params in PARAMS:
            # each member is fitted once (context + KV cache) and predicts both the validation and the test rows
            model = TabICLClassifier(**params)
            model.fit(X_tr, y_tr)
            p_va.append(model.predict_proba(X_va)[:, 1])
            p_te.append(model.predict_proba(X_test)[:, 1])
            del model
            cleanup_cuda()
        va_pred = np.clip(np.mean(p_va, axis=0).astype(np.float32), 1e-7, 1 - 1e-7)
        te_pred = np.clip(np.mean(p_te, axis=0).astype(np.float32), 1e-7, 1 - 1e-7)
        save_part(out, PAIR, k, va_pred, te_pred)
        print(f'{PAIR} fold {k}: AUC {roc_auc_score(y[va_idx], va_pred):.6f}', flush=True)
    assemble(out, PAIR, fold_id, y)


if __name__ == '__main__':
    main()
