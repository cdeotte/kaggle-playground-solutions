#!/usr/bin/env python
"""TabICLv2, zero-shot, full context, on the 21 raw features.

Context = all outer-train rows of the fold (559,708 rows) with their labels; queries = the held-out fold + all test rows in ONE
predict_proba call (the context is processed once per member; no cache needed). 8 members (TabICL's default), run one at a time
(batch_size=1; one member at full context peaks at ~78 GiB).

Precision fix: TabICL's default mixed precision (fp16 autocast in all three stages) is NOT accurate at full context. The
column-embedding stage (a set transformer whose inducing points attend over all ~560k training rows of each column) loses
precision in fp16; the error grows with context size and depends on how TabICL batches the work, so stock predictions even change
with the number of query rows in the call. Running only that stage in fp32 (COL_CONFIG use_amp=False; row and ICL stages stay
fp16) matches an all-fp32 run (fold 0, 1 member: AUC 0.957409 vs 0.957411) for ~4 s extra per member.
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, raw_arrays, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

NAME = 'tabicl2'
N_ESTIMATORS = 8
CHECKPOINT = ('jingang/TabICL', 'tabicl-classifier-v2-20260212.ckpt')


def predict(Xc, yc, Xq, fold, ckpt):
    from tabicl import TabICLClassifier
    clf = TabICLClassifier(n_estimators=N_ESTIMATORS, batch_size=1, kv_cache=False, model_path=ckpt, allow_auto_download=False,
                           device='cuda', random_state=42 + fold, inference_config={'COL_CONFIG': {'use_amp': False}})
    clf.fit(Xc, yc)
    return np.asarray(clf.predict_proba(Xq))[:, 1].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from sklearn.metrics import roc_auc_score
    ckpt = hf_hub_download(*CHECKPOINT)
    X, Xt, _ = raw_arrays()
    y = target()
    f = folds(y)
    for k in fold_list(a.folds):
        if have_part(HERE, NAME, k):
            continue
        tr, va = f != k, f == k
        p = predict(X[tr], y[tr], np.concatenate([X[va], Xt]), k, ckpt)
        nv = int(va.sum())
        save_part(HERE, NAME, k, p[:nv], p[nv:])
        print(f'fold {k}: AUC {roc_auc_score(y[va], p[:nv]):.6f}', flush=True)
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
