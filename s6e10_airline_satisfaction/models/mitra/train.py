#!/usr/bin/env python
"""Mitra-v2 (AutoGluon mitra-classifier-2, Tab2D), zero-shot, full context, on the 21 raw features.

Context = all outer-train rows of the fold (559,708 rows) with their labels; queries = the held-out fold + all test rows.
8 members. With AutoGluon's defaults a member differs from another only by a random sign flip per feature (random_mirror_x;
no feature shuffles, transforms or class shifts). Member m draws its flips from np.random.seed(10007 * (42 + fold) + m). The
preprocessor (mean-impute NaN, sign flips) is fitted on the context rows; Mitra's quantile normalisation happens inside the
model from the support set. Members' softmax probabilities are averaged, as in MitraClassifier.predict_proba.
Fine-tuning is off (AutoGluon's default would fine-tune 50 steps; all foundation models here are zero-shot).
Forward = mitra_cache.forward (exact layer-by-layer version; support processed once per member, all queries in one pass),
fp16 autocast with fp32 logits (Mitra's default bf16 is ~7x less accurate and its bf16 logits leave many tied probabilities).
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import folds, raw_arrays, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

NAME = 'mitra2'
N_MEMBERS = 8
CHUNK = 32768


def member_inputs(Xc, yc, Xq, m, fold):
    """AutoGluon Mitra preprocessing for member m (same flags as MitraClassifier's config)."""
    from autogluon.tabular.models.mitra._internal.config.enums import Task
    from autogluon.tabular.models.mitra._internal.data.preprocessor import Preprocessor
    pre = Preprocessor(dim_embedding=None, n_classes=10, dim_output=10, use_quantile_transformer=False, use_feature_count_scaling=False,
                       use_random_transforms=False, shuffle_classes=False, shuffle_features=False, random_mirror_x=True,
                       random_mirror_regression=True, task=Task.CLASSIFICATION)
    np.random.seed(10007 * (42 + fold) + m)
    pre.fit(Xc.copy(), yc)
    return pre.transform_X(Xc.copy()), pre.transform_y(yc.copy()), pre.transform_X(Xq.copy())


def predict(model, Xc, yc, Xq, fold):
    import torch
    from mitra_cache import forward
    probs = []
    for m in range(N_MEMBERS):
        xs, ys, xq = member_inputs(Xc, yc, Xq, m, fold)
        s, y, q = (torch.as_tensor(a, device='cuda')[None] for a in (xs, ys, xq))
        logits = forward(model, s, y, q, chunk=CHUNK, dtype=torch.float16, final_fp32=True)
        logits = logits[:, :2].double().cpu().numpy()  # MitraClassifier keeps the first len(unique(y)) classes, then softmax
        p = np.exp(logits - logits.max(1, keepdims=True))
        probs.append((p / p.sum(1, keepdims=True))[:, 1])
        print(f'  member {m + 1}/{N_MEMBERS} done', flush=True)
    return np.mean(probs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    import torch
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D
    from huggingface_hub import snapshot_download
    from sklearn.metrics import roc_auc_score
    model = Tab2D.from_pretrained(snapshot_download('autogluon/mitra-classifier-2'), device='cuda').to('cuda').eval()
    X, Xt, _ = raw_arrays()
    y = target()
    f = folds(y)
    for k in fold_list(a.folds):
        if have_part(HERE, NAME, k):
            continue
        tr, va = f != k, f == k
        p = predict(model, X[tr], y[tr].astype(np.int64), np.concatenate([X[va], Xt]), k)
        nv = int(va.sum())
        save_part(HERE, NAME, k, p[:nv], p[nv:])
        print(f'fold {k}: AUC {roc_auc_score(y[va], p[:nv]):.6f}', flush=True)
        torch.cuda.empty_cache()
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
