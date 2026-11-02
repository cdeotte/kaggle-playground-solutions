#!/usr/bin/env python
"""LimiX-2, zero-shot, full context, on the 21 raw features.

Context = all outer-train rows of the fold (559,708 rows) with their labels; queries = the held-out fold + all test rows.
LimiXPredictor (v2.0 code path of the LimiX GitHub repository) with the official classification config
cls_default_noretrieval_v2.json (32 preprocessing pipelines = 32 members). We keep 8 light pipelines (0, 4, 8, 13, 20, 24, 27,
31; at most 23 features each): at 25k context they lose nothing against all 32, while the wide pipelines (e.g. SVD + polynomial
features) need far more memory at full context.
limix_cache.py replaces the layer stack with an exact row-chunked version, so the training context is processed once per member.
Mixed precision: LimiX's default fp16 autocast. Categoricals are auto-detected by LimiX; NaN kept.
Set LIMIX_REPO to a clone of the LimiX repository (it provides inference.predictor and model.v2_0).
Usage: LIMIX_REPO=/path/to/LimiX python train.py [--folds 0,1,2,3,4]"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, raw_arrays, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

NAME = 'limix2'
LIMIX_REPO = os.environ.get('LIMIX_REPO', str(HERE / 'LimiX'))
PIPELINES = [0, 4, 8, 13, 20, 24, 27, 31]


def predict(model_path, cfg, Xc, yc, Xq, fold):
    import torch
    from inference.predictor import LimiXPredictor
    pred = LimiXPredictor(device=torch.device('cuda'), model_path=model_path, inference_config=cfg, seed=42 + fold,
                          test_batch_size=16384, preprocess_num_jobs=8, mix_precision=True)
    return np.asarray(pred.predict(Xc, yc, Xq, task_type='Classification'))[:, 1].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    for k, v in (('RANK', '0'), ('WORLD_SIZE', '1'), ('MASTER_ADDR', '127.0.0.1'), ('MASTER_PORT', '29500')):
        os.environ.setdefault(k, v)
    sys.path.insert(0, LIMIX_REPO)
    sys.path.insert(0, str(HERE))
    import limix_cache
    from huggingface_hub import hf_hub_download
    from sklearn.metrics import roc_auc_score
    limix_cache.install(65536)
    model_path = hf_hub_download(repo_id='stable-ai/LimiX-2', filename='LimiX-2.ckpt')
    cfg = json.load(open(f'{LIMIX_REPO}/config/cls_default_noretrieval_v2.json'))
    cfg['pipelines'] = [cfg['pipelines'][i] for i in PIPELINES]
    X, Xt, _ = raw_arrays()
    y = target()
    f = folds(y)
    for k in fold_list(a.folds):
        if have_part(HERE, NAME, k):
            continue
        tr, va = f != k, f == k
        p = predict(model_path, cfg, X[tr], y[tr].astype(np.int64), np.concatenate([X[va], Xt]), k)
        nv = int(va.sum())
        save_part(HERE, NAME, k, p[:nv], p[nv:])
        print(f'fold {k}: AUC {roc_auc_score(y[va], p[:nv]):.6f}', flush=True)
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
