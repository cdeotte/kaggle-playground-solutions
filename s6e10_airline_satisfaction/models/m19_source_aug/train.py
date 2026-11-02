#!/usr/bin/env python
"""M19 fold fine-tuning (two seeds x three snapshots, "bag2") and label-free alignment -> pair m19_source_aug_bag2_aligned.

Per outer fold k, starting from checkpoints/source.pt (pretrain.py): fine-tune encoder + head on the outer-train rows with BCE,
15% field masking, 6 epochs, batch 512, AdamW wd 0.01, OneCycle (peaks 7e-5 encoder / 7e-4 head, pct_start 0.1, default div
factors), grad clip 1, bf16. Two seeds (101700 + k, 101800 + k). The logits after epochs 4, 5 and 6 of both seeds (6 snapshots)
are averaged, then sigmoid.
Alignment, per fold model and without labels: p -> sigmoid(3 * (logit(p) - mu) / sd), with mu / sd the mean / std of that
model's own test logits. OOF = aligned fold predictions; test = mean of the 5 aligned fold vectors.
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from model import CKPT, MASK, Clf, load_codes  # noqa: E402

NAME = 'm19_source_aug_bag2_aligned'
EPOCHS, SNAPSHOTS, BATCH = 6, (4, 5, 6), 512
LR_ENC, LR_HEAD, MASK_P, WD = 7e-5, 7e-4, 0.15, 0.01
SEED_BASES = (101700, 101800)


@torch.inference_mode()
def predict(model, x, rows):
    model.eval()
    out = []
    for s in range(0, len(rows), 4096):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(model(x[torch.as_tensor(rows[s:s + 4096], device='cuda')]).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def align(pv, pt):
    """sigmoid(3 * (logit(p) - mu) / sd), mu / sd = mean / std of this fold model's test logits."""
    zv, zt = logit(np.clip(pv, 1e-7, 1 - 1e-7)), logit(np.clip(pt, 1e-7, 1 - 1e-7))
    mu, sd = zt.mean(), zt.std()
    return expit(3 * (zv - mu) / sd), expit(3 * (zt - mu) / sd)


def run_fold(k, x, yt, f, state, vocab):
    """Returns the bagged fold-k and test probabilities (before alignment)."""
    tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
    te = np.arange(len(f), len(x))
    trt = torch.as_tensor(tr, device='cuda')
    zv, zt = np.zeros(len(va)), np.zeros(len(te))
    n_snap = len(SEED_BASES) * len(SNAPSHOTS)
    for seed in SEED_BASES:
        torch.manual_seed(seed + k)
        model = Clf(vocab).cuda()
        model.load_state_dict(state)
        opt = torch.optim.AdamW([dict(params=model.enc.parameters(), lr=LR_ENC),
                                 dict(params=model.head.parameters(), lr=LR_HEAD)], weight_decay=WD)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_ENC, LR_HEAD],
                                                    total_steps=EPOCHS * math.ceil(len(tr) / BATCH), pct_start=0.1)
        for ep in range(1, EPOCHS + 1):
            model.train()
            order = trt[torch.randperm(len(trt), device='cuda')]
            for s in range(0, len(order), BATCH):
                idx = order[s:s + BATCH]
                ids = x[idx]
                inp = ids.masked_fill(torch.rand(ids.shape, device='cuda') < MASK_P, MASK)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(inp)
                loss = F.binary_cross_entropy_with_logits(out.float(), yt[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            print(f'fold {k} seed {seed + k}: epoch {ep}/{EPOCHS} BCE {loss.item():.4f}', flush=True)
            if ep in SNAPSHOTS:
                zv += predict(model, x, va) / n_snap
                zt += predict(model, x, te) / n_snap
        del model, opt, sched
        torch.cuda.empty_cache()
    return expit(zv), expit(zt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    y = target()
    f = folds(y)
    todo = [k for k in fold_list(a.folds) if not have_part(HERE, NAME, k)]
    if todo:
        assert (CKPT / 'source.pt').exists(), 'run pretrain.py first'
        codes, sizes = load_codes()
        assert sizes[0] == len(y), 'cache/codes.npz does not match train.csv; delete it'
        vocab, n_comp = int(codes.max()) + 1, int(sizes[0] + sizes[1])
        x = torch.as_tensor(codes[:n_comp].astype(np.int64), device='cuda')      # competition rows: train, then test
        yt = torch.as_tensor(y.astype(np.float32), device='cuda')
        state = torch.load(CKPT / 'source.pt', map_location='cpu', weights_only=True)
        for k in todo:
            pv, pt = run_fold(k, x, yt, f, state, vocab)
            av, at = align(pv, pt)
            save_part(HERE, NAME, k, av, at)
            print(f'fold {k}: AUC {roc_auc_score(y[f == k], av):.6f}', flush=True)
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
