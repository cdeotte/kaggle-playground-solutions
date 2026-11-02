#!/usr/bin/env python
"""M19 pretraining without competition labels; done once and shared by all folds.

Stage 1, masked-field modelling (checkpoints/mfm.pt): the features of all rows (train, test, original and both replica
worlds; no labels). Every field is replaced by MASK with probability 0.25, plus one random field per row; the masked
fields are predicted.
Encoder L8, d384, 12 heads; 20 epochs, batch 1024, AdamW lr 5e-4 (betas 0.9 / 0.95, wd 0.01), OneCycle (5% warmup, cosine),
grad clip 1, bf16. Seed 0.
Stage 2, source supervision (checkpoints/source.pt): the stage-1 encoder (loaded into the 8-head encoder) + a classification
head, fit with BCE on the labels of the original rows and both replica worlds; no competition labels. 4 epochs, batch 512,
5% field masking, AdamW wd 0.01, lr 5e-5 encoder / 5e-4 head times a per-epoch factor 0.1 + 0.9 * 0.5 * (1 + cos(pi *
(epoch - 1) / 4)), grad clip 1, bf16. Seed 101410.
Usage: python pretrain.py"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import TARGET, labels, read_original  # noqa: E402
from model import CKPT, DIM, MASK, NF, WORLDS, WORLDS_DIR, Clf, Enc, load_codes  # noqa: E402

MFM_EPOCHS, MFM_BATCH, MFM_LR, MFM_MASK = 20, 1024, 5e-4, 0.25
SRC_EPOCHS, SRC_BATCH, SRC_LR, SRC_HEAD_LR, SRC_MASK = 4, 512, 5e-5, 5e-4, 0.05


def source_labels():
    """Labels of the source rows in code order: original, M32_1M train / test, MB32_1M train / test."""
    ys = [labels(read_original()[TARGET])]
    for w in WORLDS:
        ys += [np.load(WORLDS_DIR / w / 'y.npy'), np.load(WORLDS_DIR / w / 'y_test.npy')]
    return np.concatenate([np.asarray(v).astype(np.float32).ravel() for v in ys])


def pretrain_mfm(codes, vocab):
    torch.manual_seed(0)
    enc = Enc(vocab, heads=12, random_field=True).cuda()
    mlm = nn.Linear(DIM, vocab).cuda()
    X = torch.as_tensor(codes.astype(np.int64), device='cuda')
    params = list(enc.parameters()) + list(mlm.parameters())
    opt = torch.optim.AdamW(params, lr=MFM_LR, weight_decay=0.01, betas=(0.9, 0.95))
    steps = MFM_EPOCHS * math.ceil(len(X) / MFM_BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MFM_LR, total_steps=steps, pct_start=0.05, anneal_strategy='cos')
    for ep in range(MFM_EPOCHS):
        perm = torch.randperm(len(X), device='cuda')
        for s0 in range(0, len(perm), MFM_BATCH):
            ids = X[perm[s0:s0 + MFM_BATCH]]
            mask = torch.rand(ids.shape, device='cuda') < MFM_MASK
            mask[torch.arange(len(ids), device='cuda'), torch.randint(0, NF, (len(ids),), device='cuda')] = True
            inp = ids.masked_fill(mask, MASK)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = enc(inp)[:, 1:]
                loss = F.cross_entropy(mlm(h[mask]).float(), ids[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        print(f'masked-field pretraining: epoch {ep + 1}/{MFM_EPOCHS} CE {loss.item():.4f}', flush=True)
    CKPT.mkdir(exist_ok=True)
    torch.save(enc.state_dict(), CKPT / 'mfm.pt')


def pretrain_source(codes, vocab, n_comp):
    torch.backends.cuda.matmul.allow_tf32 = True
    sy = source_labels()
    assert len(sy) == len(codes) - n_comp, 'source labels must match the original + world rows'
    X = torch.as_tensor(codes[n_comp:].astype(np.int64), device='cuda')            # source rows only
    target = torch.as_tensor(sy, device='cuda')
    torch.manual_seed(101410)
    model = Clf(vocab).cuda()
    model.enc.load_state_dict(torch.load(CKPT / 'mfm.pt', map_location='cpu', weights_only=True))
    opt = torch.optim.AdamW([dict(params=model.enc.parameters(), lr=SRC_LR),
                             dict(params=model.head.parameters(), lr=SRC_HEAD_LR)], weight_decay=0.01)
    for ep in range(1, SRC_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(X), device='cuda')
        factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (ep - 1) / SRC_EPOCHS))
        for g, lr in zip(opt.param_groups, (SRC_LR, SRC_HEAD_LR)):
            g['lr'] = lr * factor
        for s in range(0, len(order), SRC_BATCH):
            idx = order[s:s + SRC_BATCH]
            ids = X[idx]
            inp = ids.masked_fill(torch.rand(ids.shape, device='cuda') < SRC_MASK, MASK)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = model(inp)
            loss = F.binary_cross_entropy_with_logits(z.float(), target[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print(f'source pretraining: epoch {ep}/{SRC_EPOCHS} BCE {loss.item():.4f}', flush=True)
    torch.save({n: v.detach().cpu() for n, v in model.state_dict().items()}, CKPT / 'source.pt')


def main():
    codes, sizes = load_codes()
    vocab, n_comp = int(codes.max()) + 1, int(sizes[0] + sizes[1])
    print(f'rows {len(codes)} (competition {n_comp}, source {len(codes) - n_comp}) | vocab {vocab}', flush=True)
    if not (CKPT / 'mfm.pt').exists():
        pretrain_mfm(codes, vocab)
        torch.cuda.empty_cache()
    if not (CKPT / 'source.pt').exists():
        pretrain_source(codes, vocab, n_comp)


if __name__ == '__main__':
    main()
