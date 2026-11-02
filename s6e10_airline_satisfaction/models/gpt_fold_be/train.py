#!/usr/bin/env python
"""BatchEnsemble head on the frozen B0 GPT features (pair gpt_fold_be_bag2_aligned), two head seeds per fold.

Inputs of every train + test row, for fold k:
  - 639 GPT features of fold k's B0 model (extract_features.py): the fields [2 orders, 2 labels, 21, 3] (252), their
    True - False differences (126), the hidden projections averaged over the two labels (128) and their True - False
    differences (128), the 4 summed log-likelihoods (order x label), and B0's two-order LLR margin (1):
        margin = logsumexp_o ll(o, True) - logsumexp_o ll(o, False)
  - 53 numeric features and 35 categorical codes (vocabulary 7,601) from ../transfer_features.
The 53 + 639 numeric inputs are z-scored with the outer-train rows' mean / std (std floored at 1e-3) and clipped to +-12.
Head: BatchEnsemble MLP with 4 members (shared weights, per-member random-sign input / output factors and biases):
Embedding(7601, 8) for the codes, 3 layers of width 384, SiLU, dropout 0.1, a linear output per member, and B0's margin added
to every member's logit (skip connection).
Training: BCE on every member, AdamW lr 4e-4, weight decay 0.01, cosine over the 14 epochs floored at 5% (set per epoch),
batch 2048, bf16 autocast, grad clip 2. Logit = mean of the 4 members, then mean of the snapshots after epochs 10, 12, 14.
bag2: heads with seeds 104100 + k and 105100 + k, mean of their logits.
Aligned (per fold, label-free): z = logit(clip(sigmoid(bag logit), 1e-7, 1 - 1e-7)), out = sigmoid(3 (z - mu) / sd), where
mu / sd = the mean / std of z over the test rows.
Pair: OOF = the aligned held-out folds; test = mean of the 5 aligned fold vectors.
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit, logit, logsumexp
from sklearn.metrics import roc_auc_score
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

NAME = 'gpt_fold_be_bag2_aligned'
TRANSFER = HERE.parent / 'transfer_features' / 'cache'
SEEDS = (104100, 105100)                     # + fold
CFG = dict(epochs=14, snapshots=(10, 12, 14), width=384, depth=3, members=4, emb=8, dropout=0.1, lr=4e-4, wd=0.01,
           batch=2048, clip=2.0)


def gpt_features(k):
    """The 639 GPT features of fold k's B0 model and its two-order LLR margin, for all train + test rows."""
    fdir = HERE / 'features'
    assert (fdir / f'f{k}_fields.npy').exists(), f'run extract_features.py --folds {k} first'
    pf = np.load(fdir / f'f{k}_fields.npy')                                  # [n, order, label, 21, 3]
    h = np.load(fdir / f'f{k}_hidden.npy').astype(np.float32)               # [n, order, label, 64]
    n = len(pf)
    ll = pf[..., 0].sum(-1)                                                 # [n, order, label] restricted-vocab log p(fields | label)
    margin = (logsumexp(ll[:, :, 1], axis=1) - logsumexp(ll[:, :, 0], axis=1)).astype(np.float32)
    feat = np.concatenate([pf.reshape(n, -1), (pf[:, :, 1] - pf[:, :, 0]).reshape(n, -1),
                           h.mean(axis=2).reshape(n, -1), (h[:, :, 1] - h[:, :, 0]).reshape(n, -1),
                           ll.reshape(n, -1), margin[:, None]], axis=1)
    assert feat.shape[1] == 639 and np.isfinite(feat).all()
    return feat, margin


class BELinear(nn.Module):
    """BatchEnsemble linear layer: shared weight W, per-member random-sign factors r (input) and s (output), bias b."""
    def __init__(self, din, dout, members):
        super().__init__()
        self.w = nn.Linear(din, dout, bias=False)
        self.r = nn.Parameter(torch.randint(0, 2, (members, din)).float() * 2 - 1)
        self.s = nn.Parameter(torch.randint(0, 2, (members, dout)).float() * 2 - 1)
        self.b = nn.Parameter(torch.zeros(members, dout))

    def forward(self, z):                                                   # [B, 1 or members, din] -> [B, members, dout]
        return self.w(z * self.r) * self.s + self.b


class Head(nn.Module):
    def __init__(self, vocab, n_cat, n_num, cfg):
        super().__init__()
        self.dropout = cfg['dropout']
        self.cat = nn.Embedding(vocab, cfg['emb'])
        nn.init.normal_(self.cat.weight, std=0.1)
        dim = n_cat * cfg['emb'] + n_num
        self.layers = nn.ModuleList([BELinear(dim if i == 0 else cfg['width'], cfg['width'], cfg['members'])
                                     for i in range(cfg['depth'])])
        self.hw = nn.Parameter(torch.zeros(cfg['members'], cfg['width']))
        self.hb = nn.Parameter(torch.zeros(cfg['members']))

    def forward(self, cats, nums, margin):
        z = torch.cat([self.cat(cats).flatten(1), nums], 1)[:, None]
        for layer in self.layers:
            z = F.dropout(F.silu(layer(z)), self.dropout, self.training)
        return margin[:, None] + (z * self.hw).sum(-1) + self.hb           # [B, members]


def fit_head(seed, cats, nums, margin, y, tr, va, te, vocab, cfg=CFG):
    """Train one head; returns the valid and test logits (mean of the members, then of the snapshots)."""
    torch.manual_seed(seed)
    model = Head(vocab, cats.shape[1], nums.shape[1], cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    tr_t = torch.tensor(tr, device='cuda')
    target = torch.tensor(y.astype(np.float32), device='cuda')
    preds = {'valid': [], 'test': []}
    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        order = tr_t[torch.randperm(len(tr_t), device='cuda')]
        lr = cfg['lr'] * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (ep - 1) / cfg['epochs'])))
        for group in opt.param_groups:
            group['lr'] = lr
        tot = 0.0
        for j in range(0, len(order), cfg['batch']):
            idx = order[j:j + cfg['batch']]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = model(cats[idx], nums[idx], margin[idx])
                loss = F.binary_cross_entropy_with_logits(z.float(), target[idx, None].expand_as(z))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'])
            opt.step()
            tot += float(loss.detach()) * len(idx)
        print(f'seed {seed} epoch {ep}: loss {tot / len(tr):.5f}', flush=True)
        if ep in cfg['snapshots']:
            model.eval()
            with torch.inference_mode():
                for split, rows in (('valid', va), ('test', te)):
                    out = []
                    for j in range(0, len(rows), 4096):
                        idx = torch.tensor(rows[j:j + 4096], device='cuda')
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            z = model(cats[idx], nums[idx], margin[idx])
                        out.append(z.float().mean(1).cpu().numpy().astype(np.float64))
                    preds[split].append(np.concatenate(out))
    return np.mean(preds['valid'], 0), np.mean(preds['test'], 0)


def align(p_valid, p_test):
    """Label-free per-fold alignment: standardize the logits by their test mean / std, scale 3."""
    zv, zt = logit(np.clip(p_valid, 1e-7, 1 - 1e-7)), logit(np.clip(p_test, 1e-7, 1 - 1e-7))
    mu, sd = zt.mean(), zt.std()
    return expit(3 * (zv - mu) / sd), expit(3 * (zt - mu) / sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    y = target()
    f = folds(y)
    cat_all = np.load(TRANSFER / 'cat.npy', mmap_mode='r')                 # rows: train, test, then extra rows (unused here)
    num_all = np.load(TRANSFER / 'num.npy', mmap_mode='r')
    vocab = int(json.loads((TRANSFER / 'meta.json').read_text())['vocab'])  # 7,601 codes
    torch.backends.cuda.matmul.allow_tf32 = True
    for k in fold_list(a.folds):
        if have_part(HERE, NAME, k):
            continue
        feat, margin = gpt_features(k)
        n = len(feat)                                                       # train + test rows
        x = np.concatenate([np.asarray(num_all[:n], np.float32), feat], 1)  # 53 + 639
        del feat
        tr, va, te = np.flatnonzero(f != k), np.flatnonzero(f == k), np.arange(len(y), n)
        xt = x[tr]
        mean, sd = xt.mean(0), np.maximum(xt.std(0), 1e-3)
        del xt
        nums = torch.tensor(np.clip((x - mean) / sd, -12, 12), device='cuda')
        del x
        cats = torch.tensor(np.asarray(cat_all[:n], np.int64), device='cuda')
        mg = torch.tensor(margin, device='cuda')
        logits = [fit_head(s + k, cats, nums, mg, y, tr, va, te, vocab) for s in SEEDS]
        zv, zt = np.mean([v for v, _ in logits], 0), np.mean([t for _, t in logits], 0)   # bag2: mean of the raw logits
        pv, pt = align(expit(zv), expit(zt))
        save_part(HERE, NAME, k, pv, pt)
        print(f'fold {k}: AUC {roc_auc_score(y[va], pv):.6f}', flush=True)
        del cats, nums, mg
        torch.cuda.empty_cache()
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
