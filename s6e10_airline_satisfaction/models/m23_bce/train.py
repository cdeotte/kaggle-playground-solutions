#!/usr/bin/env python
"""M23 + BCE: a label-conditional masked-field transformer, then a short BCE continuation. Two stages per outer fold k.

A row is 23 tokens: the 21 features, `label` and `source` (Kaggle / original). Every distinct value of a feature column is its
own token (sorted factorization over train + test + original; numbers are categorical identities, NaN -> -999 is one more
value). Token 0 = CLS, 1 = MASK, field tokens start at 2.
1. Parent (checkpoints/parent_f{k}.pt). Rows: Kaggle train (labels known, except fold k whose labels are hidden), Kaggle test
   (label unknown) and the original data.csv rows (labels known). Masked-field modelling: every field is masked with
   probability 0.25 and predicted, except the label: a known label is masked with probability 0.5, hidden / unknown labels
   are always masked and never scored.
   Pre-norm transformer L6, d256, 8 heads, FFN 1024, 15 epochs, batch 1024, AdamW lr 5e-4 (betas 0.9 / 0.95, wd 0.01),
   OneCycle (5% warmup, cosine), bf16. Its zero-shot readout is P(label | x, source = Kaggle) at the masked label position.
2. BCE continuation of the fold-k parent on the outer-train rows: label input always MASK, source = Kaggle, BCE on the
   difference of the two label-token logits, 10% feature masking, 4 epochs, batch 512, AdamW lr 5e-5 decaying linearly to 0,
   wd 0.01. Prediction = mean of the probabilities after epochs 3 and 4.
Pair `m23_bce`: probabilities; test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds, labels, read_original, read_test, read_train, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

NAME = 'm23_bce'
CKPT = HERE / 'checkpoints'
FEATS = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
         'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service',
         'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness', 'Departure Delay in Minutes',
         'Arrival Delay in Minutes', 'Gender', 'Customer Type', 'Type of Travel', 'Class']      # train.csv order
CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
CATMAP = {'Female': 0, 'Male': 1, 'Loyal Customer': 0, 'disloyal Customer': 1, 'Business travel': 0, 'Personal Travel': 1,
          'Business': 0, 'Eco': 1, 'Eco Plus': 2}
NF = len(FEATS)
LPOS, SPOS, MASK = NF, NF + 1, 1                          # label field, source field, MASK token
DIM, LAYERS, HEADS = 256, 6, 8
P_EPOCHS, P_BATCH, P_LR, P_MASK, P_LABEL_MASK = 15, 1024, 5e-4, 0.25, 0.5   # parent
EPOCHS, BATCH, LR, MASK_P, WD = 4, 512, 5e-5, 0.10, 0.01                    # BCE continuation


class Enc(nn.Module):
    """CLS + 23 field tokens + learned field embeddings -> pre-norm transformer -> LayerNorm (one output per field);
    `mlm` maps an output to logits over the whole vocabulary."""

    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, DIM)
        self.field = nn.Parameter(torch.randn(1, NF + 3, DIM) * 0.02)
        layer = nn.TransformerEncoderLayer(DIM, HEADS, 4 * DIM, dropout=0.1, batch_first=True, norm_first=True,
                                           activation='gelu')
        self.blocks = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(DIM)
        self.mlm = nn.Linear(DIM, vocab)

    def forward(self, ids):
        cls = torch.zeros(ids.shape[0], 1, dtype=torch.long, device=ids.device)
        h = self.tok(torch.cat([cls, ids], 1)) + self.field
        return self.ln(self.blocks(h))[:, 1:]


def build_codes(y):
    """Token ids [rows, 23] of train, test and original rows (in that order) with their label / source tokens.
    Returns codes, label_known, first label token, vocabulary size (4930 with the competition data), number of Kaggle rows."""
    tr, te, og = read_train(), read_test(), read_original()
    allv = pd.concat([d[FEATS] for d in (tr, te, og)], ignore_index=True)
    for c in CATS:
        allv[c] = allv[c].map(CATMAP)
    codes = np.zeros((len(allv), NF + 2), np.int64)
    off = 2
    for j, c in enumerate(FEATS):
        code, uniq = pd.factorize(allv[c].fillna(-999), sort=True)
        codes[:, j] = code + off
        off += len(uniq)
    lab0, vocab = off, off + 4                 # label tokens lab0 (0) / lab0 + 1 (1); source tokens lab0 + 2 / lab0 + 3
    ntr, nk = len(tr), len(tr) + len(te)
    codes[:ntr, LPOS] = lab0 + y.astype(np.int64)
    codes[ntr:nk, LPOS] = MASK                                         # test: label unknown
    codes[nk:, LPOS] = lab0 + labels(og[TARGET]).astype(np.int64)
    codes[:nk, SPOS] = lab0 + 2                                        # Kaggle
    codes[nk:, SPOS] = lab0 + 3                                        # original
    known = np.ones(len(codes), bool)
    known[ntr:nk] = False
    print(f'rows {len(codes)} | vocab {vocab}', flush=True)
    return codes, known, lab0, vocab, nk


def train_parent(k, codes, known, lab0, vocab, y, f):
    """Label-conditional masked-field model with fold k's labels hidden -> checkpoints/parent_f{k}.pt."""
    va = np.flatnonzero(f == k)
    cd, kn = codes.copy(), known.copy()
    cd[va, LPOS] = MASK
    kn[va] = False
    X, KN = torch.as_tensor(cd, device='cuda'), torch.as_tensor(kn, device='cuda')
    model = Enc(vocab).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=P_LR, weight_decay=0.01, betas=(0.9, 0.95))
    steps = P_EPOCHS * math.ceil(len(X) / P_BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=P_LR, total_steps=steps, pct_start=0.05, anneal_strategy='cos')
    model.train()
    for ep in range(P_EPOCHS):
        perm = torch.randperm(len(X), device='cuda')
        for s0 in range(0, len(perm), P_BATCH):
            idx = perm[s0:s0 + P_BATCH]
            ids, kb = X[idx], KN[idx]
            mask = torch.rand(ids.shape, device='cuda') < P_MASK
            mask[:, LPOS] = torch.rand(len(ids), device='cuda') < P_LABEL_MASK
            mask[:, LPOS] |= ~kb                      # unknown labels: always masked in the input ...
            loss_mask = mask.clone()
            loss_mask[:, LPOS] &= kb                  # ... and never scored
            inp = ids.masked_fill(mask, MASK)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = model(inp)
                loss = F.cross_entropy(model.mlm(h[loss_mask]).float(), ids[loss_mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        print(f'parent fold {k}: epoch {ep + 1}/{P_EPOCHS} masked-field CE {loss.item():.4f}', flush=True)
    model.eval()                                      # zero-shot readout on fold k (printed only)
    out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for s0 in range(0, len(va), 8192):
            h = model(X[torch.as_tensor(va[s0:s0 + 8192], device='cuda')])[:, LPOS]
            lg = model.mlm(h.float())[:, lab0:lab0 + 2].float()
            out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
    print(f'parent fold {k}: zero-shot AUC {roc_auc_score(y[va], np.concatenate(out)):.6f}', flush=True)
    CKPT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CKPT / f'parent_f{k}.pt')


def train_bce(k, xk, yt, lab0, vocab, f):
    """BCE continuation of the fold-k parent on the outer-train rows; returns fold-k and test probabilities."""
    tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
    te = np.arange(len(f), len(xk))
    torch.manual_seed(21000 + k)
    model = Enc(vocab).cuda()
    model.load_state_dict(torch.load(CKPT / f'parent_f{k}.pt', map_location='cpu', weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total = EPOCHS * math.ceil(len(tr) / BATCH)

    def z(ids, training):
        """Label-token logit difference at the masked label position (bf16 in training, fp16 in prediction)."""
        with torch.autocast('cuda', dtype=torch.bfloat16 if training else torch.float16):
            h = model(ids)[:, LPOS]
        w, b = model.mlm.weight, model.mlm.bias
        return h.float() @ (w[lab0 + 1] - w[lab0]).float() + (b[lab0 + 1] - b[lab0]).float()

    @torch.inference_mode()
    def predict(rows):
        model.eval()
        out = [z(xk[torch.as_tensor(rows[s:s + 4096], device='cuda')], False).sigmoid().cpu().numpy()
               for s in range(0, len(rows), 4096)]
        return np.concatenate(out).astype(np.float64)

    trt = torch.as_tensor(tr, device='cuda')
    pv, pt, step = np.zeros(len(va)), np.zeros(len(te)), 0
    for ep in range(EPOCHS):
        model.train()
        order = trt[torch.randperm(len(trt), device='cuda')]
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            ids = xk[idx]                                                   # a copy (advanced indexing)
            ids[:, :NF].masked_fill_(torch.rand(len(ids), NF, device='cuda') < MASK_P, MASK)   # features only
            loss = F.binary_cross_entropy_with_logits(z(ids, True), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            for g in opt.param_groups:
                g['lr'] = LR * (1 - step / total)
        print(f'fold {k}: epoch {ep + 1}/{EPOCHS} BCE {loss.item():.4f}', flush=True)
        if ep >= 2:                                                         # epochs 3 and 4
            pv += predict(va) / 2
            pt += predict(te) / 2
    return pv, pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    y = target()
    f = folds(y)
    todo = [k for k in fold_list(a.folds) if not have_part(HERE, NAME, k)]
    if todo:
        codes, known, lab0, vocab, nk = build_codes(y)
        parents = [k for k in todo if not (CKPT / f'parent_f{k}.pt').exists()]
        if parents:
            torch.manual_seed(0)                       # one seed for the parent stage, folds in order
            for k in parents:
                train_parent(k, codes, known, lab0, vocab, y, f)
                torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        xk = torch.as_tensor(codes[:nk], device='cuda')                     # Kaggle rows: train, then test
        xk[:, LPOS] = MASK                                                  # label input always MASK
        yt = torch.as_tensor(y.astype(np.float32), device='cuda')
        for k in todo:
            pv, pt = train_bce(k, xk, yt, lab0, vocab, f)
            save_part(HERE, NAME, k, pv, pt)
            print(f'fold {k}: AUC {roc_auc_score(y[f == k], pv):.6f}', flush=True)
            torch.cuda.empty_cache()
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
