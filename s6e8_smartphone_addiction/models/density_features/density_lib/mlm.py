"""Masked lattice language model (MLM) over the 12 exact-value token columns, trained on train + test without labels.

Vocabulary of each column: 0 = [NA] (a real missing value), 1 = [MASK], 2.. = the exact tokens of train + test
(tok_{tr,te}.parquet) in sorted order. A row is a sequence of 12 column tokens: per-column token embedding + a learned
column vector, a pre-norm TransformerEncoder (8 heads, feed-forward 4d, dropout 0.1, GELU), a final LayerNorm and one
linear head per column (optionally an extra label token for the class-conditional model).
Training: each observed cell is replaced by [MASK] with probability mask_p (at least one per row); loss = cross-entropy of
the masked cells only ([NA] cells are never targets); AdamW (lr 1e-3, weight decay 1e-5), OneCycle (10% warm-up), batch
4096, gradient clipping 1.0, bf16 autocast."""
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .base import CACHE, FEATS

DEVICE = 'cuda'
NCOL = len(FEATS)


def load_tokens():
    return pd.read_parquet(CACHE / 'tok_tr.parquet'), pd.read_parquet(CACHE / 'tok_te.parquet')


def build_vocab(tokens):
    vocabs, ids = {}, np.zeros((len(tokens), NCOL), np.int64)
    for j, c in enumerate(FEATS):
        vals = sorted(v for v in tokens[c].unique() if v != '__NA__')
        mp = {v: i + 2 for i, v in enumerate(vals)}
        vocabs[c] = len(vals) + 2
        ids[:, j] = tokens[c].map(mp).fillna(0).astype(np.int64).values
    return vocabs, ids


def load_vocab():
    """-> tok_tr, tok_te, vocabs {column: vocabulary size}, ids [n_train + n_test, 12] token ids."""
    tok_tr, tok_te = load_tokens()
    vocabs, ids = build_vocab(pd.concat([tok_tr, tok_te], ignore_index=True))
    return tok_tr, tok_te, vocabs, ids


def token_values(tok_tr, tok_te, c):
    """Numeric value of every vocabulary id of numeric column c ([NA] and [MASK] -> 0)."""
    vals = sorted(v for v in pd.concat([tok_tr[c], tok_te[c]]).unique() if v != '__NA__')
    return np.array([0.0, 0.0] + [float(x) for x in vals])


class LatticeMLM(nn.Module):
    def __init__(self, vocabs, d=192, layers=4, heads=8, drop=0.1, with_y=False):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(vocabs[c], d) for c in FEATS])
        self.col = nn.Parameter(torch.randn(1, NCOL + (1 if with_y else 0), d) * 0.02)
        self.with_y = with_y
        if with_y:
            self.yemb = nn.Embedding(2, d)
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, drop, activation='gelu', batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleList([nn.Linear(d, vocabs[c]) for c in FEATS])

    def forward(self, x, yy=None):
        t = torch.stack([self.emb[j](x[:, j]) for j in range(NCOL)], 1)
        if self.with_y:
            t = torch.cat([t, self.yemb(yy).unsqueeze(1)], 1)
        h = self.norm(self.tr(t + self.col))
        return h[:, :NCOL]


def train_model(ids, vocabs, rows, d, layers, epochs, seed, mask_p=0.30, y_ctx=None, device=DEVICE, t0=None):
    t0 = time.time() if t0 is None else t0
    model = LatticeMLM(vocabs, d=d, layers=layers, with_y=(y_ctx is not None)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bs = 4096
    nstep = math.ceil(len(rows) / bs) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=nstep, pct_start=0.1)
    IDS = torch.from_numpy(ids)
    X = IDS[rows].to(device)
    O = (IDS != 0)[rows].to(device)
    Y = torch.from_numpy(y_ctx[rows].astype(np.int64)).to(device) if y_ctx is not None else None
    g = torch.Generator(device=device).manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(rows), generator=g, device=device)
        tot, nb = 0.0, 0
        for i in range(0, len(rows), bs):
            sl = perm[i:i + bs]
            xb, ob = X[sl].clone(), O[sl]
            mask = (torch.rand(xb.shape, device=device, generator=g) < mask_p) & ob
            none = mask.sum(1) == 0
            if none.any():   # force at least one masked observed cell per row
                first_obs = ob[none].float().argmax(1)
                mask[none, first_obs] = ob[none, first_obs]
            tgt = xb.clone()
            xb[mask] = 1   # [MASK]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = model(xb, Y[sl] if Y is not None else None)
                loss = 0.0; cnt = 0
                for j in range(NCOL):
                    mj = mask[:, j]
                    if mj.any():
                        loss = loss + nn.functional.cross_entropy(model.heads[j](h[mj, j]).float(), tgt[mj, j],
                                                                  reduction='sum')
                        cnt += int(mj.sum())
                loss = loss / max(cnt, 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss); nb += 1
        print(f'  ep{ep} loss {tot / nb:.4f} ({time.time() - t0:.0f}s)', flush=True)
    return model


@torch.no_grad()
def extract(model, ids, y_ctx_val=None, rows=None, bs=16384, device=DEVICE):
    """Per-cell log p(x_c | rest) and predictive entropy of the observed cells (each cell masked in turn; NaN where the
    cell is missing). y_ctx_val: label token for the class-conditional model."""
    rows = np.arange(len(ids)) if rows is None else rows
    LP = np.full((len(rows), NCOL), np.nan, np.float32)
    ENT = np.full((len(rows), NCOL), np.nan, np.float32)
    model.eval()
    IDS = torch.from_numpy(ids)
    X = IDS[rows]; O = (IDS != 0)[rows]
    for j in range(NCOL):
        for i in range(0, len(rows), bs):
            xb = X[i:i + bs].clone().to(device)
            ob = O[i:i + bs, j].numpy()
            tgt = xb[:, j].clone()
            xb[:, j] = 1
            yy = None
            if y_ctx_val is not None:
                yy = torch.full((len(xb),), y_ctx_val, dtype=torch.long, device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = model(xb, yy)
                logits = model.heads[j](h[:, j]).float()
            ls = torch.log_softmax(logits, -1)
            lp = ls.gather(1, tgt.unsqueeze(1).to(device)).squeeze(1).cpu().numpy()
            p = ls.exp()
            ent = (-(p * ls).sum(1)).cpu().numpy()
            LP[i:i + bs, j] = np.where(ob, lp, np.nan)
            ENT[i:i + bs, j] = np.where(ob, ent, np.nan)
    return LP, ENT


def load_mlm(checkpoint, vocabs, d, layers, device=DEVICE):
    """A trained unconditional MLM from the cache (checkpoint basename), in eval mode."""
    model = LatticeMLM(vocabs, d=d, layers=layers).to(device)
    model.load_state_dict(torch.load(CACHE / checkpoint, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def cell_logits(model, X_all, r, j, device=DEVICE):
    """Logits of column j for rows r with that cell set to [MASK]; the [NA] / [MASK] classes are blocked (-1e9)."""
    xb = X_all[r].clone().to(device)
    xb[:, j] = 1
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h = model(xb)
        logits = model.heads[j](h[:, j]).float()
    logits[:, :2] = -1e9
    return logits
