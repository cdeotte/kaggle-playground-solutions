#!/usr/bin/env python
"""Semi-supervised parent-conditional income generator.

For each income parent (income // 1000, leading BPE piece) and class c, a distribution over the exact income values of
that parent: log f_c(v) = bias[v, c] (+ rank-r context term value[v, c] . net(context)), softmax over the parent's values.
The biases start at smoothed class counts. Training maximizes the marginal likelihood
sum_c q_c(row) f_c(income | parent, context): labelled fit rows have q = their label, every other row (held-out fold and
test rows, predictors only) has q = the income-excluding context teacher's calibrated probability (teacher logits shifted
per parent so their mean matches the fit-pool parent rate). AdamW lr 0.01 (cosine to 5%), wd 0.001, 5 epochs,
batch 2048, gradient clip 5, L2 on the context values (reg) and on the bias drift (0.02).
Per query row: features [log f_0, log f_1, entropy_0, entropy_1]; prediction = sigmoid(teacher logit + log f_1 - log f_0).
Recipes: gen0 (rank 0), gen4_reg (rank 4, reg 1). Seeds 42 and 43.

  outer runs (fit = outer-train rows, query = fold k + test):  runs/<recipe>_s<seed>_f<k>/predictions.npz
      gen0 is the base of income_mixture_contrast; gen0 + gen4_reg feed the generator bank
  nested runs (fit / query = inner folds of the outer-train rows): runs/nested_<recipe>_s<seed>_f<k>_i<j>/predictions.npz
  bank (--assemble): train rows = nested features, query rows = outer features, mean of the 2 seeds;
      [lp0, lp1, h0, h1, llr = lp1 - lp0, hdiff = h1 - h0] per recipe and generator_context_delta_* = gen4_reg - gen0
      -> generative_f<k>.npz (+ .json names) in cache/stage3 (folds 0, 4) / cache/stage4 (folds 1-3)
Usage: python income_generator.py [--variants gen0,gen4_reg] [--seeds 42,43] [--folds 0,1,2,3,4]
       python income_generator.py --nested
       python income_generator.py --assemble"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold

from contrast_data import (RUNS, STAGE1, TEACHERS, IncomeParents, Stage1, fold_bank_dir, ints, split_indices,
                           write_names)

RECIPES = {
    'gen0': dict(rank=0, reg=.1, epochs=5, unlab_weight=1.),
    'gen4_reg': dict(rank=4, reg=1., epochs=5, unlab_weight=1.),
}


def calibrated(raw, parent, pi):
    """Teacher logits shifted per parent so that their mean probability matches the parent rate pi; returns the LLR."""
    n = np.bincount(parent, minlength=len(pi))
    a = np.zeros(len(pi))
    for _ in range(12):
        q = expit(raw + a[parent])
        s = np.bincount(parent, weights=q, minlength=len(pi))
        h = np.bincount(parent, weights=q * (1 - q), minlength=len(pi))
        a += np.clip((pi * n - s) / np.maximum(h, 1e-3), -1, 1)
    return raw + a[parent] - logit(pi[parent])


def prior_state(b, p, fit):
    """Fit-pool counts / positives per income value and smoothed parent rates."""
    y = b.y[fit].astype('float64')
    ct = np.bincount(p.key[fit], minlength=p.ng)
    pos = np.bincount(p.key[fit], weights=y, minlength=p.ng)
    pc = np.bincount(p.parent[fit], minlength=p.npar)
    ps = np.bincount(p.parent[fit], weights=y, minlength=p.npar)
    pi = ((ps + 20 * y.mean()) / (pc + 20)).clip(1e-4, 1 - 1e-4)
    return ct, pos, pi


def context(b):
    """Standardized row context without the exact income: numerics, one-hot categoricals, the coarse income, the source-rule
    score at the coarse income, log commute and the anxiety score."""
    d = b.base
    out = []
    for col in b.meta['numeric']:
        if col != 'Annual_Income_USD':
            out.append(d[col].to_numpy())
    for col in b.meta['categorical']:
        for val in range(1, 4):
            out.append((d[col].to_numpy() == val).astype('float32'))
    coarse = (np.floor(d.Annual_Income_USD.to_numpy('float64') / 1000) * 1000 + 500)
    raw = pd.read_parquet(STAGE1 / 'predictors.parquet')
    score = (1.2 * coarse / 1e5 + .6 * raw.Environmental_Concern_Level.to_numpy() + 2 * raw.Subsidy_Available.eq('Yes').to_numpy()
             - raw.Range_Anxiety_Level.map({'Low': 0, 'Medium': 1, 'High': 3}).to_numpy() - 5.5)
    out += [coarse, score, np.log1p(d.Daily_Commute_km.to_numpy()), d.anxiety_score.to_numpy()]
    A = np.column_stack(out).astype('float32')
    mean = A.mean(axis=0, dtype='float64')
    scale = np.maximum(A.std(axis=0, dtype='float64'), .001)
    return np.clip((A - mean) / scale, -8, 8).astype('float32')


def fit_model(b, p, A, fit, query, td, cfg, seed):
    """Fits one generator on the label pool `fit` with teacher td (fit / unlab / raw_logit); returns pred, features of query."""
    import torch
    from torch import nn
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    assert np.array_equal(fit, td['fit']) and not np.intersect1d(fit, query).size
    t = time.time()
    count, pos, pi = prior_state(b, p, fit)
    u = td['unlab']
    llr = calibrated(td['raw_logit'], p.parent[u], pi)
    teacher_logit = llr + logit(pi[p.parent[u]])
    q = np.empty(len(b.base), dtype='float64')
    q[fit] = b.y[fit]
    q[u] = expit(teacher_logit)
    logq = np.column_stack([np.log((1 - q).clip(1e-30)), np.log(q.clip(1e-30))]).astype('float32')
    nv = np.bincount(p.key, minlength=p.ng)
    active = nv > 0
    size = np.bincount(p.pv[active], minlength=p.npar)
    maxn = int(size.max())
    # distinct masked padding ids (one per padded slot); padding never affects a score
    lookup = np.arange(p.npar * maxn, dtype='int64').reshape(p.npar, maxn) + p.ng
    mask = np.zeros((p.npar, maxn), bool)
    local = np.zeros(p.ng, 'int64')
    for parent in range(p.npar):
        ix = np.flatnonzero(active & (p.pv == parent))
        lookup[parent, :len(ix)] = ix
        mask[parent, :len(ix)] = True
        local[ix] = np.arange(len(ix))
    prob = ((pos + 20 * pi[p.pv]) / (count + 20)).clip(1e-4, 1 - 1e-4)
    f = np.column_stack([nv * (1 - prob), nv * prob]).astype('float64') + .01
    for c in range(2):
        f[:, c] /= np.bincount(p.pv, weights=f[:, c], minlength=p.npar)[p.pv]
    init = np.vstack([np.log(f).astype('float32'), np.zeros((p.npar * maxn, 2), 'float32')])
    device = 'cuda'
    rank = cfg['rank']

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(init, device=device))
            if rank:
                self.value = nn.Parameter(torch.cat([torch.randn(p.ng, 2, rank, device=device) * .01,
                                                     torch.zeros(p.npar * maxn, 2, rank, device=device)], dim=0))
                self.net = nn.Sequential(nn.Linear(A.shape[1], 64), nn.GELU(), nn.Linear(64, rank), nn.Tanh()).to(device)

        def forward(self, x, ids, ok):
            score = self.bias[ids]
            regularizer = score.new_zeros(())
            if rank:
                h = self.net(x)
                e = self.value[ids]
                score = score + torch.einsum('blcr,br->blc', e, h)
                regularizer = cfg['reg'] * ((e * e) * ok[:, :, None, None]).sum() / (ok.sum() * 2 * rank)
            score = score.masked_fill(~ok[:, :, None], -1e9)
            return torch.log_softmax(score, dim=1), regularizer

    model = Generator()
    opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.001)
    xt = torch.tensor(A, device=device)
    pt = torch.tensor(p.parent, device=device)
    loc = torch.tensor(local[p.key], device=device)
    lut = torch.tensor(lookup, device=device)
    okt = torch.tensor(mask, device=device)
    qlog = torch.tensor(logq, device=device)
    rowweight = np.full(len(A), cfg['unlab_weight'], 'float32')
    rowweight[fit] = 1.
    wt = torch.tensor(rowweight, device=device)
    usable = torch.tensor(np.flatnonzero(rowweight > 0), device=device)
    batch = 2048
    steps = cfg['epochs'] * int(np.ceil(len(usable) / batch))
    step = 0
    initial = torch.tensor(init, device=device)
    for epoch in range(cfg['epochs']):
        model.train()
        order = usable[torch.randperm(len(usable), device=device)]
        total = denom = 0.
        for start in range(0, len(order), batch):
            ix = order[start:start + batch]
            parents = pt[ix]
            ids = lut[parents]
            ok = okt[parents]
            lprob, reg = model(xt[ix], ids, ok)
            picked = lprob[torch.arange(len(ix), device=device), loc[ix]]
            nll = -torch.logsumexp(qlog[ix] + picked, dim=1)
            w = wt[ix]
            biasreg = .02 * ((model.bias[ids] - initial[ids]) ** 2 * ok[:, :, None]).sum() / (ok.sum() * 2)
            loss = (nll * w).sum() / w.sum() + reg + biasreg
            lr = .01 * (.05 + .95 * .5 * (1 + np.cos(np.pi * step / max(steps - 1, 1))))
            for group in opt.param_groups:
                group['lr'] = lr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            step += 1
            total += float((nll * w).sum().detach())
            denom += float(w.sum())
        print(f'  epoch {epoch + 1}: nll {total / denom:.6f} ({time.time() - t:.0f}s)', flush=True)
    model.eval()
    features = []
    with torch.no_grad():
        for start in range(0, len(query), 4096):
            ix = torch.tensor(query[start:start + 4096], device=device)
            parents = pt[ix]
            lp, _ = model(xt[ix], lut[parents], okt[parents])
            picked = lp[torch.arange(len(ix), device=device), loc[ix]]
            entropy = -(lp.exp() * lp).sum(dim=1)
            features.append(torch.column_stack([picked, entropy]).cpu().numpy())
    feat = np.concatenate(features)
    qpos = np.searchsorted(u, query)
    assert np.array_equal(u[qpos], query)
    pred = expit(teacher_logit[qpos] + feat[:, 1] - feat[:, 0])
    return pred, feat


def outer_run(b, p, A, k, variant, seed):
    run = RUNS / f'{variant}_s{seed}_f{k}'
    if (run / 'predictions.npz').exists():
        return
    print(f'{run.name}', flush=True)
    fit, valid, query = split_indices(k)
    pred, feat = fit_model(b, p, A, fit, query, np.load(TEACHERS / f'income_f{k}_outer.npz'), RECIPES[variant], seed)
    run.mkdir(parents=True, exist_ok=True)
    np.savez(run / 'predictions.npz', index=query, prediction=pred, features=feat)


def nested_runs(b, p, A, k, variants, seeds):
    outer = split_indices(k)[0]
    for j, (a, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(outer, b.y[outer])):
        fit, query = outer[a], outer[v]
        td = np.load(TEACHERS / f'income_f{k}_inner{j}.npz')
        for variant in variants:
            for seed in seeds:
                run = RUNS / f'nested_{variant}_s{seed}_f{k}_i{j}'
                if (run / 'predictions.npz').exists():
                    continue
                print(f'{run.name}', flush=True)
                pred, feat = fit_model(b, p, A, fit, query, td, RECIPES[variant], seed)
                run.mkdir(parents=True, exist_ok=True)
                np.savez(run / 'predictions.npz', index=query, prediction=pred, features=feat)


def assemble(k, seeds):
    fit, valid, query = split_indices(k)
    allx, allq, names = [], [], []
    for variant in ('gen0', 'gen4_reg'):
        X = np.zeros((len(fit), 4))
        owner = np.zeros(len(fit), 'int8')
        for j in range(5):
            vals = []
            for seed in seeds:
                d = np.load(RUNS / f'nested_{variant}_s{seed}_f{k}_i{j}' / 'predictions.npz')
                index = d['index']
                vals.append(d['features'])
            loc = np.searchsorted(fit, index)
            assert np.array_equal(fit[loc], index)
            X[loc] = np.mean(vals, axis=0)
            owner[loc] += 1
        assert np.all(owner == 1)
        qq = []
        for seed in seeds:
            d = np.load(RUNS / f'{variant}_s{seed}_f{k}' / 'predictions.npz')
            assert np.array_equal(query, d['index'])
            qq.append(d['features'])
        Q = np.mean(qq, axis=0)

        def expand(a):
            return np.column_stack([a, a[:, 1] - a[:, 0], a[:, 3] - a[:, 2]])
        allx.append(expand(X))
        allq.append(expand(Q))
        names += [f'{variant}_{n}' for n in ('lp0', 'lp1', 'h0', 'h1', 'llr', 'hdiff')]
    X = np.column_stack(allx)
    Q = np.column_stack(allq)
    for feature in ('llr', 'h0', 'h1', 'hdiff'):
        a, c = names.index('gen4_reg_' + feature), names.index('gen0_' + feature)
        X = np.column_stack([X, X[:, a] - X[:, c]])
        Q = np.column_stack([Q, Q[:, a] - Q[:, c]])
        names.append('generator_context_delta_' + feature)
    path = fold_bank_dir(k) / f'generative_f{k}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, fit=fit, query=query, train_features=X.astype('float32'), query_features=Q.astype('float32'))
    write_names(path, names, seeds=list(seeds))
    print(f'generator bank fold {k}: {X.shape}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', default='gen0,gen4_reg')
    ap.add_argument('--seeds', default='42,43')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--nested', action='store_true', help='inner-fold runs of the generator bank')
    ap.add_argument('--assemble', action='store_true', help='build generative_f<k>.npz from the outer and nested runs')
    a = ap.parse_args()
    variants, seeds, folds = a.variants.split(','), ints(a.seeds), ints(a.folds)
    if a.assemble:
        for k in folds:
            assemble(k, seeds)
        return
    b = Stage1()
    p = IncomeParents(b)
    A = context(b)
    for k in folds:
        if a.nested:
            nested_runs(b, p, A, k, variants, seeds)
        else:
            for variant in variants:
                for seed in seeds:
                    outer_run(b, p, A, k, variant, seed)


if __name__ == '__main__':
    main()
