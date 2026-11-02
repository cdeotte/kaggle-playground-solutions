#!/usr/bin/env python
"""Label-free multinomial log-bilinear factorization of income-value x context co-occurrences (bank df8_a5 of
fine_context_contrast).

For each of 13 contexts (12 fields + subsidy x anxiety x concern), the counts of (exact income value, context value) over all
train + test rows are modelled as softmax(log parent profile + e_v . W_context) per income value, with an 8-dim income
embedding e_v shared across contexts (parent = income // 1000 + leading BPE piece). Full-batch Adam lr 0.05 (cosine to 10%),
180 iterations, L2 penalty 5 (scaled by 1 / rows), gradient clip 10, seed 42. Per row: e_v, the log lift of the row's own
context value over the parent profile and the entropy of each fitted context distribution. 'null' fits the same model with
the contexts shuffled within parents.
Writes cache/stage5/df8_a5_{real,null}.npy and df8_a5_all.npy (+ .json names).
Usage: python density_factors.py"""
import time

import numpy as np
import pandas as pd

from contrast_data import STAGE, Stage1, within_parent_permutation, write_names

RANK, PENALTY = 8, 5.
FIELDS = ['Age', 'Daily_Commute_km', 'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Number_of_Cars_Owned',
          'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Subsidy_Available', 'Range_Anxiety_Level',
          'Environmental_Concern_Level']


def data():
    b = Stage1()
    d, keys = b.base, b.keys
    key = keys.Annual_Income_USD.to_numpy()
    ng = b.meta['key_sizes']['Annual_Income_USD']
    par = pd.factorize(pd.MultiIndex.from_arrays([keys.inc1000, keys.inc_bpe0]), sort=True)[0]
    pv = np.zeros(ng, 'int32')
    pv[key] = par
    contexts = {c: pd.factorize(keys[c], sort=True)[0] for c in FIELDS}
    s = d.Subsidy_Available.to_numpy('int32')
    a = d.Range_Anxiety_Level.to_numpy('int32')
    c = d.Environmental_Concern_Level.to_numpy('int32')
    contexts['sac'] = (s * 3 + a) * 5 + c - 1
    perm = within_parent_permutation(par, len(key))
    return key, par, pv, ng, contexts, perm


def train(mode, rank=RANK, penalty=PENALTY):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    start = time.time()
    key, par, pv, ng, contexts, perm = data()
    npg = par.max() + 1
    dev = 'cuda'
    pc = np.bincount(par, minlength=npg)
    counts, priors, cs = [], [], []
    for name, c in contexts.items():
        ns = int(c.max() + 1)
        cc = c if mode == 'real' else c[perm]
        cs.append(cc)
        counts.append(torch.tensor(np.bincount(key * ns + cc, minlength=ng * ns).reshape(ng, ns), device=dev, dtype=torch.float32))
        a = np.bincount(par * ns + c, minlength=npg * ns).reshape(npg, ns)
        pr = (a + .5 / ns) / (pc[:, None] + .5)
        priors.append(torch.tensor(np.log(pr[pv]), device=dev, dtype=torch.float32))
    e = torch.nn.Parameter(torch.randn(ng, rank, device=dev) * .02)
    heads = [torch.nn.Parameter(torch.randn(c.shape[1], rank, device=dev) * .02) for c in counts]
    opt = torch.optim.Adam([e, *heads], lr=.05)
    for iteration in range(180):
        rate = .05 * (.1 + .9 * .5 * (1 + np.cos(np.pi * iteration / 179)))
        for g in opt.param_groups:
            g['lr'] = rate
        opt.zero_grad(set_to_none=True)
        loss = penalty * (e.square().sum() + sum(w.square().sum() for w in heads)) / len(key)
        for C, P, W in zip(counts, priors, heads):
            loss -= ((P + e @ W.T).log_softmax(dim=1) * C).sum() / len(key)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([e, *heads], 10.)
        opt.step()
        if iteration % 30 == 0:
            print(f'  {mode} iteration {iteration}: {float(loss.detach()):.6f}', flush=True)
    scores, entropy = [], []
    with torch.no_grad():
        for C, P, W, cc in zip(counts, priors, heads, cs):
            lp = (P + e @ W.T).log_softmax(dim=1)
            p = lp.exp()
            lift = (lp - P).cpu().numpy()
            scores.append(lift[key, cc])
            entropy.append(-(p * lp).sum(dim=1).cpu().numpy()[key])
    emb = e.detach().cpu().numpy()
    Z = np.column_stack([emb[key], *scores, *entropy]).astype('float32')
    assert np.isfinite(Z).all()
    name = f'df{rank}_a{penalty:g}_{mode}'
    STAGE[5].mkdir(parents=True, exist_ok=True)
    np.save(STAGE[5] / f'{name}.npy', Z)
    write_names(STAGE[5] / f'{name}.npy', [f'{mode}_embed{j}' for j in range(rank)] + [f'{mode}_lift_{c}' for c in contexts]
                + [f'{mode}_entropy_{c}' for c in contexts])
    print(f'{name} ({time.time() - start:.0f}s)', flush=True)


def assemble(rank=RANK, penalty=PENALTY):
    import json
    prefix = f'df{rank}_a{penalty:g}'
    Z, names = [], []
    for mode in ('real', 'null'):
        path = STAGE[5] / f'{prefix}_{mode}.npy'
        Z.append(np.load(path))
        names += json.loads(path.with_suffix('.json').read_text())['names']
    np.save(STAGE[5] / f'{prefix}_all.npy', np.column_stack(Z))
    write_names(STAGE[5] / f'{prefix}_all.npy', names)


if __name__ == '__main__':
    for mode in ('real', 'null'):
        if not (STAGE[5] / f'df{RANK}_a{PENALTY:g}_{mode}.npy').exists():
            train(mode)
    assemble()
