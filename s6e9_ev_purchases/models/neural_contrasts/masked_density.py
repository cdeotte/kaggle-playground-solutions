#!/usr/bin/env python
"""Cross-fitted masked-predictor density over all train + test predictors (bank masked_density of
masked_density_contrast). Neither the competition labels nor the original labels are read.

The 955,236 train + test predictor rows are split by KFold(5, shuffle=True, random_state=42) (independent of the
supervised folds). For each density fold k a network is trained on the other 4/5: 13 field embeddings (dim 16, scaled by
sqrt(n / (n + 20)) of the value's count), standardized numerics and a mask indicator -> 256-128 GELU encoder (dropout
0.1) -> one softmax head per field, initialized at the smoothed field prior (fit counts + 20 x original-data marginal).
Each step masks one field (step % 13) and predicts it from the others. AdamW lr 2e-3 (cosine to 5%), wd 0.001,
39 epochs, batch 4096, gradient clip 2, seed 612042 + k, TF32.
For the held-out rows, per field: the log lift of the observed value over the prior, entropy, KL to the prior, max
probability; for numeric fields the conditional mean / sd / standardized residual; for the 4 fields of the known source
rule (income, subsidy, anxiety, concern) the expected source probability under the conditional and its shift from
the prior: 81 columns.
Writes cache/stage6/masked_inputs.npz (+ .json), masked_density_f<k>.{pt,npz,json} and masked_density_all.npy (+ .json).
Not bit-reproducible: retraining gives a close, not identical, bank.
Usage: python masked_density.py --prepare
       python masked_density.py --fit [--folds 0,1,2,3,4]     (assembles the bank once all 5 folds exist)"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import read_original  # noqa: E402
from contrast_data import STAGE, STAGE1, ints  # noqa: E402

CACHE = STAGE[6]
FIELDS = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
          'Number_of_Cars_Owned', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Subsidy_Available',
          'Range_Anxiety_Level', 'Environmental_Concern_Level']
NUMERIC = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
           'Charging_Stations_Near_Work', 'Environmental_Concern_Level']
CORE = ['Annual_Income_USD', 'Subsidy_Available', 'Range_Anxiety_Level', 'Environmental_Concern_Level']
CONFIG = dict(epochs=39, batch=4096, lr=.002, wd=.001, embedding=16, widths=[256, 128], dropout=.1, density_split_seed=42,
              seed=612042, prior_strength=20.)


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(STAGE1 / 'predictors.parquet', columns=FIELDS)
    original = read_original()[FIELDS]
    x, numeric, states, cards = [], np.zeros((len(d), len(FIELDS)), dtype='float32'), {}, []
    eta = np.full(len(d), -5.5, dtype='float32')
    for j, field in enumerate(FIELDS):
        code, values = pd.factorize(d[field], sort=True)
        assert np.min(code) >= 0
        values = np.asarray(values)
        ns = len(values)
        cards.append(ns)
        x.append(code)
        original_code = pd.Categorical(original[field], categories=values).codes
        S = np.bincount(original_code[original_code >= 0], minlength=ns)
        states['original_' + str(j)] = (S + .5 / ns) / (S.sum() + .5)
        if field in NUMERIC:
            numeric[:, j] = d[field].to_numpy('float32')
            states['values_' + str(j)] = values.astype('float32')
        if field in CORE:
            if field == 'Annual_Income_USD':
                effect = values.astype('float32') * .000012
            elif field == 'Environmental_Concern_Level':
                effect = values.astype('float32') * .6
            elif field == 'Subsidy_Available':
                effect = (values == 'Yes').astype('float32') * 2
            else:
                effect = np.array([{'Low': 0., 'Medium': -1., 'High': -3.}[v] for v in values], dtype='float32')
            states['effect_' + str(j)] = effect
            eta += effect[code]
    X = np.column_stack(x).astype('int64')
    assert np.isfinite(numeric).all()
    fold = np.empty(len(d), 'int8')
    for k, (_, q) in enumerate(KFold(5, shuffle=True, random_state=42).split(X)):
        fold[q] = k
    np.savez(CACHE / 'masked_inputs.npz', X=X, numeric=numeric, eta=eta, fold=fold, **states)
    (CACHE / 'masked_inputs.json').write_text(json.dumps(dict(fields=FIELDS, numeric_fields=NUMERIC, core_fields=CORE, cards=cards,
                                                              config=CONFIG), indent=2))
    print('masked inputs', X.shape, cards, flush=True)


class Density(nn.Module):
    def __init__(self, cards, priors, reliability):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n, 16) for n in cards])
        self.encoder = nn.Sequential(nn.Linear(13 * 16 + 26, 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 128), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(128, n) for n in cards])
        for j, (head, p, r) in enumerate(zip(self.heads, priors, reliability)):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.register_buffer('prior_' + str(j), p)
            self.register_buffer('reliability_' + str(j), r)

    def forward(self, x, num, masked):
        embeddings = []
        for j, table in enumerate(self.embeddings):
            e = table(x[:, j]) * getattr(self, 'reliability_' + str(j))[x[:, j], None]
            if j == masked:
                e = e * 0
            embeddings.append(e)
        numeric = num.clone()
        numeric[:, masked] = 0
        indicator = torch.zeros_like(numeric)
        indicator[:, masked] = 1
        h = self.encoder(torch.cat([*embeddings, numeric, indicator], dim=1))
        return getattr(self, 'prior_' + str(masked)) + self.heads[masked](h)


def train(k):
    torch.backends.cuda.matmul.allow_tf32 = True
    seed = CONFIG['seed'] + k
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    start = time.time()
    bank = np.load(CACHE / 'masked_inputs.npz')
    meta = json.loads((CACHE / 'masked_inputs.json').read_text())
    X, num, fold = bank['X'], bank['numeric'], bank['fold']
    fit, query = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    cards = meta['cards']
    mean = num[fit].mean(axis=0, dtype='float64')
    scale = np.maximum(num[fit].std(axis=0, dtype='float64'), .001)
    num = np.clip((num - mean) / scale, -10, 10).astype('float32')
    xt = torch.tensor(X[fit], device='cuda')
    nt = torch.tensor(num[fit], device='cuda')
    priors, reliability = [], []
    for j, ns in enumerate(cards):
        C = np.bincount(X[fit, j], minlength=ns)
        p = (C + 20 * bank['original_' + str(j)]) / (len(fit) + 20)
        priors.append(torch.tensor(np.log(p), dtype=torch.float32, device='cuda'))
        reliability.append(torch.tensor(np.sqrt(C / (C + 20)), dtype=torch.float32, device='cuda'))
    model = Density(cards, priors, reliability).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    batch = CONFIG['batch']
    steps = CONFIG['epochs'] * int(np.ceil(len(fit) / batch))
    step = 0
    for epoch in range(CONFIG['epochs']):
        model.train()
        order = torch.randperm(len(fit), device='cuda')
        total = 0.
        for j in range(0, len(fit), batch):
            ix = order[j:j + batch]
            field = step % 13
            rate = .002 * (.05 + .95 * .5 * (1 + np.cos(np.pi * step / (steps - 1))))
            for g in opt.param_groups:
                g['lr'] = rate
            logits = model(xt[ix], nt[ix], field)
            loss = F.cross_entropy(logits, xt[ix, field])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
            step += 1
            total += float(loss.detach()) * len(ix)
        if epoch % 5 == 0:
            print(f'  density fold {k} epoch {epoch}: {total / len(fit):.6f} ({time.time() - start:.0f}s)', flush=True)
    torch.save(dict(state_dict=model.state_dict(), mean=mean, scale=scale, config=CONFIG, cards=cards, fields=FIELDS, fit=fit,
                    query=query, seed=seed), CACHE / f'masked_density_f{k}.pt')
    del xt, nt
    model.eval()
    xt = torch.tensor(X[query], device='cuda')
    nt = torch.tensor(num[query], device='cuda')
    eta = torch.tensor(bank['eta'][query], device='cuda')
    blocks = {c: [] for c in ['lift', 'shape', 'numeric', 'completion']}
    names = {c: [] for c in blocks}
    with torch.no_grad():
        for j, field in enumerate(FIELDS):
            values = []
            prior = getattr(model, 'prior_' + str(j))
            p0 = prior.exp()
            numeric_values = torch.tensor(bank['values_' + str(j)], device='cuda') if field in NUMERIC else None
            effect = torch.tensor(bank['effect_' + str(j)], device='cuda') if field in CORE else None
            batchq = 512 if cards[j] > 10000 else 4096
            for startq in range(0, len(query), batchq):
                xx = xt[startq:startq + batchq]
                lp = F.log_softmax(model(xx, nt[startq:startq + batchq], j), dim=1)
                p = lp.exp()
                observed = lp.gather(1, xx[:, j, None]).squeeze(1)
                lift = observed - prior[xx[:, j]]
                H = -(p * lp).sum(dim=1)
                KL = (p * (lp - prior)).sum(dim=1)
                pieces = [lift, H, KL, p.max(dim=1).values]
                if numeric_values is not None:
                    mu = p @ numeric_values
                    second = p @ numeric_values.square()
                    sd = (second - mu.square()).clamp_min(.001).sqrt()
                    actual = numeric_values[xx[:, j]]
                    pieces += [(mu - mean[j]) / scale[j], sd / scale[j], (actual - mu) / (sd + .01 * scale[j])]
                if effect is not None:
                    intercept = eta[startq:startq + len(xx)] - effect[xx[:, j]]
                    law = torch.special.ndtr(intercept[:, None] + effect)
                    expected = (p * law).sum(dim=1)
                    prior_expected = law @ p0
                    pieces += [expected, expected - prior_expected]
                values.append(torch.stack(pieces, dim=1).cpu().numpy())
            A = np.concatenate(values)
            blocks['lift'].append(A[:, 0])
            names['lift'].append('conditional_lift_' + field)
            for offset, label in enumerate(['entropy', 'kl_prior', 'max_probability'], 1):
                blocks['shape'].append(A[:, offset])
                names['shape'].append(label + '_' + field)
            position = 4
            if numeric_values is not None:
                for offset, label in enumerate(['conditional_mean', 'conditional_sd', 'standard_residual']):
                    blocks['numeric'].append(A[:, position + offset])
                    names['numeric'].append(label + '_' + field)
                position += 3
            if effect is not None:
                for offset, label in enumerate(['source_expectation', 'source_expectation_shift']):
                    blocks['completion'].append(A[:, position + offset])
                    names['completion'].append(label + '_' + field)
    features = np.column_stack([v for vals in blocks.values() for v in vals]).astype('float32')
    columns = [c for vals in names.values() for c in vals]
    assert features.shape == (len(query), 81) and np.isfinite(features).all()
    np.savez(CACHE / f'masked_density_f{k}.npz', index=query, features=features)
    (CACHE / f'masked_density_f{k}.json').write_text(json.dumps(dict(fold=k, seed=seed, names=columns, sections=names), indent=2))
    print(f'masked density fold {k}: {features.shape} ({time.time() - start:.0f}s)', flush=True)


def assemble():
    out = np.empty((len(np.load(CACHE / 'masked_inputs.npz')['fold']), 81), dtype='float32')
    owner = np.zeros(len(out), int)
    records = []
    for k in range(5):
        a = np.load(CACHE / f'masked_density_f{k}.npz')
        out[a['index']] = a['features']
        owner[a['index']] += 1
        records.append(json.loads((CACHE / f'masked_density_f{k}.json').read_text()))
    assert np.all(owner == 1) and all(r['names'] == records[0]['names'] for r in records)
    np.save(CACHE / 'masked_density_all.npy', out)
    (CACHE / 'masked_density_all.json').write_text(json.dumps(dict(names=records[0]['names'], sections=records[0]['sections'],
                                                                   config=CONFIG), indent=2))
    print('masked density bank', out.shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--fit', action='store_true')
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.fit:
        for k in ints(a.folds):
            if not (CACHE / f'masked_density_f{k}.npz').exists():
                train(k)
        if all((CACHE / f'masked_density_f{k}.npz').exists() for k in range(5)):
            assemble()


if __name__ == '__main__':
    main()
