#!/usr/bin/env python
"""Predictor-only joint bank (48 columns) used by the distillation students.

1. Masked-field density: per fold of the label-free KFold(5, seed 42) over all train + test rows (the 'fold' array of
   ../neural_contrasts/cache/stage6/masked_inputs.npz), a network predicts one field from the other, randomly masked
   fields (field embeddings scaled by count reliability, standardized numerics, mask indicators -> 384 -> 192 ->
   per-field heads initialized at the smoothed field prior). 65 epochs, batch 4096, AdamW lr 2e-3 wd 1e-3, cosine to 5%,
   clip 2; at each step one target field (step mod 13), half the rows get extra masks at a Uniform(0, 0.5) rate.
2. Joint completion for 6 field pairs (income or concern / anxiety with subsidy / anxiety / concern): the smaller
   field is generated first with both fields masked, then the other field conditional on each state of the first;
   the source law Phi(eta) of the original data is integrated over this approximate joint (expectation, shift vs the
   prior, variance), against the independent completion (difference, sequential KL, joint entropy) and the observed
   pair's joint log-likelihood lift vs the priors and vs independent completion.
Writes checkpoints/context_density_f{k}.pt, cache/joint_completion_f{k}.npz and cache/joint_completion_all.npy + .json.
Usage: python joint_bank.py [--folds 0,1,2,3,4]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.pairs import fold_list  # noqa: E402

CACHE = HERE/'cache'
CHECKPOINTS = HERE/'checkpoints'
MASKED = HERE.parent/'neural_contrasts'/'cache'/'stage6'/'masked_inputs.npz'
FIELDS = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
          'Number_of_Cars_Owned', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Subsidy_Available',
          'Range_Anxiety_Level', 'Environmental_Concern_Level']
CONFIG = dict(epochs=65, batch=4096, lr=.002, wd=.001, widths=[384, 192], embedding=16, dropout=.1, seed=712042,
              prior_strength=20.)
PAIRS = [(1, 10), (1, 11), (1, 12), (11, 10), (12, 10), (12, 11)]


def density_model(cards, priors, reliability):
    import torch
    from torch import nn

    class Density(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.ModuleList([nn.Embedding(n, 16) for n in cards])
            self.encoder = nn.Sequential(nn.Linear(234, 384), nn.GELU(), nn.Dropout(.1), nn.Linear(384, 192), nn.GELU())
            self.heads = nn.ModuleList([nn.Linear(192, n) for n in cards])
            for j, (head, p, r) in enumerate(zip(self.heads, priors, reliability)):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
                self.register_buffer('prior_'+str(j), p)
                self.register_buffer('reliability_'+str(j), r)

        def forward(self, x, num, field, mask):
            obs = (~mask).float()
            emb = [table(x[:, j])*getattr(self, 'reliability_'+str(j))[x[:, j], None]*obs[:, j, None]
                   for j, table in enumerate(self.embeddings)]
            h = self.encoder(torch.cat([*emb, num*obs, mask.float()], dim=1))
            return getattr(self, 'prior_'+str(field))+self.heads[field](h)
    return Density()


def train(k):
    import torch
    from torch.nn import functional as F
    path = CHECKPOINTS/f'context_density_f{k}.pt'
    if path.exists():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    seed = CONFIG['seed']+k
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    start = time.time()
    a = np.load(MASKED)
    cards = json.loads(MASKED.with_suffix('.json').read_text())['cards']
    X, num, fold = a['X'], a['numeric'], a['fold']
    fit, query = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    mean = num[fit].mean(axis=0, dtype='float64')
    scale = np.maximum(num[fit].std(axis=0, dtype='float64'), .001)
    norm = np.clip((num-mean)/scale, -10, 10).astype('float32')
    priors, reliability = [], []
    for j, ns in enumerate(cards):
        count = np.bincount(X[fit, j], minlength=ns)
        p = (count+20*a['original_'+str(j)])/(len(fit)+20)
        priors.append(torch.tensor(np.log(p), dtype=torch.float32, device='cuda'))
        reliability.append(torch.tensor(np.sqrt(count/(count+20)), dtype=torch.float32, device='cuda'))
    xt = torch.tensor(X[fit], device='cuda')
    nt = torch.tensor(norm[fit], device='cuda')
    model = density_model(cards, priors, reliability).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    batch = CONFIG['batch']
    steps = CONFIG['epochs']*int(np.ceil(len(fit)/batch))
    step = 0
    for epoch in range(CONFIG['epochs']):
        model.train()
        order = torch.randperm(len(fit), device='cuda')
        total = 0.
        for st in range(0, len(fit), batch):
            ix = order[st:st+batch]
            j = step % 13
            rate = torch.rand((len(ix), 1), device='cuda')*.5
            rate *= torch.rand((len(ix), 1), device='cuda') < .5
            mask = torch.rand((len(ix), 13), device='cuda') < rate
            mask[:, j] = True
            lr = .002*(.05+.95*.5*(1+np.cos(np.pi*step/(steps-1))))
            for g in opt.param_groups:
                g['lr'] = lr
            loss = F.cross_entropy(model(xt[ix], nt[ix], j, mask), xt[ix, j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
            step += 1
            total += float(loss.detach())*len(ix)
        if epoch % 10 == 0:
            print('context density', k, epoch, total/len(fit), round(time.time()-start), flush=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), mean=mean, scale=scale, config=CONFIG, cards=cards, fit=fit,
                    query=query, seed=seed), path)


def restore(k):
    import torch
    ck = torch.load(CHECKPOINTS/f'context_density_f{k}.pt', map_location='cpu', weights_only=False)
    s = ck['state_dict']
    model = density_model(ck['cards'], [s['prior_'+str(j)] for j in range(13)],
                          [s['reliability_'+str(j)] for j in range(13)]).cuda()
    model.load_state_dict(s)
    model.eval()
    return ck, model


def extract(k):
    import torch
    from torch.nn import functional as F
    out = CACHE/f'joint_completion_f{k}.npz'
    if out.exists():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    a = np.load(MASKED)
    ck, model = restore(k)
    query = ck['query']
    X = a['X'][query]
    num = np.clip((a['numeric'][query]-ck['mean'])/ck['scale'], -10, 10).astype('float32')
    xt = torch.tensor(X, device='cuda')
    nt = torch.tensor(num, device='cuda')
    eta = torch.tensor(a['eta'][query], device='cuda')
    blocks = {s: [] for s in ['joint_completion', 'joint_dependence', 'joint_likelihood']}
    names = {s: [] for s in blocks}
    with torch.no_grad():
        for j, c in PAIRS:
            ej = torch.tensor(a['effect_'+str(j)], device='cuda')
            ec = torch.tensor(a['effect_'+str(c)], device='cuda')
            pj0 = getattr(model, 'prior_'+str(j)).exp()
            pc0 = getattr(model, 'prior_'+str(c)).exp()
            vals = []
            suffix = FIELDS[j]+'__'+FIELDS[c]
            batch = 512 if ck['cards'][j] > 10000 else 4096
            for st in range(0, len(query), batch):
                xx = xt[st:st+batch]
                nnn = nt[st:st+batch]
                mask = torch.zeros_like(xx, dtype=torch.bool)
                mask[:, j] = True
                mask[:, c] = True
                pc = F.softmax(model(xx, nnn, c, mask), dim=1)
                pj_ind = F.softmax(model(xx, nnn, j, mask), dim=1)
                mask_j = mask.clone()
                mask_j[:, c] = False
                intercept = eta[st:st+len(xx)]-ej[xx[:, j]]-ec[xx[:, c]]
                mu = torch.zeros(len(xx), device='cuda')
                second, mu_ind, mu_prior = torch.zeros_like(mu), torch.zeros_like(mu), torch.zeros_like(mu)
                jointlp, mutual, cond_entropy = torch.zeros_like(mu), torch.zeros_like(mu), torch.zeros_like(mu)
                for state in range(ck['cards'][c]):
                    modified = xx.clone()
                    modified[:, c] = state
                    numeric = nnn.clone()
                    if 'values_'+str(c) in a:
                        numeric[:, c] = float(np.clip((a['values_'+str(c)][state]-ck['mean'][c])/ck['scale'][c], -10, 10))
                    lp = F.log_softmax(model(modified, numeric, j, mask_j), dim=1)
                    pj = lp.exp()
                    law = torch.special.ndtr(intercept[:, None]+ec[state]+ej)
                    conditional = (pj*law).sum(dim=1)
                    mu += pc[:, state]*conditional
                    second += pc[:, state]*(pj*law.square()).sum(dim=1)
                    mu_ind += pc[:, state]*(pj_ind*law).sum(dim=1)
                    mu_prior += pc0[state]*(law@pj0)
                    is_actual = (xx[:, c] == state).float()
                    jointlp += is_actual*(pc[:, state].log()+lp.gather(1, xx[:, j, None]).squeeze(1))
                    mutual += pc[:, state]*(pj*(lp-pj_ind.clamp_min(1e-30).log())).sum(dim=1)
                    cond_entropy += pc[:, state]*(-(pj*lp).sum(dim=1))
                var = (second-mu.square()).clamp_min(0)
                priorlp = pj0[xx[:, j]].log()+pc0[xx[:, c]].log()
                individual_lp = pj_ind.gather(1, xx[:, j, None]).squeeze(1).log()+pc.gather(1, xx[:, c, None]).squeeze(1).log()
                joint_h = cond_entropy-(pc*pc.clamp_min(1e-30).log()).sum(dim=1)
                vals.append(torch.stack([mu, mu-mu_prior, var, mu-mu_ind, mutual, joint_h, jointlp-priorlp,
                                         jointlp-individual_lp], dim=1).cpu().numpy())
            block = np.concatenate(vals)
            for pos, label in enumerate(['joint_source_expectation', 'joint_source_expectation_shift', 'joint_source_variance']):
                blocks['joint_completion'].append(block[:, pos])
                names['joint_completion'].append(label+'_'+suffix)
            for pos, label in [(3, 'joint_minus_independent_completion'), (4, 'sequential_dependence_kl'), (5, 'joint_entropy')]:
                blocks['joint_dependence'].append(block[:, pos])
                names['joint_dependence'].append(label+'_'+suffix)
            for pos, label in [(6, 'joint_observed_lift'), (7, 'joint_minus_independent_observed_lift')]:
                blocks['joint_likelihood'].append(block[:, pos])
                names['joint_likelihood'].append(label+'_'+suffix)
            print('joint completion', k, suffix, flush=True)
    features = np.column_stack([v for vals in blocks.values() for v in vals]).astype('float32')
    assert features.shape == (len(query), 48) and np.isfinite(features).all()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(out, index=query, features=features)
    out.with_suffix('.json').write_text(json.dumps(dict(names=[c for vals in names.values() for c in vals]), indent=2))


def assemble():
    names = [json.loads((CACHE/f'joint_completion_f{k}.json').read_text())['names'] for k in range(5)]
    n = len(np.load(MASKED)['fold'])
    out = np.empty((n, 48), 'float32')
    owner = np.zeros(n, 'int8')
    for k in range(5):
        assert names[k] == names[0]
        a = np.load(CACHE/f'joint_completion_f{k}.npz')
        out[a['index']] = a['features']
        owner[a['index']] += 1
    assert np.all(owner == 1)
    np.save(CACHE/'joint_completion_all.npy', out)
    (CACHE/'joint_completion_all.json').write_text(json.dumps(dict(names=names[0]), indent=2))
    print('joint bank', out.shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    for k in fold_list(a.folds):
        train(k)
        extract(k)
    if all((CACHE/f'joint_completion_f{k}.npz').exists() for k in range(5)):
        assemble()


if __name__ == '__main__':
    main()
