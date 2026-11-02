#!/usr/bin/env python
"""Replica prior pack: a label-free per-income prior from GReaT replicas (GPT-2 small trained on the original data only).

For every distinct train / test income v and every replica checkpoint r:
  LLR_r(v) = log p_r(' <v>,' | 'Will_Buy_EV is Yes, Annual_Income_USD is') - log p_r(' <v>,' | 'Will_Buy_EV is No, ...'),
at temperature 1 and 0.7 (the generator's temperature). Columns, mapped to every train / test row by its income:
  rep_llr_mean_T07   mean over the replicas of LLR at T = 0.7
  rep_llr_mean_T1    mean over the replicas of LLR at T = 1
  rep_llr_sd_T07     standard deviation over the replicas at T = 0.7
  rep_llr_D60_T07    the d60 replica alone at T = 0.7
Replicas: 8 checkpoints from 4 independent training streams (REPLICAS below), read from ../gpt2_replica/checkpoints/<name>.pt;
missing checkpoints are skipped (d60 is required).
Writes cache/replica_llr.parquet (per income), cache/replica_prior.npy (train rows then test rows) + replica_prior_names.json.
Usage: python replica_prior.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.bpe import tokenizer  # noqa: E402
from common.data import read_test, read_train  # noqa: E402
from common.gpt_rows import REPLICA_DIR, load_replica  # noqa: E402

CACHE = HERE / 'cache'
REPLICAS = ['e60', 'e70',                                   # stream A: 60- and 70-epoch schedules, final epoch
            'b_e55', 'b_tf32_e50', 'b_tf32_e200_ep60',      # stream B: 55 / 50-epoch schedules, epoch 60 of a 200-epoch one
            'c_e200_ep50',                                  # stream C: another training seed, epoch 50 of a 200-epoch schedule
            'd60', 'd_e200_ep70']                           # stream D: 60-epoch schedule, epoch 70 of a 200-epoch one
NAMES = ['rep_llr_mean_T07', 'rep_llr_mean_T1', 'rep_llr_sd_T07', 'rep_llr_D60_T07']
BATCH = 2048


def replica_llr(uniq):
    tok = tokenizer()
    conts = [tok.encode(' ' + str(float(v)) + ',') for v in uniq]
    pre = {y: tok.encode(f'Will_Buy_EV is {y}, Annual_Income_USD is') for y in ('Yes', 'No')}
    out = pd.DataFrame(index=uniq)
    for name in REPLICAS:
        if not (REPLICA_DIR / f'{name}.pt').exists():
            print(f'skipping {name}: no checkpoint in {REPLICA_DIR}', flush=True)
            continue
        model = load_replica(name).cuda().eval()
        res = {}
        for y, p in pre.items():
            P = len(p)
            for T in (1.0, 0.7):
                res[(y, T)] = np.zeros(len(uniq))
            for s in range(0, len(uniq), BATCH):
                cs = conts[s:s + BATCH]
                C = max(len(c) for c in cs)
                L = P + C
                ids = torch.full((len(cs), L), tok.eos_token_id)
                att = torch.zeros((len(cs), L), dtype=torch.long)
                tgt = torch.zeros((len(cs), C), dtype=torch.long)
                msk = torch.zeros((len(cs), C))
                for i, c in enumerate(cs):
                    ids[i, :P] = torch.tensor(p)
                    ids[i, P:P + len(c)] = torch.tensor(c)
                    att[i, :P + len(c)] = 1
                    tgt[i, :len(c)] = torch.tensor(c)
                    msk[i, :len(c)] = 1
                with torch.no_grad():
                    lo = model(ids.cuda(), attention_mask=att.cuda()).logits[:, P - 1:L - 1, :].float()  # continuation tokens
                    tg, mk = tgt.cuda(), msk.cuda()
                    for T in (1.0, 0.7):
                        lp = torch.log_softmax(lo / T, -1).gather(-1, tg[..., None])[..., 0]
                        res[(y, T)][s:s + len(cs)] = (lp * mk).sum(1).cpu().numpy()
        for T in (1.0, 0.7):
            out[f'{name}_T{T}'] = res[('Yes', T)] - res[('No', T)]
        print(name, 'LLR(T=0.7) mean %.3f sd %.3f' % (out[f'{name}_T0.7'].mean(), out[f'{name}_T0.7'].std()), flush=True)
    return out


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    inc = pd.concat([read_train()['Annual_Income_USD'], read_test()['Annual_Income_USD']], ignore_index=True)
    rl = replica_llr(np.unique(inc.to_numpy()))
    rl.to_parquet(CACHE / 'replica_llr.parquet')
    c07 = [c for c in rl.columns if c.endswith('_T0.7')]
    c1 = [c for c in rl.columns if c.endswith('_T1.0')]
    per_income = pd.DataFrame({'rep_llr_mean_T07': rl[c07].mean(1), 'rep_llr_mean_T1': rl[c1].mean(1),
                               'rep_llr_sd_T07': rl[c07].std(1).fillna(0.0), 'rep_llr_D60_T07': rl['d60_T0.7']})
    X = np.column_stack([inc.map(per_income[n]).to_numpy(np.float64) for n in NAMES]).astype(np.float32)
    assert np.isfinite(X).all()
    np.save(CACHE / 'replica_prior.npy', X)
    (CACHE / 'replica_prior_names.json').write_text(json.dumps(NAMES))
    print('replica prior', X.shape)


if __name__ == '__main__':
    main()
