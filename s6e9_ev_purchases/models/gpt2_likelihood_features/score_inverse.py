#!/usr/bin/env python
"""Inverse query of a frozen GReaT replica for every train / test row over K random field orders.

GReaT trained the replica on random field orders, so the target also appeared after other fields. Writing the 13 fields first and
appending ", Will_Buy_EV is" reads the generator's own P(Yes | x): logit(' Yes') - logit(' No') at the last position, fp32
activations and TF32 matmuls. No labels are read.
Writes features/inv_<replica>.npy, float32 [train + test rows, 4] = mean, sd, min, max of the per-order logit difference.
Usage: python score_inverse.py [--replica e55] [--orders rand6:7]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET  # noqa: E402
from common.gpt_rows import SOURCE_ORDER, load_replica, load_tokenizer, random_orders, read_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replica', default='e55')
    ap.add_argument('--orders', default='rand6:7')
    ap.add_argument('--batch-rows', type=int, default=384)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    frame, _ = read_frame()
    orders = random_orders(a.orders)
    sub = frame.astype(str)
    tok = load_tokenizer()
    model = load_replica(a.replica, from_config=True).cuda().eval()
    yes, no = tok.encode(' Yes'), tok.encode(' No')
    assert len(yes) == 1 and len(no) == 1
    frag = {}                                                          # first field without a separator, later ones with ', '
    for j, c in enumerate(SOURCE_ORDER):
        v = sub[c].to_numpy()
        frag[(j, 0)] = tok([f'{c} is {x}' for x in v], add_special_tokens=False)['input_ids']
        frag[(j, 1)] = tok([f', {c} is {x}' for x in v], add_special_tokens=False)['input_ids']
    suffix = tok.encode(f', {TARGET} is')
    n = len(frame)
    inv = np.zeros((n, len(orders)), np.float32)
    for oi, order in enumerate(orders):
        oj = [SOURCE_ORDER.index(c) for c in order]
        for s0 in range(0, n, a.batch_rows):
            rr = range(s0, min(n, s0 + a.batch_rows))
            seqs = []
            for r in rr:
                x = []
                for p, j in enumerate(oj):
                    x += frag[(j, 0 if p == 0 else 1)][r]
                seqs.append(x + suffix)
            L = max(len(x) for x in seqs)
            ids = np.full((len(seqs), L), tok.eos_token_id, np.int64)
            lens = np.array([len(x) for x in seqs])
            for d, x in enumerate(seqs):
                ids[d, :len(x)] = x
            att = (np.arange(L)[None, :] < lens[:, None]).astype(np.int64)
            with torch.inference_mode():
                h = model.transformer(input_ids=torch.as_tensor(ids).cuda(), attention_mask=torch.as_tensor(att).cuda()).last_hidden_state
                last = h[torch.arange(len(seqs)), torch.as_tensor(lens - 1).cuda()]
                lg = model.lm_head(last)[:, [no[0], yes[0]]].float()
            inv[s0:s0 + len(rr), oi] = (lg[:, 1] - lg[:, 0]).cpu().numpy()
        print(f'order {oi + 1}/{len(orders)} [{time.time() - t0:.0f}s]', flush=True)
    feats = np.column_stack([inv.mean(1), inv.std(1), inv.min(1), inv.max(1)]).astype(np.float32)
    (HERE / 'features').mkdir(exist_ok=True)
    np.save(HERE / 'features' / f'inv_{a.replica}.npy', feats)


if __name__ == '__main__':
    main()
