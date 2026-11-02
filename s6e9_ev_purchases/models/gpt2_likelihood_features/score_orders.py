#!/usr/bin/env python
"""Class-conditional likelihoods of every train / test row under a frozen GReaT replica for K random field orders.

The generator wrote every row target first and then the 13 fields in an order of its own choosing; the released csv hides that
order, so the row likelihood is a sum over orders. For K fixed random orders each row is scored after both target prompts
("Will_Buy_EV is No," / "Will_Buy_EV is Yes,"), prompt tokens masked, logits / 0.70, fp32 activations and TF32 matmuls. The
log-likelihood is also split by field (the field of each predicted token) and averaged over the orders. Each field fragment is
tokenized once per row and the orders are built by concatenating token ids (equal to tokenizing the joined text).
Writes features/scores/<replica>r<K>__<start>-<end>.npz: ll float32 [rows, K, 2] (No, Yes), fld float32 [rows, 2, 13], rows.
Usage: python score_orders.py [--replica e55] [--orders rand12] [--rows START:END]   (default all rows; slices can run in
parallel, build_order_features.py joins them)"""
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

TEMPERATURE = 0.70


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replica', default='e55')
    ap.add_argument('--orders', default='rand12')
    ap.add_argument('--rows', default='all')
    ap.add_argument('--batch-rows', type=int, default=192)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    frame, _ = read_frame()
    if a.rows == 'all':
        rows = np.arange(len(frame))
    else:
        s0, e0 = a.rows.split(':')
        rows = np.arange(int(s0), min(int(e0), len(frame)))
    orders = random_orders(a.orders)
    sub = frame.iloc[rows].astype(str)
    tok = load_tokenizer()
    model = load_replica(a.replica, from_config=True).cuda().eval()
    F = len(SOURCE_ORDER)
    frag = {}
    for j, c in enumerate(SOURCE_ORDER):
        v = sub[c].to_numpy()
        frag[(j, 0)] = tok([f' {c} is {x}' for x in v], add_special_tokens=False)['input_ids']
        frag[(j, 1)] = tok([f', {c} is {x}' for x in v], add_special_tokens=False)['input_ids']
    pre = [tok.encode(f'{TARGET} is No,'), tok.encode(f'{TARGET} is Yes,')]
    n = len(rows)
    out = np.zeros((n, len(orders), 2), np.float32)
    fld = np.zeros((n, 2, F), np.float64)
    pad = tok.eos_token_id
    for oi, order in enumerate(orders):
        oj = [SOURCE_ORDER.index(c) for c in order]
        for s in range(0, n, a.batch_rows):
            idx = range(s, min(n, s + a.batch_rows))
            seqs, fids = [], []
            for lab in (0, 1):
                for r in idx:
                    ids, fid = list(pre[lab]), [-1] * len(pre[lab])
                    for pos, j in enumerate(oj):
                        t = frag[(j, 0 if pos == 0 else 1)][r]
                        ids += t
                        fid += [j] * len(t)
                    seqs.append(ids)
                    fids.append(fid)
            L = max(len(x) for x in seqs)
            ids = np.full((len(seqs), L), pad, np.int64)
            fidm = np.full((len(seqs), L), -1, np.int64)
            mask = np.zeros((len(seqs), L), np.int64)
            for d, (x, fi) in enumerate(zip(seqs, fids)):
                ids[d, :len(x)] = x
                fidm[d, :len(x)] = fi
                mask[d, :len(x)] = 1
            ids_t, mask_t = torch.as_tensor(ids).cuda(non_blocking=True), torch.as_tensor(mask).cuda(non_blocking=True)
            fid_t = torch.as_tensor(fidm[:, 1:]).cuda(non_blocking=True)          # field of the predicted token
            with torch.inference_mode():
                lg = model(input_ids=ids_t, attention_mask=mask_t).logits[:, :-1]
                lg.div_(TEMPERATURE)
                logp = lg.gather(-1, ids_t[:, 1:, None]).squeeze(-1) - torch.logsumexp(lg, -1)
                keep = (fid_t >= 0) & mask_t[:, 1:].bool()
                lpk = torch.where(keep, logp, torch.zeros_like(logp)).double()
                tot = lpk.sum(1).cpu().numpy()
                fs = torch.zeros((lpk.shape[0], F), device='cuda', dtype=torch.float64)
                fs.scatter_add_(1, fid_t.clamp_min(0), lpk)
                fs = fs.cpu().numpy()
            m = len(idx)
            out[s:s + m, oi, 0], out[s:s + m, oi, 1] = tot[:m], tot[m:]
            fld[s:s + m, 0] += fs[:m] / len(orders)
            fld[s:s + m, 1] += fs[m:] / len(orders)
        print(f'order {oi + 1}/{len(orders)} [{time.time() - t0:.0f}s]', flush=True)
    tag = f'{a.replica}r{a.orders[4:].replace(":", "s")}'
    (HERE / 'features' / 'scores').mkdir(parents=True, exist_ok=True)
    np.savez(HERE / 'features' / 'scores' / f'{tag}__{rows[0]}-{rows[-1] + 1}.npz', ll=out, fld=fld.astype(np.float32), rows=rows)
    print(f'{tag}: {n} rows x {len(orders)} orders [{time.time() - t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
