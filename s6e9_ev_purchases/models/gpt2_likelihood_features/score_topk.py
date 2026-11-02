#!/usr/bin/env python
"""Top-50 support of every train / test row under frozen GReaT replicas: one map per (replica, field order).

The generator sampled with transformers' default top_k=50, so it could only emit tokens inside the top 50 of its next-token
distribution (logits / 0.70). Rows are written after both target prompts as in score_conditional.py and, per prompt, summarized
over the field tokens: the log-probability renormalized over the top 50 and floored at -20 (sum and mean), the number of tokens
outside the top 50, and the margin of the token's logit over the 50th logit (mean and min). fp32 activations, TF32 matmuls.
Writes features/topk_<replica>_<order>.npy, float64 [train + test rows, 10] = no/yes interleaved capped_sum, capped_mean,
out_count, margin_mean, margin_min.
Usage: python score_topk.py [--replicas e60,e65] [--orders source,competition,perm42]   (existing maps are skipped)"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET  # noqa: E402
from common.gpt_rows import load_replica, load_tokenizer, read_frame  # noqa: E402
from score_conditional import field_orders, serialize  # noqa: E402

TEMPERATURE, TOP_K, LOGP_FLOOR = 0.70, 50, -20.0


def score(model, tok, bodies, device):
    prefixes = [f'{TARGET} is No,', f'{TARGET} is Yes,']
    plen = [len(tok.encode(p, add_special_tokens=False)) for p in prefixes]
    docs = [prefixes[0] + ' ' + b for b in bodies] + [prefixes[1] + ' ' + b for b in bodies]
    enc = tok(docs, padding=True, truncation=True, max_length=256, return_tensors='pt', add_special_tokens=True)
    ids, att = enc['input_ids'].to(device, non_blocking=True), enc['attention_mask'].to(device, non_blocking=True)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
    labels = ids[:, 1:]
    keep = att[:, 1:].bool()
    n = len(bodies)
    position = torch.arange(labels.shape[1], device=device)[None, :]
    keep &= position >= torch.tensor([plen[0] - 1] * n + [plen[1] - 1] * n, device=device)[:, None]
    assert int(att.sum(1).max()) < 256, 'sequence truncated'
    scaled = logits.float() / TEMPERATURE
    selected = scaled.gather(-1, labels[..., None]).squeeze(-1)
    top = torch.topk(scaled, k=TOP_K, dim=-1, sorted=False).values
    kth = top.min(dim=-1).values
    supported = selected >= kth
    logp = torch.clamp(selected - torch.logsumexp(top, dim=-1), min=LOGP_FLOOR)
    total = torch.where(keep, logp, torch.zeros_like(logp)).sum(1)
    count = keep.sum(1).clamp_min(1)
    mean = total / count
    outside = torch.where(keep, ~supported, torch.zeros_like(supported)).sum(1).float()
    margin = selected - kth
    margin_mean = torch.where(keep, margin, torch.zeros_like(margin)).sum(1) / count
    margin_min = torch.where(keep, margin, torch.full_like(margin, float('inf'))).min(1).values
    value = torch.column_stack([total, mean, outside, margin_mean, margin_min])
    no, yes = value[:n].cpu().numpy().astype(np.float64), value[n:].cpu().numpy().astype(np.float64)
    return np.column_stack([no[:, 0], yes[:, 0], no[:, 1], yes[:, 1], no[:, 2], yes[:, 2], no[:, 3], yes[:, 3], no[:, 4], yes[:, 4]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replicas', default='e60,e65')
    ap.add_argument('--orders', default='source,competition,perm42')
    ap.add_argument('--batch-rows', type=int, default=128)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda')
    frame, _ = read_frame()
    orders = field_orders(list(frame.columns))
    tok = load_tokenizer()
    out = HERE / 'features'
    out.mkdir(exist_ok=True)
    for replica in a.replicas.split(','):
        todo = [o for o in a.orders.split(',') if not (out / f'topk_{replica}_{o}.npy').exists()]
        if not todo:
            continue
        model = load_replica(replica)
        model.config.pad_token_id = tok.pad_token_id
        model.config.use_cache = False
        model.to(device=device).eval()
        for order in todo:
            result = np.empty((len(frame), 10), np.float64)
            t0 = time.time()
            for b in range(0, len(frame), a.batch_rows):
                e = min(b + a.batch_rows, len(frame))
                result[b:e] = score(model, tok, serialize(frame.iloc[b:e], orders[order]), device)
                if b % (a.batch_rows * 800) == 0:
                    print(f'{replica} {order}: {e}/{len(frame)} rows [{time.time() - t0:.0f}s]', flush=True)
            assert np.isfinite(result).all()
            np.save(out / f'topk_{replica}_{order}.npy', result)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
