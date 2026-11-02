#!/usr/bin/env python
"""Class-conditional likelihoods of every train / test row under frozen GReaT replicas: one map per (replica, field order).

Each row's 13 fields are written in one fixed order and scored after both target prompts, prompt tokens masked:
    "Will_Buy_EV is No, <fields>" / "Will_Buy_EV is Yes, <fields>"  ->  log p(fields | No), log p(fields | Yes)
(sum and mean over the scored tokens, logits divided by the generator's sampling temperature 0.70). The same pass reads the
inverse query "<fields>, Will_Buy_EV is" -> logit(' Yes') - logit(' No') at the last position. The model runs in bfloat16; the
log-softmax is taken in float32. No labels are read.
Field orders: source (original column order), competition (test.csv column order), perm42 (np.random.default_rng(42)
permutation of the original order).
Writes features/raw_<replica>_<order>.npy, float64 [train + test rows, 5] = no_sum, yes_sum, no_mean, yes_mean, inverse_logit.
Usage: python score_conditional.py [--replicas e55,e60,e65,e70] [--orders source,competition,perm42]   (existing maps are skipped)"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET  # noqa: E402
from common.gpt_rows import SOURCE_ORDER, load_replica, load_tokenizer, read_frame  # noqa: E402

TEMPERATURE = 0.70


def field_orders(competition):
    return {'source': SOURCE_ORDER, 'competition': competition,
            'perm42': np.random.default_rng(42).permutation(SOURCE_ORDER).tolist()}


def serialize(frame, columns):
    return [', '.join(f'{c} is {v}' for c, v in zip(columns, row))
            for row in frame[columns].astype(str).itertuples(index=False, name=None)]


def body_likelihood(model, tok, bodies, device):
    prefixes = [f'{TARGET} is No,', f'{TARGET} is Yes,']
    plen = [len(tok.encode(p, add_special_tokens=False)) for p in prefixes]
    docs = [prefixes[0] + ' ' + b for b in bodies] + [prefixes[1] + ' ' + b for b in bodies]
    enc = tok(docs, padding=True, truncation=True, max_length=256, return_tensors='pt', add_special_tokens=True)
    ids, mask = enc['input_ids'].to(device, non_blocking=True), enc['attention_mask'].to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
    labels = ids[:, 1:]
    keep = mask[:, 1:].bool()
    n = len(bodies)
    position = torch.arange(labels.shape[1], device=device)[None, :]
    keep &= position >= torch.tensor([plen[0] - 1] * n + [plen[1] - 1] * n, device=device)[:, None]
    scaled = logits.float() / TEMPERATURE                               # log normalizers in fp32
    logp = scaled.gather(-1, labels[..., None]).squeeze(-1) - torch.logsumexp(scaled, dim=-1)
    total = torch.where(keep, logp, torch.zeros_like(logp)).sum(1)
    mean = total / keep.sum(1).clamp_min(1)
    assert int(mask.sum(1).max()) < 256, 'sequence truncated'
    total, mean = total.cpu().numpy().astype(np.float64), mean.cpu().numpy().astype(np.float64)
    return total[:n], total[n:], mean[:n], mean[n:]


def inverse_logit(model, tok, bodies, device, class_ids):
    enc = tok([b + f', {TARGET} is' for b in bodies], padding=True, truncation=True, max_length=256, return_tensors='pt',
              add_special_tokens=True)
    ids, mask = enc['input_ids'].to(device, non_blocking=True), enc['attention_mask'].to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = model.transformer(input_ids=ids, attention_mask=mask).last_hidden_state
        last = hidden[torch.arange(len(ids), device=device), mask.sum(1) - 1]
        pair = model.lm_head(last)[:, class_ids].float()
    return (pair[:, 1] - pair[:, 0]).cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replicas', default='e55,e60,e65,e70')
    ap.add_argument('--orders', default='source,competition,perm42')
    ap.add_argument('--batch-rows', type=int, default=256)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    frame, _ = read_frame()
    orders = field_orders(list(frame.columns))
    tok = load_tokenizer()
    no, yes = tok.encode(' No', add_special_tokens=False), tok.encode(' Yes', add_special_tokens=False)
    assert len(no) == 1 and len(yes) == 1
    out = HERE / 'features'
    out.mkdir(exist_ok=True)
    for replica in a.replicas.split(','):
        todo = [o for o in a.orders.split(',') if not (out / f'raw_{replica}_{o}.npy').exists()]
        if not todo:
            continue
        model = load_replica(replica)
        model.config.pad_token_id = tok.pad_token_id
        model.config.use_cache = False
        model.to(device=device, dtype=torch.bfloat16).eval()
        for order in todo:
            values = np.empty((len(frame), 5), np.float64)
            t0 = time.time()
            for b in range(0, len(frame), a.batch_rows):
                e = min(b + a.batch_rows, len(frame))
                bodies = serialize(frame.iloc[b:e], orders[order])
                values[b:e] = np.column_stack([*body_likelihood(model, tok, bodies, device),
                                               inverse_logit(model, tok, bodies, device, [no[0], yes[0]])])
                if b % (a.batch_rows * 400) == 0:
                    print(f'{replica} {order}: {e}/{len(frame)} rows [{time.time() - t0:.0f}s]', flush=True)
            assert np.isfinite(values).all()
            np.save(out / f'raw_{replica}_{order}.npy', values)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
