#!/usr/bin/env python
"""Per-field pseudo-likelihoods of every train / test row under frozen GReaT replicas: one map per (replica, held-out field).

For held field c and each label the text is
    "Will_Buy_EV is <No|Yes>, <the other 12 fields in original column order>, <c> is" + " <value of c>"
and only the value tokens of c are scored (logits / 0.70; fp32 activations, TF32 matmuls). Statistics per label: smooth
log-probability (sum, mean, min, first token, last token), log-probability renormalized over the top 50 (unsupported tokens set
to -20; sum, mean), number of tokens outside the top 50, logit margin over the 50th logit (mean, min); plus the value's token count.
Writes features/field_<replica>_c<NN>.npy, float32 [train + test rows, 21] (COLUMNS order).
Usage: python score_fields.py [--replicas e55,e60] [--fields 0,1,...,12]   (field index into the original column order;
existing maps are skipped)"""
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

TEMPERATURE, TOP_K, UNSUPPORTED_LOGP = 0.70, 50, -20.0
COLUMNS = ['no_smooth_sum', 'yes_smooth_sum', 'no_smooth_mean', 'yes_smooth_mean', 'no_smooth_min', 'yes_smooth_min',
           'no_smooth_first', 'yes_smooth_first', 'no_smooth_last', 'yes_smooth_last', 'no_topk_sum', 'yes_topk_sum',
           'no_topk_mean', 'yes_topk_mean', 'no_out_count', 'yes_out_count', 'no_margin_mean', 'yes_margin_mean',
           'no_margin_min', 'yes_margin_min', 'value_token_count']


def make_documents(frame, context, held):
    contexts = [', '.join(f'{c} is {v}' for c, v in zip(context, row))
                for row in frame[context].astype(str).itertuples(index=False, name=None)]
    values = frame[held].astype(str).tolist()
    prefixes, documents = [], []
    for label in ('No', 'Yes'):
        for ctx, v in zip(contexts, values):
            prefix = f'{TARGET} is {label}, {ctx}, {held} is'
            prefixes.append(prefix)
            documents.append(prefix + ' ' + v)
    return prefixes, documents


def prompt_statistics(smooth_logp, topk_logp, supported, margin, keep, start, last):
    count = keep.sum(1)
    assert int(count.min()) >= 1
    count = count.float()
    smooth_sum = torch.where(keep, smooth_logp, 0.0).sum(1)
    rows = torch.arange(len(smooth_logp), device=smooth_logp.device)
    topk_sum = torch.where(keep, topk_logp, 0.0).sum(1)
    return torch.column_stack([
        smooth_sum,
        smooth_sum / count,
        torch.where(keep, smooth_logp, torch.full_like(smooth_logp, float('inf'))).min(1).values,
        smooth_logp[rows, start],
        smooth_logp[rows, last],
        topk_sum,
        topk_sum / count,
        torch.where(keep, ~supported, False).sum(1).float(),
        torch.where(keep, margin, 0.0).sum(1) / count,
        torch.where(keep, margin, torch.full_like(margin, float('inf'))).min(1).values,
        count,
    ])


def score_batch(model, tok, prefixes, documents, device):
    prefix_ids = tok(prefixes, add_special_tokens=True, padding=False, truncation=False)['input_ids']
    prefix_length = torch.tensor([len(p) for p in prefix_ids], dtype=torch.long, device=device)
    enc = tok(documents, add_special_tokens=True, padding=True, truncation=True, max_length=256, return_tensors='pt')
    ids, att = enc['input_ids'].to(device, non_blocking=True), enc['attention_mask'].to(device, non_blocking=True)
    assert int(att.sum(1).max()) < 256, 'sequence truncated'
    for r, p in enumerate(prefix_ids):                                  # the prompt must tokenize the same inside the document
        assert ids[r, :len(p)].tolist() == p, r
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
    label = ids[:, 1:]
    scaled = logits.float() / TEMPERATURE
    selected = scaled.gather(-1, label[..., None]).squeeze(-1)
    smooth_logp = selected - torch.logsumexp(scaled, dim=-1)
    top = torch.topk(scaled, k=TOP_K, dim=-1, sorted=False).values
    threshold = top.min(dim=-1).values
    supported = selected >= threshold
    supported_logp = torch.clamp(selected - torch.logsumexp(top, dim=-1), min=UNSUPPORTED_LOGP)
    topk_logp = torch.where(supported, supported_logp, torch.full_like(supported_logp, UNSUPPORTED_LOGP))
    margin = selected - threshold
    start = prefix_length - 1
    last = att.sum(1) - 2
    position = torch.arange(label.shape[1], device=device)[None, :]
    keep = (position >= start[:, None]) & (position <= last[:, None])   # the value tokens only
    assert torch.equal(keep.sum(1), att.sum(1) - prefix_length)
    stats = prompt_statistics(smooth_logp, topk_logp, supported, margin, keep, start, last)
    n = len(documents) // 2
    no, yes = stats[:n], stats[n:]
    assert torch.equal(no[:, -1], yes[:, -1])
    out = torch.column_stack([x for j in range(10) for x in (no[:, j], yes[:, j])] + [no[:, 10]])
    return out.cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replicas', default='e55,e60')
    ap.add_argument('--fields', default=','.join(map(str, range(13))))
    ap.add_argument('--batch-rows', type=int, default=256)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    frame, _ = read_frame()
    tok = load_tokenizer()
    out = HERE / 'features'
    out.mkdir(exist_ok=True)
    for replica in a.replicas.split(','):
        todo = [int(j) for j in a.fields.split(',') if not (out / f'field_{replica}_c{int(j):02d}.npy').exists()]
        if not todo:
            continue
        model = load_replica(replica)
        model.config.pad_token_id = tok.pad_token_id
        model.config.use_cache = False
        model.to(device=device).eval()
        for j in todo:
            held = SOURCE_ORDER[j]
            context = [c for c in SOURCE_ORDER if c != held]
            result = np.empty((len(frame), len(COLUMNS)), np.float32)
            t0 = time.time()
            for b in range(0, len(frame), a.batch_rows):
                e = min(b + a.batch_rows, len(frame))
                result[b:e] = score_batch(model, tok, *make_documents(frame.iloc[b:e], context, held), device)
                if b % (a.batch_rows * 400) == 0:
                    print(f'{replica} {held}: {e}/{len(frame)} rows [{time.time() - t0:.0f}s]', flush=True)
            assert np.isfinite(result).all()
            np.save(out / f'field_{replica}_c{j:02d}.npy', result)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
