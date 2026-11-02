#!/usr/bin/env python
"""GPT features from the frozen B0 fold models: the input of train.py.

B0_f{k} is the GPT-2 small target-first model of fold k (../gpt2_target_first, --size small); it never saw the labels of fold k.
It reads every train and test row with the prefix "satisfaction is False," and with "satisfaction is True,", in two fixed field
orders: the first of the 30 seed-2025 scoring orders and its reverse (so every field is seen early and late).
The softmax is restricted to the 1,073 tokens that occur in the row texts (eos, both label prefixes, every 'column is value'
fragment of train + test): a feature transform, not an exact likelihood.
Per row, order and label prefix:
  fields [21, 3]  per field: the summed log-prob of all its tokens, the summed log-prob of its value tokens (the tokens after
                  'column is'), and the summed entropy of the next-token distributions that predict its value tokens;
  hidden [64]     the mean hidden state over the field tokens and the hidden state of the last token, each projected to 32 dims
                  by a fixed random orthonormal matrix (QR of a 768 x 32 Gaussian, seed 104001).
Body in fp16, LM head and softmax in fp32. Writes, for the 999,479 train + test rows (train.csv then test.csv order):
  features/f{k}_fields.npy  float32 [rows, 2 orders, 2 labels, 21, 3]
  features/f{k}_hidden.npy  float16 [rows, 2 orders, 2 labels, 64]
Usage: python extract_features.py [--folds 0,1,2,3,4]"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.gpt_rows import fast_gelu, fragment_tables, load_frame, load_tokenizer  # noqa: E402
from common.pairs import fold_list  # noqa: E402

_spec = importlib.util.spec_from_file_location('gpt2_target_first', HERE.parent / 'gpt2_target_first' / 'train.py')
TF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TF)                 # prefixes, check_tokenization, make_orders

OUT = HERE / 'features'
BATCH = 128                                  # rows per batch (x 2 label prefixes)
PROJ_SEED, PROJ_DIM = 104001, 32


def load_b0(k):
    """The fine-tuned GPT-2 small fold model of ../gpt2_target_first (fp32 state_dict), frozen."""
    from transformers import GPT2Config, GPT2LMHeadModel
    path = HERE.parent / 'gpt2_target_first' / 'checkpoints' / f'gpt2_small_target_first_f{k}.pt'
    assert path.exists(), f'{path} not found: run ../gpt2_target_first/train.py --size small first'
    model = GPT2LMHeadModel(GPT2Config.from_pretrained('gpt2'))
    res = model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=False)
    assert not [n for n in res.missing_keys if n != 'lm_head.weight'], res.missing_keys
    assert not [n for n in res.unexpected_keys if not n.endswith(('attn.bias', 'attn.masked_bias'))], res.unexpected_keys
    model.tie_weights()
    return fast_gelu(model).cuda().eval()


def feature_vocab(tabs, pre, eos):
    """Token ids that occur in the row texts: eos, both label prefixes, every fragment of train + test (1,073)."""
    return sorted({eos} | {t for p in pre for t in p} | {t for _, first, later in tabs for s in first + later for t in s})


def value_offsets(tok, tabs, cols):
    """Per field: the number of leading tokens of its first / later fragment that spell ' {c} is' / ', {c} is'.
    The remaining tokens of the fragment are its value tokens (checked on every distinct value)."""
    offs = []
    for (_, first, later), c in zip(tabs, cols):
        o = []
        for lead, frags in ((' ', first), (', ', later)):
            p = tok.encode(f'{lead}{c} is')
            assert all(s[:len(p)] == p for s in frags), c
            o.append(len(p))
        offs.append(o)
    return offs


def batch_tokens(tabs, pre, offs, rows, order, eos):
    """Both label prefixes for every row (all False texts first), right-padded with eos. Returns the token ids, the field of
    every token (-1 = label prefix / padding), the value-token mask and the position of the last real token."""
    seqs, flds, vals = [], [], []
    for lab in (0, 1):
        for r in rows:
            seq, fld, val = list(pre[lab]), [-1] * len(pre[lab]), [False] * len(pre[lab])
            for pos, j in enumerate(order):
                codes, first, later = tabs[j]
                frag = (first if pos == 0 else later)[codes[r]]
                off = offs[j][0 if pos == 0 else 1]
                seq += frag
                fld += [int(j)] * len(frag)
                val += [False] * off + [True] * (len(frag) - off)
            seqs.append(seq)
            flds.append(fld)
            vals.append(val)
    L = max(map(len, seqs))
    ids = np.full((len(seqs), L), eos, np.int64)
    fid = np.full((len(seqs), L), -1, np.int64)
    vmask = np.zeros((len(seqs), L), bool)
    for i, (s, fl, v) in enumerate(zip(seqs, flds, vals)):
        ids[i, :len(s)], fid[i, :len(s)], vmask[i, :len(s)] = s, fl, v
    end = np.array([len(s) - 1 for s in seqs])
    return [torch.as_tensor(x, device='cuda') for x in (ids, fid, vmask, end)]


@torch.inference_mode()
def features(model, W, mapping, proj, tabs, pre, offs, rows, order, eos):
    """One batch of rows in one order: fields [m, 2 labels, 21, 3] and hidden projections [m, 2 labels, 64]."""
    ids, fid, vmask, end = batch_tokens(tabs, pre, offs, rows, order, eos)
    with torch.autocast('cuda', dtype=torch.float16):                       # right padding + causal attention: no mask needed
        hidden = model.transformer(input_ids=ids, use_cache=False).last_hidden_state
    lp = F.log_softmax(F.linear(hidden[:, :-1].float(), W), -1)             # restricted-vocabulary next-token log-probs
    tgt = mapping[ids[:, 1:]]
    assert bool((tgt >= 0).all())
    obs = lp.gather(-1, tgt[..., None]).squeeze(-1)                         # log-prob of the actual next token
    ent = -(lp.exp() * lp).sum(-1)                                          # entropy of the next-token distribution
    keep, value, idx = fid[:, 1:] >= 0, vmask[:, 1:], fid[:, 1:].clamp(min=0)  # by the field of the predicted token
    nf = len(tabs)

    def per_field(x):
        return torch.zeros(len(ids), nf, device='cuda').scatter_add_(1, idx, torch.where(keep, x, 0.))
    fields = torch.stack([per_field(obs), per_field(torch.where(value, obs, 0.)), per_field(torch.where(value, ent, 0.))], -1)
    mask = (fid >= 0).float()
    mean = (hidden.float() * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
    last = hidden[torch.arange(len(ids), device='cuda'), end].float()
    emb = torch.cat([mean @ proj, last @ proj], -1)
    m = len(rows)
    return (fields.reshape(2, m, nf, 3).permute(1, 0, 2, 3).cpu().numpy(),
            emb.reshape(2, m, 2 * PROJ_DIM).permute(1, 0, 2).cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    frame, _ = load_frame()
    tok = load_tokenizer()
    tabs = fragment_tables(tok, frame, first_space=' ')
    TF.check_tokenization(tok, frame, tabs)
    pre, eos = TF.prefixes(tok), tok.eos_token_id
    offs = value_offsets(tok, tabs, list(frame.columns))
    vocab = feature_vocab(tabs, pre, eos)
    first = TF.make_orders(1, len(tabs))[0]                                 # the first of B0's 30 scoring orders
    orders = [first, first[::-1].copy()]
    proj = np.linalg.qr(np.random.default_rng(PROJ_SEED).normal(size=(768, PROJ_DIM)))[0].astype(np.float32)
    n = len(frame)
    print(f'{n} rows, {len(vocab)}-token vocabulary', flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    OUT.mkdir(exist_ok=True)
    for k in fold_list(a.folds):
        out_f, out_h = OUT / f'f{k}_fields.npy', OUT / f'f{k}_hidden.npy'
        if out_f.exists() and out_h.exists():
            continue
        model = load_b0(k)
        vocab_t = torch.tensor(vocab, device='cuda')
        W = model.lm_head.weight.detach()[vocab_t].float()
        mapping = torch.full((len(tok),), -1, dtype=torch.long, device='cuda')
        mapping[vocab_t] = torch.arange(len(vocab), device='cuda')
        proj_t = torch.tensor(proj, device='cuda')
        fields = np.zeros((n, 2, 2, len(tabs), 3), np.float32)
        hidden = np.zeros((n, 2, 2, 2 * PROJ_DIM), np.float16)
        for oi, order in enumerate(orders):
            for s in range(0, n, BATCH):
                rows = np.arange(s, min(s + BATCH, n))
                fields[s:s + len(rows), oi], hidden[s:s + len(rows), oi] = features(
                    model, W, mapping, proj_t, tabs, pre, offs, rows, order, eos)
                if s % (1024 * BATCH) == 0:
                    print(f'fold {k}: order {oi + 1}/2, rows {s + len(rows)}/{n}', flush=True)
        np.save(out_h, hidden)
        np.save(out_f, fields)
        print(f'fold {k}: wrote {out_f.name}, {out_h.name}', flush=True)
        del model, W
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
