#!/usr/bin/env python
"""GPT-2 small B0 fold model + one more epoch of generative + gentle BCE training (pair gpt_hybrid_cv6), one model per fold.

B0_f{k} is the GPT-2 small target-first model of fold k (../gpt2_target_first, --size small), which also wrote B0's 30-order
log-likelihoods of the held-out and test rows.
1. Child: continue B0_f{k} for one epoch over the outer-train rows of fold k: AdamW lr 1e-5 (no weight decay), linear decay to 0,
   batch 64, grad clip 1, bf16 autocast, dropout on. Each row is written with both label prefixes in one fresh random field order.
   Loss = generative loss + 0.1 x BCE:
     generative: mean next-token NLL of the true-label text (eos included);
     BCE:        on LLR + log(prior / (1 - prior)), LLR = log p(fields | True) - log p(fields | False) at a fixed scale,
                 prior = the training rows' positive rate (about 0.4436).
   The training softmax is restricted to the 1,073 tokens of the row texts (eos, both label prefixes, every 'column is value'
   fragment of train + test).
2. Score the child like B0 (full softmax, fp16 body + fp32 head) with the first 6 of B0's 30 orders.
3. Readout per row, each LLR pooled over its orders by log-sum-exp per label:
       z = LLR_B0(30 orders) + LLR_child(6 orders) - LLR_B0(the same 6 orders),   p = sigmoid(z)
   i.e. B0's 30-order score plus a 6-order estimate of the change made by the extra epoch.
Pair: OOF = p of the held-out folds; test = mean of the 5 fold models' p (no alignment).
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit, logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.gpt_rows import fast_gelu, fragment_tables, load_frame, load_tokenizer  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

_spec = importlib.util.spec_from_file_location('gpt2_target_first', HERE.parent / 'gpt2_target_first' / 'train.py')
TF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TF)                 # prefixes, build, check_tokenization, make_orders, score

NAME = 'gpt_hybrid_cv6'
B0_DIR, B0_NAME = HERE.parent / 'gpt2_target_first', 'gpt2_small_target_first'
LR, BATCH, BCE_WEIGHT, N_ORDERS = 1e-5, 64, 0.1, 6


def load_b0(k):
    """The fine-tuned GPT-2 small fold model of ../gpt2_target_first (fp32 state_dict)."""
    from transformers import GPT2Config, GPT2LMHeadModel
    path = B0_DIR / 'checkpoints' / f'{B0_NAME}_f{k}.pt'
    assert path.exists(), f'{path} not found: run ../gpt2_target_first/train.py --size small first'
    model = GPT2LMHeadModel(GPT2Config.from_pretrained('gpt2'))
    res = model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=False)
    assert not [n for n in res.missing_keys if n != 'lm_head.weight'], res.missing_keys
    assert not [n for n in res.unexpected_keys if not n.endswith(('attn.bias', 'attn.masked_bias'))], res.unexpected_keys
    model.tie_weights()
    return fast_gelu(model).cuda()


def b0_ll(k, split, n_rows):
    """B0's log p(fields | False / True) per scoring order: float64 [rows, 30, 2]."""
    path = B0_DIR / 'parts' / f'{B0_NAME}_f{k}_{split}_ll.npy'
    assert path.exists(), f'{path} not found: run ../gpt2_target_first/train.py --size small first'
    ll = np.load(path).astype(np.float64)
    assert ll.shape == (n_rows, 30, 2), ll.shape
    return ll


def marginal_llr(ll):
    """Order-marginal LLR of ll [n, K, 2]: log-sum-exp over the orders per label, True - False."""
    return logsumexp(ll[:, :, 1], 1) - logsumexp(ll[:, :, 0], 1)


def continue_training(model, tabs, pre, eos, vocab, tr_rows, y, fold):
    vocab_t = torch.tensor(vocab, device='cuda')
    mapping = torch.full((model.config.vocab_size,), -1, dtype=torch.long, device='cuda')
    mapping[vocab_t] = torch.arange(len(vocab), device='cuda')
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, fused=True)
    steps = math.ceil(len(tr_rows) / BATCH)
    rng = np.random.default_rng(20000 + fold)
    order = rng.permutation(tr_rows)
    prior = float(y[tr_rows].mean())
    prior_logit = math.log(prior / (1 - prior))
    model.train()
    for step, j in enumerate(range(0, len(order), BATCH)):
        rr = order[j:j + BATCH]
        m = len(rr)
        perms = [rng.permutation(len(tabs)) for _ in rr]
        seqs = [TF.build(tabs, r, o, pre[lab]) + [eos] for lab in (0, 1) for r, o in zip(rr, perms)]   # all False texts first
        L = max(map(len, seqs))
        ids = np.full((2 * m, L), eos, np.int64)
        keep = np.zeros((2 * m, L - 1), bool)                                  # positions that predict a field token
        gkeep = np.zeros_like(keep)                                            # positions that predict any real token
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = s
            keep[i, len(pre[i // m]) - 1:len(s) - 2] = True
            gkeep[i, :len(s) - 1] = True
        ids, keep, gkeep = (torch.as_tensor(x, device='cuda') for x in (ids, keep, gkeep))
        labels = torch.as_tensor(y[rr].astype(np.int64), device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):                     # right padding + causal attention: no mask needed
            h = model.transformer(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
        logits = F.linear(h.float(), model.lm_head.weight[vocab_t].float())    # restricted-vocabulary softmax
        logp = F.log_softmax(logits, -1).gather(-1, mapping[ids[:, 1:]][:, :, None]).squeeze(-1)
        ll = torch.where(keep, logp, 0.).sum(1)
        bce = F.binary_cross_entropy_with_logits(ll[m:] - ll[:m] + prior_logit, labels.float())
        sel = torch.arange(m, device='cuda') + labels * m                      # the true-label text of every row
        gen = -torch.where(gkeep[sel], logp[sel], 0.).sum() / gkeep[sel].sum()
        loss = BCE_WEIGHT * bce + gen
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for group in opt.param_groups:
            group['lr'] = LR * max(0.0, 1 - (step + 1) / steps)
        if (step + 1) % 1000 == 0 or step + 1 == steps:
            print(f'fold {fold}: step {step + 1}/{steps} bce {bce.item():.4f} gen {gen.item():.4f}', flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    y = target()
    f = folds(y)
    frame, n_train = load_frame()
    tok = load_tokenizer()
    tabs = fragment_tables(tok, frame, first_space=' ')
    TF.check_tokenization(tok, frame, tabs)
    pre, eos = TF.prefixes(tok), tok.eos_token_id
    vocab = sorted({eos} | {t for p in pre for t in p} | {t for _, first, later in tabs for s in first + later for t in s})
    orders = TF.make_orders(N_ORDERS, len(tabs))                               # = the first 6 of B0's 30 scoring orders
    te_rows = np.arange(n_train, len(frame))
    torch.backends.cuda.matmul.allow_tf32 = True
    for k in fold_list(a.folds):
        if have_part(HERE, NAME, k):
            continue
        torch.manual_seed(19000 + k)
        tr_rows, va_rows = np.flatnonzero(f != k), np.flatnonzero(f == k)
        model = load_b0(k)
        ckpt = HERE / 'checkpoints' / f'{NAME}_f{k}.pt'
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
        else:
            continue_training(model, tabs, pre, eos, vocab, tr_rows, y, k)
            ckpt.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), ckpt)
        model.eval()
        p = {}
        for split, rows in (('valid', va_rows), ('test', te_rows)):
            child = TF.score(model, tabs, pre, rows, orders, eos)               # [rows, 6, 2], full softmax
            b0 = b0_ll(k, split, len(rows))
            p[split] = expit(marginal_llr(b0) + marginal_llr(child) - marginal_llr(b0[:, :N_ORDERS]))
        save_part(HERE, NAME, k, p['valid'], p['test'])
        print(f'fold {k}: AUC {roc_auc_score(y[va_rows], p["valid"]):.6f}', flush=True)
        del model
        torch.cuda.empty_cache()
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
