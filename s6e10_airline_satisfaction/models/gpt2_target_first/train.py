#!/usr/bin/env python
"""GPT-2 target-first generative classifier (the "Storyteller"), one model per fold.

1. Start from the GReaT replica of the chosen size (../gpt2_replica, trained on the original data only).
2. Keep training it as a causal LM on the outer-train rows of the fold, written label FIRST, like the competition's generator
   sampled them: "satisfaction is True, <the 21 'column is value' fields in a fresh random order per row per step>" + eos.
   AdamW lr 5e-5 (no weight decay), linear decay without warmup, batch 32, grad clip 1, bf16 autocast, 8 epochs.
   Medium and large also keep an EMA of the weights (decay 0.9999); the EMA weights are the scored model.
3. Score with Bayes' rule. For each label, sum the log-probabilities of all field tokens after the label prefix: log p(row | label).
   Do this for 30 fixed random field orders and pool the orders by log-sum-exp per label (the generator wrote every row in a
   random order, so its density is a mixture over orders):
       LLR = logsumexp_o log p(row, o | True) - logsumexp_o log p(row, o | False)
   The prior log-odds is a constant, so it is left out (AUC does not change and the stacker learns an intercept).
   Scoring runs the transformer body in fp16 and the LM head + log-softmax in fp32 (summed log-likelihoods need the precision).
Pair: OOF = sigmoid(LLR); test = mean of the 5 fold models' sigmoid(LLR).
Also kept for the models built on these fold models (../gpt_hybrid, ../gpt_fold_be use the small ones as their "B0"):
  checkpoints/<pair>_f<k>.pt          the fine-tuned fold model (state_dict)
  parts/<pair>_f<k>_<split>_ll.npy    float32 [rows, 30, 2] log p(fields | False / True) per order (valid / test rows)
Usage: python train.py --size small|medium|large [--folds 0,1,2,3,4]"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds, target  # noqa: E402
from common.gpt_rows import fast_gelu, fragment_tables, load_frame, load_replica, load_tokenizer, pad_batch  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

EPOCHS, LR, BATCH, N_ORDERS = 8, 5e-5, 32, 30
EMA = {'small': 0.0, 'medium': 0.9999, 'large': 0.9999}


def prefixes(tok):
    return [tok.encode(f'{TARGET} is False,'), tok.encode(f'{TARGET} is True,')]


def build(tabs, r, order, pre):
    ids = list(pre)
    for pos, j in enumerate(order):
        codes, first, later = tabs[j]
        ids += (first if pos == 0 else later)[codes[r]]
    return ids


def check_tokenization(tok, frame, tabs, n=200):
    rng = np.random.default_rng(0)
    pre = prefixes(tok)
    cols = list(frame.columns)
    for r in rng.choice(len(frame), n, replace=False):
        order, lab = rng.permutation(len(cols)), int(rng.integers(2))
        text = f'{TARGET} is {"True" if lab else "False"}, ' + ', '.join(f'{cols[j]} is {frame.iat[r, j]}' for j in order)
        assert build(tabs, r, order, pre[lab]) == tok.encode(text), text


def make_orders(k, n_fields, seed=2025):
    rng = np.random.default_rng(seed)
    return [rng.permutation(n_fields) for _ in range(k)]


@torch.inference_mode()
def score(model, tabs, pre, rows, orders, eos, batch=128):
    """log p(fields | label) for every row x order x label (label prefix masked): float64 [n, K, 2]."""
    W = model.lm_head.weight
    ll = np.zeros((len(rows), len(orders), 2))
    for oi, order in enumerate(orders):
        for s0 in range(0, len(rows), batch):
            rr = rows[s0:s0 + batch]
            seqs = [build(tabs, r, order, pre[lab]) for lab in (0, 1) for r in rr]
            npre = [len(pre[lab]) for lab in (0, 1) for _ in rr]
            ids, att = pad_batch(seqs, eos)
            keep = torch.zeros(ids.shape[0], ids.shape[1] - 1, dtype=torch.bool)
            for d, x in enumerate(seqs):
                keep[d, npre[d] - 1:len(x) - 1] = True                           # positions that predict the field tokens
            ids, keep = ids.cuda(), keep.cuda()
            with torch.autocast('cuda', dtype=torch.float16):                   # right padding + causal attention: no mask needed
                h = model.transformer(input_ids=ids, use_cache=False).last_hidden_state
            lp = torch.log_softmax(torch.nn.functional.linear(h[:, :-1].float(), W.float()), -1)
            lp = lp.gather(-1, ids[:, 1:, None]).squeeze(-1)
            tot = torch.where(keep, lp, torch.zeros_like(lp)).double().sum(1).cpu().numpy()
            m = len(rr)
            ll[s0:s0 + m, oi, 0], ll[s0:s0 + m, oi, 1] = tot[:m], tot[m:]
    return ll


def finetune(model, tabs, pre, tr_rows, y_tr, eos, fold, ema_decay):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    n_steps = int(math.ceil(EPOCHS * len(tr_rows) / BATCH))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / n_steps))
    params = list(model.parameters())
    ema = [p.detach().clone().float() for p in params] if ema_decay else None
    rng = np.random.default_rng(2000 + fold)
    nf = len(tabs)
    perm, pos = rng.permutation(len(tr_rows)), 0
    model.train()
    for step in range(1, n_steps + 1):
        if pos + BATCH > len(perm):
            perm, pos = rng.permutation(len(tr_rows)), 0
        idx = perm[pos:pos + BATCH]
        pos += BATCH
        seqs = [build(tabs, tr_rows[i], rng.permutation(nf), pre[int(y_tr[i])]) + [eos] for i in idx]
        ids, att = pad_batch(seqs, eos)
        ids, att = ids.cuda(non_blocking=True), att.cuda(non_blocking=True)
        labels = ids.masked_fill(att == 0, -100)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_ids=ids, attention_mask=att, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            d = min(ema_decay, (1 + step) / (10 + step))
            with torch.no_grad():
                torch._foreach_mul_(ema, d)
                torch._foreach_add_(ema, [p.detach().float() for p in params], alpha=1 - d)
        if step % 2000 == 0 or step == n_steps:
            print(f'fold {fold}: step {step}/{n_steps} (epoch {step * BATCH / len(tr_rows):.2f}) loss {loss.item():.4f}', flush=True)
    if ema is not None:
        with torch.no_grad():
            for p, e in zip(params, ema):
                p.copy_(e)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', required=True, choices=['small', 'medium', 'large'])
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    name = f'gpt2_{a.size}_target_first'
    y = target()
    f = folds(y)
    frame, n_train = load_frame()
    tok = load_tokenizer()
    tabs = fragment_tables(tok, frame, first_space=' ')
    check_tokenization(tok, frame, tabs)
    pre, eos = prefixes(tok), tok.eos_token_id
    orders = make_orders(N_ORDERS, len(tabs))
    te_rows = np.arange(n_train, len(frame))
    for k in fold_list(a.folds):
        if have_part(HERE, name, k):
            continue
        torch.manual_seed(1000 + k)
        tr_rows, va_rows = np.flatnonzero(f != k), np.flatnonzero(f == k)
        model = load_replica(a.size)
        finetune(model, tabs, pre, tr_rows, y[tr_rows], eos, k, EMA[a.size])
        for d in ('checkpoints', 'parts'):
            (HERE / d).mkdir(exist_ok=True)
        torch.save(model.state_dict(), HERE / 'checkpoints' / f'{name}_f{k}.pt')
        model.eval()
        fast_gelu(model)
        torch.backends.cuda.matmul.allow_tf32 = True
        llr = {}
        for split, rows in (('valid', va_rows), ('test', te_rows)):
            ll = score(model, tabs, pre, rows, orders, eos)
            np.save(HERE / 'parts' / f'{name}_f{k}_{split}_ll.npy', ll.astype(np.float32))
            llr[split] = logsumexp(ll[:, :, 1], 1) - logsumexp(ll[:, :, 0], 1)
        torch.backends.cuda.matmul.allow_tf32 = False
        save_part(HERE, name, k, expit(llr['valid']), expit(llr['test']))
        print(f'fold {k}: AUC {roc_auc_score(y[va_rows], llr["valid"]):.6f}', flush=True)
        del model
        torch.cuda.empty_cache()
    assemble(HERE, name, f, y)


if __name__ == '__main__':
    main()
