#!/usr/bin/env python
"""GPT-2 label-anywhere generative classifier, one model per fold.

Same recipe as ../gpt2_target_first (replica init, causal-LM fine-tuning on the outer-train rows), except that the label is
treated like any other field, exactly as GReaT's own training text does:
  fine-tune: all 22 fields (label included) in a fresh random order per row per step,
             text "<c1> is <v1>, <c2> is <v2>, ..., <c22> is <v22>" + eos;
  score:     log p(text) of the whole row (every token after the first, eos included), once with label True and once with
             label False in the same field order, for 30 fixed random orders.
Two readouts of the same fine-tuned model:
  any : orders with the label at a random position; the per-order LLRs are pooled by their mean ('mean') or by
        log-sum-exp per label ('lse');
  last: orders with the label LAST, i.e. P(label | all 21 fields): the target-last "Detective" readout. Both labels share every
        earlier token, so one pass gives log p(fields) and log p(' False' / ' True' | fields) (the final eos term is dropped).
Configs and their pairs:
  small_8ep  : small replica,  8 epochs, lr 5e-5 -> gpt2_small_anywhere_8ep_any (lse), gpt2_small_anywhere_8ep_last (lse)
  small_24ep : small replica, 24 epochs, lr 5e-5 -> gpt2_small_anywhere_24ep (mean)
  medium_8ep : medium replica, 8 epochs, lr 1e-4 -> gpt2_medium_anywhere_8ep (mean)
AdamW (no weight decay), linear decay without warmup, batch 32, grad clip 1, bf16 autocast. Scoring: fp16 body + fp32 LM head.
Pairs hold LLRs (logits); test = mean of the 5 fold models' LLRs.
Usage: python train.py --config small_8ep|small_24ep|medium_8ep [--folds 0,1,2,3,4]"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds, target  # noqa: E402
from common.gpt_rows import fast_gelu, fragment_tables, load_frame, load_replica, load_tokenizer, pad_batch  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

CONFIGS = {
    'small_8ep': dict(size='small', epochs=8, lr=5e-5,
                      readouts=[('any', 'lse', 'gpt2_small_anywhere_8ep_any'), ('last', 'lse', 'gpt2_small_anywhere_8ep_last')]),
    'small_24ep': dict(size='small', epochs=24, lr=5e-5, readouts=[('any', 'mean', 'gpt2_small_anywhere_24ep')]),
    'medium_8ep': dict(size='medium', epochs=8, lr=1e-4, readouts=[('any', 'mean', 'gpt2_medium_anywhere_8ep')]),
}
BATCH, N_ORDERS, LABEL = 32, 30, 21     # field index 21 = the label


def label_tables(tok):
    first = tok([f'{TARGET} is False', f'{TARGET} is True'], add_special_tokens=False)['input_ids']
    later = tok([f', {TARGET} is False', f', {TARGET} is True'], add_special_tokens=False)['input_ids']
    return first, later


def build(tabs, labf, labl, r, order, lab):
    """order: a permutation of 0..21 (21 = the label); lab: the label value written into the label field."""
    ids = []
    for pos, j in enumerate(order):
        if j == LABEL:
            ids += (labf if pos == 0 else labl)[lab]
        else:
            codes, first, later = tabs[j]
            ids += (first if pos == 0 else later)[codes[r]]
    return ids


def make_orders(k, mode, seed=2025):
    rng = np.random.default_rng(seed)
    if mode == 'any':
        return [list(rng.permutation(22)) for _ in range(k)]
    return [list(rng.permutation(21)) + [LABEL] for _ in range(k)]


@torch.inference_mode()
def score_any(model, tabs, labf, labl, rows, orders, eos, batch=128):
    """log p(text) for both label values with the same order: float64 [n, K, 2]."""
    W = model.lm_head.weight
    ll = np.zeros((len(rows), len(orders), 2))
    for oi, o in enumerate(orders):
        for s0 in range(0, len(rows), batch):
            rr = rows[s0:s0 + batch]
            seqs = [build(tabs, labf, labl, r, o, lab) + [eos] for lab in (0, 1) for r in rr]
            ids, att = pad_batch(seqs, eos)
            keep = att[:, 1:].bool()                                             # every real token after the first
            ids, keep = ids.cuda(), keep.cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                h = model.transformer(input_ids=ids, use_cache=False).last_hidden_state
            lp = torch.log_softmax(torch.nn.functional.linear(h[:, :-1].float(), W.float()), -1)
            lp = lp.gather(-1, ids[:, 1:, None]).squeeze(-1)
            tot = torch.where(keep, lp, torch.zeros_like(lp)).double().sum(1).cpu().numpy()
            m = len(rr)
            ll[s0:s0 + m, oi, 0], ll[s0:s0 + m, oi, 1] = tot[:m], tot[m:]
    return ll


@torch.inference_mode()
def score_last(model, tabs, labf, labl, rows, orders, eos, batch=128):
    """Label-last orders in one pass: log p(fields) + log p(' False' / ' True' | fields): float64 [n, K, 2]."""
    W = model.lm_head.weight
    t_false, t_true = labl[0][-1], labl[1][-1]
    assert labl[0][:-1] == labl[1][:-1]
    ll = np.zeros((len(rows), len(orders), 2))
    for oi, o in enumerate(orders):
        for s0 in range(0, len(rows), batch):
            rr = rows[s0:s0 + batch]
            seqs = [build(tabs, labf, labl, r, o, 0)[:-1] for r in rr]           # text up to ', satisfaction is'
            ids, att = pad_batch(seqs, eos)
            keep = att[:, 1:].bool()
            last = torch.tensor([len(x) - 1 for x in seqs])
            ids, keep = ids.cuda(), keep.cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                h = model.transformer(input_ids=ids, use_cache=False).last_hidden_state
            lp = torch.log_softmax(torch.nn.functional.linear(h[:, :-1].float(), W.float()), -1)
            lp = lp.gather(-1, ids[:, 1:, None]).squeeze(-1)
            prefix = torch.where(keep, lp, torch.zeros_like(lp)).double().sum(1)
            hl = h[torch.arange(len(seqs), device='cuda'), last.cuda()].float()
            lab = torch.log_softmax(torch.nn.functional.linear(hl, W.float()), -1)[:, [t_false, t_true]].double()
            ll[s0:s0 + len(rr), oi, :] = (prefix[:, None] + lab).cpu().numpy()
    return ll


def finetune(model, tabs, labf, labl, tr_rows, y_tr, eos, fold, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, fused=True)
    n_steps = int(math.ceil(epochs * len(tr_rows) / BATCH))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / n_steps))
    rng = np.random.default_rng(2000 + fold)
    perm, pos = rng.permutation(len(tr_rows)), 0
    model.train()
    for step in range(1, n_steps + 1):
        if pos + BATCH > len(perm):
            perm, pos = rng.permutation(len(tr_rows)), 0
        idx = perm[pos:pos + BATCH]
        pos += BATCH
        seqs = [build(tabs, labf, labl, tr_rows[i], list(rng.permutation(22)), int(y_tr[i])) + [eos] for i in idx]
        ids, att = pad_batch(seqs, eos)
        ids, att = ids.cuda(non_blocking=True), att.cuda(non_blocking=True)
        labels = ids.masked_fill(att == 0, -100)
        with torch.autocast('cuda', dtype=torch.bfloat16):                     # right padding + causal attention: no mask needed
            loss = model(input_ids=ids, labels=labels, use_cache=False).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == n_steps:
            print(f'fold {fold}: step {step}/{n_steps} (epoch {step * BATCH / len(tr_rows):.2f}) loss {loss.item():.4f}', flush=True)
    return model


def pool(ll, how):
    if how == 'mean':
        return (ll[:, :, 1] - ll[:, :, 0]).mean(1)
    return logsumexp(ll[:, :, 1], 1) - logsumexp(ll[:, :, 0], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=list(CONFIGS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    cfg = CONFIGS[a.config]
    y = target()
    f = folds(y)
    frame, n_train = load_frame()
    tok = load_tokenizer()
    tabs = fragment_tables(tok, frame, first_space='')
    labf, labl = label_tables(tok)
    rng0, cols = np.random.default_rng(1), list(frame.columns)
    for r in rng0.choice(len(frame), 200, replace=False):                       # tokenization check vs the plain GReaT text
        o, lab = list(rng0.permutation(22)), int(rng0.integers(2))
        txt = ', '.join((f'{TARGET} is {"True" if lab else "False"}' if j == LABEL else f'{cols[j]} is {frame.iat[r, j]}') for j in o)
        assert build(tabs, labf, labl, r, o, lab) == tok.encode(txt), txt
    eos = tok.eos_token_id
    te_rows = np.arange(n_train, len(frame))
    for k in fold_list(a.folds):
        if all(have_part(HERE, name, k) for _, _, name in cfg['readouts']):
            continue
        torch.manual_seed(1000 + k)
        tr_rows, va_rows = np.flatnonzero(f != k), np.flatnonzero(f == k)
        model = load_replica(cfg['size'])
        fast_gelu(model)
        finetune(model, tabs, labf, labl, tr_rows, y[tr_rows], eos, k, cfg['epochs'], cfg['lr'])
        model.eval()
        torch.backends.cuda.matmul.allow_tf32 = True
        for mode, how, name in cfg['readouts']:
            orders = make_orders(N_ORDERS, mode)
            fn = score_last if mode == 'last' else score_any
            llr = {split: pool(fn(model, tabs, labf, labl, rows, orders, eos), how)
                   for split, rows in (('valid', va_rows), ('test', te_rows))}
            save_part(HERE, name, k, llr['valid'], llr['test'])
            print(f'fold {k} {name}: AUC {roc_auc_score(y[va_rows], llr["valid"]):.6f}', flush=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        del model
        torch.cuda.empty_cache()
    for _, _, name in cfg['readouts']:
        assemble(HERE, name, f, y)


if __name__ == '__main__':
    main()
