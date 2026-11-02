#!/usr/bin/env python
"""GPT-2 target-first generative classifier, one model per fold.

1. Start from the e55 GReaT replica (../gpt2_replica, trained on the original data only).
2. Keep training it as a causal LM on the outer-train rows of the fold, written like the generator wrote them: the label first,
   then the 13 fields in a fresh random order per row per step:
       "Will_Buy_EV is Yes, Age is 36, City_Type is Urban, ..." + eos
   12 epochs, AdamW lr 5e-5 (no weight decay), linear decay without warmup, batch 32, grad clip 1, bf16 autocast.
3. Score with Bayes' rule: for each label, the summed log-probabilities of all field tokens after the label prefix, for 6 fixed
   random field orders (fp32 activations, TF32 matmuls); the orders are pooled by log-sum-exp per label:
       LLR = logsumexp_o log p(row, o | Yes) - logsumexp_o log p(row, o | No)
Pair gpt2_target_first_12ep: OOF = sigmoid(LLR); test = mean of the 5 fold models' sigmoid(LLR).
Also kept: parts/<pair>_f<k>_<split>_ll.npy, float32 [rows, 6, 2] = log p(fields | No / Yes) per order.
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds, target  # noqa: E402
from common.gpt_rows import SOURCE_ORDER, load_replica, load_tokenizer, random_orders, read_frame  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

NAME = 'gpt2_target_first_12ep'
REPLICA, EPOCHS, LR, BATCH, ORDERS, SCORE_BATCH = 'e55', 12, 5e-5, 32, 'rand6', 192


def fragments(tok, sub):
    """Token ids of ' <c> is <v>' (first field, right after the label prefix) and ', <c> is <v>' (later fields) for every row."""
    fr = {}
    for j, c in enumerate(SOURCE_ORDER):
        v = sub[c].to_numpy()
        fr[(j, 0)] = tok([f' {c} is {x}' for x in v], add_special_tokens=False)['input_ids']
        fr[(j, 1)] = tok([f', {c} is {x}' for x in v], add_special_tokens=False)['input_ids']
    return fr


def build(fr, r, order_idx, pre):
    ids = list(pre)
    for pos, j in enumerate(order_idx):
        ids += fr[(j, 0 if pos == 0 else 1)][r]
    return ids, len(pre)


def finetune(model, tok, fr_tr, y_tr, pre, k):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    n_steps = int(math.ceil(EPOCHS * len(y_tr) / BATCH))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / n_steps))
    rng = np.random.default_rng(2000 + k)
    model.train()
    step, perm, pos = 0, rng.permutation(len(y_tr)), 0
    t0 = time.time()
    while step < n_steps:
        if pos + BATCH > len(perm):
            perm, pos = rng.permutation(len(y_tr)), 0
        idx = perm[pos:pos + BATCH]
        pos += BATCH
        seqs = [build(fr_tr, r, rng.permutation(13), pre[int(y_tr[r])])[0] + [tok.eos_token_id] for r in idx]
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), tok.eos_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for d, x in enumerate(seqs):
            ids[d, :len(x)] = torch.tensor(x)
            att[d, :len(x)] = 1
        ids, att = ids.cuda(), att.cuda()
        labels = ids.masked_fill(att == 0, -100)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_ids=ids, attention_mask=att, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 2000 == 0 or step == n_steps:
            print(f'fold {k}: step {step}/{n_steps} loss {loss.item():.4f} [{time.time() - t0:.0f}s]', flush=True)


def score(model, tok, frame, rows, oidx, pre):
    """log p(fields | label) per row x order x label (label prefix masked): float32 [n, K, 2]."""
    fr = fragments(tok, frame.iloc[rows].astype(str))
    n = len(rows)
    ll = np.zeros((n, len(oidx), 2), np.float32)
    for oi, oj in enumerate(oidx):
        for s0 in range(0, n, SCORE_BATCH):
            rr = range(s0, min(n, s0 + SCORE_BATCH))
            seqs, pl = [], []
            for lab in (0, 1):
                for r in rr:
                    x, npre = build(fr, r, oj, pre[lab])
                    seqs.append(x)
                    pl.append(npre)
            L = max(len(x) for x in seqs)
            ids = np.full((len(seqs), L), tok.eos_token_id, np.int64)
            keep = np.zeros((len(seqs), L - 1), bool)
            for d, (x, npre) in enumerate(zip(seqs, pl)):
                ids[d, :len(x)] = x
                keep[d, npre - 1:len(x) - 1] = True                    # predicted tokens = field tokens only
            att = (np.arange(L)[None, :] < np.array([len(x) for x in seqs])[:, None]).astype(np.int64)
            ids_t, att_t, keep_t = torch.as_tensor(ids).cuda(), torch.as_tensor(att).cuda(), torch.as_tensor(keep).cuda()
            with torch.inference_mode():
                lg = model(input_ids=ids_t, attention_mask=att_t).logits[:, :-1]
                logp = lg.gather(-1, ids_t[:, 1:, None]).squeeze(-1) - torch.logsumexp(lg, -1)
                tot = torch.where(keep_t, logp, torch.zeros_like(logp)).double().sum(1).cpu().numpy()
            m = len(rr)
            ll[s0:s0 + m, oi, 0], ll[s0:s0 + m, oi, 1] = tot[:m], tot[m:]
    return ll


def marginal_llr(ll):
    ll = ll.astype(np.float64)
    return logsumexp(ll[..., 1], 1) - logsumexp(ll[..., 0], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    y = target()
    f = folds(y)
    frame, n_train = read_frame()
    tok = load_tokenizer()
    pre = [tok.encode(f'{TARGET} is No,'), tok.encode(f'{TARGET} is Yes,')]
    oidx = [[SOURCE_ORDER.index(c) for c in o] for o in random_orders(ORDERS)]
    te = np.arange(n_train, len(frame))
    for k in fold_list(a.folds):
        if have_part(HERE, NAME, k):
            continue
        torch.manual_seed(1000 + k)
        tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
        model = load_replica(REPLICA, from_config=True).cuda()
        finetune(model, tok, fragments(tok, frame.iloc[tr].astype(str)), y[tr], pre, k)
        model.eval()
        torch.backends.cuda.matmul.allow_tf32 = True
        llr = {}
        for split, rows in (('valid', va), ('test', te)):
            ll = score(model, tok, frame, rows, oidx, pre)
            (HERE / 'parts').mkdir(exist_ok=True)
            np.save(HERE / 'parts' / f'{NAME}_f{k}_{split}_ll.npy', ll)
            llr[split] = marginal_llr(ll)
        torch.backends.cuda.matmul.allow_tf32 = False
        save_part(HERE, NAME, k, expit(llr['valid']), expit(llr['test']))
        print(f'fold {k}: AUC {roc_auc_score(y[va], llr["valid"]):.6f}', flush=True)
        del model
        torch.cuda.empty_cache()
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
