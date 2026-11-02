#!/usr/bin/env python
"""GPT-2 fine-tuned per fold as a classifier on rows written as text in the generator's (GReaT) serialization:
  " Age is 66, Annual_Income_USD is 92887.0, Daily_Commute_km is 23.4, ..., Range_Anxiety_Level is Low, Will_Buy_EV is"
The score of a row is the next-token logit of " Yes" minus that of " No"; training minimises the cross-entropy over these
two logits.  No engineered features: the model reads the raw values with the generator's own tokenizer.

Per outer fold: the pretrained checkpoint (gpt2 or gpt2-medium) is fine-tuned on 95% of the outer-train rows (fixed csv
column order), 3 epochs, batch 128, AdamW (weight decay 0.01, betas 0.9 / 0.98), 5% linear warmup then linear decay,
bf16 autocast, gradient clipping 1.0.  Every 1000 steps (and at the end) the AUC on the other 5% of outer-train is measured;
the best checkpoint predicts the validation fold and the test rows.  Test = mean of the 5 fold models.

Usage: python train.py all | <pair> [--folds 0,1,2,3,4] [--save-model]   runs the missing folds, then writes preds/<pair>
       python train.py --run <run> [--folds 2]                            one run (finished folds are kept in runs/parts/)"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds, read_test, read_train, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_pair, save_part  # noqa: E402

RUN_DIR = HERE / 'runs'
SETTINGS = dict(epochs=3, bs=128, orders=1, shuf=0, hold=0.05, eval_every=1000, seed=7)
RUNS = {
    'gpt2_small_lr3e5': dict(model='gpt2', lr=3e-5),
    'gpt2_medium_lr2e5': dict(model='gpt2-medium', lr=2e-5),
    'gpt2_medium_lr3e5': dict(model='gpt2-medium', lr=3e-5),
}
PAIRS = {     # pair: runs, averaged in logit space
    'gpt2_small_text_cls': ['gpt2_small_lr3e5'],
    'gpt2_text_cls_blend3': ['gpt2_small_lr3e5', 'gpt2_medium_lr2e5', 'gpt2_medium_lr3e5'],
}


def lg7(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def sg(z):
    return 1 / (1 + np.exp(-z))


def run_fold(run, k, save_model=False):
    import torch
    import torch.nn.functional as Fn
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    cfg = dict(SETTINGS, **RUNS[run])
    seed, bs, n_orders = cfg['seed'], cfg['bs'], cfg['orders']
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev, t0 = 'cuda', time.time()

    def log(*a):
        print(f'[{run} f{k} {time.time() - t0:6.0f}s]', *a, flush=True)

    # one token-id list per (row, column) clause; rows are composed in any column order
    tr, te = read_train(), read_test()
    y = target().astype(np.int64)
    f = folds(y)
    NTR, NTE = len(tr), len(te)
    COLS = [c for c in te.columns if c != 'id']
    tok = GPT2TokenizerFast.from_pretrained(cfg['model'])
    COMMA, SUFFIX = tok.encode(',')[0], tok.encode(f' {TARGET} is')
    YES, NO, EOS = tok.encode(' Yes')[0], tok.encode(' No')[0], tok.eos_token_id
    allv = pd.concat([tr[COLS], te[COLS]], ignore_index=True)
    CL = []                                              # CL[c][i]: token ids of " <col> is <value>" of row i
    for c in COLS:
        u, inv = np.unique((' ' + c + ' is ' + allv[c].astype(str)).values, return_inverse=True)
        ids = tok(list(u), add_special_tokens=False)['input_ids']
        CL.append([ids[i] for i in inv])

    def compose(rows, orders):
        """rows: indices into train rows then test rows (test row j = NTR + j); one column order per row -> padded batch"""
        seqs = []
        for i, o in zip(rows, orders):
            s = []
            for c in o:
                s += CL[c][i]
                s.append(COMMA)
            s[-1:] = SUFFIX                              # the last comma becomes " Will_Buy_EV is"
            seqs.append(s)
        n = max(len(s) for s in seqs)
        ids, att = np.full((len(seqs), n), EOS, np.int64), np.zeros((len(seqs), n), np.int64)
        last = np.zeros(len(seqs), np.int64)
        for j, s in enumerate(seqs):
            ids[j, :len(s)], att[j, :len(s)], last[j] = s, 1, len(s) - 1
        return torch.from_numpy(ids), torch.from_numpy(att), torch.from_numpy(last)

    model = GPT2LMHeadModel.from_pretrained(cfg['model']).to(dev)
    model.config.pad_token_id = EOS

    def two_logits(ids, att, last):
        h = model.transformer(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state
        logits = model.lm_head(h[torch.arange(h.shape[0], device=dev), last.to(dev)])      # at the last real token
        return torch.stack([logits[:, NO], logits[:, YES]], 1)

    @torch.no_grad()
    def predict(rows, n_ord, order_seed, pbs=512):
        """mean over n_ord column orders of the Yes-minus-No logit"""
        model.eval()
        rng_o = np.random.default_rng(order_seed)
        orders = [rng_o.permutation(len(COLS)) for _ in range(n_ord)]
        if n_ord == 1 and not cfg['shuf']:
            orders = [np.arange(len(COLS))]
        out = np.zeros(len(rows))
        for o in orders:
            for b in range(0, len(rows), pbs):
                ids, att, last = compose(rows[b:b + pbs], [o] * len(rows[b:b + pbs]))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    l2 = two_logits(ids, att, last).float()
                out[b:b + pbs] += (l2[:, 1] - l2[:, 0]).cpu().numpy() / n_ord
        model.train()
        return out

    # outer-train rows -> 95% fit / 5% holdout (checkpoint choice)
    itr, iva = np.where(f != k)[0], np.where(f == k)[0]
    perm = np.random.RandomState(1000 + k).permutation(len(itr))
    nh = int(cfg['hold'] * len(itr))
    i_hold, i_fit = itr[perm[:nh]], itr[perm[nh:]]
    steps_per_epoch = len(i_fit) // bs
    total = cfg['epochs'] * steps_per_epoch
    warm = int(0.05 * total)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.98), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (total - s) / max(1, total - warm)) if s >= warm
        else (s + 1) / max(1, warm))
    log(f'fit {len(i_fit)}, holdout {len(i_hold)}, val {len(iva)} rows; {steps_per_epoch} steps x {cfg["epochs"]} epochs')
    rng = np.random.default_rng(seed * 1000 + k)
    best, step = (-1.0, None, 0), 0
    model.train()
    for ep in range(cfg['epochs']):
        order_rows = rng.permutation(i_fit)
        for b in range(steps_per_epoch):
            rows = order_rows[b * bs:(b + 1) * bs]
            orders = [rng.permutation(len(COLS)) for _ in rows] if cfg['shuf'] else [np.arange(len(COLS))] * len(rows)
            ids, att, last = compose(rows, orders)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                l2 = two_logits(ids, att, last)
            loss = Fn.cross_entropy(l2.float(), torch.from_numpy(y[rows]).to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % cfg['eval_every'] == 0 or step == total:
                ah = roc_auc_score(y[i_hold], predict(i_hold, 1, order_seed=7))
                if ah > best[0]:
                    best = (ah, {n: v.detach().to('cpu', copy=True) for n, v in model.state_dict().items()}, step)
                log(f'ep {ep} step {step}/{total} loss {loss.item():.4f} holdout AUC {ah:.6f} (best {best[0]:.6f} @ {best[2]})')
    model.load_state_dict(best[1])
    log(f'best checkpoint: step {best[2]}, holdout AUC {best[0]:.6f}')
    if save_model:
        (HERE / 'checkpoints').mkdir(exist_ok=True)
        torch.save(best[1], HERE / 'checkpoints' / f'{run}_f{k}.pt')

    pv = sg(predict(iva, n_orders, order_seed=seed + 11))
    pt = sg(predict(NTR + np.arange(NTE), n_orders, order_seed=seed + 11))
    log(f'validation AUC {roc_auc_score(y[iva], pv):.6f}')
    return pv, pt


def write_pair(pair):
    runs = PAIRS[pair]
    if not all(have_part(RUN_DIR, r, k) for r in runs for k in range(5)):
        print(f'{pair}: not all runs are done yet', flush=True)
        return
    y = target()
    f = folds(y)
    zo = zt = 0.0
    for r in runs:
        assemble(RUN_DIR, r, f, y)                       # run pair: runs/preds/<run>_{oof,test}.npy
        o, t = (np.load(RUN_DIR / 'preds' / f'{r}_{x}.npy') for x in ('oof', 'test'))
        zo, zt = zo + lg7(o) / len(runs), zt + lg7(t) / len(runs)
    save_pair(HERE, pair, sg(zo), sg(zt))
    print(f'{pair}: OOF AUC {roc_auc_score(y, zo):.6f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', nargs='?', default='all', choices=['all'] + list(PAIRS))
    ap.add_argument('--run', choices=list(RUNS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--save-model', action='store_true', help='keep the fine-tuned weights in checkpoints/')
    a = ap.parse_args()
    pairs = list(PAIRS) if a.pair == 'all' else [a.pair]
    runs = [a.run] if a.run else list(dict.fromkeys(r for p in pairs for r in PAIRS[p]))
    for r in runs:
        for k in fold_list(a.folds):
            if not have_part(RUN_DIR, r, k):
                pv, pt = run_fold(r, k, a.save_model)
                save_part(RUN_DIR, r, k, pv, pt)
    if not a.run:
        for p in pairs:
            write_pair(p)


if __name__ == '__main__':
    main()
