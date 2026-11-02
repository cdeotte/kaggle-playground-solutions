#!/usr/bin/env python
"""GPT-Neo 125M target-first generative classifier: the recipe of ../gpt2_target_first with GPT-Neo 125M instead of GPT-2.

1. GReaT replica: `python ../gpt2_replica/train_replica.py --size neo` (EleutherAI/gpt-neo-125M, 60 epochs on data.csv only).
2. Per fold: keep training the replica as a causal LM on the outer-train rows written label first
   ("satisfaction is True, <21 'column is value' fields in a fresh random order>"), 8 epochs, AdamW lr 5e-5 (no weight decay),
   linear decay without warmup, batch 32, grad clip 1, bf16 autocast, no EMA. GPT-Neo's config has no dropout.
3. Score with Bayes' rule: LLR = logsumexp_o log p(row, o | True) - logsumexp_o log p(row, o | False).
   The held-out fold uses orders 1-30 of a fixed seed-2025 order stream (the same 30 orders as ../gpt2_target_first).
   For the test rows, fold model k uses its own orders 30k+1..30(k+1) of that stream, so the 5-fold test mean covers 150 orders.
GPT-Neo uses GPT-2's BPE tokenizer (identical token ids), so the row-to-text code is shared.
Pair: OOF and test LLRs (logits); test = mean of the 5 fold models' LLRs.
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.gpt_rows import fast_gelu, fragment_tables, load_frame, load_replica, load_tokenizer  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

_spec = importlib.util.spec_from_file_location('gpt2_target_first', HERE.parent / 'gpt2_target_first' / 'train.py')
TF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TF)                 # prefixes, build, check_tokenization, make_orders, score, finetune

NAME = 'gpt_neo_target_first'
N_ORDERS = 30


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
    orders = TF.make_orders(5 * N_ORDERS + N_ORDERS, len(tabs))   # one seed-2025 stream: 160 orders
    te_rows = np.arange(n_train, len(frame))
    for k in fold_list(a.folds):
        if have_part(HERE, NAME, k):
            continue
        torch.manual_seed(1000 + k)
        tr_rows, va_rows = np.flatnonzero(f != k), np.flatnonzero(f == k)
        model = load_replica('neo')
        TF.finetune(model, tabs, pre, tr_rows, y[tr_rows], eos, k, ema_decay=0.0)
        model.eval()
        fast_gelu(model)
        torch.backends.cuda.matmul.allow_tf32 = True
        llr = {}
        for split, rows, ords in (('valid', va_rows, orders[:N_ORDERS]),
                                  ('test', te_rows, orders[N_ORDERS * k:N_ORDERS * (k + 1)])):
            ll = TF.score(model, tabs, pre, rows, ords, eos)
            llr[split] = logsumexp(ll[:, :, 1], 1) - logsumexp(ll[:, :, 0], 1)
        torch.backends.cuda.matmul.allow_tf32 = False
        save_part(HERE, NAME, k, llr['valid'], llr['test'])
        print(f'fold {k}: AUC {roc_auc_score(y[va_rows], llr["valid"]):.6f}', flush=True)
        del model
        torch.cuda.empty_cache()
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
