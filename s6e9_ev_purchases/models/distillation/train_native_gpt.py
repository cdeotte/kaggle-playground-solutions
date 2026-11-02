#!/usr/bin/env python
"""Native GPT-2 Yes / No classifier with same-fold knowledge distillation (teacher weight 4).

Per outer fold k, starting from the original-only GPT-2 surrogate (gpt_surrogate.py), all parameters trainable:
the prompt is the 13 predictors in train.csv column order ending in ', Will_Buy_EV is' (gpt_prefix.py); the margin is
the last token's hidden state . (W[' Yes'] - W[' No']) of the tied output embedding. Each update takes 256 fit rows
with their true labels and 256 query rows (fold-k rows + test rows, cycled permutations) with the soft label of the
same-fold teacher (teacher.py), in micro-batches of 32 + 32; loss = (BCE_hard + 4 BCE_soft) / 5. AdamW peak lr 5e-5,
wd 0.01, 5% linear warmup then linear decay to 0, gradient clip 1, bf16 transformer with fp32 readout, 4 epochs, seed 42,
final epoch (~23 min per epoch on one GPU). Prediction: sigmoid of the mean margin of two views (train.csv and
original column orders). OOF = fold-k rows; test = mean of the 5 fold probabilities.
Usage: python train_native_gpt.py [--folds 0,1,2,3,4]   -> preds/gpt2_native_kd4_{oof,test}.npy"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, read_test, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from gpt_prefix import CACHE  # noqa: E402
from gpt_surrogate import SURROGATE  # noqa: E402
from teacher import teacher  # noqa: E402

PAIR = 'gpt2_native_kd4'
ORDERS = ['competition', 'source']
SEED = 42
CFG = dict(teacher_weight=4., epochs=4, microbatch=32, effective_true_batch=256, lr=5e-5, wd=.01, adam_betas=[.9, .999],
           adam_eps=1e-8, gradient_clip=1., training_order='competition', inference_batch=256)


def run_fold(k, y, f, n_test):
    import torch
    from scipy.special import expit
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from native_gpt_core import epoch_pairs, learning_rate, native_margin, token_batch, update
    torch.backends.cuda.matmul.allow_tf32 = True
    fit, valid = np.flatnonzero(f != k), np.flatnonzero(f == k)
    query = np.r_[valid, np.arange(len(f), len(f)+n_test)]
    nv = len(valid)
    soft = teacher(k, query)
    assert np.isfinite(soft).all() and np.all((soft > 0) & (soft < 1))
    fy = y[fit].astype(np.float64)
    banks = {o: (np.load(CACHE/f'gpt_prefix_{o}.npy', mmap_mode='r'), np.load(CACHE/f'gpt_prefix_{o}_lengths.npy'))
             for o in ORDERS}
    tok = AutoTokenizer.from_pretrained('gpt2')
    classes = [tok.encode(' '+w, add_special_tokens=False) for w in ['No', 'Yes']]
    assert all(len(c) == 1 for c in classes)
    no, yes = [c[0] for c in classes]
    model = AutoModelForCausalLM.from_pretrained(SURROGATE, use_safetensors=True, attn_implementation='sdpa').to('cuda')
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'], betas=tuple(CFG['adam_betas']),
                                  eps=CFG['adam_eps'], fused=True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    batch = CFG['effective_true_batch']
    total_steps = CFG['epochs']*math.ceil(len(fit)/batch)
    ids, lengths = banks[CFG['training_order']]
    step = 0
    for epoch in range(1, CFG['epochs']+1):
        started = time.time()
        fo, qo = epoch_pairs(len(fit), len(query), SEED, epoch)
        hs, ss, cnt = 0., 0., 0
        model.train()
        for start in range(0, len(fit), batch):
            end = min(start+batch, len(fit))
            fi, qi = fo[start:end], qo[start:end]
            rate = learning_rate(step, total_steps, CFG['lr'])
            for g in optimizer.param_groups:
                g['lr'] = rate
            hard, soft_loss, _ = update(model, optimizer, ids, lengths, fit[fi], query[qi], fy[fi], soft[qi], no, yes, CFG,
                                        'cuda')
            hs += hard*len(fi)
            ss += soft_loss*len(fi)
            cnt += len(fi)
            step += 1
            if step % 250 == 0:
                print(f'fold {k} epoch {epoch} step {step}: hard {hs/cnt:.5f} soft {ss/cnt:.5f}', flush=True)
        print(f'fold {k} epoch {epoch} done: hard {hs/cnt:.5f} soft {ss/cnt:.5f} ({time.time()-started:.0f} s)', flush=True)
    model.eval()
    views = []
    with torch.no_grad():
        for order in ORDERS:
            arr, lens = banks[order]
            pieces = []
            for start in range(0, len(query), CFG['inference_batch']):
                tb, tl = token_batch(arr, lens, query[start:start+CFG['inference_batch']], 'cuda')
                pieces.append(native_margin(model, tb, tl, no, yes, amp=True).cpu().numpy())
            views.append(np.concatenate(pieces))
    margin = np.column_stack(views).astype('float64').mean(axis=1)
    prediction = expit(margin)
    save_part(HERE, PAIR, k, prediction[:nv], prediction[nv:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    y = target()
    f = folds(y)
    n_test = len(read_test())
    for k in fold_list(a.folds):
        if not have_part(HERE, PAIR, k):
            run_fold(k, y, f, n_test)
    assemble(HERE, PAIR, f, y)


if __name__ == '__main__':
    main()
