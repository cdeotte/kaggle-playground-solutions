#!/usr/bin/env python
"""Factorization machine over the 12 exact value tokens (no other features, no target encodings).

Tokens (lib/tokens.py): numerics f'{x:.2f}', categoricals as strings, NaN -> '__NA__'; vocabulary over train + test.
Model (lib/fm_model.FM): global bias + per-token linear weights + k = 16 factor embeddings per token with pairwise
interactions; dropout 0.5 on the factor embeddings.
Training: AdamW lr 1e-3, weight decay 1e-5 (0 for the global bias), batch 8192, BCE, cosine schedule. Phase 1 trains on
90% of the outer-train rows (first split of StratifiedKFold(10, seed 2026 + fold)) for 200 epochs and picks the epoch
with the best AUC on the other 10%; phase 2 trains a fresh model (same seed) on all outer-train rows for that many
epochs (cosine over that many epochs). Seed 2026 + fold. Test = mean of the 5 fold models.
Usage: python train_fm.py [--folds 0,1,2,3,4]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import TARGET, folds as fold_ids, labels, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib import tokens as T  # noqa: E402
from lib.fm_model import FM  # noqa: E402

NAME = 'fm_exact_tokens'
SEED = 2026


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_epochs(model, X_tr, y_tr, X_iv, y_iv, args, epochs, device, eval_every=True, log_prefix=''):
    """Train for `epochs` epochs with a cosine LR; optionally track the inner AUC. Returns (model, best_epoch, best_auc)."""
    opt = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'bias' not in n], 'weight_decay': args.wd},
        {'params': [model.bias], 'weight_decay': 0.0},
    ], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    n = X_tr.shape[0]
    n_batches = (n + args.batch - 1) // args.batch
    best_epoch, best_auc = -1, -1.0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for b in range(n_batches):
            idx = perm[b * args.batch:(b + 1) * args.batch]
            logits = model(X_tr[idx])
            loss = loss_fn(logits, y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if eval_every and X_iv is not None:
            model.eval()
            with torch.no_grad():
                pr = torch.sigmoid(model(X_iv)).float().cpu().numpy()
            auc = roc_auc_score(y_iv, pr)
            marker = ''
            if auc > best_auc:
                best_auc, best_epoch = auc, ep
                marker = ' *'
            print(f'{log_prefix}ep {ep:3d} loss {tot / n:.5f} inner_auc {auc:.6f}{marker}', flush=True)
    return model, best_epoch, best_auc


def run_fold(f, ids_tr, ids_te, y, folds, vocab_sizes, args, device):
    trn = np.where(folds != f)[0]
    val = np.where(folds == f)[0]

    # inner split of the outer-train rows (10%) for epoch selection
    es = StratifiedKFold(10, shuffle=True, random_state=SEED + f)
    es_trn, es_val = next(iter(es.split(np.zeros(len(trn)), y[trn])))
    itrn, ival = trn[es_trn], trn[es_val]

    X_all = torch.as_tensor(ids_tr, dtype=torch.long, device=device)
    y_all = torch.as_tensor(y, dtype=torch.float32, device=device)
    Xte = torch.as_tensor(ids_te, dtype=torch.long, device=device)

    # phase 1: epoch selection on the inner split
    set_seed(SEED + f)
    model = FM(vocab_sizes, k=args.k, emb_dropout=args.emb_dropout).to(device)
    model, best_epoch, best_auc = fit_epochs(model, X_all[itrn], y_all[itrn], X_all[ival], y[ival], args, args.epochs,
                                             device, log_prefix=f'fold{f} es ')
    if best_epoch <= 0:
        best_epoch = args.epochs
    print(f'fold {f}: best_epoch={best_epoch} best_inner_auc={best_auc:.6f}', flush=True)

    # phase 2: refit on all outer-train rows for the selected number of epochs
    set_seed(SEED + f)
    model = FM(vocab_sizes, k=args.k, emb_dropout=args.emb_dropout).to(device)
    model, _, _ = fit_epochs(model, X_all[trn], y_all[trn], None, None, args, best_epoch, device, eval_every=False)

    model.eval()
    with torch.no_grad():
        oof_val = torch.sigmoid(model(X_all[val])).float().cpu().numpy()
        pr_te = torch.sigmoid(model(Xte)).float().cpu().numpy()
    return oof_val, pr_te, best_epoch, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--emb_dropout', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=8192)
    args = ap.parse_args()

    t0 = time.time()
    train, test = read_train(), read_test()
    y_int = labels(train[TARGET])
    y, folds = y_int.astype(np.float64), fold_ids(y_int)
    ids_tr, ids_te, vocab_sizes, vocabs = T.build_token_ids(train, test)
    print(f'data ready in {time.time() - t0:.1f}s; vocab_sizes={vocab_sizes.tolist()}', flush=True)

    device = torch.device('cuda')
    for f in fold_list(args.folds):
        if have_part(HERE, NAME, f):
            continue
        t0 = time.time()
        oof_val, pr_te, best_epoch, best_inner = run_fold(f, ids_tr, ids_te, y, folds, vocab_sizes, args, device)
        save_part(HERE, NAME, f, oof_val, pr_te)
        print(f'fold {f}: best_epoch={best_epoch} inner_auc={best_inner:.6f} '
              f'val AUC={roc_auc_score(y[folds == f], oof_val):.6f} ({time.time() - t0:.0f}s)', flush=True)
    assemble(HERE, NAME, folds, y)


if __name__ == '__main__':
    main()
