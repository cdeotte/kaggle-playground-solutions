#!/usr/bin/env python
"""AFN-style factorization model over the 12 exact value tokens, with a label-free numeric side input and 8 internal
heads.

Tokens (lib/tokens.py): numerics f'{x:.2f}', categoricals as strings, NaN -> '__NA__'; vocabulary over train + test.
Side input (50 columns, label-free): the 44 columns of lib/side_features.py (9 raw numerics, 9 LightGBM q50 fills,
9 fill uncertainties, 17 budget-geometry columns from the fills) and the first 6 row channels of the masked-token
density model (lib/density_model.py, built by build_density.py): pseudo-log-likelihood of the observed cells, their
mean predictive entropy, the minimum log-probability, the mean entropy of the missing cells, and the numbers of observed
and missing cells.
Model (lib/fm_model.AFN): global bias + per-token linear weights + 8 MLP heads on [the 66 pair products of 32-dim token
factors (2112), side input (50)]; each head Linear(2162, 512) - ReLU - Dropout 0.3 - Linear(512, 128) - ReLU -
Dropout 0.3 - Linear(128, 1) on its own learned scaling of the input; dropout 0.8 on the token factors. Loss = BCE over
the 8 head logits; prediction = mean of the 8 head probabilities.
Training: AdamW lr 1e-3, weight decay 1e-5, batch 8192, cosine schedule. Only parameters without 'bias' in their name
and the global bias (no weight decay) are passed to the optimizer, so the heads' layer biases keep their initial values.
Phase 1 trains on 90% of the outer-train rows (first split of StratifiedKFold(10, seed 2026 + fold)) for 80 epochs and
picks the epoch with the best AUC on the other 10%; phase 2 trains a fresh model (same seed) on all outer-train rows for
that many epochs (cosine over that many epochs). Seed 2026 + fold. Test = mean of the 5 fold models.
Usage: python train_afn.py [--folds 0,1,2,3,4]"""
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
from lib import density_model as DM  # noqa: E402
from lib import side_features as S  # noqa: E402
from lib import tokens as T  # noqa: E402
from lib.fm_model import AFN  # noqa: E402

NAME = 'afn_tokens_density_heads'
SEED = 2026


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(args, vocab_sizes, n_side):
    return AFN(vocab_sizes, k=args.k, hidden=args.hidden, emb_dropout=args.emb_dropout, mlp_dropout=args.mlp_dropout,
               n_side=n_side, hidden2=args.hidden2, n_heads=args.n_heads)


def predict_prob(model, X, N):
    """Mean predicted probability (mean over the heads)."""
    if getattr(model, 'n_heads', 1) > 1:
        return torch.sigmoid(model(X, N, return_heads=True)).mean(dim=1)
    return torch.sigmoid(model(X, N))


def fit_epochs(model, X_tr, y_tr, X_iv, y_iv, N_tr, N_iv, args, epochs, device, eval_every=True, log_prefix=''):
    """Train for `epochs` epochs with a cosine LR; optionally track the inner AUC. Returns (model, best_epoch, best_auc)."""
    opt = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'bias' not in n], 'weight_decay': args.wd},
        {'params': [model.bias], 'weight_decay': 0.0},
    ], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    nheads = getattr(model, 'n_heads', 1)

    n = X_tr.shape[0]
    n_batches = (n + args.batch - 1) // args.batch
    best_epoch, best_auc = -1, -1.0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for b in range(n_batches):
            idx = perm[b * args.batch:(b + 1) * args.batch]
            if nheads > 1:
                all_logits = model(X_tr[idx], N_tr[idx], return_heads=True)  # (B, H)
                yb = y_tr[idx].unsqueeze(1).expand_as(all_logits)
                loss = nn.functional.binary_cross_entropy_with_logits(all_logits, yb)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(model(X_tr[idx], N_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if eval_every and X_iv is not None:
            model.eval()
            with torch.no_grad():
                pr = predict_prob(model, X_iv, N_iv).float().cpu().numpy()
            auc = roc_auc_score(y_iv, pr)
            marker = ''
            if auc > best_auc:
                best_auc, best_epoch = auc, ep
                marker = ' *'
            print(f'{log_prefix}ep {ep:3d} loss {tot / n:.5f} inner_auc {auc:.6f}{marker}', flush=True)
    return model, best_epoch, best_auc


def run_fold(f, ids_tr, ids_te, y, folds, vocab_sizes, side_tr, side_te, args, device):
    trn = np.where(folds != f)[0]
    val = np.where(folds == f)[0]

    # inner split of the outer-train rows (10%) for epoch selection
    es = StratifiedKFold(10, shuffle=True, random_state=SEED + f)
    es_trn, es_val = next(iter(es.split(np.zeros(len(trn)), y[trn])))
    itrn, ival = trn[es_trn], trn[es_val]

    X_all = torch.as_tensor(ids_tr, dtype=torch.long, device=device)
    y_all = torch.as_tensor(y, dtype=torch.float32, device=device)
    Xte = torch.as_tensor(ids_te, dtype=torch.long, device=device)
    N_all = torch.as_tensor(side_tr, dtype=torch.float32, device=device)
    N_te = torch.as_tensor(side_te, dtype=torch.float32, device=device)
    n_side = side_tr.shape[1]

    # phase 1: epoch selection on the inner split
    set_seed(SEED + f)
    model = make_model(args, vocab_sizes, n_side).to(device)
    model, best_epoch, best_auc = fit_epochs(model, X_all[itrn], y_all[itrn], X_all[ival], y[ival], N_all[itrn],
                                             N_all[ival], args, args.epochs, device, log_prefix=f'fold{f} es ')
    if best_epoch <= 0:
        best_epoch = args.epochs
    print(f'fold {f}: best_epoch={best_epoch} best_inner_auc={best_auc:.6f}', flush=True)

    # phase 2: refit on all outer-train rows for the selected number of epochs
    set_seed(SEED + f)
    model = make_model(args, vocab_sizes, n_side).to(device)
    model, _, _ = fit_epochs(model, X_all[trn], y_all[trn], None, None, N_all[trn], None, args, best_epoch, device,
                             eval_every=False)

    model.eval()
    with torch.no_grad():
        oof_val = predict_prob(model, X_all[val], N_all[val]).float().cpu().numpy()
        pr_te = predict_prob(model, Xte, N_te).float().cpu().numpy()
    return oof_val, pr_te, best_epoch, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--hidden2', type=int, default=128)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--emb_dropout', type=float, default=0.8)
    ap.add_argument('--mlp_dropout', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--dens_ncol', type=int, default=6, help='number of density channels in the side input')
    args = ap.parse_args()

    device = torch.device('cuda')
    t0 = time.time()
    train, test = read_train(), read_test()
    y_int = labels(train[TARGET])
    y, folds = y_int.astype(np.float64), fold_ids(y_int)
    ids_tr, ids_te, vocab_sizes, vocabs = T.build_token_ids(train, test)

    dens_tr, dens_te = DM.density_features(train, test, device=device, cache_tag=DM.TAG)
    if args.dens_ncol:
        dens_tr, dens_te = dens_tr[:, :args.dens_ncol], dens_te[:, :args.dens_ncol]
    f_tr, f_te = S.build_numeric_side_full(train, test)
    side_tr = np.hstack([f_tr, dens_tr]).astype(np.float32)
    side_te = np.hstack([f_te, dens_te]).astype(np.float32)
    print(f'data ready in {time.time() - t0:.1f}s; vocab_sizes={vocab_sizes.tolist()}; side input {side_tr.shape[1]} '
          f'columns', flush=True)

    for f in fold_list(args.folds):
        if have_part(HERE, NAME, f):
            continue
        t0 = time.time()
        oof_val, pr_te, best_epoch, best_inner = run_fold(f, ids_tr, ids_te, y, folds, vocab_sizes, side_tr, side_te,
                                                          args, device)
        save_part(HERE, NAME, f, oof_val, pr_te)
        print(f'fold {f}: best_epoch={best_epoch} inner_auc={best_inner:.6f} '
              f'val AUC={roc_auc_score(y[folds == f], oof_val):.6f} ({time.time() - t0:.0f}s)', flush=True)
    assemble(HERE, NAME, folds, y)


if __name__ == '__main__':
    main()
