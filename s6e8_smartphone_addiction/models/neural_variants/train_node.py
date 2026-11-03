#!/usr/bin/env python
"""NODE (neural oblivious decision ensembles) on the raw + derived + original-rate matrix and the nested target
encodings.

Inputs: the 50 label-free columns of lib/features.build_all (9 raw numerics, 23 derived, 18 original-data rates),
median-imputed and standardized on train + test, plus the fold's 36 nested target encodings (not scaled).
Model (lib/node_model.py): 2 ODT layers of 128 trees, depth 6 (63 thresholds), residual connections, linear head.
Training: Adam lr 3e-3, weight decay 1e-5, batch 1024, BCE, LR warm-up 3 epochs then cosine. Phase 1 trains on 90% of
the outer-train rows (first split of StratifiedKFold(10, seed 2026 + fold)) for up to 60 epochs and picks the epoch
with the best AUC on the other 10% (patience 10). Phase 2 trains a fresh model (same seed) on all outer-train rows for
that many epochs (the schedule spans that many epochs). Seed 2026 + fold. Test = mean of the 5 fold models.
Usage: python train_node.py [--folds 0,1,2,3,4]"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib import features as FEAT  # noqa: E402
from lib.node_model import NODE  # noqa: E402

NAME = 'node_oblivious_te'
SEED = 2026


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(model, X, device, batch=8192):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i:i + batch]).float().to(device)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(preds)


def make_optim_sched(model, lr, wd, epochs, warmup=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def run_epoch(model, X_tr, y_tr, optimizer, batch_size, device):
    model.train()
    n = X_tr.shape[0]
    perm = torch.randperm(n)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = X_tr[idx].to(device)
        yb = y_tr[idx].to(device)
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        loss.backward()
        optimizer.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--n_trees', type=int, default=128)
    ap.add_argument('--n_layers', type=int, default=2)
    ap.add_argument('--depth', type=int, default=6)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    train, test, y, folds = FEAT.load_data()
    Xraw_tr, Xraw_te, te_train, te_test, te_names = FEAT.build_all(train, test, y, folds)
    print(f'built features in {time.time() - t0:.0f}s: {Xraw_tr.shape[1]} raw+derived+orig, {len(te_names)} TE',
          flush=True)

    # NODE needs dense inputs: median imputation, then standardization (both fit on train + test, label-free)
    scaler = StandardScaler()
    imp = SimpleImputer(strategy='median')
    all_raw = np.vstack([Xraw_tr.values, Xraw_te.values])
    imp.fit(all_raw)
    scaler.fit(imp.transform(all_raw))
    Xraw_s = scaler.transform(imp.transform(Xraw_tr.values)).astype(np.float32)
    Xraw_te_s = scaler.transform(imp.transform(Xraw_te.values)).astype(np.float32)

    for f in fold_list(args.folds):
        if have_part(HERE, NAME, f):
            continue
        trn = np.where(folds != f)[0]
        val = np.where(folds == f)[0]
        X_fold = np.hstack([Xraw_s, te_train[f]]).astype(np.float32)
        X_test = np.hstack([Xraw_te_s, te_test[f]]).astype(np.float32)

        es = StratifiedKFold(10, shuffle=True, random_state=SEED + f)
        itrn, ival = next(iter(es.split(np.zeros(len(trn)), y[trn])))
        itrn, ival = trn[itrn], trn[ival]

        # phase 1: epoch selection on the inner split
        set_seed(SEED + f)
        model = NODE(X_fold.shape[1], n_layers=args.n_layers, n_trees=args.n_trees, depth=args.depth).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        X_itrn = torch.from_numpy(X_fold[itrn]).float()
        y_itrn = torch.from_numpy(y[itrn]).float()
        optimizer, scheduler = make_optim_sched(model, args.lr, 1e-5, args.epochs)
        best_auc, best_epoch = -1.0, 0
        for epoch in range(args.epochs):
            run_epoch(model, X_itrn, y_itrn, optimizer, args.batch, device)
            scheduler.step()
            p = predict(model, X_fold[ival], device)
            auc = roc_auc_score(y[ival], p)
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch + 1
            elif epoch + 1 - best_epoch >= 10:
                break

        # phase 2: refit on all outer-train rows for the selected number of epochs
        set_seed(SEED + f)
        model = NODE(X_fold.shape[1], n_layers=args.n_layers, n_trees=args.n_trees, depth=args.depth).to(device)
        X_trn = torch.from_numpy(X_fold[trn]).float()
        y_trn = torch.from_numpy(y[trn]).float()
        optimizer, scheduler = make_optim_sched(model, args.lr, 1e-5, best_epoch)
        for epoch in range(best_epoch):
            run_epoch(model, X_trn, y_trn, optimizer, args.batch, device)
            scheduler.step()

        oof_val = predict(model, X_fold[val], device)
        save_part(HERE, NAME, f, oof_val, predict(model, X_test, device))
        print(f'fold {f}: best_epoch={best_epoch} inner_auc={best_auc:.6f} val AUC={roc_auc_score(y[val], oof_val):.6f} '
              f'({n_params / 1e6:.2f}M params, {time.time() - t0:.0f}s)', flush=True)
    assemble(HERE, NAME, folds, y)


if __name__ == '__main__':
    main()
