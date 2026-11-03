#!/usr/bin/env python
"""FT-Transformer on label-free imputation features, exact-value token embeddings and the nested target encodings.

Inputs (lib/ftt_features.py, lib/features.py): 47 numerics standardized on train + test (9 LightGBM-quantile fills,
9 fill uncertainties, 9 raw values with NaN -> fill, 20 budget-geometry columns from the fills) plus the fold's 36 nested
target encodings appended unscaled; 12 categoricals as embeddings (gender, stress, impact and the exact '%.2f' value
token of each numeric).
Model (lib/ftt_model.py): d_token 128, 3 layers, 8 heads, FFN factor 4/3, dropout 0.1 (attention, FFN and head), linear
numerical embedding (about 1.07M parameters).
Training: AdamW lr 3e-4, weight decay 1e-5, batch 1024, mixed precision, BCE, LR warm-up 5 epochs then cosine over
40 epochs; early stopping on 10% of the outer-train rows (first split of StratifiedKFold(10, seed 2026 + fold),
patience 10). The best-epoch weights predict the held-out fold and the test rows. Seed 2026 + fold. Test = mean of the
5 fold models.
--mc N averages N inference passes with dropout active (MC dropout); the stacked pair uses --mc 1 (no MC dropout).
Usage: python train_ftt.py [--folds 0,1,2,3,4] [--lr 3e-4] [--mc 1]"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import cache_dir  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib import features as FEAT  # noqa: E402
from lib import ftt_features as FF  # noqa: E402
from lib.ftt_model import FTTransformer  # noqa: E402

NAME = 'ftt_imputed_tokens_te'
SEED = 2026


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(model, X_num, X_cat, device, batch=4096, n_mc=1):
    if n_mc > 1:
        model.train()  # keep dropout active for MC averaging
    else:
        model.eval()
    preds = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for i in range(0, X_num.shape[0], batch):
            xb = torch.from_numpy(X_num[i:i + batch]).float().to(device)
            cb = torch.from_numpy(X_cat[i:i + batch]).long().to(device)
            if n_mc > 1:
                mc = torch.stack([torch.sigmoid(model(xb, cb)) for _ in range(n_mc)], dim=0).mean(dim=0)
                preds.append(mc.cpu().numpy())
            else:
                preds.append(torch.sigmoid(model(xb, cb)).cpu().numpy())
    return np.concatenate(preds)


def train_one_fold(model, X_num, X_cat, y, itrn, ival, epochs, batch_size, lr, wd, device, patience=10, warmup=5):
    """Train with early stopping on the inner validation rows; returns the best-epoch model, its AUC and its epoch."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    X_num_tr = torch.from_numpy(X_num[itrn]).float()
    X_cat_tr = torch.from_numpy(X_cat[itrn]).long()
    y_tr = torch.from_numpy(y[itrn]).float()
    y_va = y[ival]

    n = len(itrn)
    best_auc, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_num_tr[idx].to(device)
            cb = X_cat_tr[idx].to(device)
            yb = y_tr[idx].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                out = model(xb, cb)
                loss = loss_fn(out, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        pred = predict(model, X_num[ival], X_cat[ival], device)
        auc = roc_auc_score(y_va, pred)
        if auc > best_auc:
            best_auc, best_epoch, bad = auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_auc, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--d_token', type=int, default=128)
    ap.add_argument('--n_layers', type=int, default=3)
    ap.add_argument('--mc', type=int, default=1, help='MC dropout passes at inference (1 = off)')
    args = ap.parse_args()
    name = NAME if args.mc == 1 else f'{NAME}_mc{args.mc}'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # plain attention kernels (flash / memory-efficient SDPA can fail on odd sequence lengths)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    set_seed(SEED)

    t0 = time.time()
    train, test, y, folds = FEAT.load_data()
    X_num, X_cat, cat_card, num_names, cat_names = FF.build_features(
        train, test, cache_path=str(cache_dir('neural_variants') / 'ftt_features.npz'))
    n_tr = len(train)
    print(f'built features in {time.time() - t0:.0f}s: {X_num.shape[1]} numeric, {X_cat.shape[1]} categorical, '
          f'cat cardinalities sum {sum(cat_card)}', flush=True)

    # standardize the numerics (label-free, train + test)
    X_num = StandardScaler().fit_transform(X_num).astype(np.float32)

    # nested exact-value target encodings, given to the model as numeric inputs
    te_train, te_test, te_names = FEAT.build_te_features(train, test, y, folds)
    print(f'TE features: {len(te_names)}', flush=True)

    X_num_tr, X_cat_tr = X_num[:n_tr], X_cat[:n_tr]
    X_num_te, X_cat_te = X_num[n_tr:], X_cat[n_tr:]

    for f in fold_list(args.folds):
        if have_part(HERE, name, f):
            continue
        trn = np.where(folds != f)[0]
        val = np.where(folds == f)[0]
        X_num_fold = np.hstack([X_num_tr, te_train[f]]).astype(np.float32)
        X_num_test = np.hstack([X_num_te, te_test[f]]).astype(np.float32)
        # inner split for early stopping (10% of outer-train)
        es = StratifiedKFold(10, shuffle=True, random_state=SEED + f)
        itrn, ival = next(iter(es.split(np.zeros(len(trn)), y[trn])))
        itrn, ival = trn[itrn], trn[ival]

        set_seed(SEED + f)
        model = FTTransformer(X_num_fold.shape[1], cat_card, d_token=args.d_token, n_layers=args.n_layers, n_heads=8,
                              d_ffn_factor=4 / 3, attn_dropout=args.dropout, ffn_dropout=args.dropout,
                              head_dropout=args.dropout).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        model, best_auc, best_epoch = train_one_fold(model, X_num_fold, X_cat_tr, y, itrn, ival, args.epochs,
                                                     args.batch, args.lr, args.wd, device, patience=10)

        oof_val = predict(model, X_num_fold[val], X_cat_tr[val], device, n_mc=args.mc)
        save_part(HERE, name, f, oof_val, predict(model, X_num_test, X_cat_te, device, n_mc=args.mc))
        print(f'fold {f}: best_epoch={best_epoch} inner_auc={best_auc:.6f} val AUC={roc_auc_score(y[val], oof_val):.6f} '
              f'({n_params / 1e6:.2f}M params, {time.time() - t0:.0f}s)', flush=True)
    assemble(HERE, name, folds, y)


if __name__ == '__main__':
    main()
