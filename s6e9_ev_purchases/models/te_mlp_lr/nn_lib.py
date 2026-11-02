"""Tabular MLP and its 5-fold training loop, shared by every neural model in this folder.

Inputs (per outer fold): the standardized raw / log-count / flag columns (standardized over train + test), optionally logit/3 of
nested TEs (+ the 4 income TE differences / 3), optionally extra per-fold columns from cfg["extra_fn"](fold, fit_idx, valid_idx)
-> (fit, valid, test) arrays, standardized with the mean / std of the fit rows.
Network: Linear-BatchNorm-SiLU-Dropout blocks (default 512-256-128, dropout 0.15) -> 1 logit; optional linear skip from the inputs
to the logit (zero-initialized); optional fixed offset added to the logit (cfg["offset"], one value per train + test row).
Training: BCE, AdamW (lr 2e-3, wd 1e-5, betas 0.9 / 0.99), OneCycle (pct_start 0.15, cosine, div 20, final div 100), batch 4096,
bf16 autocast, 12 epochs; predictions after the last epoch. OOF = held-out folds; test = mean of the 5 fold models."""
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from lib.features import CHAMPION_TE, add_diffs, logit

# nested TEs of the 11 champion keys + income at m 10 + income // 100 at m 30, with the 4 income TE differences
TE_MLP = dict(num_blocks=["raw", "freq_log", "flags"], te_specs=CHAMPION_TE + [("inc", 10.0), ("inc_d100", 30.0)], te_diffs=True)


class TabMLP(nn.Module):
    def __init__(self, n_num, hidden=(512, 256, 128), dropout=0.15, skip=False):
        super().__init__()
        self.skip = nn.Linear(n_num, 1) if skip else None
        if skip:
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)
        d_in = n_num
        layers = []
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            d_in = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(d_in, 1)

    def forward(self, xn):
        out = self.out(self.mlp(xn)).squeeze(1)
        if self.skip is not None:
            out = out + self.skip(xn).squeeze(1)
        return out


def standardized_numerics(B, num_blocks):
    parts = {"raw": B.raw, "freq_log": B.freq_log, "flags": B.flags}
    num = pd.concat([parts[b] for b in num_blocks], axis=1).astype(np.float32).values
    mu, sd = num.mean(0), num.std(0) + 1e-6
    return ((num - mu) / sd).astype(np.float32)


def train_nn(ctx, cfg, log=print):
    """One seed over the 5 folds. Returns (oof, test, oof_auc, fold_aucs)."""
    B, y, folds = ctx.B, ctx.y, ctx.folds
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    num_all = standardized_numerics(B, cfg["num_blocks"])
    te_specs = cfg.get("te_specs", [])
    n_tr = B.n_tr
    oof, tpred, fold_auc = np.zeros(n_tr), np.zeros(B.n_te), []
    epochs, bs, lr, wd = cfg.get("epochs", 12), cfg.get("batch_size", 4096), cfg.get("lr", 2e-3), cfg.get("wd", 1e-5)
    dev = torch.device("cuda")
    off_all = cfg.get("offset")
    if off_all is None:
        off_all = np.zeros(B.n_tr + B.n_te, np.float32)
    for f, (a, b) in enumerate(folds):
        t0 = time.time()
        xn_fit, xn_val, xn_te = num_all[a], num_all[b], num_all[n_tr:]
        if te_specs:
            F, V, T = ctx.TE.fold(a, b, te_specs)
            if cfg.get("te_diffs", False):
                F, V, T = add_diffs(F), add_diffs(V), add_diffs(T)
                lg = lambda D: np.hstack([logit(D.values[:, :-4]) / 3, D.values[:, -4:] / 3])  # noqa: E731
                xn_fit, xn_val, xn_te = np.hstack([xn_fit, lg(F)]), np.hstack([xn_val, lg(V)]), np.hstack([xn_te, lg(T)])
            else:
                xn_fit = np.hstack([xn_fit, logit(F.values) / 3])
                xn_val = np.hstack([xn_val, logit(V.values) / 3])
                xn_te = np.hstack([xn_te, logit(T.values) / 3])
        if cfg.get("extra_fn") is not None:
            EF, EV, ET = cfg["extra_fn"](f, a, b)
            em, es = EF.mean(0), EF.std(0) + 1e-6
            xn_fit = np.hstack([xn_fit, (EF - em) / es])
            xn_val = np.hstack([xn_val, (EV - em) / es])
            xn_te = np.hstack([xn_te, (ET - em) / es])
        O = torch.tensor(np.asarray(off_all[a], np.float32), device=dev)
        O_v = torch.tensor(np.asarray(off_all[b], np.float32), device=dev)
        O_t = torch.tensor(np.asarray(off_all[n_tr:], np.float32), device=dev)
        Xn, Xn_v, Xn_t = torch.tensor(xn_fit, device=dev), torch.tensor(xn_val, device=dev), torch.tensor(xn_te, device=dev)
        Y = torch.tensor(y[a], dtype=torch.float32, device=dev)
        model = TabMLP(xn_fit.shape[1], hidden=tuple(cfg.get("hidden", (512, 256, 128))), dropout=cfg.get("dropout", 0.15),
                       skip=cfg.get("linear_skip", False)).to(dev)
        opt = torch.optim.AdamW([{"params": list(model.parameters()), "weight_decay": wd}], lr=lr, betas=(0.9, 0.99))
        steps = epochs * math.ceil(len(a) / bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr], total_steps=steps, pct_start=cfg.get("pct_start", 0.15),
                                                    anneal_strategy="cos", div_factor=20, final_div_factor=100)
        lossf = nn.BCEWithLogitsLoss()
        n = len(a)

        def predict(Xn_, O_):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                return torch.cat([torch.sigmoid(model(Xn_[s:s + 65536]).float() + O_[s:s + 65536])
                                  for s in range(0, len(Xn_), 65536)]).cpu().numpy()
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(Xn[idx])
                loss = lossf(out.float() + O[idx], Y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        model.eval()
        pv, pt = predict(Xn_v, O_v), predict(Xn_t, O_t)
        oof[b] = pv
        tpred += pt / 5
        fold_auc.append(roc_auc_score(y[b], pv))
        log(f"   fold {f}: AUC {fold_auc[-1]:.6f} ({time.time() - t0:.0f}s)")
        del Xn, Xn_v, Xn_t, Y, O, O_v, O_t, model, opt
        torch.cuda.empty_cache()
    return oof, tpred, roc_auc_score(y, oof), fold_auc
