#!/usr/bin/env python
"""DCNv2 (dcn_model.py) on a cached matrix (build_cache.py) + the 12 exact-value token codes.

Per outer fold: the channels are standardized with the outer-train mean / sd and their tails compressed
(|z| > 4 -> 4 + log1p(|z| - 4)). Epoch selection: train on 6/7 of the outer-train rows (StratifiedKFold(7, seed 777),
first split) for the full n_epochs schedule, AUC of the held-out 1/7 after every epoch, smoothed with a centred 3-point
window, argmax -> e. Refit: a fresh model (same seed) trained on all outer-train rows for the first e epochs of the same
n_epochs schedule; it predicts the outer-validation and test rows once. Test = mean of the 5 fold models.
Training: AdamW, batch 1024, linear warm-up (2% of the schedule) then cosine decay, BCE with label smoothing 0.005,
gradient-norm clip 5; learning rate groups: token embeddings lr_emb, front scale 6x lr (no weight decay), cross tower
cross_lrf x lr, the rest lr. Seed 42.

Usage: python train.py <pair> [--folds 0,1,2,3,4] [--force]"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import cache_dir, folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from dcn_model import DCN  # noqa: E402

DEV = "cuda"
BASE = dict(
    # model
    K=16, d_tok=16, emb_std=0.05, hidden=[512, 512, 256], p_drop=0.05, act="silu", plr_k=16, plr_d=4, plr_sigma=0.94,
    front_scale=1, tok_drop=0.1, n_cross=2, cross_drop=0.0, cross_gate=0.0, cross_ln=0, x0_ln=1,
    # training
    n_epochs=8, bs=1024, lr=2e-3, lr_emb=2e-3, wd=1e-2, emb_wd=1e-4, ls=0.005, warm=0.02, vf=0.15, clip=5.0,
    softclip=4.0, scale_lrf=6.0, cross_lrf=0.2, smooth_sel=1, min_ep=1, seed=42)
RECIPES = {
    "dcn_te_mlm3_plift": dict(matrix="mlm3", par=BASE),
    # the shipped fold-0 model used the 12-epoch schedule (epoch 3 selected), folds 1-4 the 8-epoch one
    "dcn_te_mlm4_banks": dict(matrix="mlm4", par=dict(BASE, cross_lrf=0.1), fold_par={0: dict(n_epochs=12)}),
}
# one cross field per engineered channel family (by column-name prefix); other columns form the "raw_geo" field
FAMILIES = ["te2_", "tes_", "teh_", "ter1_", "ter_", "teg_", "te_", "woe_", "cmp_", "dn4_", "dn5_", "cb_", "wb_", "qf_",
            "dn2_", "mf2_", "mf_", "rrd_", "plift", "pl_", "sf1_", "sf2_", "pmi_", "ollr_", "amp_", "cnt_", "imp_",
            "fill_", "na_", "share_"]

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def field_groups(cols):
    g = {}
    for j, c in enumerate(cols):
        g.setdefault(next((p for p in FAMILIES if c.startswith(p)), "raw_geo"), []).append(j)
    return [v for _, v in sorted(g.items())]


def run_fold(pair, fold):
    t0 = time.time()
    r = RECIPES[pair]
    P = dict(r["par"], **r.get("fold_par", {}).get(fold, {}))
    n_epochs, bs, seed, ls = int(P["n_epochs"]), int(P["bs"]), int(P["seed"]), float(P["ls"])

    src = cache_dir("dcn") if r["matrix"] == "mlm3" else cache_dir("dcn") / r["matrix"]
    Xtr = np.load(src / f"X_f{fold}_tr.npy")
    Xte = np.load(src / f"X_f{fold}_te.npy")
    groups = field_groups(json.load(open(src / f"cols_f{fold}.json"))["cols"])
    Ttr = np.load(cache_dir("dcn") / "tokcodes_tr.npy")
    Tte = np.load(cache_dir("dcn") / "tokcodes_te.npy")
    vocab = json.load(open(cache_dir("dcn") / "tokmeta.json"))["vocab_sizes"]
    y = target()
    fold_id = folds(y)
    trn_idx, val_idx = np.where(fold_id != fold)[0], np.where(fold_id == fold)[0]

    mu = Xtr[trn_idx].mean(0)
    sd = Xtr[trn_idx].std(0) + 1e-6

    def prep(X):   # standardize, then smooth compression of the tails
        Z = (X - mu) / sd
        c = float(P["softclip"])
        over = np.maximum(np.abs(Z) - c, 0.0)
        Z = np.sign(Z) * (np.minimum(np.abs(Z), c) + np.log1p(over))
        assert np.isfinite(Z).all()
        return Z.astype(np.float32)

    gX = torch.from_numpy(prep(Xtr)).to(DEV)
    gT = torch.from_numpy(Ttr).long().to(DEV)
    gy = torch.from_numpy(y).float().to(DEV)
    gXte = torch.from_numpy(prep(Xte)).to(DEV)
    gTte = torch.from_numpy(Tte).long().to(DEV)
    d_num = gX.shape[1]
    del Xtr, Xte

    def lr_at(prog):   # prog in [0, 1] over the full n_epochs schedule
        warm = float(P["warm"])
        if prog < warm:
            return prog / max(warm, 1e-9)
        return 0.5 * (1 + math.cos(math.pi * (prog - warm) / (1 - warm)))

    @torch.no_grad()
    def predict(model, idx=None, test=False, chunk=16384):
        model.eval()
        n = gXte.shape[0] if test else len(idx)
        out = torch.empty(n, device=DEV)
        for i in range(0, n, chunk):
            sl = slice(i, min(i + chunk, n))
            if test:
                xb, tb = gXte[sl], gTte[sl]
            else:
                j = idx[sl]
                xb, tb = gX[j], gT[j]
            out[sl] = torch.sigmoid(model(tb, xb)).mean(0)   # mean of the K members' probabilities
        return out.float().cpu().numpy()

    def build():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return DCN(vocab, d_num, P, groups).to(DEV)

    def train(model, fit_idx, n_run_epochs, eval_idx=None):
        """Trains n_run_epochs epochs of the n_epochs-long schedule; returns the per-epoch AUC on eval_idx."""
        glr = [float(P["lr_emb"]), P["lr"] * P["scale_lrf"], P["lr"] * P["cross_lrf"], P["lr"]]
        opt = torch.optim.AdamW([dict(g, lr=lr) for g, lr in zip(model.param_groups(P["wd"], P["emb_wd"]), glr)],
                                lr=P["lr"])
        gfit = torch.from_numpy(fit_idx).to(DEV)
        n = len(fit_idx)
        spe = max(1, n // bs)
        gen = torch.Generator(device=DEV)
        gen.manual_seed(seed)
        aucs = []
        for ep in range(n_run_epochs):
            model.train()
            perm = gfit[torch.randperm(n, generator=gen, device=DEV)]
            for s in range(spe):
                j = perm[s * bs:(s + 1) * bs]
                f = lr_at((ep + s / spe) / n_epochs)
                for pg, lr in zip(opt.param_groups, glr):
                    pg["lr"] = lr * f
                lg = model(gT[j], gX[j])
                tgt = gy[j] * (1 - 2 * ls) + ls
                loss = F.binary_cross_entropy_with_logits(lg, tgt.unsqueeze(0).expand_as(lg))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(P["clip"]))
                opt.step()
            if eval_idx is not None:
                aucs.append(roc_auc_score(y[eval_idx], predict(model, eval_idx)))
                print(f"  epoch {ep + 1}/{n_run_epochs}: inner AUC {aucs[-1]:.6f} loss {loss.item():.5f}", flush=True)
        return aucs

    # epoch selection on an inner split of the outer-train rows
    sp = StratifiedKFold(int(round(1 / float(P["vf"]))), shuffle=True, random_state=777)
    i_fit, i_hold = next(iter(sp.split(np.zeros(len(trn_idx)), y[trn_idx])))
    m = build()
    a = np.asarray(train(m, trn_idx[i_fit], n_epochs, trn_idx[i_hold]), dtype=np.float64)
    if int(P["smooth_sel"]) and len(a) >= 3:
        sm = np.convolve(np.r_[a[0], a, a[-1]], np.ones(3) / 3, mode="valid")
    else:
        sm = a
    min_ep = int(P["min_ep"])
    best_ep = int(np.argmax(sm[min_ep - 1:])) + min_ep
    print(f"{pair} fold {fold}: selected epoch {best_ep} (inner AUC {a[best_ep - 1]:.6f})", flush=True)
    del m
    torch.cuda.empty_cache()

    # refit on all outer-train rows
    m = build()
    train(m, trn_idx, best_ep)
    pv = predict(m, val_idx)
    pt = predict(m, test=True)
    save_part(HERE, pair, fold, pv, pt)
    print(f"{pair} fold {fold}: {d_num} channels + 12 tokens, valid AUC {roc_auc_score(y[val_idx], pv):.6f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    del m, gX, gT, gy, gXte, gTte
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(RECIPES))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    for k in fold_list(args.folds):
        if have_part(HERE, args.pair, k) and not args.force:
            print(f"{args.pair} fold {k}: done", flush=True)
            continue
        run_fold(args.pair, k)
    y = target()
    assemble(HERE, args.pair, folds(y), y)


if __name__ == "__main__":
    main()
