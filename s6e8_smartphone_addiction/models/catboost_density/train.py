#!/usr/bin/env python
"""GPU CatBoost, two input paths.

tokens:   the CatBoost-native matrix of build_cat_features.py / build_fills.py: 36 numerics (raw, codes, missing flags,
          budget geometry) + optional fills and MLM blocks, and 27 categorical string tokens (exact tokens, 0.1 tokens,
          region, pair and region crossings) encoded by CatBoost's ordered CTRs (no hand-made target encoding).
channels: the numeric per-fold channel matrix of ../xgb_density_margin/lib (nested TEs, MLM channels, lift banks; no
          categorical features) with the WOE + completion margin as CatBoost's baseline.
Per outer fold: early stopping (eval AUC, od_wait `es`) on a stratified 10% split of the outer-train rows (seed 99), then
a refit on all outer-train rows at 1.11x the best iteration count. Test = mean of the 5 fold models.

Usage: python train.py <pair> [--folds 0,1,2,3,4] [--force]
Folds can run in separate processes; the pair is written once all 5 folds are done."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "xgb_density_margin"))
from common.data import cache_dir, folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib.channels import channel_matrix, load_block  # noqa: E402

TOKEN_PAR = {"lr": 0.02, "es": 700, "ctrc": 4, "ohe": 1, "rs": 2.9, "bt": 0.26}
RECIPES = {
    "catboost_tokens_fills": dict(  # 48 numerics + 27 categoricals
        path="tokens", cfg="tok,pool,reg,pair,fill", par=TOKEN_PAR),
    "catboost_tokens_mlm3": dict(  # 158 + 27
        path="tokens", cfg="tok,pool,reg,pair,fill,mfill,mfill2,rrd,cmp3,bp2", par=TOKEN_PAR),
    "catboost_tokens_mlm4": dict(  # 169 + 27; od_wait 400
        path="tokens", cfg="tok,pool,reg,pair,fill,mfill4,cmpv4,denv4,rrd,bp2",
        par={"lr": 0.02, "depth": 6, "ctrc": 4, "rs": 2.9, "bt": 0.26, "ohe": 1}),
    "catboost_mlm4_banks_baseline": dict(  # 348 numerics, no categoricals
        path="channels",
        cfg="cnt,te,te2s,teavg,orig,pmi,imp,geo,lat,rte,mrg,cmpmrg,cmpv4,denv4,denv5,mfill4,qfill,sf,"
            "plift,cbank,wbank,wbank12,sbank,ebank",
        par={"lr": 0.005, "depth": 6, "ctrc": 4, "rs": 2.9, "bstype": "Bayesian", "bt": 0.26, "ohe": 1, "es": 1200,
             "n": 100000, "l2": 120.0, "acw": "Balanced",
             "te_smooth": 20.0, "te2s_smooth": 4.0, "mrg_scale": 0.4, "cmp_scale": 0.4}),
}
# token-path blocks, in the order they are appended after the numeric columns
TOKEN_BLOCKS = ("mfill4", "cmpv4", "denv4", "mfill", "mfill2", "rrd", "cmp3", "bp2")


def token_matrix(fold, cfg):
    src = cache_dir("catboost_density")
    Ctr, Cte = pd.read_parquet(src / "cat_tr.parquet"), pd.read_parquet(src / "cat_te.parquet")
    Ntr, Nte = pd.read_parquet(src / "num_tr.parquet"), pd.read_parquet(src / "num_te.parquet")
    ncols = [c for c in Ntr.columns if not c.startswith(("kllr", "cdfgap_", "amp_", "cnt_"))]
    A, B = Ntr[ncols].copy(), Nte[ncols].copy()
    if "fill" in cfg:
        A = pd.concat([A, pd.read_parquet(src / "fill_tr.parquet")], axis=1)
        B = pd.concat([B, pd.read_parquet(src / "fill_te.parquet")], axis=1)
    for flag in TOKEN_BLOCKS:
        if flag in cfg:
            tr, te = load_block(flag, fold)
            A, B = pd.concat([A, tr], axis=1), pd.concat([B, te], axis=1)
    ccols = []
    if "tok" in cfg:
        ccols += [c for c in Ctr.columns if c.startswith("tk_")]
    if "pool" in cfg:
        ccols += [c for c in Ctr.columns if c.startswith("t1_")]
    if "reg" in cfg:
        ccols += ["reg"]
    if "pair" in cfg:
        ccols += [c for c in Ctr.columns if c.startswith(("p_", "r_"))]
    for c in ccols:
        A[c] = Ctr[c].values
        B[c] = Cte[c].values
    return A, B, [A.columns.get_loc(c) for c in ccols]


def token_params(par):
    return dict(iterations=int(par.get("n", 20000)), learning_rate=float(par.get("lr", 0.06)),
                depth=int(par.get("depth", 6)), l2_leaf_reg=float(par.get("l2", 6.0)),
                random_strength=float(par.get("rs", 1.0)), bagging_temperature=float(par.get("bt", 1.0)),
                border_count=int(par.get("bc", 254)), one_hot_max_size=int(par.get("ohe", 4)),
                max_ctr_complexity=int(par.get("ctrc", 2)), boosting_type=par.get("boost", "Plain"),
                eval_metric="AUC", loss_function="Logloss", task_type="GPU", devices="0", random_seed=42, verbose=0,
                allow_writing_files=False)


def channel_params(par):
    p = dict(iterations=int(par.get("n", 30000)), learning_rate=float(par.get("lr", 0.02)),
             depth=int(par.get("depth", 8)), l2_leaf_reg=float(par.get("l2", 3.0)),
             random_strength=float(par.get("rs", 1.0)), border_count=int(par.get("bc", 254)),
             one_hot_max_size=int(par.get("ohe", 4)), max_ctr_complexity=int(par.get("ctrc", 2)),
             boosting_type=par.get("boost", "Plain"), bootstrap_type=par.get("bstype", "Bernoulli"),
             eval_metric="AUC", loss_function="Logloss", task_type="GPU", devices="0",
             random_seed=int(par.get("seed", 42)), verbose=0, allow_writing_files=False)
    if p["bootstrap_type"] == "Bernoulli":
        p["subsample"] = float(par.get("sub", 0.8))
    elif p["bootstrap_type"] == "Bayesian":
        p["bagging_temperature"] = float(par.get("bt", 1.0))
    if par.get("acw"):
        p["auto_class_weights"] = par["acw"]
    return p


def run_fold(pair, fold):
    t0 = time.time()
    r = RECIPES[pair]
    cfg, par = set(r["cfg"].split(",")), r["par"]
    y = target()
    fold_id = folds(y)
    trn_idx, val_idx = np.where(fold_id != fold)[0], np.where(fold_id == fold)[0]
    margin_tr = margin_te = None
    if r["path"] == "tokens":
        A, B, cat_idx = token_matrix(fold, cfg)
        params = token_params(par)
    else:
        D = channel_matrix(fold, cfg, par, post=("ebank",))
        A, B = D["A"], D["B"]
        margin_tr, margin_te = D["margin_tr"], D["margin_te"]
        for c in A.columns:
            if A[c].dtype == object:
                A[c] = pd.to_numeric(A[c], errors="coerce")
                B[c] = pd.to_numeric(B[c], errors="coerce")
        cat_idx = []
        params = channel_params(par)

    def pool(rows, label=True):
        """rows=None: the test rows. The margin (channels path) is the baseline of every pool."""
        X = B if rows is None else A.iloc[rows]
        kw = {}
        if margin_tr is not None:
            kw["baseline"] = np.asarray(margin_te if rows is None else margin_tr[rows], dtype=np.float64)
        return Pool(X, label=y[rows] if label and rows is not None else None, cat_features=cat_idx or None, **kw)

    i_tr, i_es = train_test_split(np.arange(len(trn_idx)), test_size=float(par.get("es_frac", 0.1)), random_state=99,
                                  stratify=y[trn_idx])
    m = CatBoostClassifier(**params, od_type="Iter", od_wait=int(par.get("es", 400)))
    m.fit(pool(trn_idx[i_tr]), eval_set=pool(trn_idx[i_es]), use_best_model=True)
    best = m.get_best_iteration() + 1
    inner_auc = float(m.get_best_score()["validation"]["AUC"])
    print(f"{pair} fold {fold}: best_iter {best}, inner AUC {inner_auc:.6f} ({time.time() - t0:.0f}s)", flush=True)

    m2 = CatBoostClassifier(**dict(params, iterations=int(best * float(par.get("rf", 1.11)))))
    m2.fit(pool(trn_idx), verbose=0)
    pv = m2.predict_proba(pool(val_idx, label=False))[:, 1]
    pt = m2.predict_proba(pool(None, label=False))[:, 1]
    save_part(HERE, pair, fold, pv, pt)
    print(f"{pair} fold {fold}: {A.shape[1]} columns ({len(cat_idx)} categorical), "
          f"valid AUC {roc_auc_score(y[val_idx], pv):.6f} ({time.time() - t0:.0f}s)", flush=True)


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
