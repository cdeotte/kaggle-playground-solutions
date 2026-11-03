#!/usr/bin/env python
"""GPU XGBoost on the per-fold channel matrix (lib/) with a WOE (+ completion) base margin.

Per outer fold: early stopping (eval AUC, patience `es`) on a stratified 10% split of the outer-train rows (seed 99), then
a refit on all outer-train rows at 1.11x the best number of rounds. With `prune`, the refit model's gain importances drop
the `prune` weakest columns and the model is refit once more on the rest (same rounds). The base margin is passed to
every fit and prediction. Test = mean of the 5 fold models.

Usage: python train.py <pair> [--folds 0,1,2,3,4] [--force]
Folds can run in separate processes; the pair is written once all 5 folds are done."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib.channels import channel_matrix  # noqa: E402

BASE = "cnt,te,te2s,orig,pmi,imp,geo,lat,rte,mrg"
MLM3 = "teavg,cmp3,cmpmrg,den384v3,mfill,mfill2"
MLM4 = "teavg,cmpv4,cmpmrg,denv4,mfill4"
XGB_ONLY = "sf,ares,tokemb,dsx"
PAR_MLM = {"lr": 0.02, "es": 600, "npt": 6, "sub": 0.7, "cbn": 0.6, "mrg_scale": 0.5, "cmp_scale": 0.4,
           "te_smooth": 1.0, "te2s_smooth": 20.0}
PAR_PRUNE = {**PAR_MLM, "col": 0.7, "mrg_scale": 0.55, "prune": 120}
RECIPES = {   # columns before -> after pruning
    "xgb_te_woe_margin": dict(  # 172
        cfg=BASE, par={"lr": 0.015, "es": 600, "npt": 3, "sub": 0.7, "mrg_scale": 0.5}),
    "xgb_te_mlm3_margin": dict(  # 263
        cfg=f"{BASE},{MLM3}", par=PAR_MLM),
    "xgb_te_mlm3_sf_pruned": dict(  # 337 -> 217
        cfg=f"{BASE},{MLM3},{XGB_ONLY}", par=PAR_PRUNE),
    "xgb_te_mlm3_plift": dict(  # 347 -> 227
        cfg=f"{BASE},{MLM3},{XGB_ONLY},plift", par=PAR_PRUNE),
    "xgb_te_mlm3_plift_cbank": dict(  # 356 -> 236
        cfg=f"{BASE},{MLM3},{XGB_ONLY},plift,cbank", par=PAR_PRUNE),
    "xgb_te_mlm4_plift_cbank": dict(  # 341 -> 221
        cfg=f"{BASE},{MLM4},{XGB_ONLY},plift,cbank", par=PAR_PRUNE),
    "xgb_te_mlm4_qfill_wbank": dict(  # 386 -> 266
        cfg=f"{BASE},{MLM4},{XGB_ONLY},plift,cbank,denv5,qfill,wbank", par=PAR_PRUNE),
    "xgb_te_mlm4_qfill_wbank12_sbank": dict(  # 404 -> 284
        cfg=f"{BASE},{MLM4},{XGB_ONLY},plift,cbank,denv5,qfill,wbank,wbank12,sbank", par=PAR_PRUNE),
}


def xgb_params(par):
    return dict(n_estimators=int(par.get("n", 20000)), learning_rate=par.get("lr", 0.03),
                max_depth=int(par.get("depth", 6)), min_child_weight=par.get("mcw", 20),
                subsample=par.get("sub", 0.85), colsample_bytree=par.get("col", 0.8),
                reg_lambda=par.get("lam", 4.0), reg_alpha=par.get("alp", 0.02), gamma=par.get("gam", 0.0),
                max_bin=int(par.get("max_bin", 1024)), num_parallel_tree=int(par.get("npt", 1)),
                colsample_bynode=par.get("cbn", 1.0), objective="binary:logistic", eval_metric="auc",
                tree_method="hist", device="cuda", random_state=int(par.get("seed", 42)), n_jobs=8, verbosity=0)


def run_fold(pair, fold):
    t0 = time.time()
    cfg, par = set(RECIPES[pair]["cfg"].split(",")), RECIPES[pair]["par"]
    D = channel_matrix(fold, cfg, par, post=("ares", "tokemb", "dsx"))
    A, B, y = D["A"], D["B"], D["y"]
    trn_idx, val_idx = D["trn_idx"], D["val_idx"]
    margin_tr, margin_te = D["margin_tr"], D["margin_te"]
    n_cols = A.shape[1]

    def margin(rows):
        return {} if margin_tr is None else {"base_margin": margin_tr[rows]}

    params = xgb_params(par)
    i_tr, i_es = train_test_split(np.arange(len(trn_idx)), test_size=float(par.get("es_frac", 0.1)), random_state=99,
                                  stratify=y[trn_idx])
    fit_rows, es_rows = trn_idx[i_tr], trn_idx[i_es]
    m = xgb.XGBClassifier(**params, early_stopping_rounds=int(par.get("es", 300)))
    kw = {} if margin_tr is None else dict(base_margin=margin_tr[fit_rows], base_margin_eval_set=[margin_tr[es_rows]])
    m.fit(A.iloc[fit_rows], y[fit_rows], eval_set=[(A.iloc[es_rows], y[es_rows])], verbose=False, **kw)
    best = m.best_iteration + 1
    inner_auc = float(m.best_score)

    params2 = dict(params, n_estimators=int(best * float(par.get("rf", 1.11))))
    m2 = xgb.XGBClassifier(**params2)
    m2.fit(A.iloc[trn_idx], y[trn_idx], verbose=False, **margin(trn_idx))
    if par.get("prune"):
        for ndrop in [int(x) for x in str(par["prune"]).split("+")]:
            imp = m2.get_booster().get_score(importance_type="gain")
            gains = np.array([imp.get(c, 0.0) for c in A.columns])
            keep = list(np.array(A.columns)[np.argsort(-gains)[:len(A.columns) - ndrop]])
            A, B = A[keep], B[keep]
            m2 = xgb.XGBClassifier(**params2)
            m2.fit(A.iloc[trn_idx], y[trn_idx], verbose=False, **margin(trn_idx))

    pv = m2.predict_proba(A.iloc[val_idx], **margin(val_idx))[:, 1]
    pt = m2.predict_proba(B, **({} if margin_te is None else {"base_margin": margin_te}))[:, 1]
    save_part(HERE, pair, fold, pv, pt)
    print(f"{pair} fold {fold}: {n_cols} -> {A.shape[1]} columns, best_iter {best}, inner AUC {inner_auc:.6f}, "
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
