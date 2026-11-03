#!/usr/bin/env python
"""YDF gradient boosted trees (axis-aligned splits, Newton / hessian split gain) on the 253-column mlm3 matrix of ../dcn
(build_cache.py) + the 9 cbank lift channels of ../lift_banks.

Per outer fold: early stopping on a stratified 1/7 (about 15%) of the outer-train rows (StratifiedKFold(7, seed 777),
first split), trained on the other 6/7; the model keeps the best number of trees and predicts the outer-validation and
test rows. No refit. Test = mean of the 5 fold models. CPU only.

Usage: python train.py <pair> [--folds 0,1,2,3,4] [--threads 8] [--force]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ydf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import cache_dir, folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

RECIPES = {
    "ydf_hessian_te_mlm3_cbank": dict(
        vf=0.15, seed=42,
        learner=dict(num_trees=7000, shrinkage=0.03, early_stopping_num_trees_look_ahead=100,
                     num_candidate_attributes_ratio=0.3, min_examples=20, l2_regularization=4.0, use_hessian_gain=True)),
}


def run_fold(pair, fold, threads):
    t0 = time.time()
    r = RECIPES[pair]
    src = cache_dir("dcn")
    X = np.load(src / f"X_f{fold}_tr.npy")
    Xt = np.load(src / f"X_f{fold}_te.npy")
    cols = json.load(open(src / f"cols_f{fold}.json"))["cols"]
    lb = cache_dir("lift_banks")
    c_tr = pd.read_parquet(lb / f"cbank_f{fold}_tr.parquet")
    c_te = pd.read_parquet(lb / f"cbank_f{fold}_te.parquet")
    X = np.concatenate([X, c_tr.values.astype(np.float32)], 1)
    Xt = np.concatenate([Xt, c_te.values.astype(np.float32)], 1)
    cols = cols + list(c_tr.columns)
    assert len(cols) == X.shape[1] == Xt.shape[1] and len(set(cols)) == len(cols)

    y = target().astype(np.int32)
    fold_id = folds(y)
    trn_idx, val_idx = np.where(fold_id != fold)[0], np.where(fold_id == fold)[0]
    sp = StratifiedKFold(int(round(1 / r["vf"])), shuffle=True, random_state=777)
    i_fit, i_hold = next(iter(sp.split(np.zeros(len(trn_idx)), y[trn_idx])))
    fit_idx, hold_idx = trn_idx[i_fit], trn_idx[i_hold]

    def ds(idx=None, test=False, label=True):
        src_ = Xt if test else X
        d = {c: (src_[:, j] if test else src_[idx, j]) for j, c in enumerate(cols)}
        if label and not test:
            d["__y"] = y[idx]
        return d

    learner = ydf.GradientBoostedTreesLearner(label="__y", task=ydf.Task.CLASSIFICATION, num_threads=threads,
                                              random_seed=r["seed"], **r["learner"])
    model = learner.train(ds(fit_idx), valid=ds(hold_idx))
    pv = np.asarray(model.predict(ds(val_idx, label=False)), dtype=np.float64)
    pt = np.asarray(model.predict(ds(test=True)), dtype=np.float64)
    inner = roc_auc_score(y[hold_idx], np.asarray(model.predict(ds(hold_idx, label=False)), dtype=np.float64))
    save_part(HERE, pair, fold, pv, pt)
    print(f"{pair} fold {fold}: {len(cols)} columns, {model.num_trees()} trees, inner AUC {inner:.6f}, "
          f"valid AUC {roc_auc_score(y[val_idx], pv):.6f} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(RECIPES))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    ydf.verbose(0)
    for k in fold_list(args.folds):
        if have_part(HERE, args.pair, k) and not args.force:
            print(f"{args.pair} fold {k}: done", flush=True)
            continue
        run_fold(args.pair, k, args.threads)
    y = target()
    assemble(HERE, args.pair, folds(y), y)


if __name__ == "__main__":
    main()
