#!/usr/bin/env python
"""Per-fold dense input matrices for the DCN (and for ../ydf), built with ../xgb_density_margin/lib.

Both matrices start from the same base: assemble flags te, teavg (3 inner seeds), te2s, orig, pmi, imp, geo, woe, slim with
te_smooth 1 and te2s_smooth 20, then the smooth fields (sf). Then:
  mlm3 (253 columns): cmp3, den384v3 (dn2_), mfill, mfill2, rrd, plift            -> cache/X_f{k}_{tr,te}.npy, cols_f{k}.json
  mlm4 (292 columns): cmpv4, denv4 (dn4_), denv5 (dn5_), mfill4, rrd, plift, cbank, wbank, qfill
                                                                                   -> cache/mlm4/X_f{k}_{tr,te}.npy, ...
The probability channels (te_, tes_, te2_, ter_, teg*) are turned into logits, missing values are filled with the
outer-train medians, float32. The exact-value token codes of the 12 columns (vocabulary over train + test, "__NA__" is a
token) go to cache/tokcodes_{tr,te}.npy and cache/tokmeta.json (fold-independent, written once).

Usage: python build_cache.py mlm3|mlm4 [--folds 0,1,2,3,4]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "xgb_density_margin"))
from common.data import FEATURES, cache_dir  # noqa: E402
from common.pairs import fold_list  # noqa: E402
from lib.assemble import assemble  # noqa: E402
from lib.channels import load_block  # noqa: E402
from lib.smooth_fields import add_smooth_fields  # noqa: E402

BASE_CFG = {"te", "teavg", "te2s", "orig", "pmi", "imp", "geo", "woe", "slim"}
BASE_PAR = {"te_smooth": 1.0, "te2s_smooth": 20.0}
MATRICES = {
    "mlm3": ("cmp3", "den384v3", "mfill", "mfill2", "rrd", "plift"),
    "mlm4": ("cmpv4", "denv4", "denv5", "mfill4", "rrd", "plift", "cbank", "wbank", "qfill"),
}
PROB_PREFIXES = ("te_", "tes_", "te2_", "ter_", "teg")


def out_dir(matrix):
    d = cache_dir("dcn")
    if matrix != "mlm3":
        d = d / matrix
        d.mkdir(parents=True, exist_ok=True)
    return d


def build_token_codes():
    d = cache_dir("dcn")
    if (d / "tokmeta.json").exists():
        return
    src = cache_dir("density_features")
    tok_tr = pd.read_parquet(src / "tok_tr.parquet")
    tok_te = pd.read_parquet(src / "tok_te.parquet")
    feats = list(FEATURES)
    codes_tr = np.zeros((len(tok_tr), len(feats)), dtype=np.int32)
    codes_te = np.zeros((len(tok_te), len(feats)), dtype=np.int32)
    vocab_sizes = []
    for j, c in enumerate(feats):
        cat = pd.Categorical(pd.concat([tok_tr[c], tok_te[c]], ignore_index=True))
        codes = cat.codes.astype(np.int32)
        assert codes.min() >= 0
        codes_tr[:, j] = codes[:len(tok_tr)]
        codes_te[:, j] = codes[len(tok_tr):]
        vocab_sizes.append(len(cat.categories))
    np.save(d / "tokcodes_tr.npy", codes_tr)
    np.save(d / "tokcodes_te.npy", codes_te)
    json.dump({"cols": feats, "vocab_sizes": vocab_sizes}, open(d / "tokmeta.json", "w"), indent=1)
    print(f"token codes, vocabulary sizes {vocab_sizes}", flush=True)


def build_fold(matrix, fold):
    t0 = time.time()
    D = assemble(fold, BASE_CFG, BASE_PAR)
    A, B, trn_idx = D["A"], D["B"], D["trn_idx"]
    A, B = add_smooth_fields(A, B, D["y"], trn_idx, D["val_idx"], D["train"], D["test"])
    for flag in MATRICES[matrix]:
        tr, te = load_block(flag, fold)
        A, B = pd.concat([A, tr], axis=1), pd.concat([B, te], axis=1)
    cols = list(A.columns)
    assert list(B.columns) == cols, "train/test column mismatch"
    pcols = [c for c in cols if c.startswith(PROB_PREFIXES)]
    for Df in (A, B):
        for c in pcols:
            p = np.clip(Df[c].astype(np.float64), 1e-4, 1 - 1e-4)
            Df[c] = np.log(p / (1 - p)).astype(np.float32)
    med = A.iloc[trn_idx].median()   # outer-train medians
    Af = A.fillna(med).astype(np.float32)
    Bf = B.fillna(med).astype(np.float32)
    assert np.isfinite(Af.values).all() and np.isfinite(Bf.values).all()
    d = out_dir(matrix)
    np.save(d / f"X_f{fold}_tr.npy", Af.values)
    np.save(d / f"X_f{fold}_te.npy", Bf.values)
    json.dump({"cols": cols}, open(d / f"cols_f{fold}.json", "w"), indent=1)
    print(f"{matrix} fold {fold}: {Af.shape[1]} columns ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix", choices=list(MATRICES))
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    build_token_codes()
    for k in fold_list(args.folds):
        build_fold(args.matrix, k)


if __name__ == "__main__":
    main()
