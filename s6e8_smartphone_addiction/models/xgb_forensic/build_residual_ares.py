#!/usr/bin/env python
"""Source-tempered exact-token response cache of each outer fold (lib/residual_ares.py), read by the residual-refit
and periodic-RBF XGBoost models as their 14 'ares' contrasts.

Outer-training rows: 5-way complement cross-fit over splitmix64(id) mod 5 groups; validation and test rows: fit on all
outer-training rows. Writes cache/ares_production/fold_{k}/{train,valid,test}_features.npy (44 columns),
{train,valid}_indices.npy and manifest.json (feature_names).
Usage: python build_residual_ares.py [--folds 0,1,2,3,4]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import TARGET, folds as fold_ids, read_original, read_test, read_train  # noqa: E402
from common.pairs import fold_list  # noqa: E402
from models.xgb_forensic.lib.external import xgb_cache  # noqa: E402
from models.xgb_forensic.lib.residual_ares import fit_apply, tempered_names  # noqa: E402
from models.xgb_forensic.lib.residual_surfaces import splitmix_group  # noqa: E402


def build_fold(fold, train, test, source, outer):
    target = train[TARGET].to_numpy(np.int8)
    ids = train["id"].to_numpy(np.int64)
    train_index = np.flatnonzero(outer != fold)
    valid_index = np.flatnonzero(outer == fold)
    fit = train.iloc[train_index].reset_index(drop=True)
    valid = train.iloc[valid_index].reset_index(drop=True)
    fit_y = target[train_index]
    groups = splitmix_group(ids[train_index])
    names = tempered_names()
    train_features = np.empty((len(fit), len(names)), dtype=np.float32)
    for group in range(5):
        apply_mask = groups == group
        train_features[apply_mask] = fit_apply(fit.loc[~apply_mask].reset_index(drop=True), fit_y[~apply_mask],
                                               fit.loc[apply_mask].reset_index(drop=True), source)
    out = xgb_cache() / "ares_production" / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_indices.npy", train_index)
    np.save(out / "valid_indices.npy", valid_index)
    np.save(out / "train_features.npy", train_features)
    np.save(out / "valid_features.npy", fit_apply(fit, fit_y, valid, source))
    np.save(out / "test_features.npy", fit_apply(fit, fit_y, test, source))
    (out / "manifest.json").write_text(json.dumps({"fold": fold, "feature_names": names}, indent=1) + "\n")
    print(f"ares fold {fold}: {len(names)} columns", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    train, test, source = read_train(), read_test(), read_original()
    outer = fold_ids(train[TARGET].to_numpy(np.int8))
    for fold in fold_list(args.folds):
        build_fold(fold, train, test, source, outer)


if __name__ == "__main__":
    main()
