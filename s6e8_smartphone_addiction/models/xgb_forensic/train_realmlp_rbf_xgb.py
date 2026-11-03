#!/usr/bin/env python
"""XGBoost on the 290-column frontier matrix plus RealMLP-style coordinates, strict region lifts and semantic
interaction constraints (479 features), with the completion base margin and the selector / full-refit rule.

Per outer fold, fitted on the outer-training rows only (lib/region_lift.py, lib/realmlp_embeddings.py):
  9 strict region x exact-value interaction lifts (complement-cross-fitted over splitmix64(id) mod 5 owner groups)
  36 periodic coordinates (2 random frequencies x sin / cos of the 9 robust-scaled raw numerics, seed 250314)
  144 RBF coordinates (16 quantile-placed Gaussian bumps per raw numeric)
Interaction constraints: the 42 parent columns are a shared core; each feature family of the frontier matrix
(imputer, screen response, residual source channels, lift banks, temperature, global density, fills, repair,
completion, ares / dsx, imputed screen response, PMI, copula, unsupported response) and each new block (periodic,
region lifts, RBF) forms one clique with the core.
Base margin: logit(prior) + 0.55 raw_te_woe_sum + 0.40 (completion_logit_mean - logit(prior)).
XGBoost (GPU hist): lr 0.02, depth 6, max_bin 1024, min_child_weight 20, subsample 0.7, colsample_bytree 0.7,
colsample_bynode 0.6, lambda 4, 6 parallel trees per round, seed 42; selector (StratifiedKFold(10, seed 91027) split 0,
early stopping 350, cap 5,000) then a refit on all outer-training rows for ceil(best x 10 / 9) rounds.
Test = mean of the 5 fold models.
Usage: python train_realmlp_rbf_xgb.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from models.xgb_forensic.lib.frontier_features import FRONTIER_PARAMS, completion_margin, frontier_parts  # noqa: E402
from models.xgb_forensic.lib.realmlp_embeddings import (PeriodicEmbedding, RawRBFBasis,  # noqa: E402
                                                        semantic_constraints)
from models.xgb_forensic.lib.refit_xgb import fit_selector_refit  # noqa: E402
from models.xgb_forensic.lib.region_lift import FIELDS, build_strict_lift9  # noqa: E402
from models.xgb_forensic.lib.strong import build_strong_fold  # noqa: E402

PAIR = "xgb_forensic_periodic_rbf_regions"


class RegionRBFFeatures:
    """Region lifts + periodic + RBF coordinates, fitted on the outer-training frontier frame."""

    def fit_transform(self, X, y, ids):
        self.fit_frame = X[list(FIELDS)].reset_index(drop=True).copy()
        self.fit_target = np.asarray(y, np.int8).copy()
        self.fit_ids = np.asarray(ids, np.int64).copy()
        region_fit, _, self.region_names = build_strict_lift9(X, X, self.fit_target, self.fit_ids)
        self.periodic, self.rbf = PeriodicEmbedding(), RawRBFBasis(16)
        periodic_fit = self.periodic.fit_transform(X)
        rbf_fit = self.rbf.fit_transform(X)
        augmented = pd.concat([X.reset_index(drop=True),
                               pd.DataFrame(region_fit, columns=self.region_names, dtype=np.float32),
                               periodic_fit, rbf_fit], axis=1)
        groups, core = semantic_constraints(list(X.columns))
        groups += [core + list(periodic_fit.columns), core + self.region_names, core + list(rbf_fit.columns)]
        return augmented, groups

    def transform(self, X):
        _, region_apply, names = build_strict_lift9(self.fit_frame, X, self.fit_target, self.fit_ids)
        return pd.concat([X.reset_index(drop=True), pd.DataFrame(region_apply, columns=names, dtype=np.float32),
                          self.periodic.transform(X), self.rbf.transform(X)], axis=1)


def run_fold(fold):
    s = build_strong_fold(fold, extra_parts=frontier_parts)
    margins = [completion_margin(frame, s.prior) for frame in (s.train, s.valid, s.test)]
    features = RegionRBFFeatures()
    train_ids = s.raw_train["id"].to_numpy(np.int64)[s.train_idx]
    train_frame, constraints = features.fit_transform(s.train, s.y_train, train_ids)
    params = dict(FRONTIER_PARAMS, interaction_constraints=constraints)
    model = fit_selector_refit(train_frame, s.y_train, margins[0], params)
    valid_pred = np.asarray(model.predict_proba(features.transform(s.valid), base_margin=margins[1])[:, 1], np.float64)
    test_pred = np.asarray(model.predict_proba(features.transform(s.test), base_margin=margins[2])[:, 1], np.float64)
    save_part(HERE, PAIR, fold, valid_pred, test_pred)
    from sklearn.metrics import roc_auc_score
    print(f"{PAIR} fold {fold}: AUC {roc_auc_score(s.y_valid, valid_pred):.6f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    for fold in fold_list(args.folds):
        if not have_part(HERE, PAIR, fold):
            run_fold(fold)
    y = target()
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
