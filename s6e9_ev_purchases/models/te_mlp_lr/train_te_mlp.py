#!/usr/bin/env python
"""TE MLP (nn_lib.py): standardized raw / log-count / flag columns + logit/3 of 13 nested TEs (the 11 champion keys, income at
m 10, income // 100 at m 30) + the 4 income TE differences / 3; 512-256-128; 12 epochs. Arms (pair name):
  mlp_te_diffs         as above; seeds 0-4
  mlp_te_rule_rte      + linear skip, + logit(p_rule) as a fixed offset on the output logit, + 3 residual income TEs (y - p_rule)
                       as extra inputs; seeds 0-19
  mlp_te_rule_art_rte  mlp_te_rule_rte + the artifact bundle (12 columns) as extra inputs; seeds 0-19
Pair = mean over the seeds of OOF / test (test = mean of the 5 fold models).
Usage: python train_te_mlp.py <pair> [--seeds 0,1,2]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "lr_margin_xgb"))
sys.path.insert(0, str(HERE))
from common.pairs import save_pair  # noqa: E402
from lib.features import RTE_INC, ResidTE, artifact_features, logit, rule_prob, setup  # noqa: E402
from nn_lib import TE_MLP, train_nn  # noqa: E402

ARMS = {
    "mlp_te_diffs": dict(skip=False, rule=False, art=False, rte=False, seeds=list(range(5))),
    "mlp_te_rule_rte": dict(skip=True, rule=True, art=False, rte=True, seeds=list(range(20))),
    "mlp_te_rule_art_rte": dict(skip=True, rule=True, art=True, rte=True, seeds=list(range(20))),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(ARMS))
    ap.add_argument("--seeds", default="", help="comma-separated; default: the arm's seeds")
    args = ap.parse_args()
    arm = ARMS[args.pair]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else arm["seeds"]
    ctx = setup()
    B, y, n_tr = ctx.B, ctx.y, ctx.B.n_tr
    p_rule = rule_prob(ctx.both)

    cfg = dict(TE_MLP, epochs=12)
    if arm["skip"]:
        cfg["linear_skip"] = True
    if arm["rule"]:
        cfg["offset"] = logit(p_rule).astype(np.float32)
    if arm["art"] or arm["rte"]:
        ART = artifact_features(ctx.both).values.astype(np.float32)
        RTE = ResidTE(B, y, y - p_rule[:n_tr])

        def extra_fn(f, a, b):
            pf, pv, pt = [], [], []
            if arm["art"]:
                pf.append(ART[a])
                pv.append(ART[b])
                pt.append(ART[n_tr:])
            if arm["rte"]:
                Rf, Rv, Rt = RTE.fold(a, b, RTE_INC)
                pf.append(Rf.values)
                pv.append(Rv.values)
                pt.append(Rt.values)
            return np.hstack(pf), np.hstack(pv), np.hstack(pt)
        cfg["extra_fn"] = extra_fn

    t0 = time.time()
    oof, test = np.zeros(len(y)), np.zeros(B.n_te)
    for s in seeds:
        c = dict(cfg, seed=s)
        o, t, auc, _ = train_nn(ctx, c, log=lambda m: None)
        oof += o / len(seeds)
        test += t / len(seeds)
        print(f"  seed {s}: OOF AUC {auc:.6f} ({time.time() - t0:.0f}s)", flush=True)
    print(f"{args.pair}: {len(seeds)}-seed OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)
    save_pair(HERE, args.pair, oof, test)


if __name__ == "__main__":
    main()
