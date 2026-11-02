#!/usr/bin/env python
"""MLP on the pair-TE space (nn_lib.py with a linear skip, 12 epochs, seeds 0-9). Inputs: the standardized raw / log-count /
flag columns + logit of the 215 single / pair TEs (pair_te.py) + 11 residual TEs (y - p_rule) of the champion keys.
Arms (pair name):
  mlp_pair_te                 as above; 512-256-128
  mlp_pair_te_rule_wide       + logit(p_rule) as a fixed offset on the output logit; 1024-512-256
  mlp_pair_te_rule_wide_bpe   mlp_pair_te_rule_wide + logit of nested TEs of the income's GPT-2 BPE leading piece (m 1 and 20) and of
                              its pairs with 15 other keys (m 10), + a residual TE of the piece (m 5)
Pair = mean over the seeds of OOF / test (test = mean of the 5 fold models).
Usage: python train_pair_mlp.py <pair> [--seeds 0,1,2]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "lr_margin_xgb"))
sys.path.insert(0, str(HERE))
from common.pairs import save_pair  # noqa: E402
from lib.features import CHAMPION_TE, ResidTE, income_piece, logit, rule_prob, setup  # noqa: E402
from nn_lib import TE_MLP, train_nn  # noqa: E402
from pair_te import load_or_build, pair_logits, selected  # noqa: E402

ARMS = {
    "mlp_pair_te": dict(rule=False, hidden=None, bpe=False),
    "mlp_pair_te_rule_wide": dict(rule=True, hidden=(1024, 512, 256), bpe=False),
    "mlp_pair_te_rule_wide_bpe": dict(rule=True, hidden=(1024, 512, 256), bpe=True),
}
SEEDS = list(range(10))


def register_bpe_keys(ctx):
    """Nested-TE keys of the income's BPE leading piece: the piece (m 1, 20) and the piece paired with 15 other keys (m 10).
    Returns (TE specs, extra residual-TE specs)."""
    both = ctx.both
    piece = pd.Series(income_piece(both)).astype(str)
    partners = {"age": both.Age, "gender": both.Gender, "city": both.City_Type, "cars": both.Number_of_Cars_Owned,
                "ctype": both.Current_Car_Type, "km": both.Daily_Commute_km, "sth": both.Charging_Stations_Near_Home,
                "stw": both.Charging_Stations_Near_Work, "home": both.Home_Charging_Possible, "env": both.Environmental_Concern_Level,
                "sub": both.Subsidy_Available, "anx": both.Range_Anxiety_Level, "km_i": both.Daily_Commute_km // 1,
                "km_5": both.Daily_Commute_km // 5, "age_5": both.Age // 5}
    ctx.B.register("t_inc_bpe", piece)
    for k, s in partners.items():
        ctx.B.register(f"p_inc_bpe_{k}", piece + "|" + s.astype(str))
    specs = [("t_inc_bpe", 1.0), ("t_inc_bpe", 20.0)] + [(f"p_inc_bpe_{k}", 10.0) for k in partners]
    return specs, [("t_inc_bpe", 5.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(ARMS))
    ap.add_argument("--seeds", default="", help="comma-separated; default 0-9")
    args = ap.parse_args()
    arm = ARMS[args.pair]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    ctx = setup()
    B, y, n_tr = ctx.B, ctx.y, ctx.B.n_tr
    cache = load_or_build(ctx)
    te_folds = cache["folds"]
    sel = selected(cache)
    p_rule = rule_prob(ctx.both)
    RTE = ResidTE(B, y, y - p_rule[:n_tr])
    rspecs, bspecs = list(CHAMPION_TE), []
    if arm["bpe"]:
        bspecs, rte_extra = register_bpe_keys(ctx)
        rspecs += rte_extra
    print(f"{args.pair}: {len(sel)} pair-TE keys, {len(bspecs)} BPE TE keys, {len(rspecs)} residual TEs", flush=True)

    def extra_fn(f, a, b):
        F, V, T = te_folds[f]
        Rf, Rv, Rt = RTE.fold(a, b, rspecs)
        parts = [(pair_logits(F, sel), Rf.values), (pair_logits(V, sel), Rv.values), (pair_logits(T, sel), Rt.values)]
        if bspecs:
            Bf, Bv, Bt = ctx.TE.fold(a, b, bspecs)
            parts = [(p[0], logit(Bx.values).astype(np.float32), p[1]) for p, Bx in zip(parts, (Bf, Bv, Bt))]
        return tuple(np.hstack(p) for p in parts)

    cfg = dict(TE_MLP, epochs=12, linear_skip=True, te_specs=[], te_diffs=False, extra_fn=extra_fn)
    if arm["rule"]:
        cfg["offset"] = logit(p_rule).astype(np.float32)
    if arm["hidden"]:
        cfg["hidden"] = arm["hidden"]

    t0 = time.time()
    oofs, tests = [], []
    for s in seeds:
        oof, tpred, auc, _ = train_nn(ctx, dict(cfg, seed=s), log=lambda m: None)
        oofs.append(oof)
        tests.append(tpred)
        print(f"  seed {s}: OOF AUC {auc:.6f} ({time.time() - t0:.0f}s)", flush=True)
    oof, test = np.mean(oofs, axis=0), np.mean(tests, axis=0)
    print(f"{args.pair}: {len(seeds)}-seed OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)
    save_pair(HERE, args.pair, oof, test)


if __name__ == "__main__":
    main()
