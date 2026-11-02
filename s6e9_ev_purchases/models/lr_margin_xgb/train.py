#!/usr/bin/env python
"""XGBoost on the champion matrix (raw + log counts + flags + 11 nested TEs + 4 income TE differences) with a base margin.
Recipes (pair name):
  xgb_lr_margin              in-fold LR base margin (lib/lr_margin.py, LR max_iter 1000); fast parameters; 10 seeds
  xgb_lr_margin_rte          + 3 residual income TEs (target y - p_rule); 20 seeds
  xgb_lr_margin_rte_art      + the artifact bundle (12 columns); 20 seeds
  xgb_rule_margin_art        xgb_lr_margin_rte_art with base margin = logit(p_rule) instead of the LR and slow parameters; 20 seeds
  xgb_lr_margin_rte_art_bpe  xgb_lr_margin_rte_art + nested TEs of the income's GPT-2 BPE leading piece (m 1 and 20; trees and LR)
                             + a residual TE of the piece (m 5); 20 seeds
Fast parameters: lr 0.03, depth 7, colsample 0.3; slow: lr 0.015, depth 6, colsample 0.3, lambda 5 (lib/xgb_dress.py).
Two passes (lib/xgb_dress.py): pass 1 (first seed, early stopping) sets the number of rounds, pass 2 trains every seed with it.
Usage: python train.py <pair> [--seeds 42,7,2024,...]"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import save_pair  # noqa: E402
from lib.features import CHAMPION_TE, RTE_INC, ResidTE, artifact_features, income_piece, logit, rule_prob, setup  # noqa: E402
from lib.lr_margin import build  # noqa: E402
from lib.xgb_dress import FAST, SLOW, dress  # noqa: E402

SEEDS10 = [42, 7, 2024, 1, 2, 3, 4, 5, 6, 8]
SEEDS20 = SEEDS10 + list(range(9, 19))
BPE_TE = [("inc_bpe", 1.0), ("inc_bpe", 20.0)]
RECIPES = {
    "xgb_lr_margin": dict(lr_max_iter=1000, rte=None, art=False, bpe=False, margin="lr", params=FAST, seeds=SEEDS10),
    "xgb_lr_margin_rte": dict(lr_max_iter=2000, rte=RTE_INC, art=False, bpe=False, margin="lr", params=FAST, seeds=SEEDS20),
    "xgb_lr_margin_rte_art": dict(lr_max_iter=2000, rte=RTE_INC, art=True, bpe=False, margin="lr", params=FAST, seeds=SEEDS20),
    "xgb_rule_margin_art": dict(lr_max_iter=2000, rte=RTE_INC, art=True, bpe=False, margin="rule", params=SLOW, seeds=SEEDS20),
    "xgb_lr_margin_rte_art_bpe": dict(lr_max_iter=2000, rte=RTE_INC + [("inc_bpe", 5.0)], art=True, bpe=True, margin="lr", params=FAST,
                                      seeds=SEEDS20),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(RECIPES))
    ap.add_argument("--seeds", default="", help="comma-separated; default: the recipe's seeds")
    args = ap.parse_args()
    r = RECIPES[args.pair]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else r["seeds"]
    ctx = setup()
    B, y, folds, n_tr = ctx.B, ctx.y, ctx.folds, ctx.B.n_tr

    te_specs = list(CHAMPION_TE)
    if r["bpe"]:
        B.register("inc_bpe", income_piece(ctx.both))
        te_specs += BPE_TE
    base = build(ctx, te_specs, max_iter=r["lr_max_iter"])
    p_rule = rule_prob(ctx.both)
    if r["rte"] is not None:
        RTE = ResidTE(B, y, y - p_rule[:n_tr])
        base["r_folds"] = [RTE.fold(a, b, r["rte"]) for a, b in folds]
    if r["margin"] == "rule":
        z = logit(p_rule)
        base["margins"] = [(z[a], z[b], z[n_tr:]) for a, b in folds]
    extra_tr = extra_te = None
    if r["art"]:
        art = artifact_features(ctx.both)
        extra_tr, extra_te = art.iloc[:n_tr].reset_index(drop=True), art.iloc[n_tr:].reset_index(drop=True)

    oof, test = dress(ctx, args.pair, base, seeds, params=r["params"], extra_tr=extra_tr, extra_te=extra_te)
    save_pair(HERE, args.pair, oof, test)


if __name__ == "__main__":
    main()
