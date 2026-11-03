#!/usr/bin/env python
"""pytabkit RealMLP on Pipeline A features (lib/pipeline_a.py) and on the extended Pipeline A block (lib/pipeline_a3.py).

Recipes (pair name):
  realmlp_fold_pipeline_a_te_*         Pipeline A fitted per fold + TargetEncoder of the Race x Compound / Race x Year codes
                                       (39 columns); original rows outside the fold appended; silu, 4 epochs,
                                       PLR 32 / 16 with sigma 1, batch 512, one model per fold
  realmlp_pipeline_a_mish_<n>seed*     Pipeline A fitted on all train rows (37 columns), no target encoding; original rows
                                       outside the fold appended; mish with parametric activations, [768, 384, 192],
                                       n_ens 32, lr 0.025, label smoothing 0.005; n model seeds averaged per fold
  realmlp_fold_pipeline_a_arith_mish   extended Pipeline A fitted per fold (41 columns), no target encoding; original rows
                                       outside the fold appended; mish, 6 epochs, lr 0.0175, wd 0.017, plr_sigma 0.94
  realmlp_shape_state_te_no_original   extended Pipeline A + string combinations, rounded numerics, lap-delta shape and
                                       race-state blocks, 30 target-encoded keys (113 columns); no original rows
Usage: python train.py <pair>"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import save_pair  # noqa: E402
from lib import pipeline_a, pipeline_a3  # noqa: E402

FOLD_TE = dict(use_global_fe=False, original_mode="fold", target_encode_all_combos=True, model_seed_offset=100)
LARGE_PLR = {"batch_size": 512, "plr_hidden_1": 32, "plr_hidden_2": 16, "plr_sigma": 1.0}
MISH = dict(use_global_fe=True, original_mode="fold", target_encode_all_combos=False, model_seed_offset=100)
MISH_PARAMS = {"n_ens": 32, "batch_size": 512, "lr": 0.025, "p_drop": 0.03, "wd": 0.014, "first_layer_lr_factor": 0.125,
               "plr_hidden_1": 32, "plr_hidden_2": 16, "plr_sigma": 1.0, "plr_lr_factor": 0.16,
               "hidden_sizes": [768, 384, 192], "act": "mish", "use_parametric_act": True, "act_lr_factor": 0.1,
               "ls_eps": 0.005}
TEST_RUN = dict(seed=2026, n_splits=20, model_seed_offset=10100)


def fold_te(**params):
    return dict(harness=pipeline_a, cfg=dict(FOLD_TE, params_override={**LARGE_PLR, **params}))


def mish(n_seeds, **params):
    return dict(harness=pipeline_a, cfg=dict(MISH, ensemble_seed_offsets=[100 * i for i in range(n_seeds)],
                                            params_override={**MISH_PARAMS, **params}), test_cfg=TEST_RUN)


RECIPES = {
    "realmlp_fold_pipeline_a_te_nens20": fold_te(n_ens=20, p_drop=0.04, wd=0.016),
    "realmlp_fold_pipeline_a_te_nens24": fold_te(n_ens=24, p_drop=0.04, wd=0.016),
    "realmlp_fold_pipeline_a_te_light_reg": fold_te(n_ens=24, p_drop=0.03, wd=0.014),
    "realmlp_fold_pipeline_a_te_light_reg_wide": fold_te(n_ens=24, p_drop=0.03, wd=0.014, hidden_sizes=[768, 384, 192]),
    "realmlp_pipeline_a_mish_2seed": mish(2),
    "realmlp_pipeline_a_mish_2seed_ls0075": mish(2, ls_eps=0.0075),
    "realmlp_pipeline_a_mish_3seed": mish(3),
    "realmlp_pipeline_a_mish_3seed_drop04": mish(3, p_drop=0.04),
    "realmlp_pipeline_a_mish_4seed": mish(4),
    "realmlp_pipeline_a_mish_4seed_drop04": mish(4, p_drop=0.04),
    "realmlp_pipeline_a_mish_5seed": mish(5),
    "realmlp_pipeline_a_mish_5seed_drop04": mish(5, p_drop=0.04),
    "realmlp_fold_pipeline_a_arith_mish": dict(harness=pipeline_a3, cfg=dict(
        original_mode="fold", target_encode_all_combos=False, model_seed_offset=100,
        params_override={"act": "mish", "act_lr_factor": 0.1, "batch_size": 512, "first_layer_lr_factor": 0.125,
                         "hidden_sizes": [768, 384, 192], "lr": 0.0175, "ls_eps": 0.005, "n_ens": 32, "p_drop": 0.05,
                         "plr_hidden_1": 32, "plr_hidden_2": 16, "plr_lr_factor": 0.08, "plr_sigma": 0.94,
                         "use_parametric_act": True, "wd": 0.017})),
    "realmlp_shape_state_te_no_original": dict(harness=pipeline_a3, cfg=dict(
        original_mode="none", add_extra_combos=True, add_round_cats=True, add_shape_numeric=True, add_shape_cats=True,
        add_driver_initial=True, add_state_numeric=True, add_state_cats=True, target_encode_extra_combos=True,
        target_encode_all_combos=True, model_seed_offset=102200,
        params_override={"act": "silu", "act_lr_factor": 0.10, "batch_size": 512, "first_layer_lr_factor": 0.20,
                         "hidden_sizes": [512, 512, 256], "lr": 0.0105, "ls_eps": 0.010, "max_one_hot_cat_size": 24,
                         "n_ens": 24, "p_drop": 0.040, "plr_hidden_1": 24, "plr_hidden_2": 12, "plr_lr_factor": 0.105,
                         "plr_sigma": 2.20, "use_parametric_act": False, "wd": 0.014})),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(RECIPES))
    args = ap.parse_args()
    r = RECIPES[args.pair]
    harness = r["harness"]
    cfg = harness.ExpConfig(**r["cfg"])
    oof, test = harness.run_experiment(cfg)
    if "test_cfg" in r:
        _, test = harness.run_experiment(replace(cfg, **r["test_cfg"]))
    save_pair(HERE, args.pair, oof, test)


if __name__ == "__main__":
    main()
