#!/usr/bin/env python
"""GPU XGBoost on the fold-safe target-encoding harness (lib/harness.py): 6 recipes.

Every recipe appends all original rows to the training side of each fold (sample weight below) and early-stops on the
validation fold. Base columns / target-encoded string keys of each recipe in our runs:
  xgb_relrank_next_lap_original_w05                 499 / 13   relative ranks + next-lap values; TE s 5, 20, 100
  xgb_relrank_pipea_formulas_smooth_te_original_w05 568 / 29   relative ranks + Pipeline A + formulas; TE s 20, 100, 500
  xgb_relrank_pipea_formulas_next_lap_original_w05  582 / 29   relative ranks + Pipeline A + formulas + next-lap values
  xgb_relrank_original_w035                         485 / 13   relative ranks; original weight 0.35
  xgb_group_residuals_original_w08                  855 / 27   + 13-grouping residual block + source flag; TE s 20 on keys
                                                               with <= 25,000 levels; colsample 0.82 / 0.62, lambda 7.5
  xgb_hazard_pit_window_cadence_original_w08       1478 / 116  + hazard bins, CatBoost-style block, pit-window keys, pit
                                                               cadence, hash keys; TE s 20 on keys with <= 750 levels;
                                                               lr 0.006, up to 23,000 rounds, patience 1000
Usage: python train.py <pair> [--folds 0,1,2,3,4] [--force]
Folds can run in separate processes (each rebuilds the base matrix); the pair is written once all 5 folds are done."""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import fold_list  # noqa: E402
from lib.harness import Config, run  # noqa: E402

RECIPES = {
    "xgb_relrank_next_lap_original_w05": dict(
        external_sample_weight=0.50,
        use_relative_rank_features=True,
        use_sequence_next_features=True,
    ),
    "xgb_relrank_pipea_formulas_smooth_te_original_w05": dict(
        external_sample_weight=0.50,
        te_smooths=[20, 100, 500],
        use_relative_rank_features=True,
        use_pipeline_a_features=True,
        use_public_formula_features=True,
    ),
    "xgb_relrank_pipea_formulas_next_lap_original_w05": dict(
        external_sample_weight=0.50,
        use_relative_rank_features=True,
        use_pipeline_a_features=True,
        use_public_formula_features=True,
        use_sequence_next_features=True,
    ),
    "xgb_relrank_original_w035": dict(
        external_sample_weight=0.35,
        use_relative_rank_features=True,
    ),
    "xgb_group_residuals_original_w08": dict(
        external_sample_weight=0.80,
        use_relative_rank_features=True,
        use_pipeline_a_features=True,
        use_public_formula_features=True,
        use_source_indicator=True,
        use_extended_group_residuals=True,
        te_smooths=[20],
        max_te_cardinality=25000,
        xgb_params_override={
            "colsample_bytree": 0.82,
            "colsample_bynode": 0.62,
            "reg_lambda": 7.5,
        },
    ),
    "xgb_hazard_pit_window_cadence_original_w08": dict(
        external_sample_weight=0.80,
        use_hazard_features=True,
        use_relative_rank_features=True,
        use_pipeline_a_features=True,
        use_public_formula_features=True,
        use_source_indicator=True,
        use_catboost_style_features=True,
        use_extended_group_residuals=True,
        use_pit_window_key_features=True,
        use_pit_cadence_features=True,
        use_artifact_hash_keys=True,
        te_smooths=[20],
        max_te_cardinality=750,
        xgb_params_override={
            "learning_rate": 0.006,
            "n_estimators": 23000,
            "min_child_weight": 18.0,
            "subsample": 0.88,
            "colsample_bytree": 0.70,
            "colsample_bynode": 0.54,
            "reg_lambda": 15.0,
            "gamma": 0.7,
            "max_leaves": 96,
            "early_stopping_rounds": 1000,
        },
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(RECIPES))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    run(Config(pair=args.pair, **RECIPES[args.pair]), HERE, fold_list(args.folds), force=args.force)


if __name__ == "__main__":
    main()
