#!/usr/bin/env python
"""GPU CatBoost on the domain / artifact-category / original-prior feature frames of lib/cat_harness.py.

Recipes (pair name):
  catboost_artifact_cats_priors_depth9         base frame; OP priors of 6 keys (smoothing 25); depth 9, lr 0.025,
                                               l2 10, Balanced class weights
  catboost_artifact_cats_next_lap_sequence     + previous / next row sequence features within Year x Race x Driver;
                                               depth 8, lr 0.03, l2 7, no class weights
  catboost_pit_state_cats_original_w115        + public formulas, pipeline-A floors / counts, within-group ranks,
                                               compound hazard, pit-state and strategy crosses; OP priors of 6 keys
                                               (smoothing 75); original weight 1.15; depth 9, lr 0.02, l2 14, Balanced
  catboost_native_combos_sequence_rich_priors  + rank categories, sequence within Year x Race x Driver x Stint,
                                               per-source counts, native-combo categories, wide group statistics,
                                               stint end points; OP priors of 37 keys (smoothing 55); original weight
                                               0.9; depth 9, lr 0.02, l2 14, one_hot_max_size 12, max_ctr_complexity 4,
                                               Balanced, seed offset 240
Every recipe appends all original rows to the training side of every fold and early-stops on the validation fold
(use_best_model=True). Test = mean of the 5 fold models (probabilities clipped to [1e-7, 1 - 1e-7]).

Usage: python train.py <pair> [--folds 0,1,2,3,4] [--force]
Folds can run in separate processes; the pair is written once all 5 folds are done."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds  # noqa: E402
from common.pairs import save_part  # noqa: E402
from models.catboost.lib.cat_harness import SEED, build_frames, fit_fold, seed_everything  # noqa: E402
from models.catboost.lib.runner import FOLDER, finish, fold_parser, todo_folds  # noqa: E402

BASE_PRIORS = [
    "Race_Year",
    "Race_Compound_Stint",
    "Driver_Compound",
    "TyreAgeRatio_str",
    "Compound_TyreLifeBin",
    "RacePhase_TyreLifeBin",
]
ARTIFACT_PRIORS = [
    "CAT_LapTime (s)_r1",
    "CAT_LapTime_Delta_r2",
    "CAT_Cumulative_Degradation_r2",
    "CAT_RaceProgress_r3",
    "CAT_TyreAgeRatio_r2",
    "CAT_DegPerTyreLap_r2",
    "CAT_DeltaAbs_r2",
]
NATIVE_COMBO_PRIORS = [
    "Race_Year",
    "Race_Compound_Stint",
    "Driver_Compound",
    "Compound_TyreLifeBin",
    "RacePhase_TyreLifeBin",
    "CAT2_Race_Year_Compound_Stint",
    "CAT2_Race_Compound_TyreFracBin",
    "CAT2_Compound_Stint_TyreFracBin",
    "CAT2_Compound_LapsToLifeBin",
    "CAT2_Race_Compound_LapsToLifeBin",
    "CAT2_Compound_RacePhase_RiskAgeBin",
    "CAT2_LapTime_Delta_r2__Compound_CAT2_TyreFracBin",
    "CAT2_Cumulative_Degradation_r2__Compound_CAT2_TyreFracBin",
    "CAT2_CAT2_LapsToCompoundLife_r0__Compound_RacePhase",
]
NATIVE_ARTIFACT_PRIORS = [
    "CAT2_LapTime_s_r1",
    "CAT2_LapTime_Delta_r2",
    "CAT2_Cumulative_Degradation_r2",
    "CAT2_RaceProgress_r3",
    "CAT2_CAT2_RiskAgeProgress_r2",
    "CAT2_CAT2_RiskDegAge_r1",
    "CAT2_CAT2_RiskDeltaAge_r1",
    "CAT2_CAT2_LapsToCompoundLife_r0",
]
RICH_PRIORS = list(dict.fromkeys(
    BASE_PRIORS
    + ARTIFACT_PRIORS
    + NATIVE_COMBO_PRIORS
    + NATIVE_ARTIFACT_PRIORS
    + [
        "Driver_Race",
        "Race_Compound",
        "Compound_Stint",
        "CB2_Compound_Stint_TyreWindow",
        "CB2_Race_Compound_ProgressWindow",
        "CB3_Race_Year_Compound_Stint",
        "CB3_Driver_Race_Stint_TyreWindow",
    ]
))

RECIPES = {
    "catboost_artifact_cats_priors_depth9": dict(  # 309 columns, 96 categorical
        original_priors=BASE_PRIORS, prior_smooth=25.0, original_weight=1.0, params="deeper_reg"),
    "catboost_artifact_cats_next_lap_sequence": dict(  # 357 columns, 96 categorical
        sequence_features=True,
        original_priors=BASE_PRIORS, prior_smooth=25.0, original_weight=1.0, params="no_balanced"),
    "catboost_pit_state_cats_original_w115": dict(  # 541 columns, 141 categorical
        public_formula_features=True, pipeline_a_features=True, relative_rank_features=True,
        compound_hazard_light=True, pit_state_cats=True, strategy_cats=True,
        original_priors=BASE_PRIORS, prior_smooth=75.0, original_weight=1.15, params="deeper_reg_soft"),
    "catboost_native_combos_sequence_rich_priors": dict(  # 1562 columns, 244 categorical
        public_formula_features=True, pipeline_a_features=True, relative_rank_features=True,
        relative_category_features=True, sequence_features=True, sequence_stint_features=True,
        source_freq_features=True, compound_hazard_light=True, pit_state_cats=True, strategy_cats=True,
        native_combo_features=True, wide_group_stats=True, stint_endpoint_features=True,
        original_priors=RICH_PRIORS, prior_smooth=55.0, original_weight=0.9, params="deeper_reg_soft_ctr4",
        seed_offset=240),
}


def main():
    args = fold_parser(RECIPES).parse_args()
    cfg = RECIPES[args.pair]
    todo = todo_folds(args.pair, args.folds, args.force)
    if todo:
        seed_everything(SEED)
        frames = build_frames(cfg)
        fold_id = folds(frames["y"].to_numpy())
        for k in todo:
            val_pred, test_pred = fit_fold(cfg, frames, fold_id, k, name=args.pair)
            save_part(FOLDER, args.pair, k, val_pred, test_pred)
    finish(args.pair)


if __name__ == "__main__":
    main()
