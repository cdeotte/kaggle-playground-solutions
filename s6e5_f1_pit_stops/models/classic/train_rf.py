#!/usr/bin/env python
"""cuML random forest on the shared harness of ../neural_nets/lib/shared_harness.py (pair rf_domain_group_stats_combo_te).

Features (built once on train + test + original, use_global_fe; encoders fitted on train): the 14 raw columns,
16 domain ratios / pressures and 3 binned race-phase / tyre-age / position categoricals (add_domain_features), the ratios
LapNumber / RaceProgress and TyreLife / LapNumber, floor-value codes of every numeric column, count encodings of the
string columns and of Year / PitStop, a 200-bin RaceProgress quantile code, the Race x Compound and Race x Year combos,
and group statistics (count, value - group mean and z-score of LapTime (s), LapTime_Delta, Cumulative_Degradation,
TyreLife within Race, Compound, Driver and Race x Compound; add_group_stats).
Per outer fold: the fold-matched 4/5 of the original rows are appended to the training rows (original_mode 'fold'),
the two combos are target-encoded (sklearn TargetEncoder, 5 shuffled inner folds, seed 42 + fold), string and
low-cardinality (< 100 values) columns become vocabulary codes and the other numerics are standardized; the forest gets
the concatenated matrix (56 codes + 57 numerics).
Model: cuML RandomForestClassifier, 950 trees, max_depth 20, max_features 0.45, min_samples_leaf 8, 128 bins,
random_state 42 + fold (1..5). Test = mean of the 5 fold models. The neural-net fields of params_override are unused by
the forest; they are kept as in our run.
Usage: python train_rf.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import model_dir  # noqa: E402
from models.neural_nets.lib.shared_harness import ExpConfig, run_experiment  # noqa: E402

CFG = ExpConfig(
    pair='rf_domain_group_stats_combo_te',
    folds_to_run=[1, 2, 3, 4, 5],
    seed=42,
    n_splits=5,
    use_global_fe=True,
    original_mode='fold',
    target_encode_all_combos=True,
    params_override={'epochs': 14, 'batch_size': 4096, 'eval_batch_size': 16384, 'lr': 0.0007, 'min_lr': 2e-05,
                     'weight_decay': 0.0001, 'patience': 4, 'dropout': 0.08, 'emb_dropout': 0.03, 'num_emb_dim': 0,
                     'label_smoothing': 0.008, 'scaler': 'standard', 'num_workers': 0, 'model_type': 'rf',
                     'rf_n_estimators': 950, 'rf_max_depth': 20, 'rf_max_features': 0.45, 'rf_min_samples_leaf': 8,
                     'rf_n_bins': 128},
    add_domain_features=True,
    add_group_stats=True,
)


if __name__ == '__main__':
    run_experiment(CFG, model_dir('classic'))
