#!/usr/bin/env python
"""TabM-mini (pytabkit) on the shared feature / training harness of ../neural_nets (lib/shared_harness.py)
(pair tabm_mini_domain_group_source).

Features built once on all competition rows (use_global_fe): the harness base block (floored-value categories, value counts,
RaceProgress quantile bins, Race x Compound and Race x Year), domain features (estimated total laps, laps remaining, tyre-age /
pit-window / degradation / lap-time-delta ratios and race-phase, tyre-life and position bins), the source flag, and group
statistics (group size, value minus group mean and z-score of LapTime (s), LapTime_Delta, Cumulative_Degradation, TyreLife
within Race, Compound, Driver, Race x Compound) over train + test + original rows; no target encodings.
Training rows per fold: the competition rows of the other 4 folds + the fold-matched original subset (original_mode 'fold').
Model (model_type 'tabm'): TabM_D_Classifier, tabm-mini, k 16, 2 blocks x 256, d_embedding 8, dropout 0.18, lr 2.5e-4,
batch 512, 8 epochs, best epoch on the validation fold; categoricals = string columns and numerics with < 100 values;
seed 42 + fold. Test = mean of the 5 fold models.
Usage: python train_domain_group_source.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import model_dir  # noqa: E402
from models.neural_nets.lib.shared_harness import ExpConfig, run_experiment  # noqa: E402

cfg = ExpConfig(
    pair='tabm_mini_domain_group_source',
    folds_to_run=[1, 2, 3, 4, 5],
    use_global_fe=True,
    original_mode='fold',
    target_encode_all_combos=False,
    params_override={'epochs': 8, 'batch_size': 4096, 'eval_batch_size': 16384, 'lr': 0.0008, 'min_lr': 2e-05,
                     'weight_decay': 0.0001, 'patience': 5, 'dropout': 0.08, 'emb_dropout': 0.03, 'num_emb_dim': 0,
                     'label_smoothing': 0.008, 'scaler': 'quantile', 'num_workers': 0, 'model_type': 'tabm',
                     'tabm_epochs': 8, 'tabm_batch_size': 512, 'tabm_lr': 0.00025, 'tabm_k': 16, 'tabm_d_embedding': 8,
                     'tabm_d_block': 256, 'tabm_n_blocks': 2, 'tabm_dropout': 0.18},
    add_domain_features=True,
    add_group_stats=True,
    add_source_indicator=True,
)


if __name__ == '__main__':
    run_experiment(cfg, model_dir('tabm'))
