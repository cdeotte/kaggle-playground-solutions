#!/usr/bin/env python
"""Embedding networks, factorization machines, TabTransformer, attention graph MLP and TabNet on the shared fold harness
`lib/shared_harness.py` (its docstring describes the feature blocks, column typing and training loop). One config per pair;
the config fields are the harness flags, `params_override` the model / training parameters.

Usage: python train.py <pair> [--folds 0,1,2,3,4]"""
import argparse
import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import model_dir  # noqa: E402
from common.pairs import fold_list  # noqa: E402
from models.neural_nets.lib.shared_harness import ExpConfig, run_experiment  # noqa: E402

CONFIGS = [
    # residual MLP (d 640, 5 blocks) on categorical embeddings + periodic sin / cos numeric embeddings; low-cardinality
    # numerics are both categorical and numeric inputs; base block fitted per fold, group + extended group stats, source
    # flag, all original rows appended to every fold
    ExpConfig(
        pair='emb_resnet_periodic_group_stats_all_original',
        folds_to_run=[1, 2, 3, 4, 5],
        use_global_fe=False,
        original_mode='all',
        add_source_indicator=True,
        add_group_stats=True,
        add_extended_group_stats=True,
        model_seed_offset=2200,
        params_override={
            'model_type': 'emb_resnet',
            'epochs': 46,
            'batch_size': 4096,
            'lr': 6.0e-4,
            'weight_decay': 1.2e-4,
            'dropout': 0.08,
            'emb_dropout': 0.02,
            'd_model': 640,
            'n_blocks': 5,
            'block_hidden_mult': 2.0,
            'num_emb_dim': 8,
            'duplicate_low_card_num_as_num': True,
            'scaler': 'quantile',
            'original_weight': 1.0,
            'num_workers': 2,  # 2 persistent loader workers, as trained (they change the torch random stream)
        },
    ),
    # TabTransformer (d 96, 2 blocks, 4 heads): domain features, group stats, source flag, fold-matched original rows
    ExpConfig(
        pair='tabtransformer_domain_group_stats_fold_original',
        folds_to_run=[1, 2, 3, 4, 5],
        use_global_fe=True,
        original_mode='fold',
        target_encode_all_combos=False,
        params_override={'epochs': 18, 'batch_size': 4096, 'eval_batch_size': 16384, 'lr': 0.0007, 'min_lr': 2e-05,
                         'weight_decay': 0.0001, 'patience': 5, 'dropout': 0.07, 'emb_dropout': 0.03, 'num_emb_dim': 0,
                         'label_smoothing': 0.008, 'scaler': 'quantile', 'num_workers': 0, 'model_type': 'tabtransformer',
                         'd_model': 96, 'n_blocks': 2, 'n_heads': 4, 'transformer_ff': 256, 'head_hidden': 384},
        add_domain_features=True,
        add_group_stats=True,
        add_source_indicator=True,
    ),
    # TabTransformer (d 128, 3 blocks, 4 heads): domain features, group stats, source flag, all original rows
    ExpConfig(
        pair='tabtransformer_medium_domain_group_stats_all_original',
        folds_to_run=[1, 2, 3, 4, 5],
        seed=42,
        n_splits=5,
        use_global_fe=True,
        original_mode='all',
        target_encode_all_combos=False,
        params_override={'epochs': 15, 'batch_size': 4096, 'eval_batch_size': 16384, 'lr': 0.00055, 'min_lr': 2e-05,
                         'weight_decay': 0.0001, 'patience': 5, 'dropout': 0.09, 'emb_dropout': 0.03, 'num_emb_dim': 0,
                         'label_smoothing': 0.008, 'scaler': 'quantile', 'num_workers': 0, 'model_type': 'tabtransformer',
                         'd_model': 128, 'n_blocks': 3, 'n_heads': 4, 'transformer_ff': 384, 'head_hidden': 512},
        add_domain_features=True,
        add_group_stats=True,
        add_source_indicator=True,
    ),
    # TabNet (n_d = n_a = 32, 3 steps) on the base block + target-encoded Race x Compound / Race x Year combos,
    # standardized numerics, fold-matched original rows
    ExpConfig(
        pair='tabnet_combo_te',
        folds_to_run=[1, 2, 3, 4, 5],
        use_global_fe=True,
        original_mode='fold',
        target_encode_all_combos=True,
        params_override={'epochs': 35, 'batch_size': 8192, 'eval_batch_size': 16384, 'lr': 0.0008, 'min_lr': 2e-05,
                         'weight_decay': 1e-05, 'patience': 8, 'dropout': 0.08, 'emb_dropout': 0.03, 'num_emb_dim': 0,
                         'label_smoothing': 0.008, 'scaler': 'standard', 'num_workers': 0, 'model_type': 'tabnet',
                         'tabnet_n_d': 32, 'tabnet_n_a': 32, 'tabnet_n_steps': 3, 'tabnet_gamma': 1.4, 'tabnet_lr': 0.002},
    ),
    # field-gated FM network (dim 16, interactions of the first 26 fields, MLP 512-384-192): domain features, group
    # stats, source flag, all original rows
    ExpConfig(
        pair='ffm_domain_group_stats_all_original',
        folds_to_run=[1, 2, 3, 4, 5],
        seed=42,
        n_splits=5,
        use_global_fe=True,
        original_mode='all',
        target_encode_all_combos=False,
        params_override={'epochs': 16, 'batch_size': 4096, 'eval_batch_size': 16384, 'lr': 0.0006, 'min_lr': 2e-05,
                         'weight_decay': 0.0002, 'patience': 5, 'dropout': 0.12, 'emb_dropout': 0.03, 'num_emb_dim': 0,
                         'label_smoothing': 0.012, 'scaler': 'quantile', 'num_workers': 0, 'model_type': 'ffm',
                         'ffm_dim': 16, 'ffm_max_pair_fields': 26, 'hidden': [512, 384, 192]},
        add_domain_features=True,
        add_group_stats=True,
        add_source_indicator=True,
    ),
    # attention graph MLP over the field embeddings (d 128, MLP 512-256-128): domain features, group + extended group
    # stats, fold-matched original rows
    ExpConfig(
        pair='graph_mlp_domain_extended_group_stats',
        folds_to_run=[1, 2, 3, 4, 5],
        seed=42,
        n_splits=5,
        use_global_fe=True,
        original_mode='fold',
        target_encode_all_combos=False,
        params_override={'epochs': 18, 'batch_size': 4096, 'eval_batch_size': 16384, 'lr': 0.0007, 'min_lr': 2e-05,
                         'weight_decay': 0.0001, 'patience': 4, 'dropout': 0.08, 'emb_dropout': 0.03, 'num_emb_dim': 0,
                         'label_smoothing': 0.008, 'scaler': 'quantile', 'num_workers': 0, 'model_type': 'graph_mlp',
                         'd_model': 128, 'hidden': [512, 256, 128]},
        add_domain_features=True,
        add_group_stats=True,
        add_extended_group_stats=True,
    ),
    # factorization machine (dim 48, head 384): domain features, group stats, source flag, fold-matched original rows
    ExpConfig(
        pair='fm_domain_group_stats_fold_original',
        folds_to_run=[1, 2, 3, 4, 5],
        seed=42,
        n_splits=5,
        use_global_fe=True,
        original_mode='fold',
        target_encode_all_combos=False,
        params_override={'epochs': 16, 'batch_size': 4096, 'eval_batch_size': 16384, 'lr': 0.0006, 'min_lr': 2e-05,
                         'weight_decay': 0.0002, 'patience': 5, 'dropout': 0.1, 'emb_dropout': 0.03, 'num_emb_dim': 0,
                         'label_smoothing': 0.012, 'scaler': 'quantile', 'num_workers': 0, 'model_type': 'fm',
                         'fm_dim': 48, 'fm_hidden': 384},
        add_domain_features=True,
        add_group_stats=True,
        add_source_indicator=True,
    ),
]
CONFIGS = {cfg.pair: cfg for cfg in CONFIGS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=sorted(CONFIGS))
    ap.add_argument('--folds', default='0,1,2,3,4', help='repo fold ids 0-4 (harness fold numbers 1-5)')
    a = ap.parse_args()
    cfg = dataclasses.replace(CONFIGS[a.pair], folds_to_run=[k + 1 for k in fold_list(a.folds)])
    run_experiment(cfg, model_dir('neural_nets'))


if __name__ == '__main__':
    main()
