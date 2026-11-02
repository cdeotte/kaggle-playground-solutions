#!/usr/bin/env python
"""RealMLP (pytabkit RealMLP_TD_Classifier) on response encodings, income / commute window residuals and the original-data
source bank.

--pair realmlp_response_geometry: 209 inputs (25 categorical)
  raw, tokens, coarse         the 13 raw features; every numeric value as a categorical token; fixed-width bins (features.py)
  cache/base                  source_full_global (113), response_single_loo (30), response_local_loo (3),
                              response_target_gender_home (1)
  factor hinges               max(z - t, 0), t in (-1, -0.5, 0, 0.5, 1, 2), z = (sigmoid(resp_logit) - 0.175) / 0.2, for income,
                              environmental concern, home charging, subsidy and range anxiety (30)
--pair realmlp_multires: the same + the 6 families of cache/multires (184 columns), 393 inputs.
Scaling: predictor-only numeric columns (raw numerics, source bank): outer-train median / IQR, then x / sqrt(1 + x^2 / 9).
Target-derived columns: fixed constants only (logits: (x - logit(0.175)) / 3, residuals and multires columns: x / 3, log counts:
x / log1p(#outer-train rows), lifts: x / 0.2, rates: (x - 0.175) / 0.2), then the same soft clip. Categoricals: one-hot up
to 18 levels, else embeddings.
RealMLP: 16-member ensemble, 9 epochs; the epoch is picked on an internal 15% split of the outer-train rows, then 2 refits on
all outer-train rows (n_refit 2); seed 202609 + fold. Test = mean of the 5 fold models.
Usage: python train_realmlp.py --pair realmlp_response_geometry|realmlp_multires [--folds 0,1,2,3,4]"""
import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, read_test, read_train, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from features import CACHE, build_row_local, read_manifest  # noqa: E402

PAIRS = ('realmlp_response_geometry', 'realmlp_multires')
PARENT_BLOCKS = ('source_full_global', 'response_single_loo', 'response_local_loo', 'response_target_gender_home')
MULTIRES_FAMILIES = ('exact_single', 'exact_avg8', 'core_single', 'core_avg8', 'wide_avg8', 'response_summaries')
HINGE_FIELDS = ('Annual_Income_USD', 'Environmental_Concern_Level', 'Home_Charging_Possible', 'Subsidy_Available',
                'Range_Anxiety_Level')
HINGE_THRESHOLDS = (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
SOURCE_PRIOR = 0.175
DIVISOR = 3.0
SEED_BASE = 202609
PARAMS = {
    'act': 'mish', 'act_lr_factor': 0.1, 'add_front_scale': True, 'batch_size': 1024,
    'bias_init_mode': 'neg-uniform-dynamic-2', 'early_stopping_additive_patience': 10,
    'early_stopping_multiplicative_patience': 1, 'embedding_size': 6, 'ens_av_before_softmax': True,
    'first_layer_config': {'scale_wd_factor': 0.4082482904638631}, 'first_layer_lr_factor': 0.125,
    'hidden_sizes': [768, 384, 192], 'lr': {'': 0.0175, '^/tfms/emb': 0.035}, 'lr_sched': 'lin_cos_log_15', 'ls_eps': 0.0,
    'ls_eps_sched': 'sqrt_cos', 'max_one_hot_cat_size': 18, 'n_cv': 1, 'n_ens': 16, 'n_epochs': 9, 'n_refit': 2,
    'n_threads': 4, 'num_emb_type': 'pblrd', 'p_drop': 0.05, 'p_drop_sched': 'invsqrtp1e-3', 'plr_act_name': 'gelu',
    'plr_hidden_1': 16, 'plr_hidden_2': 8, 'plr_lr_factor': 0.08, 'plr_sigma': 0.94, 'predict_batch_size': 8192,
    'scale_lr_factor': 6, 'sq_mom': 0.99, 'tfms': ['one_hot', 'embedding'], 'use_early_stopping': False, 'use_ls': True,
    'val_fraction': 0.15, 'val_metric_name': '1-auc_ovr', 'verbosity': 0, 'wd': 0.008,
}


def soft_clip(x):
    return x / np.sqrt(1.0 + x * x / 9.0)


def robust_predictor_scale(frames, names):
    """Median / IQR of the outer-train rows (frames[0]), then the soft clip."""
    for name in names:
        fit = frames[0][name].to_numpy(np.float64)
        median = float(np.nanmedian(fit))
        if not np.isfinite(median):
            median = 0.0
        filled = np.nan_to_num(fit, nan=median, posinf=median, neginf=median)
        q25, q75 = np.quantile(filled, [0.25, 0.75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0:
            spread = float(np.ptp(filled))
            scale = spread / 2.0 if spread > 0 else 1.0
        for frame in frames:
            values = np.nan_to_num(frame[name].to_numpy(np.float64), nan=median, posinf=median, neginf=median)
            frame[name] = soft_clip((values - median) / scale).astype(np.float32)


def fixed_parent_response_scale(frames, names, n_outer_train):
    prior_logit = float(np.log(SOURCE_PRIOR / (1.0 - SOURCE_PRIOR)))
    for name in names:
        for frame in frames:
            values = frame[name].to_numpy(np.float64)
            if 'logcount' in name:
                scaled = values / np.log1p(float(n_outer_train))
            elif name.startswith('resp_logit__'):
                scaled = (values - prior_logit) / DIVISOR
            elif name.startswith('resp_delta__'):
                scaled = values / DIVISOR
            elif 'pair_lift' in name:
                scaled = values / 0.20
            elif name.startswith('resp_p_'):
                scaled = (values - SOURCE_PRIOR) / 0.20
            else:
                raise ValueError(f'no scaling rule for {name}')
            frame[name] = soft_clip(scaled).astype(np.float32)


def fixed_response_scale(frames, names):
    for name in names:
        for frame in frames:
            frame[name] = soft_clip(frame[name].to_numpy(np.float64) / DIVISOR).astype(np.float32)


def append(frame, values, names):
    return pd.concat([frame, pd.DataFrame(np.asarray(values), columns=names, dtype=np.float32)], axis=1, copy=False)


def build_inputs(pair, k, f):
    train, test = read_train(), read_test()
    base, categorical = build_row_local(train)
    base_test, _ = build_row_local(test)
    tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
    frames = [base.iloc[tr].reset_index(drop=True).copy(), base.iloc[va].reset_index(drop=True).copy(),
              base_test.reset_index(drop=True).copy()]
    roles = ('train', 'valid', 'test')
    predictor = [c for c in frames[0].columns if c not in categorical]
    parent_response = []
    folder = CACHE / 'base' / f'fold_{k}'
    manifest = read_manifest(folder)
    for block in PARENT_BLOCKS:
        names = list(manifest[f'{block}_names'])
        frames = [append(frame, np.load(folder / f'{block}_{role}.npy'), names) for frame, role in zip(frames, roles)]
        (parent_response if block.startswith('response_') else predictor).extend(names)

    hinge_names = [f'resp_delta__factor_hinge__{field}__{str(t).replace("-", "m").replace(".", "p")}'
                   for field in HINGE_FIELDS for t in HINGE_THRESHOLDS]
    for i, frame in enumerate(frames):
        columns = []
        for field in HINGE_FIELDS:
            logit_ = np.clip(frame[f'resp_logit__{field}'].to_numpy(np.float64), -30.0, 30.0)
            z = (1.0 / (1.0 + np.exp(-logit_)) - SOURCE_PRIOR) / 0.20
            columns += [np.maximum(z - t, 0.0) for t in HINGE_THRESHOLDS]
        frames[i] = append(frame, np.column_stack(columns).astype(np.float32), hinge_names)
    parent_response += hinge_names

    response = []
    if pair == 'realmlp_multires':
        folder = CACHE / 'multires' / f'fold_{k}'
        families = read_manifest(folder)['families']
        for family in MULTIRES_FAMILIES:
            names = list(families[family]['names'])
            frames = [append(frame, np.load(folder / f'{family}_{role}.npy'), names) for frame, role in zip(frames, roles)]
            response.extend(names)

    robust_predictor_scale(frames, predictor)
    fixed_parent_response_scale(frames, parent_response, len(tr))
    fixed_response_scale(frames, response)
    return frames, categorical, tr, va


def run_fold(pair, k, y, f):
    import torch
    from pytabkit import RealMLP_TD_Classifier
    torch.set_float32_matmul_precision('high')
    (x_train, x_valid, x_test), categorical, tr, va = build_inputs(pair, k, f)
    print(f'fold {k}: {x_train.shape[1]} inputs ({len(categorical)} categorical)', flush=True)
    seed = SEED_BASE + k
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = RealMLP_TD_Classifier(**PARAMS, device='cuda', random_state=seed)
    model.fit(x_train, y[tr], cat_col_names=categorical)
    return model.predict_proba(x_valid)[:, 1], model.predict_proba(x_test)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True, choices=PAIRS)
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    y = target()
    f = folds(y)
    for k in fold_list(a.folds):
        if have_part(HERE, a.pair, k):
            print(f'fold {k}: done', flush=True)
            continue
        valid_pred, test_pred = run_fold(a.pair, k, y, f)
        print(f'fold {k}: AUC {roc_auc_score(y[f == k], valid_pred):.6f}', flush=True)
        save_part(HERE, a.pair, k, valid_pred, test_pred)
    assemble(HERE, a.pair, f, y)


if __name__ == '__main__':
    main()
