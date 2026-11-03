"""The RealMLP shared by every step of the chain.

Base matrix (135 columns, 12 categorical):
  row-local blocks (lib/rowwise.py): raw (9 numerics + 3 categoricals), 9 exact-value categorical tokens,
  19 screen-geometry columns                                                                      -> 40
  95 columns of the forensic fold matrix (models/lgbm_margin cache, see lib/paths.py), selected by name:
    nested target encodings of the 9 numerics, TE / WOE sums, the selected base margin / probability,
    daily minus known parts, imputed screen values + if-missing indicators, screen-response logits
    (q16 / q64, observed and imputed), 18 observed joint-PMI pair channels, unsupported-pair response,
    original-data binary LLRs (round2 / q16 / exact) and original exact / nearest label priors          -> 95
  (every target-derived column of that matrix is 5-way inner cross-fitted inside the outer-training rows)
Each step appends a 'source' block and the 10 completion aggregates after the base matrix.
Preprocessing: categoricals as strings ('__MISSING__' for NaN); numeric NaN / inf -> outer-training median.
Model: pytabkit RealMLP_TD_Classifier, hidden [768, 384, 192], PBLRD numeric embeddings (16, 8), Mish, dropout 0.05,
wd 0.017, batch 256, label smoothing 0.005, 32 internal ensemble members, 9 epochs with a 15% inner validation split
choosing the epoch (8 in every fold) followed by a refit on all outer-training rows; seed 142 + fold."""
import json
import os
import re

import numpy as np
import pandas as pd

from .paths import fold_indices, forensic_dir
from .rowwise import BLOCKS, build_rowwise_features

MODEL_PARAMS = {
    'act': 'mish', 'act_lr_factor': 0.1, 'add_front_scale': True, 'batch_size': 256,
    'bias_init_mode': 'neg-uniform-dynamic-2', 'early_stopping_additive_patience': 10,
    'early_stopping_multiplicative_patience': 1, 'embedding_size': 6,
    'first_layer_config': {'scale_wd_factor': 0.4082482904638631}, 'first_layer_lr_factor': 0.125,
    'hidden_sizes': [768, 384, 192], 'lr': {'': 0.0175, '^/tfms/emb': 0.035}, 'lr_sched': 'lin_cos_log_15',
    'ls_eps': 0.005, 'ls_eps_sched': 'sqrt_cos', 'max_one_hot_cat_size': 18, 'n_cv': 1, 'n_ens': 32, 'n_epochs': 9,
    'n_refit': 1, 'n_threads': 4, 'num_emb_type': 'pblrd', 'p_drop': 0.05, 'p_drop_sched': 'invsqrtp1e-3',
    'plr_act_name': 'gelu', 'plr_hidden_1': 16, 'plr_hidden_2': 8, 'plr_lr_factor': 0.08, 'plr_sigma': 0.94,
    'predict_batch_size': 8192, 'scale_lr_factor': 6.0, 'sq_mom': 0.99,
    'tfms': ['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding', 'l2_normalize'],
    'use_early_stopping': False, 'use_ls': True, 'val_fraction': 0.15, 'val_metric_name': '1-auc_ovr',
    'verbosity': 2, 'wd': 0.017,
}
NUMERICS = '(?:age|daily_screen_time_hours|social_media_hours|gaming_hours|work_study_hours|sleep_hours|' \
           'notifications_per_day|app_opens_per_day|weekend_screen_time)'
FORENSIC_COLUMNS = [
    f'^strong__{NUMERICS}__te$',
    '^strong__(?:raw_te_woe_sum|screen_te_woe_sum|selected_base_margin|selected_base_probability|daily_minus_known_parts)$',
    '^strong__imputer_v2_l2_(?:imputed|if_missing)__',
    '^strong__(?:imputed_)?screen_response_logit__',
    '^strong__observed_joint_pmi__',
    '^strong__unsupported_pair_response__',
    '^fixed_source__binary_llr__.*__(?:round2|q16|exact)$',
    '^cat_numeric__source_alignment__.*__(?:exact_label_prior|nearest_label_prior)$',
]


# positions of those 95 columns in the 333-column forensic matrix (the same in every fold); used when the matrix's
# manifest.json with its column names is not available
FORENSIC_POSITIONS = list(range(29, 41)) + list(range(42, 82)) + [100, 101, 102, 103, 104] + [
    108, 116, 120, 128, 132, 140, 144, 152, 156, 164, 168, 176, 180, 188, 192, 200, 204, 212, 214, 216] + [
    243, 244, 250, 251, 257, 258, 264, 265, 271, 272, 278, 279, 285, 286, 292, 293, 299, 300]


def forensic_columns(fold):
    path = forensic_dir(fold) / 'manifest.json'
    if not path.exists():
        return FORENSIC_POSITIONS, [f'forensic__{i}' for i in FORENSIC_POSITIONS]
    names = json.loads(path.read_text())['base_numeric_names']
    patterns = [re.compile(p) for p in FORENSIC_COLUMNS]
    index = [i for i, n in enumerate(names) if any(p.search(n) for p in patterns)]
    return index, [names[i] for i in index]


def _append(frame, matrix, names):
    values = np.asarray(matrix, dtype=np.float32)
    assert values.shape == (len(frame), len(names))
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(values, columns=names)], axis=1)


def prepare_frames(train_frame, valid_frame, test_frame, categorical):
    numeric = [c for c in train_frame.columns if c not in set(categorical)]
    for c in categorical:
        for frame in (train_frame, valid_frame, test_frame):
            if frame is not None:
                frame[c] = frame[c].astype('string').fillna('__MISSING__').astype(object)
    for c in numeric:
        train_values = pd.to_numeric(train_frame[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
        median = float(train_values.median())
        if not np.isfinite(median):
            median = 0.0
        train_frame[c] = train_values.fillna(median).astype(np.float32)
        for frame in (valid_frame, test_frame):
            if frame is not None:
                frame[c] = pd.to_numeric(frame[c], errors='coerce').replace([np.inf, -np.inf], np.nan) \
                    .fillna(median).astype(np.float32)
    return train_frame, valid_frame, test_frame


def seed_everything(seed):
    import random

    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fit_fold(fold, train, test, y, extra_blocks, predict_test=True):
    """extra_blocks: list of (names, {'train', 'valid', 'test'} float32 matrices) appended after the base matrix.
    Returns (outer-valid probabilities, test probabilities or None)."""
    import torch
    from pytabkit import RealMLP_TD_Classifier

    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    fold_train, fold_valid = fold_indices(fold)
    columns, column_names = forensic_columns(fold)
    base_full, categorical = build_rowwise_features(train, BLOCKS)
    frames = {'train': base_full.iloc[fold_train].reset_index(drop=True).copy(),
              'valid': base_full.iloc[fold_valid].reset_index(drop=True).copy()}
    if predict_test:
        frames['test'] = build_rowwise_features(test, BLOCKS)[0]
    for role in list(frames):
        cache = np.load(forensic_dir(fold) / f'{role}_numeric.npy', mmap_mode='r')
        frames[role] = _append(frames[role], cache[:, columns], column_names)
        for names, matrices in extra_blocks:
            frames[role] = _append(frames[role], matrices[role], names)
    train_frame, valid_frame, test_frame = prepare_frames(frames['train'], frames['valid'], frames.get('test'),
                                                          categorical)
    print(f'fold {fold}: {train_frame.shape[1]} features ({len(categorical)} categorical)', flush=True)

    params = dict(MODEL_PARAMS, device='cuda:0', random_state=142 + fold)
    seed_everything(142 + fold)
    model = RealMLP_TD_Classifier(**params)
    model.fit(train_frame, y[fold_train], cat_col_names=categorical)
    valid_prediction = model.predict_proba(valid_frame)[:, 1].astype(np.float64)
    test_prediction = model.predict_proba(test_frame)[:, 1].astype(np.float64) if test_frame is not None else None
    del model
    torch.cuda.empty_cache()
    return valid_prediction, test_prediction
