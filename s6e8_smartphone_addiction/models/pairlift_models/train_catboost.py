#!/usr/bin/env python
"""pairlift_catboost: the reconstruction CatBoost (24,000 GPU trees, forensic base margin as baseline) + the 10
pair-lift columns with CatBoost feature weight 0.1.

Features per outer fold, in this order (290 after the drops, 103 categorical):
  forensic parent of ../catboost_forensic (lib/strong_cat.py): the 102 forensic columns as float32 strong__* (+ the
      margin columns), the 114 fixed-original univariate LLR / log-count channels, native source-fragment categories
      (raw_cat, mask, exact_value, semantic_state: 20) and semantic interaction categories (source_qbins,
      source_q16_pairs, imputed_screen, source_rule: 53)
  10 pair-lift columns (feature weight 0.1; with weight 1 they overfit this learner)
  reconstruction__<compact 71>: cross-fitted XGBoost / masked-autoencoder reconstruction (fills if missing,
      disagreement, uncertainty, source LLR, error aggregates, completed screen geometry)
  density_response__<10 aggregates> of the density-completion responses
  30 reconstruction categories (lib/reconstruction_views.py, core view)
then strong__selected_base_margin / _probability and the 120 columns of DROP_FEATURES are removed.
Baseline = logit(prior) + 0.40 * raw_te_woe_sum + 0.10 * unsupported-pair response, added to the training pool and
every prediction pool. Depth 6, lr 0.01, l2 30, random strength 0.3, Bernoulli subsample 0.9, 254 borders, one-hot up
to 16 levels, max CTR complexity 3, NewtonL2, model size reg 0, random_seed 42 + fold, no early stopping;
test = mean of the 5 fold models.
Usage: python train_catboost.py [--folds 0,1,2,3,4]"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import folds, read_test, read_train, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from models.catboost_forensic.lib.strong_cat import (build_strong_application, cat_semantic_categories,  # noqa: E402
                                                     fixed_source_features, native_categories, source_test_features)
from lib import pairlift  # noqa: E402
from lib.reconstruction_views import reconstruction_categories  # noqa: E402
from lib.upstream import (MARGIN, PROBABILITY, density_completion_names, load_density_completion,  # noqa: E402
                          load_reconstruction, reconstruction_names)

PAIR = 'pairlift_catboost'
PAIRLIFT_FEATURE_WEIGHT = 0.1
CAT_GROUPS = ['raw_cat', 'mask', 'exact_value', 'semantic_state']
SOURCE_NUMERIC_GROUPS = ['univariate_llr_logcount']
SEMANTIC_GROUPS = ['source_qbins', 'source_q16_pairs', 'imputed_screen', 'source_rule']
PARAMS = {
    'iterations': 24000, 'depth': 6, 'learning_rate': 0.01, 'l2_leaf_reg': 30.0, 'random_strength': 0.3,
    'bootstrap_type': 'Bernoulli', 'subsample': 0.9, 'border_count': 254, 'one_hot_max_size': 16,
    'max_ctr_complexity': 3, 'score_function': 'NewtonL2', 'model_size_reg': 0.0,
}
DROP_FEATURES = (
    'fixed_source__binary_llr__weekend_screen_time__q16', 'strong__imputer_v2_l2_if_missing__work_study_hours',
    'fixed_source__binary_llr__weekend_screen_time__q8',
    'fixed_source__source_logcount__academic_work_impact__exact', 'fixed_source__binary_llr__social_media_hours__q8',
    'fixed_source__source_logcount__work_study_hours__q8', 'strong__academic_work_impact',
    'fixed_source__source_logcount__age__q32', 'fixed_source__binary_llr__academic_work_impact__exact',
    'fixed_source__binary_llr__daily_screen_time_hours__q16', 'fixed_source__source_logcount__stress_level__exact',
    'fixed_source__source_logcount__age__q64', 'fixed_source__source_logcount__social_media_hours__q8',
    'fixed_source__binary_llr__work_study_hours__q8', 'fixed_source__source_logcount__gender__exact',
    'fixed_source__source_logcount__age__q128', 'fixed_source__binary_llr__daily_screen_time_hours__q32',
    'strong__stress_level', 'fixed_source__source_logcount__daily_screen_time_hours__q8', 'strong__gender',
    'fixed_source__binary_llr__daily_screen_time_hours__q64', 'fixed_source__binary_llr__social_media_hours__q16',
    'fixed_source__binary_llr__social_media_hours__q32', 'fixed_source__binary_llr__gender__exact',
    'strong__imputer_v2_l2_if_missing__weekend_screen_time',
    'fixed_source__source_logcount__daily_screen_time_hours__q16', 'fixed_source__binary_llr__age__q64',
    'fixed_source__source_logcount__social_media_hours__q16', 'fixed_source__binary_llr__gaming_hours__q8',
    'fixed_source__source_logcount__gaming_hours__q8',
    'strong__exact_copula_density_ratio18_unsupported__daily_screen_time_hours__sleep_hours',
    'fixed_source__source_logcount__age__q8', 'fixed_source__source_logcount__sleep_hours__q8',
    'fixed_source__source_logcount__work_study_hours__q16', 'fixed_source__source_logcount__weekend_screen_time__q8',
    'fixed_source__source_logcount__notifications_per_day__q8', 'fixed_source__binary_llr__sleep_hours__q16',
    'fixed_source__source_logcount__age__q16', 'fixed_source__binary_llr__daily_screen_time_hours__q128',
    'strong__screen_response_logit__strict_slack__q16',
    'strong__exact_copula_density_ratio18_unsupported__daily_screen_time_hours__social_media_hours',
    'fixed_source__binary_llr__sleep_hours__q8',
    'strong__exact_copula_density_ratio18_unsupported__daily_screen_time_hours__work_study_hours',
    'strong__imputer_v2_l2_if_missing__daily_screen_time_hours',
    'strong__exact_copula_density_ratio18_unsupported__daily_screen_time_hours__gaming_hours',
    'fixed_source__binary_llr__weekend_screen_time__q32',
    'fixed_source__source_logcount__daily_screen_time_hours__q64', 'fixed_source__source_logcount__age__round2',
    'fixed_source__binary_llr__age__round2', 'fixed_source__binary_llr__work_study_hours__q16',
    'fixed_source__binary_llr__age__q8', 'fixed_source__source_logcount__social_media_hours__q32',
    'fixed_source__binary_llr__stress_level__exact',
    'strong__exact_copula_density_ratio18_unsupported__sleep_hours__weekend_screen_time',
    'strong__exact_copula_density_ratio18_unsupported__social_media_hours__notifications_per_day',
    'fixed_source__binary_llr__age__q128',
    'strong__exact_copula_density_ratio18_unsupported__social_media_hours__weekend_screen_time',
    'fixed_source__source_logcount__gaming_hours__q32', 'strong__age__frequency',
    'fixed_source__source_logcount__weekend_screen_time__q16', 'fixed_source__source_logcount__gaming_hours__q16',
    'fixed_source__binary_llr__daily_screen_time_hours__q8', 'fixed_source__binary_llr__weekend_screen_time__q64',
    'fixed_source__source_logcount__social_media_hours__q64', 'fixed_source__source_logcount__sleep_hours__q16',
    'fixed_source__binary_llr__notifications_per_day__q8', 'strong__gaming_hours',
    'fixed_source__source_logcount__daily_screen_time_hours__q32', 'fixed_source__binary_llr__age__q32',
    'strong__imputer_v2_l2_if_missing__gaming_hours', 'fixed_source__source_logcount__app_opens_per_day__q16',
    'fixed_source__binary_llr__gaming_hours__q16', 'strong__imputer_v2_l2_if_missing__social_media_hours',
    'fixed_source__source_logcount__social_media_hours__round2',
    'fixed_source__source_logcount__app_opens_per_day__q8',
    'fixed_source__source_logcount__weekend_screen_time__q32', 'fixed_source__source_logcount__gaming_hours__q64',
    'fixed_source__binary_llr__app_opens_per_day__q8',
    'strong__exact_copula_density_ratio18_unsupported__gaming_hours__work_study_hours',
    'fixed_source__source_logcount__weekend_screen_time__q64', 'strong__age', 'fixed_source__binary_llr__age__q16',
    'fixed_source__source_logcount__weekend_screen_time__round2',
    'fixed_source__source_logcount__daily_screen_time_hours__q128',
    'strong__exact_copula_density_ratio18_unsupported__work_study_hours__sleep_hours',
    'fixed_source__binary_llr__social_media_hours__q64', 'fixed_source__source_logcount__work_study_hours__q32',
    'strong__exact_copula_density_ratio18_unsupported__gaming_hours__sleep_hours', 'strong__work_study_hours',
    'strong__imputed_screen_response_logit__v2_l2__strict_slack__q16',
    'fixed_source__binary_llr__app_opens_per_day__q16',
    'strong__exact_copula_density_ratio18_unsupported__social_media_hours__work_study_hours',
    'strong__screen_response_logit__strict_slack__q64', 'fixed_source__source_logcount__social_media_hours__q128',
    'fixed_source__source_logcount__weekend_screen_time__q128',
    'fixed_source__binary_llr__weekend_screen_time__round2', 'strong__imputer_v2_l2_imputed__work_study_hours',
    'strong__scaled_complete_budget_remaining', 'fixed_source__binary_llr__sleep_hours__q32',
    'strong__exact_copula_density_ratio18_unsupported__daily_screen_time_hours__notifications_per_day',
    'strong__screen_response_logit__partial_slack__q16', 'fixed_source__binary_llr__social_media_hours__q128',
    'fixed_source__binary_llr__notifications_per_day__q16', 'fixed_source__source_logcount__gaming_hours__q128',
    'strong__weekend_minus_daily', 'fixed_source__source_logcount__app_opens_per_day__q64',
    'fixed_source__source_logcount__sleep_hours__q64',
    'fixed_source__source_logcount__notifications_per_day__round2',
    'fixed_source__source_logcount__notifications_per_day__q16',
    'fixed_source__binary_llr__notifications_per_day__q32',
    'strong__exact_copula_density_ratio18_unsupported__notifications_per_day__weekend_screen_time',
    'strong__exact_copula_density_ratio18_unsupported__daily_screen_time_hours__app_opens_per_day',
    'fixed_source__binary_llr__gaming_hours__q64', 'fixed_source__binary_llr__notifications_per_day__q64',
    'strong__imputer_v2_l2_imputed__gaming_hours', 'strong__imputed_screen_response_logit__v2_l2__strict_slack__q64',
    'strong__screen_response_logit__component_share_daily__q16', 'fixed_source__source_logcount__sleep_hours__q32',
    'strong__observed_joint_pmi__pmi_top18_shrink5__daily_screen_time_hours__social_media_hours__count_shrink_5',
    'fixed_source__binary_llr__weekend_screen_time__q128',
)


def parent_frames(fold):
    """Forensic CatBoost parent: (frames train / valid / test, categorical names, baselines, y, train_idx, valid_idx).
    Column order: strong__*, fixed-original numerics, native categories, fixed-original categories (none), semantic
    categories."""
    (strong_train, strong_valid, strong_test, y, train_idx, valid_idx,
     margin_train, margin_valid, margin_test) = build_strong_application(fold)
    test_rows = len(y) + np.arange(len(strong_test), dtype=np.int64)
    native = [native_categories(rows, CAT_GROUPS) for rows in (train_idx, valid_idx, test_rows)]
    cat_names = list(native[0][1])
    semantic_train, semantic_apply, semantic_names = cat_semantic_categories(
        strong_train, pd.concat([strong_valid, strong_test], ignore_index=True), SEMANTIC_GROUPS)
    semantic = [semantic_train, semantic_apply.iloc[:len(valid_idx)].reset_index(drop=True),
                semantic_apply.iloc[len(valid_idx):].reset_index(drop=True)]
    source_train, source_cat_train, source_cat_names = fixed_source_features(train_idx, SOURCE_NUMERIC_GROUPS)
    source_valid, source_cat_valid, _ = fixed_source_features(valid_idx, SOURCE_NUMERIC_GROUPS)
    source_test, source_cat_test = source_test_features(source_train.columns.tolist(), source_cat_names)
    cat_names += semantic_names + source_cat_names
    frames = {}
    for role, strong, source, nat, source_cat, sem in zip(
            ('train', 'valid', 'test'), (strong_train, strong_valid, strong_test),
            (source_train, source_valid, source_test), native, (source_cat_train, source_cat_valid, source_cat_test),
            semantic):
        frames[role] = pd.concat([strong.reset_index(drop=True), source.reset_index(drop=True), nat[0],
                                  source_cat.reset_index(drop=True), sem], axis=1)
    baselines = {'train': margin_train, 'valid': margin_valid, 'test': margin_test}
    return frames, cat_names, baselines, y, train_idx, valid_idx


def run_fold(fold, train_raw_all, test_raw):
    from catboost import CatBoostClassifier, Pool

    frames, cat_names, baselines, y, train_idx, valid_idx = parent_frames(fold)
    roles = ('train', 'valid', 'test')
    for role in roles:
        frames[role] = pairlift.append(frames[role], fold, role)

    recon_names = reconstruction_names()
    recon_full, recon_test = load_reconstruction(fold, train_idx, valid_idx, len(y))
    density_full, density_test = load_density_completion(fold, train_idx, valid_idx, len(y))
    density_idx = [i for i, n in enumerate(density_completion_names()) if '__aggregate__' in n]
    recon = {'train': recon_full[train_idx], 'valid': recon_full[valid_idx], 'test': recon_test}
    density = {'train': density_full[train_idx], 'valid': density_full[valid_idx], 'test': density_test}
    recon_columns = [f'reconstruction__{n}' for n in recon_names[:71]]
    density_columns = [f'density_response__{density_completion_names()[i]}' for i in density_idx]
    for role in roles:
        frames[role] = pd.concat([frames[role], pd.DataFrame(recon[role][:, :71], columns=recon_columns),
                                  pd.DataFrame(density[role][:, density_idx], columns=density_columns)], axis=1)

    raw = {'train': train_raw_all.iloc[train_idx].reset_index(drop=True),
           'valid': train_raw_all.iloc[valid_idx].reset_index(drop=True), 'test': test_raw}
    cat_train, cat_apply, recon_cat_names = reconstruction_categories(
        raw['train'], [raw['valid'], raw['test']], recon['train'], [recon['valid'], recon['test']], recon_names, 'core')
    for role, extra in zip(roles, [cat_train] + cat_apply):
        frames[role] = pd.concat([frames[role], extra], axis=1).drop(
            columns=[MARGIN, PROBABILITY, *DROP_FEATURES])
    cat_names = cat_names + recon_cat_names
    feature_names = frames['train'].columns.tolist()

    pools = {role: Pool(frames[role], label=y[train_idx] if role == 'train' else None, cat_features=cat_names,
                        feature_names=feature_names, baseline=baselines[role]) for role in roles}
    del frames, recon_full, density_full
    gc.collect()
    params = dict(PARAMS, feature_weights={name: PAIRLIFT_FEATURE_WEIGHT for name in pairlift.FEATURE_NAMES})
    model = CatBoostClassifier(**params, loss_function='Logloss', eval_metric='AUC', task_type='GPU', devices='0',
                               random_seed=42 + fold, thread_count=8, allow_writing_files=False, verbose=500)
    model.fit(pools['train'])
    valid_pred = np.asarray(model.predict_proba(pools['valid'])[:, 1], np.float64)
    test_pred = np.asarray(model.predict_proba(pools['test'])[:, 1], np.float64)
    print(f'fold {fold}: {len(feature_names)} features ({len(cat_names)} categorical), '
          f'AUC {roc_auc_score(y[valid_idx], valid_pred):.6f}', flush=True)
    return valid_pred, test_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    train_raw_all, test_raw = read_train(), read_test()
    for fold in fold_list(args.folds):
        if have_part(HERE, PAIR, fold):
            continue
        valid_pred, test_pred = run_fold(fold, train_raw_all, test_raw)
        save_part(HERE, PAIR, fold, valid_pred, test_pred)
    y = target()
    assemble(HERE, PAIR, folds(y), y)


if __name__ == '__main__':
    main()
