"""Readers of the upstream fold caches the four parent recipes are built from (locations in lib/paths.py).

FoldCache: the shared forensic fold matrices (numeric / categories / category frequencies / category summaries).
  Column groups, by name prefix:
    strong        strong__*        104 columns: the forensic XGBoost core (raw columns as numbers, screen-budget
                                   features, frequencies, nested target encodings and WOE sums, imputer fills, screen
                                   responses, observed-pair PMI, source copula ratios, unsupported-pair response) plus
                                   strong__selected_base_margin / strong__selected_base_probability
    fixed_source  fixed_source__*  114 fixed-original likelihood channels (binary LLR and source log-count per value
                                   bin)
    native_base   native__raw_cat__ / native__mask__ / native__exact_value__ / native__semantic_state__ (20 categories)
    semantic      semantic__*      53 source-bin / imputed-bin / source-rule categories
Reconstruction cache: 189 columns per row (names from reconstruction_names()); the first 71 are the compact view, the
  TabM completion and the CatBoost reconstruction categories also read the detailed / full columns.
Density-completion cache: 30 response columns (names from density_completion_names()); 10 are aggregates.
Autoencoder fill view (IMPUTATION_FILL_NAMES, 25 reconstruction columns): completed numerics, categorical probabilities
  if missing and the completed screen geometry of the masked autoencoder."""
import json

import numpy as np

from lib import paths

NUMERIC_FIELDS = ('age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours',
                  'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time')
CATEGORICAL_LEVELS = {'gender': ('Female', 'Male', 'Other'), 'stress_level': ('High', 'Low', 'Medium'),
                      'academic_work_impact': ('No', 'Yes')}
SCREEN_FIELDS = ('daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours',
                 'weekend_screen_time')
MARGIN, PROBABILITY = 'strong__selected_base_margin', 'strong__selected_base_probability'
NUMERIC_GROUP_PREFIXES = {'strong': ('strong__',), 'fixed_source': ('fixed_source__',)}
CATEGORY_GROUP_PREFIXES = {
    'native_base': ('native__raw_cat__', 'native__mask__', 'native__exact_value__', 'native__semantic_state__'),
    'semantic': ('semantic__',),
}


class FoldCache:
    """One outer fold of the shared forensic feature cache."""

    def __init__(self, fold):
        self.root = paths.fold_cache_dir(fold)
        manifest = json.loads((self.root / 'manifest.json').read_text())
        self.numeric_names = list(manifest['base_numeric_names'])
        self.category_names = list(manifest['category_names'])
        self.frequency_names = list(manifest['category_frequency_names'])
        self.summary_names = list(manifest['category_summary_names'])

    def array(self, role, kind):
        """kind in numeric / categories / category_frequency / category_summary; role in train / valid / test."""
        return np.load(self.root / f'{role}_{kind}.npy', mmap_mode='r')

    def numeric_indices(self, groups):
        return [i for g in groups for i, n in enumerate(self.numeric_names) if n.startswith(NUMERIC_GROUP_PREFIXES[g])]

    def category_indices(self, groups):
        return [i for g in groups for i, n in enumerate(self.category_names)
                if n.startswith(CATEGORY_GROUP_PREFIXES[g])]


def reconstruction_names():
    names = []
    for f in NUMERIC_FIELDS:
        names += [f'hybrid__if_missing__{f}', f'hybrid__disagreement_norm_if_missing__{f}',
                  f'hybrid__ae_uncertainty_if_missing__{f}', f'hybrid__source_llr_if_missing__{f}']
    for f in CATEGORICAL_LEVELS:
        names += [f'hybrid__cat_disagreement_if_missing__{f}', f'hybrid__cat_entropy_if_missing__{f}',
                  f'hybrid__cat_source_llr_if_missing__{f}']
    for label in ('all', 'screen', 'nonscreen'):
        names += [f'reconagg__{kind}__{label}' for kind in ('xgb_abs_error_observed', 'ae_abs_error_observed',
                                                             'disagreement_missing', 'uncertainty_missing',
                                                             'source_llr_missing')]
    names += ['reconagg__categorical_disagreement_missing', 'reconagg__categorical_entropy_missing',
              'reconagg__categorical_source_llr_missing']
    geometry = ('screen_parts_sum', 'screen_slack', 'screen_slack_share', 'screen_social_share', 'screen_gaming_share',
                'screen_work_share', 'weekend_minus_daily', 'source_rule_probability')
    names += [f'hybridgeom__{g}' for g in geometry]                                  # compact view: 71 columns
    for f in NUMERIC_FIELDS:
        names += [f'xgb__if_missing__{f}', f'ae__if_missing__{f}', f'xgb__source_llr_if_missing__{f}',
                  f'ae__source_llr_if_missing__{f}']
    for f, levels in CATEGORICAL_LEVELS.items():
        for mechanism in ('xgb', 'ae'):
            names += [f'{mechanism}__cat_probability_if_missing__{f}__{level}' for level in levels]
    for f in NUMERIC_FIELDS:
        names += [f'xgbfull__residual_norm_observed__{f}', f'aefull__residual_norm_observed__{f}',
                  f'xgbfull__completed__{f}', f'aefull__completed__{f}', f'hybridfull__completed__{f}']
    names += [f'aefull__latent__{s}' for s in ('norm', 'mean', 'std', 'abs_mean', 'abs_max')]
    names += [f'xgbfullgeom__{g}' for g in geometry] + [f'aefullgeom__{g}' for g in geometry]
    return names


def density_completion_names():
    names = []
    for f in SCREEN_FIELDS:
        names += [f'density_completion__{f}__{kind}' for kind in ('context_expected_logit', 'observed_or_context_logit',
                                                                   'context_response_std', 'context_entropy')]
    names += ['density_completion__aggregate__' + kind for kind in (
        'completion_logit_mean', 'completion_logit_std', 'completion_logit_min', 'completion_logit_max',
        'context_logit_mean', 'context_logit_std', 'missing_context_logit_mean', 'missing_context_logit_std',
        'missing_field_count', 'context_confidence_mean')]
    return names


IMPUTATION_FILL_NAMES = (
    [f'aefull__completed__{f}' for f in NUMERIC_FIELDS]
    + [f'ae__cat_probability_if_missing__{f}__{level}' for f, levels in CATEGORICAL_LEVELS.items() for level in levels]
    + [f'aefullgeom__{g}' for g in ('screen_parts_sum', 'screen_slack', 'screen_slack_share', 'screen_social_share',
                                    'screen_gaming_share', 'screen_work_share', 'weekend_minus_daily',
                                    'source_rule_probability')])


def _full_and_test(root, train_idx, valid_idx, n_train, width):
    parts = {role: np.load(root / f'{role}_features.npy', mmap_mode='r') for role in ('train', 'valid', 'test')}
    assert all(m.shape[1] == width for m in parts.values()), f'{root}: expected {width} columns per row'
    full = np.empty((n_train, width), dtype=np.float32)
    full[train_idx] = parts['train']
    full[valid_idx] = parts['valid']
    return full, np.asarray(parts['test'], dtype=np.float32)


def load_reconstruction(fold, train_idx, valid_idx, n_train):
    """(train.csv-ordered 189-column matrix: outer-training rows row-excluded cross-fit, outer-validation rows from the
    outer-training state; test matrix)."""
    return _full_and_test(paths.reconstruction_dir(fold), train_idx, valid_idx, n_train, 189)


def load_density_completion(fold, train_idx, valid_idx, n_train):
    """(train.csv-ordered 30-column density-completion response matrix; test matrix)."""
    return _full_and_test(paths.density_completion_dir(fold), train_idx, valid_idx, n_train, 30)


def load_imputation_fill(fold, role):
    """The 25-column masked-autoencoder fill view of one role (train / valid / test rows of the fold)."""
    names = reconstruction_names()
    values = np.load(paths.reconstruction_dir(fold) / f'{role}_features.npy', mmap_mode='r')
    assert values.shape[1] == len(names), 'the reconstruction cache must hold all 189 columns'
    return np.asarray(values[:, [names.index(n) for n in IMPUTATION_FILL_NAMES]], dtype=np.float32)
