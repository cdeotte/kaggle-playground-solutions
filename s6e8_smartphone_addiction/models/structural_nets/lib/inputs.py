"""Fold matrices made by other folders of this repo (run those folders first).

../lgbm_margin     cache/features/fold_<k>/{train,valid,test}_numeric.npy  the 333-column fold-specific forensic matrix
                   (104 forensic-core columns incl. inner-cross-fitted target-encoding WoE sums and the selected base
                   margin, 115 source-alignment / fragment / domain columns, 114 fixed original-data likelihood and
                   count channels); column names in cache/features/fold_<k>/manifest.json (base_numeric_names).
../realmlp_repair  cache/reconstruction-production-v1/fold_<k>/{train,valid,test}_features.npy: the first 71 of its
                   189 columns are the compact reconstruction channels (masked-autoencoder / XGBoost field
                   reconstructions).
Both have one block per role: the outer-train rows of fold k (train.csv order), the held-out rows of fold k
(train.csv order) and all test rows."""
import json

import numpy as np

from common.data import cache_dir

FORENSIC_WIDTH = 333
RECONSTRUCTION_COMPACT = 71
# column positions in the 333-column forensic matrix, used when its manifest.json is not available
KNOWN_POSITIONS = {'strong__raw_te_woe_sum': 38, 'strong__screen_te_woe_sum': 39, 'strong__selected_base_margin': 102,
                   'strong__selected_base_probability': 103}


def forensic_dir(fold):
    return cache_dir('lgbm_margin') / 'features' / f'fold_{fold}'


def reconstruction_dir(fold):
    return cache_dir('realmlp_repair') / 'reconstruction-production-v1' / f'fold_{fold}'


def forensic_columns(fold, names):
    """Positions of `names` among the 333 forensic columns of fold `fold`."""
    path = forensic_dir(fold) / 'manifest.json'
    if path.exists():
        all_names = list(json.loads(path.read_text())['base_numeric_names'])
        if len(all_names) != FORENSIC_WIDTH:
            raise RuntimeError(f'expected {FORENSIC_WIDTH} forensic columns, found {len(all_names)}')
        return [all_names.index(name) for name in names]
    return [KNOWN_POSITIONS[name] for name in names]


def load_forensic(fold, role):
    """role: 'train' (outer-train rows), 'valid' (held-out rows) or 'test'."""
    value = np.load(forensic_dir(fold) / f'{role}_numeric.npy', mmap_mode='r')
    if value.shape[1] != FORENSIC_WIDTH:
        raise RuntimeError(f'forensic matrix has {value.shape[1]} columns, expected {FORENSIC_WIDTH}')
    return value


def load_reconstruction_compact(fold, role):
    """The 71 compact reconstruction channels (the first 71 columns if a wider matrix is stored), memory-mapped."""
    value = np.load(reconstruction_dir(fold) / f'{role}_features.npy', mmap_mode='r')
    if value.shape[1] < RECONSTRUCTION_COMPACT:
        raise RuntimeError(f'reconstruction matrix has {value.shape[1]} columns, expected {RECONSTRUCTION_COMPACT}')
    return value[:, :RECONSTRUCTION_COMPACT]
