"""Cache locations and fold helpers.

Inputs made by another repo folder:
  forensic fold matrices of models/lgbm_margin (its build_fold_features.py):
    cache_dir('lgbm_margin')/features/fold_<k>/{train,valid,test}_numeric.npy  (outer-train rows, rows of the fold,
                                                                                test rows; 333 columns)
    cache_dir('lgbm_margin')/features/fold_<k>/manifest.json                   (key 'base_numeric_names'; without
                                                                                it lib/realmlp.py uses fixed column
                                                                                positions)
Everything else is written by the scripts of this folder to cache_dir('realmlp_repair') (layout in README.md); the
reconstruction and density-completion fold caches are also read by other repo folders."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.data import cache_dir, folds, model_dir, read_original, read_test, read_train, target  # noqa: E402,F401

NAME = 'realmlp_repair'
MODEL_DIR = model_dir(NAME)


def forensic_dir(fold):
    return cache_dir('lgbm_margin') / 'features' / f'fold_{fold}'


def own_dir(*parts):
    d = cache_dir(NAME).joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


_FOLD_IDS = []


def fold_indices(fold):
    """(outer-train row indices, outer-valid row indices) in train.csv order."""
    if not _FOLD_IDS:
        _FOLD_IDS.append(folds())
    f = _FOLD_IDS[0]
    return np.flatnonzero(f != fold).astype(np.int64), np.flatnonzero(f == fold).astype(np.int64)


def save_arrays(directory, **arrays):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        if value is not None:
            np.save(directory / f'{name}.npy', value)


def save_fold_cache(directory, matrices, names, fold, **info):
    """Fold cache read by other repo folders: {train,valid,test}_features.npy (outer-training rows, rows of the fold,
    test rows), train_indices.npy / valid_indices.npy (train.csv rows) and manifest.json (feature_names, ...)."""
    fold_train, fold_valid = fold_indices(fold)
    save_arrays(directory, train_indices=fold_train, valid_indices=fold_valid,
                **{f'{role}_features': m for role, m in matrices.items()})
    (Path(directory) / 'manifest.json').write_text(json.dumps(dict(feature_names=list(names), **info), indent=1))


def load_fold_cache(directory, columns=None):
    """-> (names, {'train', 'valid', 'test'} float32 matrices) of a fold cache, optionally a column subset."""
    names = json.loads((Path(directory) / 'manifest.json').read_text())['feature_names']
    columns = list(range(len(names))) if columns is None else list(columns)
    out = {role: np.asarray(load_array(directory, f'{role}_features')[:, columns], dtype=np.float32)
           for role in ('train', 'valid', 'test')}
    return [names[i] for i in columns], out


def load_array(directory, name, mmap=True):
    return np.load(Path(directory) / f'{name}.npy', mmap_mode='r' if mmap else None)
