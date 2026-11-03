"""Column lists, data loading and folds shared by the density-feature scripts."""
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.data import CATEGORICALS, DATA_DIR, NUMERICS, ORIGINAL_FILE, TARGET, cache_dir, folds, read_test, read_train

NUMS = list(NUMERICS)
CATS = list(CATEGORICALS)
FEATS = NUMS + CATS
COMP = ['social_media_hours', 'gaming_hours', 'work_study_hours']   # the components of the daily screen-time budget
CACHE = cache_dir('density_features')


def load_comp():
    return read_train(), read_test()


def load_orig():
    """The 7,500 original rows. keep_default_na=False keeps the literal 'None' level of addiction_level; numerics and the
    label are coerced to numbers."""
    orig = pd.read_csv(DATA_DIR / ORIGINAL_FILE, keep_default_na=False)
    for c in NUMS + [TARGET]:
        orig[c] = pd.to_numeric(orig[c], errors='coerce')
    return orig


def get_folds(y):
    """StratifiedKFold(5, shuffle=True, random_state=42) fold id of every train row."""
    f = folds(y)
    assert (f >= 0).all()
    return f


def auc(y, p):
    return float(roc_auc_score(y, p))


def val_str(s):
    """Exact-value string tokens; NaN -> '__NA__'."""
    return s.astype(object).where(s.notna(), '__NA__').astype(str)
