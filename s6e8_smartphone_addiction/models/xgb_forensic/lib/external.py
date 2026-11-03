"""Fold-owned external blocks of the forensic pipeline.

screen_imputer(fold, ...)   cache/screen_imputer_v2/fold_{k}_{train,test}.npy (build_screen_imputer.py): 5 screen
                            fields imputed by LightGBM regressors fitted on the fold's outer-training predictors;
                            the 'l2' columns (squared-error model): imputed value and value-if-missing (0 if observed)
screen_response_slack(fold) the 6 slack / share logits of ../forensic_features (q16, q64)
A block is (names, train, valid, test) with train / valid = the fold's outer-training rows / rows of the fold."""
import json

import numpy as np

from common.data import cache_dir
from models.forensic_features.lib.blocks import screen_response

SLACK_KEYS = ("strict_slack", "partial_slack", "component_share_daily")


def xgb_cache():
    return cache_dir("xgb_forensic")


def screen_imputer(fold, train_idx, valid_idx, selection="l2"):
    root = xgb_cache() / "screen_imputer_v2"
    names_all = list(json.loads((root / "manifest.json").read_text())["feature_names"])
    keep = [i for i, name in enumerate(names_all) if name.startswith(f"imputer_v2_{selection}_")]
    train_all = np.load(root / f"fold_{fold}_train.npy", mmap_mode="r")
    test_all = np.load(root / f"fold_{fold}_test.npy", mmap_mode="r")
    return ([names_all[i] for i in keep], np.asarray(train_all[train_idx][:, keep], np.float32),
            np.asarray(train_all[valid_idx][:, keep], np.float32), np.asarray(test_all[:, keep], np.float32))


def screen_response_slack(fold):
    train, valid, test, names = screen_response(fold)
    expected = {f"screen_response_logit__{key}__q{bins}" for key in SLACK_KEYS for bins in (16, 64)}
    keep = [i for i, name in enumerate(names) if name in expected]
    return ([names[i] for i in keep], np.asarray(train[:, keep], np.float32), np.asarray(valid[:, keep], np.float32),
            np.asarray(test[:, keep], np.float32))
