"""The 102-column forensic feature pipeline of one outer fold (the 'strong' matrix used by every model of this folder
and by ../catboost_forensic, ../lgbm_margin).

Column order (the XGBoost colsample draws depend on it):
  0-41    parent (lib/parent.py): 29 label-free columns, 9 nested TEs, 2 WOE sums, 2 partial-budget residuals
  42-51   l2 screen imputer: imputed value and value-if-missing of daily / social / gaming / work / weekend
  52-57   screen-response slack logits (q16, q64)
          [extra blocks of the later models are inserted here, see build_strong_fold(extra_parts=...)]
  58-63   imputed screen response (6)
  64-81   observed-joint PMIs of the PMI_PAIRS18 pairs shrunk by count / (count + 5)
  82-99   exact original copula density ratios, unsupported pairs only
  100-101 unsupported pair response: weighted mean, available fraction
Base margin 'synthetic_unsupported:0.40:0.10': logit(prior) + 0.40 raw_te_woe_sum + 0.10 unsupported weighted mean.
The outer-training rows get cross-fitted target statistics; the fold's rows and the test rows are transformed with the
state fitted on all outer-training rows (validation and test are processed together as one application set)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.data import folds as fold_ids

from . import parent
from .external import screen_imputer, screen_response_slack, xgb_cache
from .pair_evidence import (exact_copula_unsupported, imputed_screen_response, load_observed_joint,
                            observed_joint_pmi18, unsupported_pair_response)

MARGIN_VARIANT = "synthetic_unsupported:0.40:0.10"
XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist", "device": "cuda",
    "enable_categorical": True, "max_bin": 2048, "n_estimators": 12_000, "learning_rate": 0.006875, "max_depth": 6,
    "grow_policy": "depthwise", "max_leaves": 0, "min_child_weight": 1.0, "subsample": 0.80, "colsample_bytree": 0.80,
    "colsample_bynode": 1.0, "reg_alpha": 0.05, "reg_lambda": 10.0, "gamma": 0.0, "scale_pos_weight": 1.0,
    "max_delta_step": 0.0, "sampling_method": "uniform", "max_cat_threshold": 64, "random_state": 42, "n_jobs": 8,
    "verbosity": 0,
}


@dataclass
class StrongFold:
    fold: int
    train: pd.DataFrame       # outer-training rows
    valid: pd.DataFrame       # rows of the fold
    test: pd.DataFrame        # all test rows
    target: np.ndarray        # int8 target of all train rows
    train_idx: np.ndarray
    valid_idx: np.ndarray
    raw_train: pd.DataFrame   # train.csv
    raw_test: pd.DataFrame    # test.csv

    @property
    def prior(self):
        return float(self.target[self.train_idx].mean())

    @property
    def y_train(self):
        return self.target[self.train_idx]

    @property
    def y_valid(self):
        return self.target[self.valid_idx]


def logit(probability):
    values = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def base_margin(frame, prior, variant=MARGIN_VARIANT):
    """'synthetic:a' = logit(prior) + a raw_te_woe_sum; 'synthetic_unsupported:a:b' adds b x the unsupported-pair
    weighted mean."""
    pieces = variant.split(":")
    margin = float(logit(prior)) + float(pieces[1]) * frame["raw_te_woe_sum"].to_numpy(np.float64)
    if pieces[0] == "synthetic_unsupported":
        response = [c for c in frame if c.startswith("unsupported_pair_response__") and c.endswith("__weighted_mean")]
        assert len(response) == 1
        margin = margin + float(pieces[2]) * frame[response[0]].to_numpy(np.float64)
    elif pieces[0] != "synthetic":
        raise ValueError(variant)
    return margin


def _append(frame, values, names):
    return pd.concat([frame, pd.DataFrame(np.asarray(values, np.float32), columns=list(names), index=frame.index)],
                     axis=1)


def build_strong_fold(fold, encoder_class=None, extra_parts=None, verbose=True):
    """Build the forensic matrix of outer fold `fold`.
    encoder_class: the nested-TE encoder (default cuML's TargetEncoder; any class with the same fit_transform(x, y,
      fold_ids=...) / transform(x) interface).
    extra_parts: optional callable (fold, train_idx, valid_idx) -> list of (names, train, valid, test) arrays,
      appended right after the screen-response block (before the imputed screen response)."""
    raw_train, raw_test, X, X_test, y = parent.prepare_data()
    target = y.to_numpy(np.int8)
    outer = fold_ids(target)
    train_idx = np.flatnonzero(outer != fold).astype(np.int64)
    valid_idx = np.flatnonzero(outer == fold).astype(np.int64)
    n_valid = len(valid_idx)
    priors = parent.compute_component_share_priors(X, X_test)

    application = pd.concat([X.iloc[valid_idx], X_test], ignore_index=True)
    model_train, model_app = parent.add_repeated_inner_target_encoding(
        X.iloc[train_idx].copy(), y.iloc[train_idx], application, outer_fold=fold + 1, encoder_class=encoder_class,
        repeats=3)
    model_train, model_app = parent.add_partial_budget_features(model_train, model_app, priors)
    model_train, model_app = model_train.reset_index(drop=True), model_app.reset_index(drop=True)
    if verbose:
        print(f"fold {fold}: parent {model_train.shape[1]} columns", flush=True)

    blocks = [screen_imputer(fold, train_idx, valid_idx), screen_response_slack(fold)]
    if extra_parts is not None:
        blocks += list(extra_parts(fold, train_idx, valid_idx))
    for names, train_values, valid_values, test_values in blocks:
        model_train = _append(model_train, train_values, names)
        model_app = _append(model_app, np.concatenate([valid_values, test_values]), names)

    y_train = target[train_idx]
    imputed_train, imputed_app = imputed_screen_response(model_train, model_app, y_train, train_idx)
    model_train = pd.concat([model_train, imputed_train], axis=1)
    model_app = pd.concat([model_app, imputed_app], axis=1)

    joint_train, joint_test, joint_names, complete_counts = load_observed_joint(xgb_cache())
    joint_app_rows = np.concatenate([np.asarray(joint_train[valid_idx]), np.asarray(joint_test)])
    joint_train_rows = np.asarray(joint_train[train_idx])
    model_train = pd.concat([model_train, observed_joint_pmi18(joint_train_rows, joint_names)], axis=1)
    model_app = pd.concat([model_app, observed_joint_pmi18(joint_app_rows, joint_names)], axis=1)

    raw_train_rows = raw_train.iloc[train_idx].reset_index(drop=True)
    raw_app_rows = pd.concat([raw_train.iloc[valid_idx], raw_test], ignore_index=True)
    model_train = pd.concat(
        [model_train, exact_copula_unsupported(raw_train_rows, joint_train_rows, joint_names, complete_counts)], axis=1)
    model_app = pd.concat(
        [model_app, exact_copula_unsupported(raw_app_rows, joint_app_rows, joint_names, complete_counts)], axis=1)

    response_train, response_app = unsupported_pair_response(raw_train_rows, raw_app_rows, y_train, train_idx)
    model_train = pd.concat([model_train, response_train], axis=1)
    model_app = pd.concat([model_app, response_app], axis=1)
    if verbose:
        print(f"fold {fold}: forensic matrix {model_train.shape[1]} columns, train {len(model_train)}, "
              f"valid {n_valid}, test {len(model_app) - n_valid}", flush=True)
    return StrongFold(fold=fold, train=model_train, valid=model_app.iloc[:n_valid].reset_index(drop=True),
                      test=model_app.iloc[n_valid:].reset_index(drop=True), target=target, train_idx=train_idx,
                      valid_idx=valid_idx, raw_train=raw_train, raw_test=raw_test)
