"""Selector + full refit XGBoost of the frontier models.

Selector: the first split of StratifiedKFold(10, shuffle, random_state 91027) over the outer-training rows; an XGBoost
with 6 parallel trees per round (boosted random forest), at most 5,000 rounds, early stopping (350 rounds, AUC) on the
held-out tenth (with its base margin). Refit: the same parameters on all outer-training rows for
ceil(best rounds x 10 / 9) rounds."""
import math

import numpy as np
from sklearn.model_selection import StratifiedKFold

SELECTOR_CAP = 5000
EARLY_STOPPING_ROUNDS = 350
PARALLEL_TREES = 6


def fit_selector_refit(X, y, base_margin, params, verbose=True):
    """Returns the refitted model (an xgboost.XGBClassifier); predict with model.predict_proba(X, base_margin=...)."""
    import xgboost as xgb
    y = np.asarray(y, np.int8)
    margin = np.asarray(base_margin, np.float64)
    fit_index, hold_index = next(StratifiedKFold(n_splits=10, shuffle=True, random_state=91027).split(
        np.zeros(len(y)), y))
    selector_params = dict(params)
    selector_params.update({"n_estimators": min(int(selector_params.get("n_estimators", SELECTOR_CAP)), SELECTOR_CAP),
                            "num_parallel_tree": PARALLEL_TREES, "early_stopping_rounds": EARLY_STOPPING_ROUNDS})
    selector = xgb.XGBClassifier(**selector_params)
    selector.fit(X.iloc[fit_index], y[fit_index], base_margin=margin[fit_index],
                 eval_set=[(X.iloc[hold_index], y[hold_index])], base_margin_eval_set=[margin[hold_index]],
                 verbose=False)
    best_round = int(selector.best_iteration) + 1
    final_params = dict(params)
    final_params.update({"n_estimators": max(1, int(math.ceil(best_round * (10.0 / 9.0)))),
                         "num_parallel_tree": PARALLEL_TREES})
    final_params.pop("early_stopping_rounds", None)
    del selector
    model = xgb.XGBClassifier(**final_params)
    model.fit(X, y, base_margin=margin, verbose=False)
    if verbose:
        print(f"selector best round {best_round}, refit rounds {final_params['n_estimators']}, "
              f"{X.shape[1]} features", flush=True)
    return model
