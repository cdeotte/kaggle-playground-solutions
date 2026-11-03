#!/usr/bin/env python
"""GPU CatBoost with native categoricals on the snap / decimal-string combo frame
(pair catboost_snap_combo_native_cats).

Inputs: the raw data plus the snap / decimal-string columns of models/snap_features (combo.csv: decimal strings of the
charges and their first / second decimal digits, the charges snapped to the nearest original value and the snap
differences). lib/snap_combo_frames.py builds feature blocks over train + test; the model uses 498 of them (315
CatBoost-native categoricals):
  - categorical-like columns: the 16 raw categoricals; tenure bins and 14 quantile bins of the charges and snapped
    charges; 16 pair and 4 triple keys of categoricals, bins and decimal digits; 28 bigrams and 4 trigrams of the main
    categoricals; integer / decimal digit strings of the charges
  - frequency / percentile rank of the numerics and snapped charges; TotalCharges - tenure x MonthlyCharges and its
    absolute value, the MonthlyCharges / TotalCharges ratios, TotalCharges / (tenure + 1) and MonthlyCharges minus
    it; service Yes / No / other counts and their balance
  - counts and rare flags (<= 20) of the categorical-like columns; means of the numerics per Contract,
    InternetService, PaymentMethod and tenure bin and the deviations from them
  - original churn rate and count of the 16 categoricals, distances of the numerics to the original churner /
    non-churner means; train-vs-test frequency ratio of the categorical-like columns
  - distances to the original churner / non-churner centroids of the standardized numerics and a logistic score of
    their gap
CatBoost (GPU): depth 8, lr 0.02, l2 6, min_data_in_leaf 20, Bernoulli subsample 0.85, random_strength 0.6,
max_ctr_complexity 4, up to 50,000 iterations with early stopping (600) on the validation fold (use_best_model=True),
seed 42 + fold number (1-5). Test = mean of the 5 fold models.

Requires models/snap_features to have run first.
Usage: python train_snap_combo.py"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.pairs import save_pair  # noqa: E402
from models.catboost.lib.cv import run_folds  # noqa: E402
from models.catboost.lib.snap_combo_frames import (  # noqa: E402
    BASE_CATS, SEED, TARGET, add_orig_aux_model_features, build_feature_frames, native_combo_columns, read_data,
    seed_everything,
)

PAIR = 'catboost_snap_combo_native_cats'


def cb_params(seed):
    return dict(
        loss_function='Logloss',
        eval_metric='AUC',
        iterations=50000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=6.0,
        min_data_in_leaf=20,
        random_seed=seed,
        task_type='GPU',
        devices='0',
        bootstrap_type='Bernoulli',
        subsample=0.85,
        random_strength=0.6,
        grow_policy='SymmetricTree',
        max_ctr_complexity=4,
        verbose=250,
    )


def main():
    seed_everything(SEED)
    train, test, orig, y, y_orig = read_data()
    tr, te = build_feature_frames(train, test, orig, y_orig)
    aux_tr, aux_te = add_orig_aux_model_features(tr, te, orig.drop(columns=[TARGET], errors='ignore'), y_orig)
    tr = pd.concat([tr, aux_tr], axis=1)
    te = pd.concat([te, aux_te], axis=1)

    use_cols = native_combo_columns(tr)
    X = tr[use_cols].copy()
    Xtest = te[use_cols].copy()
    cat_cols = []
    for c in use_cols:
        dt = str(X[c].dtype)
        if c in BASE_CATS or c.endswith('_bin') or '__' in c or c.startswith('NGRAM_') or dt in ('object', 'string',
                                                                                                'category'):
            cat_cols.append(c)
    cat_cols = list(dict.fromkeys(cat_cols))
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    for df in [X, Xtest]:
        for c in cat_cols:
            df[c] = df[c].astype('string').fillna('NA')
        for c in [col for col in use_cols if col not in cat_cols]:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('float32')
    print(f'features: {len(use_cols)} ({len(cat_cols)} categorical)', flush=True)

    def fit_fold(fold, itr, iva, seed):
        Xtr = X.iloc[itr].copy()
        Xva = X.iloc[iva].copy()
        model = CatBoostClassifier(**cb_params(seed + fold))
        model.fit(Xtr, y[itr], cat_features=cat_idx, eval_set=(Xva, y[iva]), use_best_model=True,
                  early_stopping_rounds=600, verbose=250)
        va_pred = model.predict_proba(Xva)[:, 1].astype(np.float32)
        te_pred = model.predict_proba(Xtest)[:, 1].astype(np.float32)
        print(f'fold {fold}: best iteration {model.get_best_iteration()}', flush=True)
        del Xtr, Xva, model
        gc.collect()
        return va_pred, te_pred

    oof, test_pred = run_folds(fit_fold, y, len(Xtest), name=PAIR)
    print(f'{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
