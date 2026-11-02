#!/usr/bin/env python
"""XGBoost on the fold matrices of xgb_inputs.py, started from the nested LR margin; one model per outer fold and seed.

  --model mixture       145 columns (stage-1 features + exact-income EM), seeds 42-49  -> xgb_income_mixture
  --model shape         + 9 income context-shape columns, seeds 42-49                  -> xgb_income_context_shape
  --model digits_diffs  259 columns (+ ../neural_contrasts banks, digits, income TE contrasts), seeds 42-57
                                                                                     -> xgb_full_context_contrasts
Per fold: sigmoid(mean over seeds of logit(clip(p, 1e-7))); OOF = fold rows, test = mean of the 5 fold probabilities.
Each run is kept in runs/<stage>/<tag>_s{seed}_f{k}/predictions.npz (index = fold-k rows then test rows, prediction,
margin); the digits_diffs runs are also the teacher of ../distillation.
Usage: python train_xgb.py --model mixture [--folds 0,1,2,3,4] [--seeds 42,43]"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.pairs import fold_list, save_pair  # noqa: E402
from layout import RUNS, STAGE1  # noqa: E402
from xgb_inputs import inputs  # noqa: E402

PARAMS = dict(objective='binary:logistic', tree_method='hist', device='cuda', eta=.03, max_depth=7, min_child_weight=10,
              subsample=.85, colsample_bytree=.85, colsample_bynode=.8, reg_lambda=5, reg_alpha=.1, max_bin=512,
              num_parallel_tree=4)
ROUNDS = 400
# model -> (runs subfolder, run tag, seeds, pair)
SPECS = {'mixture': ('stage1', 'matched_semi1', list(range(42, 50)), 'xgb_income_mixture'),
         'shape': ('stage2', 'shape', list(range(42, 50)), 'xgb_income_context_shape'),
         'digits_diffs': ('stage8', 'digits_diffs', list(range(42, 58)), 'xgb_full_context_contrasts')}


def run_dir(model, k, seed):
    stage, tag, _, _ = SPECS[model]
    return RUNS/stage/f'{tag}_s{seed}_f{k}'


def fit_fold(model, k, seeds):
    import xgboost as xgb
    todo = [s for s in seeds if not (run_dir(model, k, s)/'predictions.npz').exists()]
    if not todo:
        return
    X, Q, names, types, fw, fit, query, nv, mf, mq = inputs(k, model)
    y = np.load(STAGE1/'y.npy')
    dt = xgb.DMatrix(X, label=y[fit], base_margin=mf, feature_names=names, feature_types=types, feature_weights=fw,
                     enable_categorical=True)
    dq = xgb.DMatrix(Q, base_margin=mq, feature_names=names, feature_types=types, enable_categorical=True)
    for seed in todo:
        print(f'{model} fold {k} seed {seed}: {X.shape}', flush=True)
        booster = xgb.train({**PARAMS, 'seed': seed}, dt, num_boost_round=ROUNDS, verbose_eval=False)
        pred = booster.predict(dq)
        margin = booster.predict(dq, output_margin=True)
        assert np.isfinite(pred).all() and ((pred > 0) & (pred < 1)).all()
        run = run_dir(model, k, seed)
        run.mkdir(parents=True, exist_ok=True)
        np.savez(run/'predictions.npz', index=query, prediction=pred, margin=margin)


def assemble(model):
    _, _, seeds, pair = SPECS[model]
    if not all((run_dir(model, k, s)/'predictions.npz').exists() for k in range(5) for s in seeds):
        print(f'{model}: not all folds / seeds are done yet', flush=True)
        return
    from sklearn.metrics import roc_auc_score
    y = np.load(STAGE1/'y.npy')
    folds = np.load(STAGE1/'fold_id.npy')
    oof, tests = np.full(len(y), np.nan), []
    for k in range(5):
        valid = np.flatnonzero(folds == k)
        zz = []
        for s in seeds:
            d = np.load(run_dir(model, k, s)/'predictions.npz')
            assert np.array_equal(d['index'][:len(valid)], valid)
            zz.append(logit(d['prediction'].astype('float64').clip(1e-7, 1-1e-7)))
        q = expit(np.mean(zz, axis=0))
        oof[valid] = q[:len(valid)]
        tests.append(q[len(valid):])
    assert np.isfinite(oof).all()
    save_pair(HERE, pair, oof, np.mean(tests, axis=0))
    print(f'{pair}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(SPECS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--seeds', default='')
    a = ap.parse_args()
    seeds = fold_list(a.seeds) if a.seeds else SPECS[a.model][2]
    for k in fold_list(a.folds):
        fit_fold(a.model, k, seeds)
    assemble(a.model)


if __name__ == '__main__':
    main()
