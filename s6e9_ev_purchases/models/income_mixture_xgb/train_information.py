#!/usr/bin/env python
"""XGBoost on the information-aware income EM features (cache/em_info, see em_pipeline.py), seeds 42-49.

167 columns: the compact base view (static blocks without the source channels, log counts, target encodings of the raw
fields, income 100 / 1000 bins and the leading BPE piece, with train-side noise sd 0.04), the source rule and 12
inner-held source-conditional intercepts (compact.py), the 4 exact-income EM columns (weight 0.5) and the 14
information columns (weight 0.5). Base margin = 0.5 x the nested response margin of ../response_features + 0.5 x the
EM Bayes posterior margin. Per fold: sigmoid(mean over seeds of clipped logits); test = mean of the 5 fold probabilities.
Runs: runs/em_info/production_information/fold_{k}/seed_{s}/{valid,test}.npy.
Usage: python train_information.py [--folds 0,1,2,3,4]   -> preds/xgb_information_mixture_{oof,test}.npy"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.pairs import fold_list, save_pair  # noqa: E402
import bayes_margin  # noqa: E402
import compact  # noqa: E402
from layout import RUNS  # noqa: E402
from lib import BASE_DIR, FEATURES, INFORMATION, Data, train_noise  # noqa: E402

PAIR = 'xgb_information_mixture'
SEEDS = list(range(42, 50))
ROUNDS = 400
PARAMS = {'objective': 'binary:logistic', 'tree_method': 'hist', 'device': 'cuda', 'max_depth': 7, 'eta': .03,
          'num_parallel_tree': 4, 'min_child_weight': 10, 'lambda': 5, 'alpha': .1, 'subsample': .85,
          'colsample_bytree': .85, 'colsample_bynode': .8, 'max_bin': 512}
OUT = RUNS/'em_info'/'production_information'


def z(p):
    p = np.clip(np.asarray(p, np.float64), 1e-15, 1-1e-15)
    return np.clip(np.log(p)-np.log1p(-p), -30, 30)


def fold_inputs(d, fold):
    root = BASE_DIR/f'fold_{fold}'
    names = list(json.loads((root/'manifest.json').read_text())['names'])
    a = np.load(root/'train_indices.npy')
    X = train_noise(np.load(root/'train.npy'), names, fold, sd=.04)
    Q = np.load(root/'query.npy')
    ix = compact.select(names)
    X, Q, names = X[:, ix], Q[:, ix], [names[j] for j in ix]
    A, B, newnames = compact.build(d, fold, True)
    A = train_noise(A, newnames, fold, sd=.04)
    X, Q = np.column_stack([X, A]), np.column_stack([Q, B])
    names += newnames
    weights = np.ones(len(names), np.float32)
    fr = FEATURES/f'fold_{fold}'
    fn = json.loads((fr/'manifest.json').read_text())['names']
    ix = [j for j, n in enumerate(fn) if n.startswith('inc_global__')]
    X = np.column_stack([X, np.load(fr/'train.npy')[:, ix]])
    Q = np.column_stack([Q, np.load(fr/'query.npy')[:, ix]])
    names += [fn[j] for j in ix]
    weights = np.r_[weights, np.full(len(ix), .5, np.float32)]
    fr = INFORMATION/f'fold_{fold}'
    fn = json.loads((fr/'manifest.json').read_text())['names']
    X = np.column_stack([X, np.load(fr/'train.npy')])
    Q = np.column_stack([Q, np.load(fr/'query.npy')])
    names += fn
    weights = np.r_[weights, np.full(len(fn), .5, np.float32)]
    mt, mv, mtest = d.margin(fold)
    mq = np.r_[mv, mtest]
    bt, bq = bayes_margin.build(d, fold)
    mt = .5*mt+.5*bt[:, 0]
    mq = .5*mq+.5*bq[:, 0]
    return a, X, Q, names, weights, mt, mq


def train_fold(d, fold):
    import xgboost as xgb
    todo = [s for s in SEEDS if not (OUT/f'fold_{fold}'/f'seed_{s}'/'test.npy').exists()]
    if not todo:
        return
    a, X, Q, names, weights, mt, mq = fold_inputs(d, fold)
    nv = int((d.folds == fold).sum())
    dt = xgb.DMatrix(X, label=d.y[a], base_margin=mt, feature_weights=weights)
    dq = xgb.DMatrix(Q, base_margin=mq)
    for seed in todo:
        print(f'fold {fold} seed {seed}: {X.shape}', flush=True)
        model = xgb.train({**PARAMS, 'seed': seed}, dt, num_boost_round=ROUNDS)
        p = model.predict(dq).astype(np.float64)
        out = OUT/f'fold_{fold}'/f'seed_{seed}'
        out.mkdir(parents=True, exist_ok=True)
        np.save(out/'valid.npy', p[:nv])
        np.save(out/'test.npy', p[nv:])


def assemble(d):
    if not all((OUT/f'fold_{f}'/f'seed_{s}'/'test.npy').exists() for f in range(5) for s in SEEDS):
        print('not all folds / seeds are done yet', flush=True)
        return
    from sklearn.metrics import roc_auc_score
    oof, tests = np.full(d.n, np.nan), []
    for f in range(5):
        b = np.flatnonzero(d.folds == f)
        zv = [z(np.load(OUT/f'fold_{f}'/f'seed_{s}'/'valid.npy')) for s in SEEDS]
        zt = [z(np.load(OUT/f'fold_{f}'/f'seed_{s}'/'test.npy')) for s in SEEDS]
        oof[b] = expit(np.mean(zv, axis=0))
        tests.append(expit(np.mean(zt, axis=0)))
    assert np.isfinite(oof).all()
    save_pair(HERE, PAIR, oof, np.mean(tests, axis=0))
    print(f'{PAIR}: OOF AUC {roc_auc_score(d.y, oof):.6f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    d = Data()
    for fold in fold_list(a.folds):
        train_fold(d, fold)
    assemble(d)


if __name__ == '__main__':
    main()
