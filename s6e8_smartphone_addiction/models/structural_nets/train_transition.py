#!/usr/bin/env python
"""source_transition_likelihood: class-conditional original-to-synthetic transition likelihood (lib/transition.py).

Per outer fold the base histograms and the per-relation transition tables are fitted on the outer-train rows only;
the held-out fold and the test rows are scored with them. Variant k4_w050 (the pair): the 4 nearest donors per
relation, retrieval temperature 1.25, Dirichlet pseudo-count 48, donor-identity event on, final logit = base + 0.5 x
transition. Other variants (--variant): control (base histograms only), k1 / k4 / k8 donors with transition weight
0.10 / 0.25 / 0.50 (k1: temperature 1.5, pseudo-count 24, no identity event; k8: 0.75, 96, identity event).
Test = mean of the 5 fold predictions. Needs cache/donors (build_donors.py). CPU only.
Usage: python train_transition.py [--folds 0,1,2,3,4] [--variant k4_w050]"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib.donors import load_donors  # noqa: E402
from lib.transition import VARIANTS, build_source_schema, fit_likelihood_state, predict_likelihood  # noqa: E402


def pair_name(variant):
    return 'source_transition_likelihood' if variant == 'k4_w050' else f'source_transition_{variant}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--variant', default='k4_w050', choices=list(VARIANTS))
    a = ap.parse_args()
    name = pair_name(a.variant)
    train, test = read_train(), read_test()
    y = labels(train[TARGET])
    f = folds(y)
    schema = build_source_schema(read_original(keep_level=True))
    train_view, test_view = load_donors('train'), load_donors('test')
    for k in fold_list(a.folds):
        if have_part(HERE, name, k):
            print(f'fold {k}: done', flush=True)
            continue
        tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
        state = fit_likelihood_state(train, y, train_view, tr, schema, a.variant)
        pv, base_v, transition_v = predict_likelihood(train, train_view, va, schema, state)
        pt, _, _ = predict_likelihood(test, test_view, np.arange(len(test), dtype=np.int64), schema, state)
        save_part(HERE, name, k, pv, pt)
        msg = f'fold {k}: AUC {roc_auc_score(y[va], pv):.6f} (base {roc_auc_score(y[va], base_v):.6f}'
        if state.transition is not None:
            msg += f', transition {roc_auc_score(y[va], transition_v):.6f}'
        print(msg + ')', flush=True)
    assemble(HERE, name, f, y)


if __name__ == '__main__':
    main()
