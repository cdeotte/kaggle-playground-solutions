#!/usr/bin/env python
"""Level-2 stacker: two multinomial logistic regressions (GPU, PyTorch) and their blend.

Pairs:
  models/<model>/preds/<pair>_oof.npy + <pair>_test.npy, and models/public_models/**/preds/ (any depth).
  OOF (n_train, 3) or (5, n_train, 3) for nested models; test (n_test, 3); class order Low, Medium, High.
Stack A uses the pairs whose name ends in `_a`; stack B uses all the others (names ending in `_b` and the public models).

Each stack: the inputs are the per-class logits log(p / (1 - p)) of every pair (p clipped to [1e-15, 1 - 1e-15], logits
clipped to +-30; nested OOFs are averaged over their 5 slices in logit space), 3 inputs per pair. The model is
nn.Linear(3 * n_pairs, 3) with softmax cross-entropy, trained full-batch with Adam (lr 0.01, 1000 epochs). As in an L2
logistic regression, the weights (not the bias) get weight decay 1 / (C * n_train) with C = 10, and the rows are weighted
with balanced class weights, because the metric is balanced accuracy. The stack runs StratifiedKFold(5) with 5 seeds
(42-46); OOF and test probabilities are averaged over the seeds (test = mean of the 25 fold models).

Final prediction: argmax(P_A + 0.25 * P_B).
Usage: python stack.py [--models-dir ../models] [--weight-b 0.25] [--C 10] [--epochs 1000] [--seeds 5] [--out submission.csv]"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.data import CLASSES, ID, TARGET, labels, read_test, read_train  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.zeros(1, device=DEVICE)  # initialize the device before the large arrays are loaded


def oof_files(models_dir):
    found = glob.glob(os.path.join(models_dir, '*', 'preds', '*_oof.npy'))
    found += glob.glob(os.path.join(models_dir, 'public_models', '**', 'preds', '*_oof.npy'), recursive=True)
    return sorted(set(found))


def to_logits(p):
    p = np.clip(np.asarray(p, np.float64), 1e-15, 1 - 1e-15)
    z = np.clip(np.log(p / (1 - p)), -30, 30).astype(np.float32)
    return z.mean(axis=0) if z.ndim == 3 else z


def find_pairs(models_dir, n_train, n_test):
    pairs = []
    for oof_path in oof_files(models_dir):
        test_path = oof_path[:-len('_oof.npy')] + '_test.npy'
        name = f'{os.path.relpath(Path(oof_path).parents[1], models_dir)}/{Path(oof_path).name[:-len("_oof.npy")]}'
        if not os.path.exists(test_path):
            print(f'skipping {name}: no test file', flush=True)
            continue
        o, t = to_logits(np.load(oof_path)), to_logits(np.load(test_path))
        assert o.shape == (n_train, 3) and t.shape == (n_test, 3), f'{name}: wrong shape {o.shape} / {t.shape}'
        pairs.append((name, o, t))
    return pairs


def fit_stack(X, X_test, y, C, epochs, seeds):
    X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
    oof, test = np.zeros((len(y), 3)), np.zeros((len(X_test), 3))
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        oof_seed = np.zeros((len(y), 3))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
            cw = compute_class_weight('balanced', classes=np.arange(3), y=y[tr])
            sw = torch.tensor(cw[y[tr]], dtype=torch.float32, device=DEVICE)
            tr_t, va_t = torch.tensor(tr, device=DEVICE), torch.tensor(va, device=DEVICE)
            X_tr, y_tr = X[tr_t], yt[tr_t]
            model = nn.Linear(X.shape[1], 3).to(DEVICE)
            opt = optim.Adam([{'params': model.weight, 'weight_decay': 1.0 / (C * len(tr))},
                              {'params': model.bias, 'weight_decay': 0.0}], lr=0.01)
            ce = nn.CrossEntropyLoss(reduction='none')
            model.train()
            for _ in range(epochs):
                opt.zero_grad()
                loss = (ce(model(X_tr), y_tr) * sw).mean()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                oof_seed[va] = torch.softmax(model(X[va_t]), 1).cpu().numpy()
                test += torch.softmax(model(X_test), 1).cpu().numpy() / (5 * len(seeds))
        oof += oof_seed / len(seeds)
        print(f'  seed {seed}: balanced accuracy {balanced_accuracy_score(y, oof_seed.argmax(1)):.5f}', flush=True)
    return oof, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', default=str(HERE.parent / 'models'))
    ap.add_argument('--weight-b', type=float, default=0.25)
    ap.add_argument('--C', type=float, default=10.0)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--out', default=str(HERE / 'submission.csv'))
    a = ap.parse_args()
    seeds = list(range(42, 42 + a.seeds))

    train, test = read_train(), read_test()
    y = labels(train[TARGET])
    pairs = find_pairs(a.models_dir, len(train), len(test))
    groups = {'A': [p for p in pairs if p[0].endswith('_a')], 'B': [p for p in pairs if not p[0].endswith('_a')]}
    probs = {}
    for g, members in groups.items():
        if not members:
            continue
        print(f'\nstack {g}: {len(members)} pairs')
        for name, o, _ in members:
            print(f'  {name:70s} balanced accuracy {balanced_accuracy_score(y, o.argmax(1)):.5f}')
        X = np.concatenate([o for _, o, _ in members], axis=1)
        X_test = np.concatenate([t for _, _, t in members], axis=1)
        probs[g] = fit_stack(X, X_test, y, a.C, a.epochs, seeds)
        np.save(HERE / f'oof_stack_{g.lower()}.npy', probs[g][0].astype(np.float32))
        np.save(HERE / f'test_stack_{g.lower()}.npy', probs[g][1].astype(np.float32))
        print(f'stack {g}: CV balanced accuracy {balanced_accuracy_score(y, probs[g][0].argmax(1)):.5f}')
    w = {'A': 1.0, 'B': a.weight_b if 'A' in probs else 1.0}
    oof = sum(w[g] * probs[g][0] for g in probs)
    pred = sum(w[g] * probs[g][1] for g in probs)
    print(f'\nblend {" + ".join(f"{w[g]:g} x {g}" for g in probs)}: CV balanced accuracy '
          f'{balanced_accuracy_score(y, oof.argmax(1)):.5f}')
    pd.DataFrame({ID: test[ID], TARGET: np.array(CLASSES)[pred.argmax(1)]}).to_csv(a.out, index=False)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
