#!/usr/bin/env python
"""Turn the chain runs of train.py (parts/chain_<config>_s<seed>_f<k>_{valid,test}.npy, raw LLRs) into the pairs.
Each pair: OOF = sigmoid(raw LLR of its OOF run); test = sigmoid(mean over the 5 folds and its test runs of the raw test LLR)
(raw clipped to +-30).
  gpt2_chain_e55      OOF chain_e55_s42, test chain_e55_s42
  gpt2_chain_d60_s42  OOF chain_d60_s42, test chain_d60_s42 ... chain_d60_s46 (25 fold models)
  gpt2_chain_d60_s43  OOF chain_d60_s43, test chain_d60_s43
Usage: python assemble.py"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.gpt_rows import fold_parts  # noqa: E402
from common.pairs import have_part, save_pair  # noqa: E402

PAIRS = {
    'gpt2_chain_e55': ('chain_e55_s42', ['chain_e55_s42']),
    'gpt2_chain_d60_s42': ('chain_d60_s42', [f'chain_d60_s{s}' for s in range(42, 47)]),
    'gpt2_chain_d60_s43': ('chain_d60_s43', ['chain_d60_s43']),
}


def main():
    y = target()
    f = folds(y)
    sig = lambda z: expit(np.clip(z, -30, 30))  # noqa: E731
    for name, (oof_run, test_runs) in PAIRS.items():
        missing = [r for r in {oof_run, *test_runs} if not all(have_part(HERE, r, k) for k in range(5))]
        if missing:
            print(f'{name}: waiting for {sorted(missing)}', flush=True)
            continue
        oof, test = np.zeros(len(y)), []
        for k in range(5):
            oof[f == k] = fold_parts(HERE, oof_run, k)[0]
            test.append(np.mean([fold_parts(HERE, r, k)[1] for r in test_runs], 0))
        save_pair(HERE, name, sig(oof), sig(np.mean(test, 0)))
        print(f'{name}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)


if __name__ == '__main__':
    main()
