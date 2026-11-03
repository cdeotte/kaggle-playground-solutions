#!/usr/bin/env python
"""Step 2a: one masked lattice density transformer per outer fold (lib/lattice.py), fit on the predictors of the
outer-training rows only (no target, no validation / test rows): width 384, 4 layers, 8 heads, ff x3, dropout 0.08,
AdamW lr 0.001 / wd 2e-5, batch 2048, 10 epochs, Bernoulli-0.15 masks, bf16 autocast, seed 60142 + fold.
Writes cache/fold_density/fold_<k>/model_state.pt.
Usage: python build_fold_density.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import fold_list  # noqa: E402
from lib.lattice import FOLD_CONFIG, FOLD_SEED, LatticeEncoder, train_model  # noqa: E402
from lib.paths import fold_indices, own_dir, read_train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    import torch
    train = read_train()
    for fold in fold_list(args.folds):
        fold_train, _ = fold_indices(fold)
        fit = train.iloc[fold_train]
        encoder = LatticeEncoder(fit)
        tokens, scalars = encoder.transform(fit)
        print(f'fold {fold}: density state on {len(fit)} outer-training rows', flush=True)
        model = train_model(torch, torch.from_numpy(tokens).cuda(), torch.from_numpy(scalars).cuda(),
                            encoder.cardinalities, FOLD_CONFIG, seed=FOLD_SEED + fold)
        torch.save(model.state_dict(), own_dir('fold_density', f'fold_{fold}') / 'model_state.pt')
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
