#!/usr/bin/env python
"""Train the label-free masked-token density model (d 192, 6 layers, 8 heads, 30 epochs; lib/density_model.py) on the
train + test predictors and cache its weights and row channels in cache/ (density_d192L6h8e30.pt,
density_feats_d192L6h8e30.npz) for train_afn.py.
Usage: python build_density.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import read_test, read_train  # noqa: E402
from lib import density_model as DM  # noqa: E402


def main():
    train, test = read_train(), read_test()
    model, _, _, _ = DM.train_density(train, test, cache_tag=DM.TAG, device='cuda', **DM.CONFIG)
    DM.density_features(train, test, model=model, device='cuda', cache_tag=DM.TAG)


if __name__ == '__main__':
    main()
