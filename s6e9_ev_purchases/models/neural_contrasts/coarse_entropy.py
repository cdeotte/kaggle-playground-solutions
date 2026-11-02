#!/usr/bin/env python
"""Coarse income-parent entropy bank (coarse_context_contrast; also part of the masked_density_contrast baseline).

For each of the 13 contexts of density_factors.py: the entropy of the context distribution inside the row's income parent
(income // 1000 + leading BPE piece; 0.5 pseudo-counts), the same entropy times r = 1 / sqrt(count of the exact income value
+ 20), and r itself: 27 label-free columns over all train + test rows.
Writes cache/stage5/coarse_entropy_all.npy (+ .json names).
Usage: python coarse_entropy.py"""
import numpy as np

from contrast_data import STAGE, write_names
from density_factors import data


def main():
    key, par, pv, ng, contexts, perm = data()
    pc = np.bincount(par)
    n = np.bincount(key, minlength=ng)
    hs = []
    for name, c in contexts.items():
        ns = int(c.max() + 1)
        counts = np.bincount(par * ns + c, minlength=len(pc) * ns).reshape(len(pc), ns)
        p = (counts + .5 / ns) / (pc[:, None] + .5)
        hs.append(-(p * np.log(p)).sum(axis=1)[par])
    H = np.column_stack(hs)
    r = 1 / np.sqrt(n[key] + 20)
    A = np.column_stack([H, H * r[:, None], r]).astype('float32')
    names = [f'parent_entropy_{c}' for c in contexts] + [f'parent_entropy_count_{c}' for c in contexts] + ['value_count_uncertainty']
    STAGE[5].mkdir(parents=True, exist_ok=True)
    np.save(STAGE[5] / 'coarse_entropy_all.npy', A)
    write_names(STAGE[5] / 'coarse_entropy_all.npy', names)
    print(f'coarse entropy bank {A.shape}', flush=True)


if __name__ == '__main__':
    main()
