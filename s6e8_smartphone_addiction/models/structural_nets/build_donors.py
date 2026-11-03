#!/usr/bin/env python
"""Target-free original-row donors for every train.csv and test.csv row (GPU, torch) -> cache/donors/{train,test}/.

Six semantic field blocks (lib/donors.py BLOCKS): screen core (daily, social, gaming, work/study hours), screen + weekend
(daily, social, weekend), engagement (daily, weekend, notifications, app opens), behavior (social, gaming, notifications,
app opens), lifestyle (age, daily, work/study, sleep) and profile (age, sleep + gender, stress, academic impact). For each
block the 16 original rows with the lowest score are kept: mean robust-scaled squared distance over the row's observed
block fields (categorical mismatch 0.75) minus 0.75 x the exact-match fraction. Labels are not read.
Files per split: indices.npy (uint16 [rows, 6, 16]), score.npy and exact_fraction.npy (float16 [rows, 6, 16]),
coverage.npy (observed fields per block) and pattern.npy (block missingness bits), both uint8 [rows, 6].
Usage: python build_donors.py [--split train|test|all] [--chunk-size 4096]"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import TARGET, read_original, read_test, read_train  # noqa: E402
from lib.donors import build_donor_arrays, donor_dir, fit_retrieval_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='all', choices=['train', 'test', 'all'])
    ap.add_argument('--chunk-size', type=int, default=4096)
    a = ap.parse_args()
    state = fit_retrieval_state(read_original())
    for split in (['train', 'test'] if a.split == 'all' else [a.split]):
        frame = (read_train() if split == 'train' else read_test()).drop(columns=[TARGET], errors='ignore')
        print(f'{split}: {len(frame)} rows', flush=True)
        arrays = build_donor_arrays(frame, state, device='cuda', chunk_size=a.chunk_size)
        out = donor_dir(split)
        out.mkdir(parents=True, exist_ok=True)
        for name, value in arrays.items():
            np.save(out / f'{name}.npy', value)
        print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
