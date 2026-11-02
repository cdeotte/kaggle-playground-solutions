#!/usr/bin/env python
"""Sample labelled rows from a GReaT replica checkpoint (GPT-2 small trained on the original data only), for the footprint
pool. GReaT sampling with the default label-first start ("Will_Buy_EV is Yes/No, ..."), temperature 0.7, max_length 256.
The generation batch size k does not change the distribution (both prompts have the same length, rows are independent).
Needs be_great 0.0.3 and the environment of ../gpt2_replica. Reads ../gpt2_replica/checkpoints/<name>.pt.
Writes features/replica_samples/<name>_s<seed>_n<rows>.csv.gz.
Usage: python sample_replica.py --checkpoint e60 --rows 50000 --seed 426500 [--temperature 0.7] [--k 500]"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

np.float = float  # be_great 0.0.3 uses the removed np.float alias

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, read_original  # noqa: E402

CHECKPOINTS = HERE.parent / 'gpt2_replica' / 'checkpoints'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--rows', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--k', type=int, default=500)
    ap.add_argument('--max-length', type=int, default=256)
    a = ap.parse_args()
    from be_great import GReaT
    out = HERE / 'features' / 'replica_samples' / f'{a.checkpoint}_s{a.seed}_n{a.rows}.csv.gz'
    out.parent.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    source = read_original()
    model = GReaT('gpt2')
    model._update_column_information(source)
    model._update_conditional_information(source, TARGET)
    model.model.load_state_dict(torch.load(CHECKPOINTS / f'{a.checkpoint}.pt', map_location='cpu'))
    rows = model.sample(n_samples=a.rows, k=a.k, temperature=a.temperature, max_length=a.max_length, device='cuda')
    assert len(rows) == a.rows, len(rows)
    rows.to_csv(out, index=False, compression='gzip')
    print(f'wrote {out} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
