#!/usr/bin/env python
"""GReaT replicas of the data generator: GPT-2 small fine-tuned with be-great 0.0.3 on the 10,000 original rows only (Buyer_ID
dropped, Will_Buy_EV kept as the last column with its Yes / No values).

  e55, e60, e65, e70: legacy recipe, one run per replica with an E-epoch schedule (the learning rate decays linearly from 5e-5 to 0
                      at epoch E); the replica is the model at the end of the run. Trainer seed 42 and data_seed 42, 2 dataloader
                      workers, TF32 on.
  d60:                organizer-default Trainer RNG path: only the GReaT arguments below are passed (Trainer defaults seed 42,
                      data_seed None, 0 dataloader workers), TF32 off, 60 epochs.
Both: batch 16, a fresh random field order every time a row is read (be-great's GReaTDataset), and the transformers 4.26.1 Trainer
defaults (AdamW, no warmup, weight decay 0, grad clip 1, fp32). train.csv and test.csv are never read.
Writes checkpoints/<replica>.pt (the GPT2LMHeadModel state dict); the classifiers load it through common.gpt_rows.load_replica.
Usage: python train_replica.py --replica e55|e60|e65|e70|d60"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

np.float = float  # be-great 0.0.3 uses the removed numpy alias

import torch  # noqa: E402
from be_great import GReaT  # noqa: E402
from be_great.great_dataset import GReaTDataset  # noqa: E402
from transformers import set_seed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, read_original  # noqa: E402

EPOCHS = {'e55': 55, 'e60': 60, 'e65': 65, 'e70': 70, 'd60': 60}


def batch_getitems(self, keys):
    """torch 2 DataLoaders fetch a list of keys at once; read them one by one through be-great's own _getitem (same rows, same
    random.shuffle calls as the per-row path)."""
    return [self._getitem(key) for key in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replica', required=True, choices=list(EPOCHS))
    a = ap.parse_args()
    org_default = a.replica == 'd60'
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    set_seed(42)
    GReaTDataset.__getitems__ = batch_getitems
    torch.backends.cuda.matmul.allow_tf32 = not org_default
    torch.backends.cudnn.allow_tf32 = not org_default

    data = read_original()
    assert data.columns[-1] == TARGET
    kwargs = dict(logging_strategy='epoch', save_strategy='no', disable_tqdm=True, report_to=[])
    if not org_default:
        kwargs.update(dataloader_num_workers=2, seed=42, data_seed=42)
    model = GReaT(llm='gpt2', experiment_dir=str(HERE / 'runs' / a.replica), epochs=EPOCHS[a.replica], batch_size=16, **kwargs)
    model.fit(data, conditional_col=TARGET)
    out = HERE / 'checkpoints' / f'{a.replica}.pt'
    out.parent.mkdir(exist_ok=True)
    torch.save(model.model.state_dict(), out)
    print(f'saved {out}', flush=True)


if __name__ == '__main__':
    main()
