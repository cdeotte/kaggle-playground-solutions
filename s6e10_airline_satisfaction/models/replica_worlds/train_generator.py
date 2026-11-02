#!/usr/bin/env python
"""Generator of a replica world: GReaT GPT-2 medium fine-tuned on a NOISED copy of the original data (data.csv).

The trainer of ../gpt2_replica/train_replica.py (be-great 0.0.3 + transformers 4.26.1, be-great's default Trainer path with
seed 42, a random field order per row, AdamW (torch implementation), TF32) with three changes:
  1. the training copy of data.csv is noised first (noisy_original: label flips, then +-1 jitter of the 13 ratings);
  2. batch 8 and a 40-epoch schedule (learning rate 5e-5, linear decay to 0 at epoch 40);
  3. the weights at the end of epoch 32 are saved and training stops there (the rest of the schedule cannot change them).
train.csv and test.csv are never read.
Writes checkpoints/<world>/epoch_032.pt (GPT2LMHeadModel state dict).
Usage: python train_generator.py --world M32_1M|MB32_1M"""
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
from be_great.great_trainer import GReaTTrainer  # noqa: E402
from transformers import TrainerCallback, set_seed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, labels, read_original  # noqa: E402
from recipe import BATCH, EPOCHS, LLM, MILESTONE, RATING_JITTER, WORLDS  # noqa: E402


def noisy_original(world):
    """data.csv as the world's generator sees it in training.
    Labels: one uniform draw u per row (seed 1234); a False label flips if u < p_neg, a True label if u < p_pos.
    Ratings: seed 4321, one uniform draw per rating column in data.csv order; -1 if u < 0.05, +1 if u > 0.95, clipped to
    the column's min/max."""
    p_neg, p_pos = WORLDS[world]['label_flip']
    data = read_original()
    y = labels(data[TARGET]).astype(bool)
    u = np.random.default_rng(1234).random(len(data))
    flip = np.where(y, u < p_pos, u < p_neg)
    data[TARGET] = np.where(flip, ~y, y)
    rng = np.random.default_rng(4321)
    ratings = [c for c in data.columns if c != TARGET and data[c].dtype.kind in 'iu' and data[c].nunique() <= 6]
    assert len(ratings) == 13, ratings
    for c in ratings:
        u = rng.random(len(data))
        step = np.where(u < RATING_JITTER, -1, np.where(u > 1 - RATING_JITTER, 1, 0))
        data[c] = np.clip(data[c] + step, data[c].min(), data[c].max())
    print(f'{world}: {int(flip.sum())} labels flipped; training prior {data[TARGET].mean():.6f} '
          f'(prompt prior {WORLDS[world]["prior"]})', flush=True)
    return data[[c for c in data.columns if c != TARGET] + [TARGET]]  # label last, as in data.csv


def batch_getitems(self, keys):
    """Tokenize a batch at once, with the same row order and random.shuffle calls as be-great's per-row path (same tokens)."""
    texts = []
    for key in keys:
        row = self._data.fast_slice(key, 1)
        idx = list(range(row.num_columns))
        random.shuffle(idx)
        texts.append(', '.join('%s is %s' % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip()) for i in idx))
    enc = self.tokenizer(texts, padding=False)
    return [{field: values[p] for field, values in enc.items()} for p in range(len(keys))]


class SaveAndStop(TrainerCallback):
    """At the end of epoch MILESTONE: save the model weights and stop training."""

    def __init__(self, path):
        self.path = path

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if int(round(state.epoch or 0)) == MILESTONE:
            torch.save(model.state_dict(), self.path)
            print(f'saved {self.path}', flush=True)
            control.should_training_stop = True
        return control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', required=True, choices=list(WORLDS))
    a = ap.parse_args()
    out = HERE / 'checkpoints' / a.world
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    set_seed(42)
    GReaTDataset.__getitems__ = batch_getitems
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data = noisy_original(a.world)
    callback = SaveAndStop(out / f'epoch_{MILESTONE:03d}.pt')
    train = GReaTTrainer.train

    def train_with_callback(self, *args, **kwargs):  # be-great creates its Trainer inside fit()
        self.add_callback(callback)
        return train(self, *args, **kwargs)

    GReaTTrainer.train = train_with_callback
    model = GReaT(llm=LLM, experiment_dir=str(out / 'trainer'), epochs=EPOCHS, batch_size=BATCH, logging_strategy='epoch',
                  save_strategy='no', disable_tqdm=True, report_to=[], optim='adamw_torch')
    model.fit(data, conditional_col=TARGET)


if __name__ == '__main__':
    main()
