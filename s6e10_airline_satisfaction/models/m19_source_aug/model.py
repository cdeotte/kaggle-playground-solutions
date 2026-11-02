"""M19 field transformer: token codes, encoder and classifier (used by pretrain.py and train.py).

A row is 21 tokens, one per feature. Each column is factorized (sorted) over all rows, so every distinct value of a column is
its own token; numbers are categorical identities, NaN and unknown categories become -999 (one more value). Token 0 = CLS,
1 = MASK, field tokens start at 2. Rows, in this order: competition train, test, original data.csv, then the replica worlds
M32_1M train / test and MB32_1M train / test (3,127,133 rows and 4,957 tokens with our files)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import read_original, read_test, read_train  # noqa: E402

WORLDS_DIR = HERE.parent / 'replica_worlds' / 'worlds'
WORLDS = ('M32_1M', 'MB32_1M')
CACHE, CKPT = HERE / 'cache', HERE / 'checkpoints'
FEATS = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
         'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service',
         'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness', 'Departure Delay in Minutes',
         'Arrival Delay in Minutes', 'Gender', 'Customer Type', 'Type of Travel', 'Class']      # train.csv order
CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
CATMAP = {'Female': 0, 'Male': 1, 'Loyal Customer': 0, 'disloyal Customer': 1, 'Business travel': 0, 'Personal Travel': 1,
          'Business': 0, 'Eco': 1, 'Eco Plus': 2}
NF, MASK = len(FEATS), 1
DIM, LAYERS = 384, 8


def load_codes():
    """Token ids [rows, 21] (int32) and the row count of each of the 7 parts; built once, cached in cache/codes.npz.
    The vocabulary size is codes.max() + 1."""
    path = CACHE / 'codes.npz'
    if not path.exists():
        frames = [read_train()[FEATS], read_test()[FEATS], read_original()[FEATS]]
        for w in WORLDS:
            frames += [pd.read_csv(WORLDS_DIR / w / f'{s}.csv', usecols=FEATS)[FEATS] for s in ('train', 'test')]
        allv = pd.concat(frames, ignore_index=True)
        for c in CATS:
            allv[c] = allv[c].map(CATMAP)
        codes = np.empty((len(allv), NF), np.int32)
        off = 2
        for j, c in enumerate(FEATS):
            code, uniq = pd.factorize(allv[c].fillna(-999), sort=True)
            codes[:, j] = code + off
            off += len(uniq)
        CACHE.mkdir(exist_ok=True)
        np.savez(path, codes=codes, sizes=np.array([len(d) for d in frames]))
    z = np.load(path)
    return z['codes'], z['sizes']


class Enc(nn.Module):
    """CLS + 21 field tokens + learned field embeddings -> pre-norm transformer (L8, d384, FFN 1536, GELU, dropout 0.1)
    -> LayerNorm; returns all 22 outputs.
    Heads: 12 in the masked-field pretraining, 8 in the source and fold stages, which load the 12-head weights unchanged
    (no weight shape depends on the head count). This is how the model was trained."""

    def __init__(self, vocab, heads=8, random_field=False):
        super().__init__()
        self.tok = nn.Embedding(vocab, DIM)
        # random init when trained from scratch; zeros when the weights are loaded afterwards
        self.field = nn.Parameter(torch.randn(1, NF + 1, DIM) * 0.02 if random_field else torch.zeros(1, NF + 1, DIM))
        layer = nn.TransformerEncoderLayer(DIM, heads, 4 * DIM, dropout=0.1, batch_first=True, norm_first=True,
                                           activation='gelu')
        self.blocks = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(DIM)

    def forward(self, ids):
        cls = torch.zeros(ids.shape[0], 1, dtype=torch.long, device=ids.device)
        return self.ln(self.blocks(self.tok(torch.cat([cls, ids], 1)) + self.field))


class Clf(nn.Module):
    """8-head encoder; head on [CLS output, mean of the 21 field outputs] -> 256 (GELU, dropout 0.1) -> 1 logit."""

    def __init__(self, vocab):
        super().__init__()
        self.enc = Enc(vocab)
        self.head = nn.Sequential(nn.Linear(2 * DIM, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, ids):
        h = self.enc(ids)
        return self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)).squeeze(-1)
