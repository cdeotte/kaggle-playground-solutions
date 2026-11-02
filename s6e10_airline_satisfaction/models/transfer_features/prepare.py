#!/usr/bin/env python
"""Label-free feature cache over the competition rows and the labelled source rows (original data + 2 replica worlds).

Used by ../field_be and ../pair_be (all files), ../gpt_fold_be (cat, num, meta) and ../m19_source_aug (pre_y).
Rows, in this order (meta['blocks'] has the sizes):
  competition train (train.csv order), competition test (test.csv order), original data.csv,
  world M32_1M train, M32_1M test, world MB32_1M train, MB32_1M test   (../replica_worlds/worlds/<world>/)
Rows 0 .. n_train + n_test - 1 are the competition rows; rows from meta['pre_start'] on are the source rows.

Only predictors enter the features; value counts and category codes are taken over all rows.
  cat   int32 [N, 35]: category ids with a per-field offset (one shared embedding table): the 21 raw values, the first 2
        GPT-2 tokens of Age / Flight Distance / both delays (written like the generator wrote them), and floor bins of Age
        (5, 10) and Flight Distance (10, 50, 100, 500)
  pair  int32 [N, 154]: exact ids of field pairs: every pair of the 13 ratings + 4 categoricals, and Flight Distance / its
        first GPT-2 token / Age bin 5 x (the 4 categoricals, wifi, online boarding); a global offset per pair
  num   float32 [N, 53]: log value frequency of every field; raw value, signed log (distance, delays) and missing flag of
        the numeric fields; mean / std / range of 3 rating groups; number of zero ratings; arrival minus departure delay.
        Unscaled (the models fit their own scaler)
  y, folds  int64 [n_train]: competition labels and fold ids (common.data)
  pre_y     int64 [N - pre_start]: labels of the source rows (original, M32 train, M32 test, MB32 train, MB32 test)
Usage: python prepare.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402

OUT = HERE / 'cache'
WORLDS = HERE.parent / 'replica_worlds' / 'worlds'
RATINGS = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location',
           'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service',
           'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness']
CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
LOG_COLS = ('Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes')
BPE_COLS = ('Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes')
BINS = (('Age', (5, 10)), ('Flight Distance', (10, 50, 100, 500)))
RATING_GROUPS = (('booking', [0, 1, 2, 3, 5]), ('comfort', [4, 6, 7, 12]), ('service', [8, 9, 10, 11]))  # into RATINGS
ROUTE_FIELDS = ('Flight Distance', 'Flight Distance:bpe0', 'Age:bin5')
CONTEXT_FIELDS = CATS + ['Inflight wifi service', 'Online boarding']


def generator_strings(v, as_float):
    """Values as the GReaT generator wrote them: ints '36', floats '12.0', NaN 'None'."""
    return v.map(lambda z: 'None' if pd.isna(z) else str(float(z)) if as_float else str(int(z)))


def load_rows():
    tr, te, og = read_train(), read_test(), read_original()
    cols = [c for c in tr.columns if c not in ('id', TARGET)]
    blocks = [('train', tr[cols]), ('test', te[cols]), ('original', og[cols])]
    pre_y = [labels(og[TARGET])]
    for world in ('M32_1M', 'MB32_1M'):
        d = WORLDS / world
        blocks.append((f'{world}_train', pd.read_csv(d / 'train.csv')[cols]))
        pre_y.append(np.load(d / 'y.npy'))
        blocks.append((f'{world}_test', pd.read_csv(d / 'test.csv')[cols]))
        pre_y.append(np.load(d / 'y_test.npy'))
    frame = pd.concat([b for _, b in blocks], ignore_index=True)
    pre_y = np.concatenate([np.asarray(p).astype(np.int64) for p in pre_y])
    y = labels(tr[TARGET]).astype(np.int64)
    assert len(pre_y) == len(frame) - len(tr) - len(te)
    return frame, cols, blocks, y, pre_y


def main():
    frame, cols, blocks, y, pre_y = load_rows()
    ntr, nte = len(blocks[0][1]), len(blocks[1][1])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')

    catcols, catnames, cards, numcols, numnames = [], [], [], [], []

    def addcat(name, values):
        code, uniq = pd.factorize(pd.Series(values).fillna('__MISSING__'), sort=True)
        catcols.append(code.astype(np.int32))
        catnames.append(name)
        cards.append(len(uniq))

    def addnum(name, values):
        numcols.append(np.asarray(values, dtype=np.float32))
        numnames.append(name)

    for c in cols:
        v = frame[c]
        addcat(c, v.astype(str))
        cnt = v.fillna('__MISSING__').value_counts()
        addnum(c + ':logfreq', np.log1p(v.fillna('__MISSING__').map(cnt)))
        if c not in CATS:
            x = v.fillna(0).to_numpy(dtype=np.float32)
            addnum(c + ':raw', x)
            if c in LOG_COLS:
                addnum(c + ':log', np.sign(x) * np.log1p(np.abs(x)))
            if v.isna().any():
                addnum(c + ':missing', v.isna())
        if c in BPE_COLS:
            # first 2 GPT-2 tokens of ' <value>' (-1 if there is no second token)
            strs = generator_strings(v, as_float=(c == 'Arrival Delay in Minutes'))
            uniq = strs.unique()
            toks = tok([' ' + s for s in uniq], add_special_tokens=False)['input_ids']
            for j in range(2):
                addcat(f'{c}:bpe{j}', strs.map({s: (t[j] if len(t) > j else -1) for s, t in zip(uniq, toks)}))
    for c, scales in BINS:
        for s in scales:
            addcat(f'{c}:bin{s}', np.floor(frame[c].to_numpy() / s).astype(np.int64))
    r = frame[RATINGS].to_numpy(dtype=np.float32)
    for name, idx in RATING_GROUPS:
        z = r[:, idx]
        addnum(name + ':mean', z.mean(1))
        addnum(name + ':std', z.std(1))
        addnum(name + ':range', z.max(1) - z.min(1))
    addnum('ratings:zeros', (r == 0).sum(1))
    addnum('delays:difference', frame['Arrival Delay in Minutes'].fillna(0) - frame['Departure Delay in Minutes'])

    cards = np.array(cards, dtype=np.int64)
    offsets = np.r_[0, cards[:-1].cumsum()]
    cat = (np.stack(catcols, 1) + offsets).astype(np.int32)

    # exact pair ids: rating / category pairs, and route (Flight Distance) / age context pairs
    low = RATINGS + CATS
    pairs = [(c, d) for i, c in enumerate(low) for d in low[i + 1:]]
    pairs += [(c, d) for c in ROUTE_FIELDS for d in CONTEXT_FIELDS]
    paircols, paircards, poff = [], [], 0
    for c, d in pairs:
        i, j = catnames.index(c), catnames.index(d)
        key = (cat[:, i].astype(np.int64) - offsets[i]) * cards[j] + (cat[:, j].astype(np.int64) - offsets[j])
        code, uniq = pd.factorize(key, sort=True)
        paircols.append((code + poff).astype(np.int32))
        paircards.append(len(uniq))
        poff += len(uniq)
    pair = np.stack(paircols, 1)
    num = np.stack(numcols, 1).astype(np.float32)
    assert np.isfinite(num).all()

    OUT.mkdir(exist_ok=True)
    for name, arr in (('cat', cat), ('pair', pair), ('num', num), ('y', y), ('folds', folds(y).astype(np.int64)),
                      ('pre_y', pre_y)):
        np.save(OUT / f'{name}.npy', arr)
    meta = dict(n_train=ntr, n_test=nte, n_all=len(frame), pre_start=ntr + nte,
                blocks=[[name, len(b)] for name, b in blocks],
                cat_names=catnames, cards=cards.tolist(), offsets=offsets.tolist(), vocab=int(cards.sum()),
                num_names=numnames, pairs=pairs, pair_cards=paircards, pair_vocab=int(poff))
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'rows: {meta["blocks"]} (total {len(frame)}, source rows {len(pre_y)})')
    print(f'cat {cat.shape} vocab {meta["vocab"]} | pair {pair.shape} vocab {meta["pair_vocab"]} | num {num.shape}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
