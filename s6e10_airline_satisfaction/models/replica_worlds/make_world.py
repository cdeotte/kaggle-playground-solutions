#!/usr/bin/env python
"""Assemble a world (worlds/<world>/) from its raw decodes, like a GReaT sample turned into a competition:
  1. parse each text with GReaT's rules: split on ',', each piece on ' is '; the first occurrence of a column wins; rows
     missing a column are dropped; 'None' -> NaN; numeric columns must parse, else the row is dropped. Categorical values
     are not checked (a few malformed ones are kept as generated);
  2. keep the first n parsed rows, then drop rows with duplicate features (keep the first);
  3. train_test_split(test_size=0.3, random_state=0); ids number the train rows first, then the test rows;
  4. M32_1M only: symmetric post-generation label flips at rate 0.009 (seed 11; train mask first, then test mask).
Writes train.csv (the competition's column order, satisfaction True/False), test.csv (no label), y.npy / y_test.npy (0/1
labels of the train / test rows), folds.npy (StratifiedKFold(5, shuffle=True, random_state=42) on the train labels before
step 4) and sizes.json.
Usage: python make_world.py --world M32_1M|MB32_1M"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import DATA_DIR, TARGET, read_original  # noqa: E402
from recipe import WORLDS  # noqa: E402


def parse(path, cols, max_rows):
    """Raw decodes -> DataFrame of value strings (only rows with every column present), prompt labels, number of decodes."""
    rows, ys, n_raw = [], [], 0
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        for line in fh:
            n_raw += 1
            o = json.loads(line)
            row = dict.fromkeys(cols)
            for piece in o['text'].split(','):
                kv = piece.strip().split(' is ')
                if kv[0] in row and row[kv[0]] is None and len(kv) > 1:
                    row[kv[0]] = kv[1]
            if any(v is None for v in row.values()):
                continue
            rows.append(row)
            ys.append(o['y'])
            if len(rows) >= max_rows:
                break
    return pd.DataFrame(rows, columns=cols), np.array(ys), n_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', required=True, choices=list(WORLDS))
    ap.add_argument('--n', type=int, default=1_000_000)
    a = ap.parse_args()
    og = read_original()
    cols = og.columns.tolist()
    feats = [c for c in cols if c != TARGET]
    num = og.select_dtypes(include=np.number).columns.tolist()
    ints = [c for c in num if og[c].dtype.kind in 'iu']
    order = pd.read_csv(DATA_DIR / 'train.csv', nrows=0).columns.tolist()  # id, numeric, categorical, label

    # GReaT keeps sampling until it has n valid rows; parsing stops at 1.02 n rows with every column present
    df, prompt_y, n_raw = parse(HERE / 'raw' / a.world / 'raw.jsonl.gz', cols, int(a.n * 1.02))
    n_complete = len(df)
    df = df.replace('None', np.nan)
    ok = np.ones(len(df), bool)
    for c in num:
        v = pd.to_numeric(df[c], errors='coerce')
        ok &= v.notna().to_numpy() | df[c].isna().to_numpy()
        if c in ints:  # the int cast below needs finite values (none were missing in our worlds)
            ok &= np.isfinite(v.to_numpy(dtype=float))
        df[c] = v
    df, prompt_y = df[ok].reset_index(drop=True), prompt_y[ok]
    assert set(df[TARGET].unique()) <= {'True', 'False'}
    assert (df[TARGET].eq('True').to_numpy() == (prompt_y == 1)).all()  # the label is the prompt's
    df = df.iloc[:a.n]
    n_valid = len(df)
    df = df.drop_duplicates(subset=feats).reset_index(drop=True)
    tr, te = train_test_split(df, test_size=0.3, random_state=0, shuffle=True)

    y_tr, y_te = tr[TARGET].eq('True').to_numpy(), te[TARGET].eq('True').to_numpy()
    folds = np.zeros(len(tr), np.int64)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y_tr, y_tr.astype(int))):
        folds[va] = k
    q = WORLDS[a.world]['posthoc_flip']
    if q > 0:
        rng = np.random.default_rng(11)
        y_tr = y_tr ^ (rng.random(len(y_tr)) < q)
        y_te = y_te ^ (rng.random(len(y_te)) < q)

    out = HERE / 'worlds' / a.world
    out.mkdir(parents=True, exist_ok=True)
    for name, d, y, start in (('train', tr, y_tr, 0), ('test', te, y_te, len(tr))):
        d = d.copy()
        d[ints] = d[ints].astype(int)
        d[TARGET] = y
        d.insert(0, 'id', np.arange(start, start + len(d)))
        d[order if name == 'train' else [c for c in order if c != TARGET]].to_csv(out / f'{name}.csv', index=False)
        np.save(out / ('y.npy' if name == 'train' else 'y_test.npy'), y.astype(int))
    np.save(out / 'folds.npy', folds)
    (out / 'sizes.json').write_text(json.dumps(dict(n_train=len(tr), n_test=len(te))))
    print(json.dumps(dict(world=a.world, decodes=n_raw, complete=n_complete, valid=n_valid, after_dedup=len(df),
                          n_train=len(tr), n_test=len(te), positive_rate_train=round(float(y_tr.mean()), 6))), flush=True)


if __name__ == '__main__':
    main()
