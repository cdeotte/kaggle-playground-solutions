#!/usr/bin/env python
"""Token banks of the native GPT KD model: every train + test row (train first, csv row order) written as
'<field> is <value>, ...' with the 13 predictors in their exact csv spelling (empty -> 'None'), ending in
', Will_Buy_EV is', tokenized by GPT-2 and right-padded to 160 tokens. Two column orders: 'competition' (train.csv)
and 'source' (the original dataset).
Writes cache/gpt_prefix_{competition,source}.npy (uint16) and cache/gpt_prefix_{order}_lengths.npy.
Usage: python gpt_prefix.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import DATA_DIR, TARGET  # noqa: E402
from gpt_surrogate import original_rows, tokenizer  # noqa: E402

CACHE = HERE/'cache'
ORDERS = ['source', 'competition']
PREFIX_END = ', Will_Buy_EV is'
PAD_LENGTH = 160


def main():
    source_fields, _ = original_rows()
    fields = [c for c in source_fields if c != TARGET]
    train = pd.read_csv(DATA_DIR/'train.csv', usecols=fields, dtype=str, keep_default_na=False)
    test = pd.read_csv(DATA_DIR/'test.csv', usecols=fields, dtype=str, keep_default_na=False)
    assert list(train.columns) == list(test.columns)
    table = pd.concat([train, test], ignore_index=True)
    tok = tokenizer()
    assert len(tok.encode(' Yes', add_special_tokens=False)) == 1 and len(tok.encode(' No', add_special_tokens=False)) == 1
    CACHE.mkdir(parents=True, exist_ok=True)
    for order in ORDERS:
        path = CACHE/f'gpt_prefix_{order}.npy'
        if path.exists():
            continue
        columns = fields if order == 'source' else list(train.columns)
        data = table[columns].to_numpy(dtype=object)
        ids = np.lib.format.open_memmap(path, mode='w+', dtype='uint16', shape=(len(data), PAD_LENGTH))
        ids[:] = tok.pad_token_id
        lengths = np.zeros(len(data), dtype='int16')
        for start in range(0, len(data), 8192):
            rows = data[start:start+8192]
            docs = [', '.join(f'{field} is {str(value).strip() or "None"}' for field, value in zip(columns, row))+PREFIX_END
                    for row in rows]
            encoded = tok(docs, add_special_tokens=False, truncation=False)['input_ids']
            for offset, values in enumerate(encoded):
                assert 0 < len(values) <= PAD_LENGTH, len(values)
                ids[start+offset, :len(values)] = values
                lengths[start+offset] = len(values)
        ids.flush()
        np.save(CACHE/f'gpt_prefix_{order}_lengths.npy', lengths)
        print('token bank', order, int(lengths.min()), int(lengths.max()), flush=True)


if __name__ == '__main__':
    main()
