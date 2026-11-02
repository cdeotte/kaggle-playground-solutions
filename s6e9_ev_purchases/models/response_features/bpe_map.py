#!/usr/bin/env python
"""Leading GPT-2 BPE piece of every income: cache/bpe/income_to_first_piece.csv (columns income, first_piece).

One row per distinct income over train + test (sorted); first_piece is the first BPE piece of " <income as float>" without the
space, e.g. 32910.0 -> " 329" "10" "." "0" -> 329 (common/bpe.py). Label-free. Needs transformers (GPT-2 tokenizer).
Usage: python bpe_map.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.bpe import income_bpe_table  # noqa: E402
from common.data import read_test, read_train  # noqa: E402

OUT = HERE / 'cache' / 'bpe' / 'income_to_first_piece.csv'


def main():
    income = np.concatenate([read_train()['Annual_Income_USD'].to_numpy(np.float64),
                             read_test()['Annual_Income_USD'].to_numpy(np.float64)])
    table = income_bpe_table(income)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'income': table.index.to_numpy(np.float64), 'first_piece': table['p1'].to_numpy()}).to_csv(OUT, index=False)
    print(f'{len(table)} incomes, {table["p1"].nunique()} leading pieces -> {OUT}')


if __name__ == '__main__':
    main()
