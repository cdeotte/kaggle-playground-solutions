"""The 12 columns as exact value tokens, encoded as integer ids (label-free, cached).

Numerics -> f'{x:.2f}', categoricals -> the raw string, NaN -> '__NA__' (per column, no imputation). The vocabulary of a
column is the sorted set of its tokens over train + test; id = position in that list."""
import numpy as np
import pandas as pd

from common.data import CATEGORICALS as _CATS, NUMERICS as _NUMS, cache_dir

NUMERICS = list(_NUMS)
CATEGORICALS = list(_CATS)
FIELDS = NUMERICS + CATEGORICALS  # 12 fields, fixed order
NA_TOKEN = '__NA__'
FOLDER = 'factorization_machines'


def _tokens(s, is_numeric):
    if is_numeric:
        return s.map(lambda x: f"{x:.2f}" if pd.notna(x) else NA_TOKEN)
    return s.fillna(NA_TOKEN).astype(str)


def build_token_ids(train, test):
    """Returns (ids_train, ids_test, vocab_sizes, vocabs): int32 ids of shape (n_rows, 12), the 12 vocabulary sizes and
    a dict field -> list of tokens (id = position)."""
    cache_path = cache_dir(FOLDER) / 'fm_tokens.npz'
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        return z['ids_train'], z['ids_test'], z['vocab_sizes'], z['vocabs'].item()

    ids_tr = np.zeros((len(train), len(FIELDS)), dtype=np.int32)
    ids_te = np.zeros((len(test), len(FIELDS)), dtype=np.int32)
    vocab_sizes = np.zeros(len(FIELDS), dtype=np.int64)
    vocabs = {}
    for j, c in enumerate(FIELDS):
        is_num = c in NUMERICS
        tok_tr = _tokens(train[c], is_num)
        tok_te = _tokens(test[c], is_num)
        vocab = sorted(set(tok_tr.unique()) | set(tok_te.unique()))
        tok2id = {t: i for i, t in enumerate(vocab)}
        ids_tr[:, j] = tok_tr.map(tok2id).values
        ids_te[:, j] = tok_te.map(tok2id).values
        vocab_sizes[j] = len(vocab)
        vocabs[c] = vocab
    np.savez_compressed(cache_path, ids_train=ids_tr, ids_test=ids_te, vocab_sizes=vocab_sizes, vocabs=vocabs)
    print(f'built token ids: train {ids_tr.shape}, test {ids_te.shape}, total vocab {vocab_sizes.sum()}', flush=True)
    return ids_tr, ids_te, vocab_sizes, vocabs
