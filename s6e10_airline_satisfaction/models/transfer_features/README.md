# Transfer features: the feature cache of the source-pretrained models

One label-free feature table over the competition rows and the labelled **source rows**: the original `data.csv` and two
synthetic replica worlds (`M32_1M`, `MB32_1M` from `../replica_worlds`, sampled from GReaT replicas trained on `data.csv`
only). Category codes, value frequencies and pair ids are computed over all rows; labels never enter the features.

```
python prepare.py
```

Needs `data/` (train.csv, test.csv, data.csv), `../replica_worlds/worlds/{M32_1M,MB32_1M}/{train.csv,test.csv,y.npy,y_test.npy}`
and the GPT-2 tokenizer (`gpt2` from Hugging Face). Writes `cache/` (about 3 GB). Read by `../field_be` and `../pair_be`
(everything) and `../gpt_fold_be` (`cat`, `num`, `meta.json`; competition rows only).

## Cache

Rows of `cat`, `pair` and `num`, in this order (3,127,133 rows with our worlds; `blocks` in `meta.json` has the sizes):

| Rows | Block |
|---|---|
| 0 - 699,634 | competition train (train.csv order) |
| 699,635 - 999,478 | competition test (test.csv order) |
| 999,479 - 1,129,358 | original data.csv (129,880 rows) |
| 1,129,359 - 1,828,583 | world M32_1M train (699,225) |
| 1,828,584 - 2,128,252 | world M32_1M test (299,669) |
| 2,128,253 - 2,827,468 | world MB32_1M train (699,216) |
| 2,827,469 - 3,127,132 | world MB32_1M test (299,664) |

| File | dtype [shape] | Content |
|---|---|---|
| `cat.npy` | int32 [N, 35] | categorical ids with a per-field offset (one embedding table of size `vocab`): the 21 raw values, the first 2 GPT-2 tokens of Age, Flight Distance and both delays (written like the generator wrote them), floor bins of Age (5, 10) and Flight Distance (10, 50, 100, 500) |
| `pair.npy` | int32 [N, 154] | exact ids of field pairs (one table of size `pair_vocab`): all 136 pairs of the 13 ratings and 4 categoricals, and Flight Distance / its first GPT-2 token / the 5-year Age bin x the 4 categoricals, wifi and online boarding |
| `num.npy` | float32 [N, 53] | log value frequency of every field; raw value, signed log (distance, delays) and missing flag (arrival delay) of the numeric fields; mean / std / range of 3 rating groups; number of zero ratings; arrival minus departure delay. Unscaled |
| `y.npy`, `folds.npy` | int64 [699,635] | competition labels and fold ids (the same as `common.data`) |
| `pre_y.npy` | int64 [2,127,654] | labels of the source rows, i.e. rows `pre_start` to `n_all - 1` (original, M32 train, M32 test, MB32 train, MB32 test) |

`meta.json`: `n_train`, `n_test`, `n_all` (all rows), `pre_start` (first source row = `n_train + n_test`), `blocks`
([name, rows] in row order), `cat_names`, `cards`, `offsets`, `vocab` (7,601 categorical ids with our worlds), `num_names`,
`pairs`, `pair_cards`, `pair_vocab` (79,703 pair ids with our worlds).
