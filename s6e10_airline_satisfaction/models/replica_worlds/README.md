# Replica worlds (M32_1M, MB32_1M)

Two synthetic "worlds" of about 1M labelled rows each, made the way the competition data was made: a GReaT GPT-2 medium
generator fine-tuned on the original data (`data.csv`), sampled with a label-first prompt, then parsed, deduplicated and
split 70/30 into a train and a test file. They give other models many more labelled rows from nearly the competition's
distribution: [field_be](../field_be), [pair_be](../pair_be) and [m19_source_aug](../m19_source_aug) pretrain on them (plus
`data.csv`) before fine-tuning on the competition folds, and [gpt_fold_be](../gpt_fold_be) adds their rows to the rows its
feature vocabularies are built from (categorical codes, log-frequency features). The generators never see `train.csv` or
`test.csv`.

Each generator trains on a noised copy of `data.csv`. The noise settings come from our reconstruction of the competition's
generator: of the worlds we sampled, these two matched the competition data best.

| World | Label flips in the training copy | Prompt P(True) | Post-generation label flips | Rows in our run (train + test) |
|---|---|---:|---:|---|
| M32_1M | 7% of all labels | 0.444179 | 0.9% | 699,225 + 299,669 |
| MB32_1M | 8.5% False→True, 9.0% True→False | 0.443455 | none | 699,216 + 299,664 |

Both worlds:
- **Noise:** label flips from one uniform draw per row (seed 1234), then ±1 jitter of the 13 service ratings (seed 4321,
  5% each way, clipped to the column's range). The prompt P(True) is the positive rate of this noisy copy.
- **Generator:** GPT-2 medium, batch 8, a 40-epoch linear learning-rate schedule. The worlds use the weights after epoch 32
  of that schedule (not a 32-epoch schedule), so training stops there.
- **Sampling:** prompt `satisfaction is True,` or `satisfaction is False,`, temperature 0.7, top-k 50, at most 200 tokens,
  fp16 weights, seed 7, 1M decodes.
- **Assembly:** GReaT's parsing rules, rows with duplicate features dropped, `train_test_split(test_size=0.3, random_state=0)`.

## Steps

Run this folder before the four models above.

```
python train_generator.py --world M32_1M     # legacy environment (requirements.txt), one GPU
python train_generator.py --world MB32_1M
python sample_world.py --world M32_1M        # one GPU: 1M raw decodes
python sample_world.py --world MB32_1M
python make_world.py --world M32_1M          # CPU: parse, dedup, split, post-generation flips (M32_1M only)
python make_world.py --world MB32_1M
```

## Output

```
worlds/<WORLD>/train.csv          id, the 21 features and satisfaction (True/False), in the competition's column order
worlds/<WORLD>/test.csv           id and the 21 features
worlds/<WORLD>/y.npy              train labels (0/1)
worlds/<WORLD>/y_test.npy         test labels (0/1)
worlds/<WORLD>/folds.npy          StratifiedKFold(5, shuffle=True, random_state=42) ids of the train rows (not used downstream)
worlds/<WORLD>/sizes.json         row counts
checkpoints/<WORLD>/epoch_032.pt  generator weights (GPT2LMHeadModel state dict)
raw/<WORLD>/raw.jsonl.gz          raw decodes with their prompt labels
```

## Notes
- **Not bit-reproducible.** TF32 training and fp16 sampling differ across GPUs and library versions, so a rerun gives an
  equivalent world, not the same rows.
- **Categorical values are not checked.** As in our worlds, a few malformed values (such as `Class` = `Male`) are kept as
  generated.
- **Environment.** `train_generator.py` needs the old pins in `requirements.txt` (be-great 0.0.3, transformers 4.26.1), like
  `../gpt2_replica`. Train with a single visible GPU: with several, the HF Trainer would use all of them and multiply the
  batch size. `sample_world.py` and `make_world.py` run in the base environment (any recent transformers).
