# Combo XGB (XGBoost on 692 fold-local columns, 32 seeds)

Our best single XGBoost. It trains on a 692-column design built per outer fold:
- the 642-column "combo" design (`build_combo.py`): static features, 168 GPT-2 acceptance columns, single-field and
  multi-resolution income target encodings, 127 differences between income TEs at neighbouring resolutions, and a TE of
  the income's first GPT-2 BPE piece;
- income digits / modulus / rounding (14), digit-bin income TEs with 4 contrasts (8), the replica footprint (10: how
  often each income appears in the competition data vs in rows sampled from GReaT replicas), and the exact-income EM /
  information columns of the income-mixture model (4 + 14).

Every label-derived column is fold-local: outer-train rows are encoded by an inner cross-fit, valid / test rows by the
state fitted on all outer-train rows. XGBoost starts from a nested fold-local logistic-regression margin (`base_margin`)
and uses per-block feature weights. Depth 6, eta 0.03, 600 rounds, `num_parallel_tree` 6. Each fold averages 32 XGB
seeds in logit space; every seed draws fresh train-only Gaussian noise (sd 0.04) on the response-encoding columns.
Test = mean of the 5 fold models.

## Steps

```
python build_combo.py          # 642-column combo design per fold -> features/combo/fold_k/
python build_blocks.py all     # digits pack, digit_te block, income_em / income_info blocks -> features/
python sample_replica.py --checkpoint e60 --rows 50000 --seed 426500   # replica samples, one command per row of the table below
python footprint.py            # replica pool + footprint pack -> features/footprint.npy
python train.py                # 5 folds x 8 parts x 4 seeds -> preds/combo_xgb_32seeds_{oof,test}.npy
```

`train.py --folds 0 --parts 0,1` runs a subset (one part = 4 seeds of one fold, about 20 minutes on one GPU); finished
parts are kept in `parts/` and skipped, and the pair is written once all 40 parts exist.

Replica samples for the footprint pool (GReaT sampling, temperature 0.7, label-first start; `sample_replica.py` runs in
the be_great environment of `../gpt2_replica`):

| Checkpoint | Seed | Rows |
|---|---:|---:|
| e55 | 425500 | 50,000 |
| e55 | 425555 | 10,000 |
| e60 | 426500 | 50,000 |
| e60 | 426060 | 10,000 |
| e65 | 426550 | 50,000 |
| e65 | 426565 | 10,000 |
| e70 | 427500 | 50,000 |
| e70 | 427070 | 10,000 |
| d60 | 426500 | 50,000 |

## Dependencies

- `../response_features`: its caches for `build_combo.py`: static blocks (`cache/static`), fold caches (`cache/base`,
  response logits and nested margin in `cache/static/fold_k`, `cache/multires`) and the acceptance pack
  (`cache/acceptance`); the paths are set at the top of `build_combo.py`.
- `../income_mixture_xgb`: `cache/em_info/{features_gpu,information_features}` (exact-income EM and information
  features) for `build_blocks.py`.
- `../gpt2_replica`: the replica checkpoints for `sample_replica.py`.
- `../te_bank_lr` reads `features/digits.npy`, `features/footprint.npy` and the `income_em` / `income_info` blocks.

Packages: xgboost (GPU), transformers (GPT-2 tokenizer for the BPE-prefix block), be_great 0.0.3 for the replica samples.

## Pair

| Pair | Model | OOF AUC |
|---|---|---:|
| combo_xgb_32seeds | XGBoost, 692 columns, depth 6, 600 rounds, 32 seeds | 0.946574 |

## Gaps

Our footprint pool had 835,000 replica rows from 37 sample sets (checkpoints at epochs 40-70 of several GPT-2-small
replica runs). The 9 sets above (290,000 rows) come from checkpoints in `../gpt2_replica`; the other 28 came from replica
runs with other seeds and training settings that are not in this repository, so the pool (and the replica-count footprint
columns) can only be approximated.
