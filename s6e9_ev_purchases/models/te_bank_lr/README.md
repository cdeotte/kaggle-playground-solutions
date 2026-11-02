# TE-bank logistic regressions and RealMLP

Three models on the same fold-local target-encoding banks (`build_banks.py`). Keys: the 13 raw fields, income // 10 ...
// 10,000, the first and first two GPT-2 BPE pieces of the income, income digits, commute and age bins, and 64 bins of the
generator-formula score; 29 single keys, 105 pairs, 36 income x context pairs and 41 triples. Each key is target-encoded
per outer fold (outer-train rows by a 5-fold inner cross-fit, valid / test rows from all outer-train rows), at alpha 20,
5 and 1.

- **te_bank_lr**: elastic-net logistic regression (cuML, l1_ratio 0.5, C 0.02) on 669 bank columns + label-free packs
  (formula score / logit with BPE piece count and income digits, more income digits, replica footprint, replica prior)
  + 168 GPT-2 acceptance columns + the exact-income EM / information columns, standardized on outer-train, plus GAM-style
  hinges at 4 outer-train quantiles of the single-key TEs, the formula columns and the EM columns (1,201 columns).
- **te_bank_lr_residual**: the same keys encoded as the smoothed mean of the formula residual y - Phi(rule), with the
  formula columns and hinges; elastic net, C 0.05 (341 columns).
- **te_bank_realmlp**: pytabkit RealMLP (32-member ensemble, 4 epochs, the S6E8 1st-place settings) on te_bank_lr's 889
  inputs without hinges; 3 seeds averaged in logit space.

The replica prior (`replica_prior.py`) is a label-free per-income prior: the log-likelihood ratio of the income's text
given "Will_Buy_EV is Yes" vs "No" under GReaT replicas trained on the original data only (mean and sd over the replicas
at temperature 0.7, mean at temperature 1, and the d60 replica alone).

## Steps

```
python build_banks.py               # TE banks, formula / digit pack, income_bpe_pieces.csv -> cache/   (CuPy)
python replica_prior.py             # replica prior pack -> cache/replica_prior.npy                     (GPU)
python lr.py te_bank_lr             # -> preds/te_bank_lr_{oof,test}.npy                                (cuML)
python lr.py te_bank_lr_residual    # -> preds/te_bank_lr_residual_{oof,test}.npy
python realmlp.py                   # seeds 0,1,2 x 5 folds -> preds/te_bank_realmlp_{oof,test}.npy
```

`realmlp.py --seeds 0 --folds 0` runs one part; finished parts are kept in `parts/` and skipped.

## Dependencies

- `../combo_xgb`: run its `build_blocks.py all` and `footprint.py` first (digits pack, footprint pack, `income_em` /
  `income_info` blocks in `../combo_xgb/features/`).
- `../response_features`: the GPT-2 acceptance pack (`cache/acceptance`).
- `../gpt2_replica`: the replica checkpoints for `replica_prior.py`.

Packages: cupy, cuML, pytabkit, torch, transformers.

## Pairs

| Pair | Model | OOF AUC |
|---|---|---:|
| te_bank_lr | elastic-net LR on all banks + packs + acceptance + EM columns, hinges | 0.946499 |
| te_bank_lr_residual | elastic-net LR on the formula-residual banks, hinges | 0.945328 |
| te_bank_realmlp | RealMLP on te_bank_lr's inputs, n_ens 32, 3 seeds | 0.946442 |

## Gaps

- The replica prior averaged 8 replica checkpoints from 4 training streams (names in `replica_prior.py`). Only e60, e70
  and d60 are in `../gpt2_replica`; the other five were further replicas trained the same way with other seeds and
  settings. `replica_prior.py` uses the checkpoints it finds.
- The original builder of the 4-column replica pack was not kept; `replica_prior.py` rebuilds it from the column
  definitions above.
