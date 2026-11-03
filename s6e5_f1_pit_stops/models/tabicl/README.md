# TabICL v2: zero-shot and fine-tuned in-context models

Five TabICL models on the 14 raw competition columns (no feature engineering). TabICL is a pretrained tabular
in-context transformer: the fold's training rows are its context, and the fine-tuned variants additionally update the
pretrained weights on those rows. All five use the TabICL v2 classifier checkpoint `tabicl-classifier-v2-20260212.ckpt`,
which the `tabicl` package downloads from the Hugging Face model [jingang/TabICL](https://huggingface.co/jingang/TabICL)
on first use (cached afterwards).

**Zero-shot** (`train_zero_shot.py`): per outer fold, eight `TabICLClassifier` models (`n_estimators=2`,
`kv_cache=True`, `random_state` 0..7) get the fold's training rows as context (Driver, Compound and Race passed as
strings; TabICL encodes them itself) and predict the fold's validation rows and the test rows. The 8 probabilities are
averaged and clipped to [1e-7, 1 - 1e-7].

**Fine-tuned** (`train_finetune.py`, `FinetunedTabICLClassifier`): per outer fold, a `ColumnTransformer` fitted on the
fold's training rows ordinal-encodes the string columns (unknown / missing -> -1) and median-imputes the numeric ones
(float32, categoricals first). Eight members (`random_state` 0..7) each fine-tune the checkpoint on the fold's training
rows only: the library splits off its own validation set (`validation_split_ratio` 0.08) for early stopping on ROC AUC;
the outer validation rows are never passed to `fit`. Base settings: 8 epochs, lr 1e-5 with warm-up (10%) and scheduler,
weight decay 0.01, grad clip 1, AMP, `max_data_size` 12,000, `finetune_ctx_query_ratio` 0.2, patience 3
(min_delta 2e-5), 2 estimators for fine-tuning / validation / inference, latin feature shuffles, shift class shuffles,
softmax temperature 0.9, inference with KV cache and batch size 2. Member probabilities are clipped to [1e-7, 1 - 1e-7]
and averaged.
- `tabicl_v2_finetune_lowcard_cats`: Year, PitStop, LapNumber, Stint, TyreLife, Position and Position_Change are cast
  to strings and ordinal-encoded like Driver, Compound and Race; LapTime (s), LapTime_Delta, Cumulative_Degradation and
  RaceProgress stay numeric.
- `tabicl_v2_finetune_original_append`: all 101,371 rows of the original dataset (`Normalized_TyreLife` dropped) are
  appended to every fold's training rows (never to the validation rows); the encoder is fitted on the combined rows.
- `tabicl_v2_finetune_bigger_context`: `max_data_size` 14,000, 10 epochs, lr 8e-6, patience 4, inference without KV
  cache and with batch size 1 (and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) to fit in GPU memory.

Test = mean of the 5 fold models for every pair (probabilities).

```
python train_zero_shot.py
python train_finetune.py tabicl_v2_finetune_8seed
python train_finetune.py tabicl_v2_finetune_lowcard_cats
python train_finetune.py tabicl_v2_finetune_original_append
python train_finetune.py tabicl_v2_finetune_bigger_context
```
No other repo folders are needed (the scripts read `data/` through `common/data.py`). One GPU per run; a fine-tuned fold
(8 fine-tunes) took 80-125 minutes. `--folds 0,1` runs a subset (e.g. one process per GPU with
`CUDA_VISIBLE_DEVICES`); finished folds are kept in `parts/` and `preds/<pair>_{oof,test}.npy` is written when the last
fold is done.

| Pair | Model | OOF AUC |
|---|---|---:|
| tabicl_v2_zero_shot_8seed | zero-shot TabICLClassifier, 8 seeds x 2 estimators, raw columns | 0.949538 |
| tabicl_v2_finetune_8seed | fine-tuned, 8 members, raw columns (ordinal categoricals, median-imputed numerics) | 0.950652 |
| tabicl_v2_finetune_lowcard_cats | fine-tuned, 7 low-cardinality numeric columns as categoricals | 0.948527 |
| tabicl_v2_finetune_original_append | fine-tuned, raw columns, all original rows appended to every fold's training rows | 0.950827 |
| tabicl_v2_finetune_bigger_context | fine-tuned, max_data_size 14,000, 10 epochs, lr 8e-6, patience 4 | 0.950681 |

**Packages.** The two scripts ran in different environments:
- zero-shot: `tabicl` 2.0.3 (`TabICLClassifier`), torch (CUDA), huggingface_hub, scikit-learn, pandas, numpy.
- fine-tuned: `tabicl` 2.1.1 with the fine-tuning extra, `pip install "tabicl[finetune]==2.1.1"` (provides
  `FinetunedTabICLClassifier`; pulls in transformers and wandb, which stays off: `wandb_kwargs=None` and
  `WANDB_DISABLED=true`), torch 2.11 (CUDA 12.6), scikit-learn 1.6.1, pandas 2.3.3, numpy 2.2.6.
