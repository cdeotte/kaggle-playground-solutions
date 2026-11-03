# TabM: snapped numeric proxies and rare flags

One pytabkit `TabM_D_Classifier` model. Folds: the 5 folds of `common/data.py`; training rows = competition rows only (the
original data is not used); the fold's validation rows are passed to `fit` and TabM keeps its best epoch on them.
Test = mean of the 5 fold models.

Credit: the TabM settings follow mahoganybuttstrings's public notebook
[PG S5E8 TabM](https://www.kaggle.com/code/mahoganybuttstrings/pg-s5e8-tabm-cv-0-976810-pb-0-97750).

- **`train_mini_snapped_proxies_rare_flags.py`** (25 columns): the 16 categoricals as strings; for tenure,
  MonthlyCharges and TotalCharges a per-fold snapper (values seen < 25 times in the fold's training rows -> nearest
  frequent value, ties to the larger one) giving the snapped value (`__cat`) and the rare flag (`__is_rare`) as
  categories; the 3 numerics (NaN -> train median) standardized on the fold's training rows. TabM `tabm-mini`, k 16,
  2 blocks x 256, piecewise-linear embeddings (d 8), dropout 0.2, lr 3e-4, batch 512, 5 epochs, random_state 100.

```
python train_mini_snapped_proxies_rare_flags.py
```
Runs its 5 folds in one process on one GPU and writes `preds/tabm_mini_snapped_proxies_rare_flags_{oof,test}.npy`
(probabilities). Only `data/` is read; no other repo folder has to run first.

| Pair | Model | OOF AUC |
|---|---|---:|
| tabm_mini_snapped_proxies_rare_flags | TabM-mini on the categoricals plus snapped proxies and rare flags of the 3 numerics | 0.916734 |

Requirements: pytabkit, torch (CUDA), scikit-learn, pandas, numpy.
