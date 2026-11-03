# TabM: snapped numeric proxies, original priors and target encodings

Four pytabkit `TabM_D_Classifier` models, one script each. Stack A and stack B are the two stacks of `stack/`; the
pair names end in `_a` / `_b` accordingly. Folds: the 5 folds of `common/data.py`; training rows = competition rows
only (the original dataset only feeds features); the fold's validation rows are passed to `fit` and TabM keeps its
best epoch on them. Test = mean of the fold models.

- **`train_mini_snapped_proxies_rare_flags_a.py`** (41 columns): the 8 categoricals; for each of the 11 numerics a
  per-fold snapper (values seen < 25 times in the fold's training rows -> nearest frequent value) giving the snapped
  value (`__cat`) and the rare flag (`__is_rare`) as categories; the 11 numerics standardized. TabM `tabm-mini`, k 16,
  2 blocks x 256, piecewise-linear embeddings (d 8), dropout 0.2, lr 3e-4, batch 512, 5 epochs, random_state 100.
- **`train_original_priors_te_ngram_a.py`** (216 columns): the 11 numerics and their frequencies; 6 water / soil
  ratios and products; per-class original target rates of the 8 categoricals and of 4 numerics rounded to 2 decimals;
  percentile ranks / z-scores of Previous_Irrigation_mm and Rainfall_mm within the original classes; mod 10, mod 100,
  decimals and is-integer flag of 4 numerics; per fold scikit-learn `TargetEncoder` (multiclass, smooth auto, 5-fold
  cross fitting, seed 42) of the 11 numerics as rounded value strings, the 8 categoricals, 15 bigrams and 4 trigrams
  (114 columns); numerics standardized, categoricals ordinal-encoded. TabM `tabm-mini-normal`, k 32, 3 blocks x 256,
  piecewise-linear embeddings (d 24), dropout 0.2, lr 1e-3, weight decay 1e-3, batch 512, up to 50 epochs, patience
  10, random_state 42.
- **`train_mini_snapped_formula_priors_b.py`** (66 columns): the 8 categoricals, the snapped proxies / rare flags of
  all 11 numerics (ties snap to the smaller value), the 11 numerics, a formula score (Soil_Moisture < 25, Rainfall_mm < 300,
  Temperature_C > 30, Wind_Speed_kmh > 10, growth stage Harvest / Sowing, mulching) and per-class original rates of the
  8 categoricals, standardized. TabM `tabm-mini`, k 16, 2 blocks x 256, piecewise-linear embeddings (d 8), dropout
  0.2, lr 3e-4, batch 1024, 40 epochs (best epoch on the validation cross-entropy), random_state 100.
- **`train_nested_binary_te_ngram_b.py`**: the 11 numerics; frequencies of the 19 raw columns over train + original;
  3 ratios and 3 flags; per-class original rates of the 19 raw columns; Soil_Moisture percentile ranks within the
  original classes; per outer fold, per-class (one-vs-rest) target means of the 8 categoricals, the 11 numeric value
  strings, 15 bigrams and 4 trigrams (114 columns), out-of-fold over 5 folds (seed 42 + outer fold) for the outer
  training rows. Nested 5 x 5: outer folds = the common folds, inner `StratifiedKFold(5, shuffle=True,
  random_state=1000 + outer fold)`; TabM `tabm-mini-normal` with the settings of `train_original_priors_te_ngram_a.py`.
  The OOF has shape `(5, 630000, 3)`: slice k = the inner-validation predictions on outer fold k's training rows + the
  mean of its 5 inner models on its held-out rows. Test = mean of the 25 models.

```
python train_mini_snapped_proxies_rare_flags_a.py
python train_original_priors_te_ngram_a.py
python train_mini_snapped_formula_priors_b.py
python train_nested_binary_te_ngram_b.py
```
Each script runs its folds in one process on one GPU and writes `preds/<pair>_{oof,test}.npy` (probabilities, class
order Low, Medium, High). Only `data/` is read; no other repo folder has to run first. Logged run times: about 20
minutes (`train_mini_snapped_proxies_rare_flags_a`), 1 hour (`train_original_priors_te_ngram_a`) and 2 hours
(`train_mini_snapped_formula_priors_b`); the nested script trains 25 models.

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| tabm_mini_snapped_proxies_rare_flags_a | A | TabM-mini on the categoricals plus snapped proxies and rare flags of the 11 numerics | 0.965773 |
| tabm_original_priors_te_ngram_a | A | TabM-mini-normal on original priors / distribution / digit features and TEs of numeric values, categoricals and n-grams | 0.972872 |
| tabm_mini_snapped_formula_priors_b | B | TabM-mini on snapped proxies and rare flags, formula score and original priors, 40 epochs | 0.964803 |
| tabm_nested_binary_te_ngram_b | B | TabM-mini-normal, nested 5 x 5, per-class target means of categoricals, numeric values and n-grams | 0.972351 |

For the nested model the score is that of the mean of the 5 OOF slices.

Requirements: pytabkit (the runs used 1.7.3), torch (CUDA), scikit-learn (>= 1.3 for `TargetEncoder`), pandas, numpy.
