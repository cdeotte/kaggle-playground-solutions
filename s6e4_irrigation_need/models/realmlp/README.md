# RealMLP: pytabkit RealMLP models and a nested PyTorch RealMLP

Eleven RealMLP models, one script each: ten pytabkit `RealMLP_TD_Classifier` models and one PyTorch RealMLP with 8
vectorized members in nested 5 x 5 cross-validation. Stack A and stack B are the two stacks of `stack/`; the pair names
end in `_a` / `_b` accordingly. Unless noted: folds = the 5 folds of `common/data.py`, the fold's validation rows are
passed to `fit` as the validation set, training rows = competition rows only (the original dataset only feeds
features), test = mean of the 5 fold models.

Credit: `train_torch_nested_combo_te_b.py` adapts yunsuxiaozi's public notebook
[RealMLP from scratch](https://www.kaggle.com/code/yunsuxiaozi/realmlp-from-scratchcv-0-91908).

## Stack A

- **`train_3config_digits_bins_rare_original_a.py`**: for each of the 11 numerics a NaN flag, the base-10 digits of the
  value rounded to its step and a sign flag; quantile (4, 5, 10, 20), equal-width (5, 10, 20) and rounded bins; every
  raw column as a (rounded) string category; rare flags (< 25 rows) of the rounded numerics. The training rows'
  features are computed over the fold's training + original + test rows, the validation / test rows' features in a
  second pass that includes the validation rows. All 10,000 original rows are appended to every fold's training rows.
  Three tuned RealMLP configurations (mish, PBLD embeddings, n_ens 8, up to 32 epochs, batch 1024), averaged.
- **`train_digit_cats_crosses_rank_priors_a.py`**: 9 water / soil row features; last digit, mod 100, two decimals and
  1-decimal value of 6 numerics as categories; every numeric rounded to its step as a category; quantile bins of 4
  numerics; 5 categorical pairs and Crop_Type x Season x Irrigation_Type; frequency and percentile rank over
  train + test + original; per-class original rates of the 8 categoricals. Original rows appended. RealMLP silu
  [512, 256, 128], n_ens 8, lr 0.075 (flat_anneal), batch 256, 3 epochs.
- **`train_pair_te_logit_priors_drift_a.py`**: label-free block over train + test + original (frequency, rank, log,
  sqrt, inverse of the numerics; water formulas; practice flags; original-quantile bins; 6 pairs + 1 triple with counts
  and rare flags; per-class original rates of the 19 columns; train-vs-original drift; digits, bins and rounded
  categories of 5 numerics; ranks of Previous_Irrigation_mm in the original classes) plus, per fold, smoothed (20)
  multiclass target encodings of the 28 categorical pairs (inner 5 folds, seed 42 + fold) as logit, logit^2, logit^3,
  standardized. Original rows appended (pair encodings 0). RealMLP 2 x 512 mish, PBLD, p_drop 0.42, n_ens 8, 30 epochs.
- **`train_nested_te_stats_ngram_a.py`**: frequencies, 5 interactions, 3 practice flags, original target means per
  value, Previous_Irrigation_mm against the original class distributions (percentile ranks, z-gap, quartile distances),
  digit features of 4 numerics, 6 numerics rounded to 2 decimals as categories, 15 bigrams + 4 trigrams; per fold,
  nested (5 inner folds) label std / min / max and Laplace-smoothed class probabilities of the 14 category keys and
  class probabilities of the 19 n-grams. RealMLP 4 x 512 mish, n_ens 1, lr 0.13, up to 64 epochs with early stopping.
- **`train_kbins_value_codes_combo_te_a.py`**: 7 ratios / products, digit flags, KBinsDiscretizer quantile and k-means
  bins of Rainfall_mm (5000, 400) and Previous_Irrigation_mm (300), every numeric factorized as a category, the
  Crop_Type x Season x Irrigation_Type code and its per-fold scikit-learn `TargetEncoder` (3 columns). RealMLP settings
  of `train_digit_cats_crosses_rank_priors_a.py`.
- **`train_snapped_proxies_rare_flags_a.py`**: the 19 raw columns plus, for Rainfall_mm, Previous_Irrigation_mm and
  Soil_Moisture, a per-fold snapper (values seen < 25 times in the fold's training rows -> nearest frequent value): the
  snapped value and the rare flag as categories (25 columns). RealMLP 3 x 384 mish, n_ens 8, lr 0.03, up to 80 epochs
  with early stopping.

## Stack B

- **`train_3config_digits_bins_formula_original_b.py`**: formula score (Soil_Moisture < 25, Rainfall_mm < 300,
  Temperature_C > 30, Wind_Speed_kmh > 10, growth stage Harvest / Sowing, mulching) and two flags; the digit / bin /
  rounded-category features above (other rounding steps), rare flags of 4 numerics; computed once over train +
  original + test. Folds: `StratifiedKFold(5, shuffle=True, random_state=42)` over the 640,000 competition + original
  rows (not the common folds); a fold trains on the rows of the 4 other folds and is validated on its competition rows.
  The three RealMLP configurations of the stack A model, averaged.
- **`train_digits_pairs_priors_drift_b.py`**: 3 integer digits and the first decimal of the 11 numerics, 4 numerics
  rounded to 1 and 5, 6 categorical pairs, formula score, per-class original rates of the 8 categoricals and 6 pairs,
  frequencies, counts / rare flags, drift. RealMLP 2 x 512 mish, PBLD, p_drop 0.42, n_ens 8, 30 epochs.
- **`train_nested_te_stats_ngram_formula_b.py`**: frequencies, 3 ratios, formula score, per-class original rates,
  percentile ranks of the 11 numerics within the original classes, mod 10 / mod 100 / fraction, the numeric values as
  categories, 15 bigrams + 4 trigrams; per fold, nested (5 inner folds) label std / min / max of the 19 category keys
  and label means of the 19 n-grams. RealMLP 4 x 512 mish, n_ens 1, early stopping on the validation cross-entropy.
- **`train_formula_original_priors_b.py`**: the 19 raw columns, formula score and flags, per-class original rates of
  the 8 categoricals (46 columns). RealMLP 3 x 384 mish, n_ens 8, lr 0.03, up to 80 epochs with early stopping.
- **`train_torch_nested_combo_te_b.py`**: PyTorch RealMLP: 19 numerics (raw, ratios, decimals, mods) and 31 integer
  codes (categoricals, digit flags, KBins bins, factorized numerics, Crop_Type x Season x Region); per inner fold a
  multiclass `TargetEncoder` of the codes with more than 8 levels and robust scaling + smooth clipping; PBLD numeric
  embeddings, one-hot / 6-d code embeddings, NTP layers 512 -> 64 -> 128 (GELU, dropout 0.05), 8 members, AdamW with
  flat-anneal schedule, label smoothing 0.04, balanced class weights, batch 4096, 5 epochs (best inner-validation
  epoch). Nested 5 x 5: outer folds = the common folds, inner `StratifiedKFold(5, shuffle=True, random_state=1000 +
  outer fold)`. The OOF has shape `(5, 630000, 3)`: slice k = the inner-validation predictions on outer fold k's
  training rows + the mean of its 5 inner models on its held-out rows. Test = mean of the 25 models.

```
python train_3config_digits_bins_rare_original_a.py
python train_digit_cats_crosses_rank_priors_a.py
python train_pair_te_logit_priors_drift_a.py
python train_nested_te_stats_ngram_a.py
python train_kbins_value_codes_combo_te_a.py
python train_snapped_proxies_rare_flags_a.py
python train_3config_digits_bins_formula_original_b.py
python train_digits_pairs_priors_drift_b.py
python train_nested_te_stats_ngram_formula_b.py
python train_formula_original_priors_b.py
python train_torch_nested_combo_te_b.py
```
Each script runs its folds in one process on one GPU and writes `preds/<pair>_{oof,test}.npy` (probabilities, class
order Low, Medium, High). Only `data/` is read; no other repo folder has to run first. Logged run times: about 20
minutes (`train_snapped_proxies_rare_flags_a`, `train_formula_original_priors_b`), 40 minutes (both
`train_nested_te_stats_*`) and 1 hour (`train_digits_pairs_priors_drift_b`).

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| realmlp_3config_digits_bins_rare_original_a | A | 3 RealMLP configurations on digit / bin / rounded-category / rare features, original rows appended | 0.968247 |
| realmlp_digit_cats_crosses_rank_priors_a | A | RealMLP on digit categories, crosses, rank / frequency and original priors, original rows appended | 0.971830 |
| realmlp_pair_te_logit_priors_drift_a | A | RealMLP on 28 pairwise multiclass TEs (logit powers), original priors and drift, original rows appended | 0.972210 |
| realmlp_nested_te_stats_ngram_a | A | RealMLP with nested label-stat / class-probability TEs of categories, numeric values and n-grams | 0.971327 |
| realmlp_kbins_value_codes_combo_te_a | A | RealMLP on KBins bins, numeric value codes and a Crop_Type x Season x Irrigation_Type TE | 0.971071 |
| realmlp_snapped_proxies_rare_flags_a | A | compact RealMLP on the raw columns plus snapped proxies and rare flags of 3 numerics | 0.966372 |
| realmlp_3config_digits_bins_formula_original_b | B | 3 RealMLP configurations on formula / digit / bin / rounded-category features, folds over competition + original rows | 0.965882 |
| realmlp_digits_pairs_priors_drift_b | B | RealMLP on digits, roundings, 6 pairs, formula score, original priors, counts and drift | 0.967651 |
| realmlp_nested_te_stats_ngram_formula_b | B | RealMLP with nested label std / min / max TEs and n-gram means, formula score, original priors and ranks | 0.971956 |
| realmlp_formula_original_priors_b | B | RealMLP on the raw columns, formula score and per-class original priors | 0.960693 |
| realmlp_torch_nested_combo_te_b | B | PyTorch RealMLP (8 members), nested 5 x 5, inner-fold TEs of the high-cardinality codes | 0.978822 |

For the nested model the score is that of the mean of the 5 OOF slices.

Requirements: pytabkit (the runs used 1.7.3), torch (CUDA; the runs used 2.11), scikit-learn (>= 1.3 for
`TargetEncoder`), pandas, numpy.
