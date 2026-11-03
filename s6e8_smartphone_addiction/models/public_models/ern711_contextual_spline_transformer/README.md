# Contextualized spline transformer by ern711 (public notebook)

Ern711's (@ern711) neural network from the public notebook
[Contextualized Deep Univariate Spline Transformer](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer)
(version 3). Thank you, Ern711!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).
`spline_transformer.py` is the notebook's code up to its cross-validation loop (only the two data paths changed);
`train.py` runs that loop on our folds through the notebook's own `train_one_fold`.

- **Features:** the 9 numeric predictors, missing count and 9 screen-time ratios; per fold, nested target and
  frequency encodings of the 12 raw predictors as strings (inner 5-fold out-of-fold for the training rows, smoothing 10;
  validation / test rows get the mean of the inner maps); numerics standardized on the training rows; the 12 raw
  predictors and 9 missing flags as embedded categoricals.
- **Model:** a small nonlinear context network adjusts each standardized numeric value, every numeric feature is then
  encoded on its own (multi-resolution spline, MLP and raw-value experts with quantile knots from the training rows),
  one transformer block mixes the feature tokens and the final classifier predicts from the full representation (an
  additive head is trained alongside). AdamW lr 7e-4, batch 4096, up to 35 epochs with patience 7 on the validation AUC,
  seed 21 + fold. The final head's predictions are used; test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook splits with `StratifiedKFold(5, shuffle=True, random_state=21)`; our run
uses our folds. Nothing else changed.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/ern711_contextual_spline_transformer_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv,
test.csv); no other repo folder is needed. One GPU, about 5 to 6 hours per fold in our run (folds can run in parallel
processes). Packages: torch, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| ern711_contextual_spline_transformer | the notebook's contextualized spline transformer on our 5 folds | 0.966833 |
