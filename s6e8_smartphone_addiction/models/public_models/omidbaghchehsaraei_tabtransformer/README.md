# TabTransformer by omidbaghchehsaraei (public notebook)

OMID BAGHCHEH SARAEI's (@omidbaghchehsaraei) TabTransformer from the public notebook
[TabTransformer : Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction)
(version 1). Thank you, Omid!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses). `tabtransformer.py` is the notebook (cells 1-9; the body of its fold loop is the
function `train_fold`); `train.py` runs the 5 folds.

- **Features:** the 9 numeric predictors, missing count and 25 ratio / intensity / log / sqrt features (NaN -> 0); per
  fold a frequency encoding and a smoothed target encoding (smoothing 10) of the 12 raw predictors; everything
  quantile-normalized (normal output) on the fold's training rows.
- **Model:** per-feature piecewise-linear and periodic embeddings projected to 64-d tokens plus column embeddings,
  3 transformer blocks (4 heads), an MLP on the flattened tokens plus the raw inputs, and a multi-sample dropout head.
  Mixup (alpha 0.2), label smoothing 0.005, AdamW lr 1e-3, weight decay 0.03, cosine warm restarts, mixed precision,
  EMA 0.999 of the weights, batch 256, up to 120 epochs with patience 18 on the EMA model's validation AUC; the best EMA
  weights predict. Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook encoded each fold's training rows with the map fitted on those same
rows; our run gives them 5-fold inner out-of-fold target encodings (`compute_fold_encodings` in `tabtransformer.py`;
validation and test rows keep the fold's training-row map, frequencies unchanged). The notebook seeds once (42) and
trains its folds in one loop, so `train.py` runs the 5 folds in order in one process, as in our run.

```
python train.py            # all 5 folds in order
```

Writes `preds/omidbaghchehsaraei_tabtransformer_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv,
test.csv); no other repo folder is needed (shared fold loop: `../lib/runner.py`). One GPU. Packages: torch,
scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| omidbaghchehsaraei_tabtransformer | the notebook's TabTransformer on our 5 folds | 0.967496 |
