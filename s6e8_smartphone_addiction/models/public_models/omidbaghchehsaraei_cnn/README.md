# CNN by omidbaghchehsaraei (public notebook)

OMID BAGHCHEH SARAEI's (@omidbaghchehsaraei) tabular CNN from the public notebook
[CNN for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/cnn-for-predicting-smartphone-addiction)
(version 1). Thank you, Omid!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses). `cnn.py` is the notebook (cells 1-9; the body of its fold loop is the function
`train_fold`); `train.py` runs the 5 folds.

- **Features:** the 9 numeric predictors, missing count and 25 ratio / intensity / log / sqrt features (NaN -> 0); per
  fold a frequency encoding and a smoothed target encoding (smoothing 10) of the 12 raw predictors; everything
  quantile-normalized (normal output) on the fold's training rows.
- **Model:** per-feature piecewise-linear and periodic embeddings, a 1-D CNN over the feature axis (two conv layers,
  squeeze-excitation, average and max pooling), an MLP on the pooled features plus the raw inputs, and a multi-sample
  dropout head. AdamW lr 0.002, weight decay 0.015, cosine warm restarts, label smoothing 0.01, mixed precision,
  batch 128, up to 100 epochs with patience 15 on the validation AUC; the best epoch's weights predict.
  Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook encoded each fold's training rows with the map fitted on those same
rows; our run gives them 5-fold inner out-of-fold target encodings (`compute_fold_encodings` in `cnn.py`; validation and
test rows keep the fold's training-row map, frequencies unchanged). The notebook seeds once (42) and trains its folds in
one loop, so `train.py` runs the 5 folds in order in one process, as in our run.

```
python train.py            # all 5 folds in order
```

Writes `preds/omidbaghchehsaraei_cnn_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no
other repo folder is needed (shared fold loop: `../lib/runner.py`). One GPU. Packages: torch, scikit-learn, pandas,
numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| omidbaghchehsaraei_cnn | the notebook's tabular CNN on our 5 folds | 0.967646 |
