# RealMLP by nawfeelrahman1124444 (public notebook)

The custom RealMLP of nawfeelrahman1124444 (display name "(ノ◡ <)") from the public notebook
[RealMLP --> 0.97014](https://www.kaggle.com/code/nawfeelrahman1124444/realmlp-0-97014) (version 3).
Thank you, nawfeelrahman1124444!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).
`realmlp.py` is the notebook's code up to its global target encoding (only the two data paths changed); `train.py` is
the fold loop.

- **Features (label-free, built once on train + test):** the 12 raw predictors as factorized categories; missing count
  and 9 screen-time ratio features; log of weekend screen time and social media hours; 5 hour columns rounded to tens,
  quantile bins (40 or 50) of 5 columns and the 3 pairs of the categoricals, as categories; ones / tenths / hundredths
  digits of the 9 numeric columns and their floor / round(0.1) / mod-10 / 20ths-of-fraction values as categories;
  activity mean / std / min / max / range / argmax.
- **Encodings per fold:** the notebook's `apply_encodings_cv` gives every column a smoothed target encoding
  (smoothing 10) and a frequency encoding: 5-fold out-of-fold (seed 63) on the fold's training rows, mean of the 5
  inner maps for the validation and test rows.
- **Model:** one RealMLP with 16 vectorized ensemble members: PBLD periodic embeddings of the numeric columns, one-hot
  (up to 8 levels) or 16-d embeddings of the categoricals, 3 x 512 GELU layers, dropout 0.06; AdamW lr 0.01 with a
  flat-cos schedule, EMA 0.997875, label smoothing 0.04, balanced class weights; 4 epochs, batch 256. The validation
  probabilities of the best epoch are the OOF; the restored model predicts test. Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook computed the target / frequency encodings once on all training rows
before its CV (and split with seed 63); our run calls the same encoder inside each of our folds, on the fold's training
rows only. Every fold starts from the notebook's seed 63 (each fold ran as a separate process in our run).

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/nawfeelrahman1124444_realmlp_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv);
no other repo folder is needed (shared fold loop: `../lib/runner.py`). One GPU. Packages: torch, scikit-learn, pandas,
numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| nawfeelrahman1124444_realmlp | the notebook's RealMLP (16 members) on our 5 folds | 0.968255 |
