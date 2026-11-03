# RealMLP by zhenruiweng (public notebook)

Zhenrui.Weng's (@zhenruiweng) custom RealMLP from the public notebook
[RealMLP for Predicting Smartphone Addiction](https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction)
(version 1). Thank you, Zhenrui!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses). `realmlp.py` is the notebook's code (cells 6-22, unchanged); `train.py` is its `main()`
(cell 24).

- **Features (label-free):** missing-value indicators of the 9 numeric predictors; numeric NaN -> train median; the 3
  categoricals (missing -> 'missing') and every numeric value factorized as categories; 10 quantile bins of daily screen
  time and social media hours.
- **Encodings per fold:** a scikit-learn `TargetEncoder` (cv=5, smooth 'auto', shuffle, seed 42) of all categorical
  columns except the bins, fitted on the fold's training rows.
- **Model:** RealMLP with 8 vectorized ensemble members: PBLD periodic numeric embeddings, one-hot (up to 8 levels) or
  8-d embeddings of the categoricals, 3 x 512 SiLU layers, dropout 0.06; AdamW lr 0.01 with a flat-cos schedule, EMA
  0.997875, label smoothing 0.04, balanced class weights; 8 epochs, batch 256. The validation probabilities of the best
  epoch are the OOF; the restored model predicts test. Test = mean of the 5 fold models.

**Changed to run it on our folds:** nothing in the model or features; `train.py` takes the folds from our fold
assignment (the same split) and keeps the predictions in memory instead of writing CSV files. The notebook seeds once
(42) and trains its folds in one loop, so `train.py` runs the 5 folds in order in one process.

```
python train.py            # all 5 folds in order
```

Writes `preds/zhenruiweng_realmlp_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other
repo folder is needed (shared fold loop: `../lib/runner.py`). One GPU. Packages: torch, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| zhenruiweng_realmlp | the notebook's RealMLP (8 members) on our 5 folds | 0.968180 |
