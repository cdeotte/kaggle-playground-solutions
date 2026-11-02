# LightGBM by kirill0212 (public notebook)

Credit: kirill0212, Kaggle notebook [s6e9-lightgbm](https://www.kaggle.com/code/kirill0212/s6e9-lightgbm) (version 2).
This is the notebook's model (its active `model_type = 'lgb'` path) run on our 5 folds
(`StratifiedKFold(5, shuffle=True, random_state=42)`, the same split the notebook uses), with the fold predictions saved
as an OOF / test pair. The features, encoders and parameters are the author's.

- **Features:** the 13 raw columns (categoricals mapped to integers), the digits of every column at the 10^-4 ... 10^3
  places (constant and duplicate columns dropped) and the frequency of every raw value over train + test: 49 label-free
  columns.
- **Target encodings per fold:** two catstat `TargetEncoder` mean views of the 13 raw columns (smooth 10 and 'auto'),
  5-fold cross-fitted on the training rows: 75 features.
- **Model:** LightGBM (learning rate 0.02, depth 5, 247 leaves, max_bin 1024, column sample 0.30) with the notebook's early
  stopping (patience 500); the best iterations were 1,182 to 1,867 in our run. Test = mean of the 5 fold models.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/lgbm_kirill0212_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other model
folder is needed. CPU only: 24 LightGBM threads (`n_jobs` in `PARAMS`), about a minute per fold.
Needs `lightgbm` and `catstat` (`pip install catstat==0.5.2`). We used lightgbm 4.6.0, catstat 0.5.2, scikit-learn 1.8.0,
pandas 2.3.3 and numpy 2.2.6.

| Pair | OOF AUC |
|---|---:|
| lgbm_kirill0212 | 0.945832 |
