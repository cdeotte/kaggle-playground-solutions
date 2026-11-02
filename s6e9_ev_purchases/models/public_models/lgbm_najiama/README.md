# LightGBM by najiama (public notebook)

Credit: najiama, Kaggle notebook
[pure-lgbm-model-cv-0-94606-lb-0-94637](https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94606-lb-0-94637)
(version 1). This is the notebook's model run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the same
split the notebook uses), with the fold predictions saved as an OOF / test pair. The features, encoders and parameters are
the author's.

- **Features:** `Number_of_Cars_Owned` is dropped. The digits of the 6 other numeric columns at the 10^-4 ... 10^3 places;
  for each of the 12 raw columns, the mean target of that value in the original 10,000-row dataset; every numeric and
  digit column also as a string category, with its frequency over train + test; 4 fixed rules on income and environmental
  concern. Constant and duplicate columns are dropped: 105 columns.
- **Target encodings per fold:** two scikit-learn `TargetEncoder`s (smooth 'auto' and 10, 5-fold cross-fitted on the
  training rows) replace the 37 string columns: 142 features.
- **Model:** LightGBM (learning rate 0.005, depth 5, 32 leaves, max_bin 1024, column sample 0.3) with the notebook's early
  stopping (patience 700); the best iterations were 4,814 to 7,381 in our run. Test = mean of the 5 fold models.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/lgbm_najiama_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv and the original
`EV_Adoption_and_Range_Anxiety_Dataset.csv`); no other model folder is needed. CPU only: 24 LightGBM threads (`n_jobs` in
`PARAMS`), 2 to 8 minutes per fold. We used lightgbm 4.6.0, scikit-learn 1.8.0, pandas 2.3.3 and numpy 2.2.6.

| Pair | OOF AUC |
|---|---:|
| lgbm_najiama | 0.945870 |
