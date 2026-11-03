# Lattice target encoding XGBoost by szymonkapiski (public notebook)

Szymon Kłapiński's (@szymonkapiski) XGBoost from the public notebook
[S6E8 full lattice target encoding with XGBoost](https://www.kaggle.com/code/szymonkapiski/s6e8-full-lattice-target-encoding-with-xgboost)
(version 1). Thank you, Szymon!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses). `lattice.py` is the notebook's feature and encoder code (cells 3-9, only the data-path
search removed); `train.py` is its training cell.

- **Features (label-free, train + test):** the 9 numeric predictors; constrained imputation of the screen-time block
  (an iterative imputer clipped to the bounds of daily >= social + gaming + work_study: imputed values, bounds, bound
  widths, missing flags, reconstructed other-screen time and component share, number missing); category codes; 3 domain
  relations.
- **Lattice keys:** every predictor at full resolution, every numeric predictor rounded to 0.1 and floored, and 37 pair
  cells (all 36 numeric pairs and stress level x academic impact) at floor and at 0.1 resolution: 104 keys.
- **Encodings per fold:** every key gets a smoothed target encoding (smoothing 20) and its cell count; inner 4-fold
  out-of-fold (seed 143) for the training rows, the fold's training-row map for validation and test.
- **Model:** XGBoost, lr 0.03, depth 8, min_child_weight 10, subsample 0.85, colsample_bytree 0.7, lambda 2, hist,
  seed 42, up to 3,000 rounds with early stopping 100 on the validation AUC (best iterations 991 to 1,289 in our run).
  Test = mean of the 5 fold models.

**Changed to run it on our folds:** nothing in the model or features. The notebook loads its saved predictions by
default (`RUN_TRAINING = False`); we run its training code.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/szymonkapiski_lattice_te_xgb_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv);
no other repo folder is needed. CPU, 12 threads (`THREADS`). Packages: xgboost, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| szymonkapiski_lattice_te_xgb | the notebook's lattice-TE XGBoost on our 5 folds | 0.967663 |
