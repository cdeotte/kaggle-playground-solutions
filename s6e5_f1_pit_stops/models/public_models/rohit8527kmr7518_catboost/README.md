# CatBoost by rohit8527kmr7518 (public notebook)

Rohit Kumar's (@rohit8527kmr7518) CatBoost from the public notebook
[PS S6E5: CatBoost 10-Fold CV](https://www.kaggle.com/code/rohit8527kmr7518/ps-s6e5-catboost-10-fold-cv). We ran
version 6, published then as `rohit8527kmr7518/ps-s6e5-catboost`, which trained one model on an 80/20 holdout split
(later versions train 10 folds). Thank you, Rohit!

This is version 6's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).

- **Features:** an `IsOriginalData` flag; 38 domain features (estimated total laps, laps remaining, tyre-age ratios,
  degradation and lap-time delta per tyre / race lap, pit-window / position / stint pressures, products, sign flags,
  early / mid / late race flags, race-phase / lap / tyre-life / position bins) and 9 string crosses (Race_Year,
  Compound_Stint, Driver_Race, Driver_Compound, Race_Compound, Race_Compound_Stint, Compound_RacePhase,
  Compound_TyreLifeBin, RacePhase_TyreLifeBin); integer digits 1-3 and decimal digits 1-2 of 22 numeric columns;
  digits 1-5 of round(|x| x 100) of 13 float columns as string categories (kept in a frame only where they vary);
  rounded-string copies of RaceProgress, EstimatedTotalLaps and TyreAgeRatio.
- **Statistics per fold** (over the fold's training rows + test + all original rows): count and frequency of 15
  categorical columns; mean, std and difference-from-mean of LapTime_Delta, Position_Change, RaceProgress and TyreLife
  within Race_Year, Race_Compound_Stint, Driver_Race and Compound_Stint; missing values filled with the mode (string
  columns) or median (numeric columns) of those rows. 301 features, the 77 string columns are CatBoost categoricals.
- **Original rows:** all 101,371 original rows are appended to each fold's training rows.
- **Model:** CatBoost on one GPU, Logloss, up to 9,000 iterations, lr 0.033, depth 8, l2_leaf_reg 7, random_strength
  0.8, Bayesian bootstrap (bagging temperature 0.55), balanced class weights, early stopping 350 on the validation AUC
  with the best model kept, seed 43 + fold. Predictions clipped to [1e-6, 1 - 1e-6]; test = mean of the 5 fold models.

**Changed to run it on our folds:** version 6 early-stopped one model on a stratified 80/20 holdout (seed 42) and
predicted test with a refit on all training + original rows (1.08 x the best iteration, at least 1,800, seed 1041); it
built the frequency / group statistics and fill values on all training + test + original rows (after downcasting the raw
columns) and used two GPUs. Our run trains one model per fold (early stopping on the fold's validation rows), builds the
statistics and fill values on the fold's training rows + test + original rows (raw columns not downcast), and averages
the 5 fold models for test on one GPU.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/rohit8527kmr7518_catboost_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv,
f1_strategy_dataset_v4.csv); no other repo folder is needed (shared fold loop: `../lib/runner.py`). One GPU. Packages:
catboost, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| rohit8527kmr7518_catboost | version 6's CatBoost (301 features, original rows appended) on our 5 folds | 0.952703 |
