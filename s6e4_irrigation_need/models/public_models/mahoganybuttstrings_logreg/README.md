# Logistic regression by mahoganybuttstrings (public notebook)

The logistic regression from @mahoganybuttstrings's public notebook
[PG S6E4 - LR - \[CV 0.97652 LB 0.97487\]](https://www.kaggle.com/code/mahoganybuttstrings/pg-s6e4-lr-cv-0-97652-lb-0-97487)
(its formula indicators come from @cdeotte's notebook
[original-data-exact-formula](https://www.kaggle.com/code/cdeotte/original-data-exact-formula)). Thank you!

This is the notebook's model and features on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses).

- **Features:** 4 binary formula indicators (Soil_Moisture < 25, Temperature_C > 30, Rainfall_mm < 300,
  Wind_Speed_kmh > 10); one-hot Crop_Growth_Stage and Mulching_Used; every pairwise combination of the 19 raw columns
  that has at most half as many distinct values as there are train + test + original rows (135 pairs), factorized over
  those rows. All other raw columns are dropped (145 columns before encoding).
- **Encodings per fold:** a multiclass `TargetEncoder` (cv=5, seed 42) fitted on the fold's training rows replaces
  each pair by its 3 class encodings; `StandardScaler` fitted on the training rows.
- **Model:** scikit-learn `LogisticRegression` (multinomial, lbfgs, max_iter 10,000, balanced class weights, seed 42).
  No original rows are trained on. Test = mean of the 5 fold models.

**Changed to run it on our folds:** nothing in the model. The notebook pins scikit-learn 1.5.2 with a `pip install`,
which we leave out; the script needs a scikit-learn version that still accepts `multi_class` (it was removed in 1.8).

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/mahoganybuttstrings_logreg_oof.npy` and `preds/mahoganybuttstrings_logreg_test.npy` (probabilities in
class order Low, Medium, High). Needs only `data/` (train.csv, test.csv, irrigation_prediction.csv) and `common/`
(shared fold loop: `../lib/runner.py`). CPU only. Packages: scikit-learn (>= 1.4, < 1.8), pandas, numpy.

The pair is a member of stack B (stack A and stack B are the two stacks of `stack/`).

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| mahoganybuttstrings_logreg | B | multinomial LR on formula indicators + one-hot + 135 pair target encodings | 0.976519 |
