# heuljax XGB (public notebook)

Paul Bryan Elefante's (@heuljax) XGBoost from his public notebook
[KPS6E09 XGB Sample](https://www.kaggle.com/code/heuljax/kps6e09-xgb-sample?scriptVersionId=352387662). Thank you, Paul!

This is his model and his 173 features (raw values, original-data income priors, digits, and fold-local income / commute
target encodings, GAM coordinates with a GAM base margin, an income-group XGB prior, gate mixtures and composition-curve
features), run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`) instead of his 10. The round count
comes from a stratified 10% holdout of the training folds (features rebuilt on the other 90%, early stopping with
patience 750); the model is then refit on all training rows for 1.1x the best round count. Test = mean of the 5 fold
models.

```
python train.py        # -> preds/heuljax_xgb_{oof,test}.npy
```

No other repo folders are needed. Packages: xgboost (GPU), numba, scipy, threadpoolctl.

| Pair | Model | OOF AUC |
|---|---|---:|
| heuljax_xgb | his XGBoost on our 5 folds | 0.946190 |
