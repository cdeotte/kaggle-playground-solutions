# Paul's logistic regression (public notebook)

Paul Bryan Elefante's (@heuljax) logistic regression from his public notebook
[KPS6E09 Logistic Regression Sample](https://www.kaggle.com/code/heuljax/kps6e09-logistic-regression-sample?scriptVersionId=353829637).
Thank you, Paul!

This is his model and his features (229 features, 300 columns after one-hot coding; sklearn LogisticRegression, lbfgs,
C 0.1, max_iter 250, tol 2e-4), run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`) instead of his
10. His notebook code runs as published, on CPU sklearn instead of cuml.accel. Outer-train rows get their label-derived
features from 5 inner folds, valid / test rows from the state fitted on all outer-train rows. Test = mean of the 5 fold
models.

```
python train.py        # -> preds/paul_lr_{oof,test}.npy   (CPU)
```

No other repo folders are needed. Packages: scikit-learn, numba, scipy, threadpoolctl.

| Pair | Model | OOF AUC |
|---|---|---:|
| paul_lr | his logistic regression on our 5 folds | 0.946152 |
