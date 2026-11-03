# gcForest cascade on nested target encodings

`train.py` -> pair `gcforest_rf_et_te`: the cascade part of gcForest (Deep Forest;
`lib/gcforest.py`). Every level fits a RandomForest (seed 777 + fold) and an ExtraTrees (seed 778 + fold), 200 trees
each, min_samples_leaf 5, max_features 0.5, full depth, on [features, class-probability vectors of all previous levels]
(in-sample vectors for the training rows). The cascade grows (at most 5 levels, patience 1) on 90% of the outer-train
rows, the first split of `StratifiedKFold(10, shuffle=True, random_state=777 + fold)`; the AUC on the other 10% picks the
depth. The prediction is the mean P(y=1) of the best level's two forests. There is no refit on all outer-train rows.

Features (`lib/features.py`, 86 columns):
- the 9 raw numerics and 23 label-free derived columns: budget slack `daily - (social + gaming + work)`, component sum,
  utilization `sum / daily`, their cent parts, exact-budget flag, `sleep + daily`, at-20 flag, the 5 per-daily-hour
  ratios, `weekend - daily`, `weekend / daily`, 3 products, repair flag (utilization in [0.85, 0.95]), ordinal codes of
  the 3 categoricals;
- 18 original-data rates: per-value label rate and 'Severe' addiction-level rate of each numeric in the 7,500-row
  original dataset;
- these 50 columns are median-imputed (fit on train + test);
- 36 nested target encodings of exact value tokens (`%.2f`, NaN -> `__NA__`; smoothing m = 30 for the 12 columns and
  slack / sum / sleep + daily, 50 for utilization, 10 and 200 for 5 key columns, 100 for 10 column pairs): outer-train
  rows inner 5-fold cross-fitted (seed 777), held-out and test rows from all outer-train rows.

Test = mean of the 5 fold models. Each fold is saved in `parts/` (finished folds are skipped); the pair is written to
`preds/gcforest_rf_et_te_{oof,test}.npy` (probabilities) when all 5 folds are done.

```
python train.py        # --folds 0,1 runs a subset of the folds
```

No other repo folders are needed.

| Pair | Model | OOF AUC |
|---|---|---:|
| gcforest_rf_et_te | gcForest cascade (RandomForest + ExtraTrees, 200 trees each) on raw + derived + original rates + nested TEs | 0.963966 |

Packages: scikit-learn, pandas, numpy.
