# YDF by siukeitin (discussion post)

broccoli beef's (@siukeitin) YDF gradient boosted trees from the discussion post
[YDF gives pretty good CV score off default parameters](https://www.kaggle.com/competitions/playground-series-s6e3/discussion/679983).
Thank you, broccoli beef!

This is the post's model run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).

- **Features:** the 19 raw competition columns; YDF handles the categorical (string) columns natively. No feature
  engineering, no original rows.
- **Model:** ydf `GradientBoostedTreesLearner` (CPU) with the post's parameters: max_depth 2, `BEST_FIRST_GLOBAL`
  growth, `RANDOM` categorical splits, shrinkage 0.1, up to 10,000 trees with early stopping (300-tree look-ahead) on
  the validation fold; everything else is YDF's default. Test = mean of the fold models.

**Changed to run it on our folds:** the post checks the model with 5-fold `StratifiedKFold(random_state=0)` (scores
only); our run uses our 5 folds and predicts test with the fold models.

```
python train.py            # finished folds are kept in parts/ and skipped when the script is restarted
```

Writes `preds/siukeitin_ydf_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other repo
folder is needed (shared fold loop: `../lib/runner.py`). CPU only. Packages: ydf (our run: 0.15.0), scikit-learn,
pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| siukeitin_ydf | the post's YDF GBDT (max_depth 2, raw columns) on our 5 folds | 0.917976 |
