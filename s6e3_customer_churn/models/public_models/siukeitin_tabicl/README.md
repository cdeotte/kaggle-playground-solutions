# TabICL by siukeitin (discussion comment)

broccoli beef's (@siukeitin) TabICL voting ensemble from
[a comment](https://www.kaggle.com/competitions/playground-series-s6e3/discussion/681229#3422698) on yunsuxiaozi's
(@yunsuxiaozi) discussion post
[TabICL baseline CV:0.91385,LB:0.91067](https://www.kaggle.com/competitions/playground-series-s6e3/discussion/681229),
which introduced TabICL to this competition. Thank you both!

This is the comment's model run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).

- **Features:** the 19 raw competition columns (TabICL encodes the string columns itself). No feature engineering, no
  original rows.
- **Model:** TabICL, a pretrained tabular foundation model that predicts by in-context learning (no gradient training:
  the fold's training rows are the context). A voting ensemble of 8 `TabICLClassifier` models (`n_estimators=2` each,
  so 16 ensemble members in all; `kv_cache=True`; `random_state` 0-7), each given all of the fold's training rows; the
  models run one after another on one GPU (freeing GPU memory in between) and their probabilities are averaged. Test =
  mean of the fold models.

**Changed to run it on our folds:** the comment cross-validates 4 voting members with `StratifiedKFold(5,
random_state=0)` (scores only); our run uses 8 members, our 5 folds and predicts test with the fold models.

```
python train.py            # finished folds are kept in parts/ and skipped when the script is restarted
```

Writes `preds/siukeitin_tabicl_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other repo
folder is needed (shared fold loop: `../lib/runner.py`). One GPU with plenty of memory (the full fold of about 475,000
rows is the context). The first run downloads the library's default TabICLv2 classifier checkpoint from the Hugging
Face Hub. Packages: tabicl, torch, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| siukeitin_tabicl | TabICL voting ensemble (8 x 2 members, raw columns, all fold training rows as context) | 0.915195 |
