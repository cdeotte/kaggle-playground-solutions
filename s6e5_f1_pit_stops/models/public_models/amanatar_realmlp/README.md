# RealMLP (pytabkit) by amanatar (public notebook)

Aman Atar's (@amanatar) pytabkit RealMLP ("Pipeline A") from the public notebook
[3-Model Ensemble: RealMLP + CatBoost + XGBoost](https://www.kaggle.com/code/amanatar/3-model-ensemble-realmlp-catboost-xgboost)
(version 1). Thank you, Aman! The pipeline builds on the feature engineering and parameters of
[PS|S6|E5: RealMLP · PyTabKit](https://www.kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytabkit).

This is the notebook's RealMLP and its features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`,
the same split the notebook uses, for the competition rows and for the original rows). Only the RealMLP is used; the
notebook's CatBoost and XGBoost pipelines and its optimized blend are not.

- **Features (label-free, fitted once on all competition training rows):** LapNumber / RaceProgress and TyreLife /
  LapNumber; every numeric column and the two ratios floored and factorized as string categories; counts of Driver /
  Compound / Race / Year / PitStop; 200 quantile bins of RaceProgress; Race x Compound and Race x Year combos.
- **Original rows:** the fold-matched 4/5 of the original dataset (the same `StratifiedKFold` on its own target) is
  appended to each fold's training rows.
- **Encodings per fold:** a scikit-learn `TargetEncoder` (cv=5, smooth 'auto', shuffle, seed 42) of the two combo
  columns, fitted on the training rows (competition + original).
- **Model:** pytabkit `RealMLP_TD_Classifier`: 8 members, 512-256-128 SiLU layers, PLR embeddings (16 / 8, sigma
  2.33), one-hot up to 18 levels, batch 256, 4 epochs, lr 0.03 (`lin_cos_log_15`), wd 0.018, label smoothing 0.01, no
  early stopping, seed 42. The notebook seeds once (42) and trains its folds in one loop, so `train.py` runs the 5 folds
  in order in one process. Test = mean of the 5 fold models.

**Changed to run it on our folds:** nothing in the model or features; OOF and test are clipped to [1e-7, 1 - 1e-7].

```
python train.py            # all 5 folds in order
```

Writes `preds/amanatar_realmlp_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv,
f1_strategy_dataset_v4.csv); no other repo folder is needed (shared fold loop: `../lib/runner.py`). One GPU. Packages:
pytabkit, torch, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| amanatar_realmlp | the notebook's RealMLP (Pipeline A) on our 5 folds | 0.953442 |
