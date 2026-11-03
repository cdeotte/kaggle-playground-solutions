# RealMLP (pytabkit) by yekenot (public notebook)

Vladimir Demidov's (@yekenot) pytabkit RealMLP from the public notebook
[PS|S6|E5: RealMLP · PyTabKit](https://www.kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytabkit) (versions 11 and 16).
Thank you, Vladimir!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses, for the competition rows and for the original rows). One script runs both versions.

- **Features (label-free):** every numeric column floored and factorized as a string category; LapNumber / RaceProgress
  and TyreLife / LapNumber; 200 quantile bins of RaceProgress; Race x Compound and Race x Year combo categories.
  Version 16 adds LapTime x Cumulative_Degradation (signed and absolute) and LapTime / |Cumulative_Degradation|, the two
  ratios floored as categories, counts of Driver / Compound / Race / Year / PitStop and 7 quantile bins of LapTime.
- **Original rows:** the fold-matched 4/5 of the original dataset (the same `StratifiedKFold` on its own target; outer
  fold k appends the original rows whose fold is not k) is appended to each fold's training rows.
- **Encodings per fold:** a scikit-learn `TargetEncoder` (cv=5, smooth 'auto', shuffle, seed 42) of the two combo
  columns, fitted on the training rows (competition + original).
- **Model:** pytabkit `RealMLP_TD_Classifier` with the notebook's parameters: 512-256-128 SiLU layers, PLR embeddings
  (16 / 8, sigma 2.33), one-hot up to 18 levels, batch 256, `lin_cos_log_15` schedule, label smoothing 0.01, no early
  stopping, seed 42. Version 11: 8 members, 4 epochs, lr 0.03, wd 0.018. Version 16: 24 members, 6 epochs, lr 0.01,
  wd 0.016. Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook fits its feature engineering (the category maps, quantile bins, combos
and counts) once on all competition training rows; our run fits it on each fold's competition training rows and applies
it to that fold's validation, test and original rows (values unseen in the fold's training rows map to -1). Each fold
re-seeds the global random generators with 43 + fold (the notebook seeds 42 once).

```
python train.py --version 11    # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
python train.py --version 16
```

Writes `preds/yekenot_realmlp_pytabkit_v11_{oof,test}.npy` and `preds/yekenot_realmlp_pytabkit_v16_{oof,test}.npy`
(probabilities). Reads only `data/` (train.csv, test.csv, f1_strategy_dataset_v4.csv); no other repo folder is needed
(shared fold loop: `../lib/runner.py`). One GPU. Packages: pytabkit, torch, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| yekenot_realmlp_pytabkit_v11 | version 11: RealMLP, 8 members, 4 epochs | 0.953258 |
| yekenot_realmlp_pytabkit_v16 | version 16: + degradation arithmetic, counts and LapTime bins; 24 members, 6 epochs | 0.954088 |
