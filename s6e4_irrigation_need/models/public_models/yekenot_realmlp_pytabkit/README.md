# RealMLP (pytabkit) by yekenot (public notebook)

Vladimir Demidov's (@yekenot) pytabkit RealMLP from the public notebook
[PS|S6|E4: RealMLP · PyTabKit](https://www.kaggle.com/code/yekenot/ps-s6-e4-realmlp-pytabkit). Thank you, Vladimir!

This is the notebook's model and features on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses).

- **Features (label-free part fitted once on all competition training rows, as in the notebook):** every numeric
  column floored and factorized as a category; the decimal part (rounded to 0.01) of Temperature_C, Organic_Carbon,
  Soil_Moisture, Soil_pH and Sunlight_Hours; for Organic_Carbon, Soil_Moisture, Rainfall_mm and Previous_Irrigation_mm
  the mean target of the original rows with the same value (the overall original mean where there is none) and 300
  quantile bins of it as a category; 6 combo categories (Soil_Type x Soil_pH, Crop_Type x Electrical_Conductivity,
  Crop_Type x Humidity, Crop_Growth_Stage x Rainfall_mm, Season x Organic_Carbon, Water_Source x Organic_Carbon).
- **Encodings per fold:** a scikit-learn `TargetEncoder` (cv=5, smooth 'auto', shuffle, seed 42) of the 6 combos,
  fitted on the fold's training rows (3 class columns per combo). No original rows are appended.
- **Model:** pytabkit `RealMLP_TD_Classifier` with the notebook's parameters: 512-256-128 SiLU layers, PLR embeddings
  (16 / 8, sigma 2.33), one-hot up to 18 levels, 8 ensemble members, 4 epochs, batch 256, lr 0.05 with `cos_anneal`,
  wd 0.0236, label smoothing 0.01, early stopping on the validation fold's balanced accuracy, seed 42. The global
  random generators are seeded with 42 once and the 5 folds run in order in one process, as in the notebook. Test =
  mean of the 5 fold models.

**Changed to run it on our folds:** nothing; the notebook already uses our split and averages its 5 fold models for
test. Only the file paths and the outputs differ.

```
python train.py            # the 5 folds in one process
```

Writes `preds/yekenot_realmlp_pytabkit_oof.npy` and `preds/yekenot_realmlp_pytabkit_test.npy` (probabilities in class
order Low, Medium, High). Needs only `data/` (train.csv, test.csv, irrigation_prediction.csv) and `common/` (shared fold
loop: `../lib/runner.py`). One GPU. Packages: pytabkit, torch, scikit-learn, pandas, numpy.

The pair is a member of stack B (stack A and stack B are the two stacks of `stack/`).

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| yekenot_realmlp_pytabkit | B | RealMLP (8 members, 4 epochs) on floored-numeric categories, decimals, original-data target means + 6 combo target encodings | 0.973410 |
