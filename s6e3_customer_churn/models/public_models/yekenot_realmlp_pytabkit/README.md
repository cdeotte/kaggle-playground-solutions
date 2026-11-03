# RealMLP (pytabkit) by yekenot (public notebook)

Vladimir Demidov's (@yekenot) pytabkit RealMLP from the public notebook
[PS|S6|E3: RealMLP · PyTabKit](https://www.kaggle.com/code/yekenot/ps-s6-e3-realmlp-pytabkit): an earlier version
(preset `mish`) and a [later version](https://www.kaggle.com/code/yekenot/ps-s6-e3-realmlp-pytabkit?scriptVersionId=307208257)
(preset `silu`). Thank you, Vladimir!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).
One script runs both versions.

- **Features (label-free, fitted once on all train rows and applied to test):** MonthlyCharges / TotalCharges,
  TotalCharges / tenure, MonthlyCharges / (TotalCharges / tenure), TotalCharges / MonthlyCharges and tenure^2; the
  thousands digit of TotalCharges; quantile bins of TotalCharges and MonthlyCharges; every numeric column (with
  SeniorCitizen) factorized as a category (TotalCharges rounded to an integer first); the Contract x InternetService x
  PaymentMethod combo as a category.
  - `mish`: TotalCharges in 4,000 and 500 bins, MonthlyCharges in 200 and 100 bins. 34 columns.
  - `silu`: also a tenure >= 24 flag, the MonthlyCharges cents, TotalCharges mod 100 and mod 1000 and an
    is-multiple-of-10 flag; quantile and k-means bins of TotalCharges (5,000, 400) and MonthlyCharges (300).
    41 columns.
- **Encodings per fold:** a scikit-learn `TargetEncoder` of the combo (cv equal to the number of folds, as in the
  notebook; smooth 'auto', shuffle, seed 42), fitted on the training rows.
- **Model:** pytabkit `RealMLP_TD_Classifier`, 8 members, 3 epochs, batch 256, lr 0.075, wd 0.0236, PLR embeddings
  (16 / 8, sigma 2.33), one-hot up to 18 levels, label smoothing 0.01, early stopping on the validation fold's AUC,
  seed 42.
  - `mish`: 4 x 352 mish layers, `flat_cos` dropout schedule, `coslog4` label smoothing schedule, patience 20.
  - `silu`: 512-256-128 SiLU layers, `flat_anneal` learning-rate schedule, `expm4t` dropout schedule, `sqrt_cos` label
    smoothing schedule, GELU PLR activation, patience 10.

  The notebook seeds once (42) and trains its folds in one loop, so `train.py` runs the folds in order in one process.
  Test = mean of the fold models.

**Changed to run it on our folds:** our 5 folds replace the notebook's folds (its latest version trains 20 folds, with
`TargetEncoder(cv=20)`).

```
python train.py --preset mish    # all folds in order in one process
python train.py --preset silu
```

Writes `preds/yekenot_realmlp_pytabkit_mish_{oof,test}.npy` and `preds/yekenot_realmlp_pytabkit_silu_{oof,test}.npy`
(probabilities). Reads only `data/` (train.csv, test.csv); no other repo folder is needed (shared fold loop:
`../lib/runner.py`). One GPU. Packages: pytabkit (our runs: 1.7.3), torch, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| yekenot_realmlp_pytabkit_mish | earlier version: RealMLP 4 x 352 mish, 34 columns | 0.918679 |
| yekenot_realmlp_pytabkit_silu | later version: RealMLP 512-256-128 SiLU, + digit / modulo features and k-means bins | 0.919118 |
