# RealMLP by omidbaghchehsaraei (public notebook)

OMID BAGHCHEH SARAEI's (@omidbaghchehsaraei) pytabkit RealMLP from the public notebook
[RealMLP for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/realmlp-for-predicting-smartphone-addiction)
(version 2). Thank you, Omid!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses).

- **Features** (`../lib/ratio_te.py`): the 12 raw predictors as string categories, missing count and 9 screen-time
  ratio features (NaN -> 0), and per raw predictor a smoothed target encoding (smoothing 10) and a frequency encoding.
- **Model:** pytabkit `RealMLP_TD_Classifier` with the notebook's tuned parameters: 8 ensemble members, 4 x 384 mish
  layers, PLR embeddings, lr 0.04, label smoothing 0.0115, dropout 0.073, weight decay 0.0237, batch 128, up to 100
  epochs with early stopping on the validation 1-AUC (additive patience 20), seed 42. Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook computed the target / frequency encodings once on all training rows
(5-fold out-of-fold) before its CV loop. Our run computes them inside each fold: the training rows get 5-fold inner
out-of-fold target encodings, the validation and test rows the map fitted on all training rows of the fold, and the
frequencies are counted on the fold's training rows.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/omidbaghchehsaraei_realmlp_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no
other repo folder is needed (shared code: `../lib/`). One GPU. Packages: pytabkit, torch, scikit-learn, pandas.

| Pair | Model | OOF AUC |
|---|---|---:|
| omidbaghchehsaraei_realmlp | the notebook's pytabkit RealMLP on our 5 folds | 0.968680 |
