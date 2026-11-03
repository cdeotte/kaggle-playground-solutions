# TabNet by omidbaghchehsaraei (public notebook)

OMID BAGHCHEH SARAEI's (@omidbaghchehsaraei) TabNet from the public notebook
[TabNet for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/tabnet-for-predicting-smartphone-addiction)
(version 1). Thank you, Omid!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses).

- **Features** (`../lib/ratio_te.py`): the 12 raw predictors as label codes (sorted train + test levels) for TabNet's
  categorical embeddings, missing count and 9 screen-time ratio features (NaN -> 0), and per raw predictor a smoothed
  target encoding (smoothing 10) and a frequency encoding; the continuous columns are standardized on the fold's
  training rows.
- **Model:** pytorch-tabnet `TabNetClassifier`, n_d = n_a = 8, 3 steps, gamma 1.3, sparsemax masks, categorical
  embedding dim 1, Adam lr 0.02 (weight decay 1e-5) with StepLR(10, 0.5), batch 1024 / virtual batch 128, up to 200
  epochs with patience 10 on the validation AUC, seed 42. Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook computed the target / frequency encodings once on all training rows
(5-fold out-of-fold) before its CV loop. Our run computes them inside each fold: the training rows get 5-fold inner
out-of-fold target encodings, the validation and test rows the map fitted on all training rows of the fold, and the
frequencies are counted on the fold's training rows.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/omidbaghchehsaraei_tabnet_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no
other repo folder is needed (shared code: `../lib/`). One GPU. Packages: pytorch-tabnet (we used 4.1.0), torch,
scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| omidbaghchehsaraei_tabnet | the notebook's TabNet on our 5 folds | 0.967774 |
