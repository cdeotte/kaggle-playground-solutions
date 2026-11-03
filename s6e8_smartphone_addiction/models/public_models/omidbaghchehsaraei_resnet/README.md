# ResNet by omidbaghchehsaraei (public notebook)

OMID BAGHCHEH SARAEI's (@omidbaghchehsaraei) pytabkit ResNet from the public notebook
[ResNet for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/resnet-for-predicting-smartphone-addiction)
(version 1). Thank you, Omid!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses).

- **Features** (`../lib/ratio_te.py`): the 12 raw predictors as string categories, missing count and 9 screen-time
  ratio features (NaN -> 0), and per raw predictor a smoothed target encoding (smoothing 10) and a frequency encoding.
- **Model:** pytabkit `Resnet_RTDL_D_Classifier`, width 128, 2 residual layers, hidden factor 2, categorical
  embeddings of size 8, residual dropout 0.1, lr 1e-3, batch 128, seed 42, fitted with the fold's validation rows (best
  epoch on the validation set). Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook computed the target / frequency encodings once on all training rows
(5-fold out-of-fold) before its CV loop. Our run computes them inside each fold: the training rows get 5-fold inner
out-of-fold target encodings, the validation and test rows the map fitted on all training rows of the fold, and the
frequencies are counted on the fold's training rows.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/omidbaghchehsaraei_resnet_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no
other repo folder is needed (shared code: `../lib/`). One GPU. Packages: pytabkit (its RTDL models also need skorch; we
used skorch 1.4.0), torch, scikit-learn, pandas.

| Pair | Model | OOF AUC |
|---|---|---:|
| omidbaghchehsaraei_resnet | the notebook's pytabkit ResNet on our 5 folds | 0.966670 |
