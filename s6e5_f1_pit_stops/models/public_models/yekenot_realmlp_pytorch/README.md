# RealMLP (PyTorch) by yekenot (public notebook)

Vladimir Demidov's (@yekenot) PyTorch RealMLP from the public notebook
[PS|S6|E5: RealMLP · PyTorch](https://www.kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytorch) (version 2). Thank you,
Vladimir! The notebook adapts a PyTorch re-implementation of pytabkit's RealMLP from an
[S6E4 RealMLP notebook by mahoganybuttstrings](https://www.kaggle.com/code/mahoganybuttstrings/pg-s6e4-realmlp-cv-0-97802-lb-0-97685)
to binary classification, with the feature engineering of
[PS|S6|E5: RealMLP · PyTabKit](https://www.kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytabkit) (our run of it:
[../yekenot_realmlp_pytabkit](../yekenot_realmlp_pytabkit)).

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses, for the competition rows and for the original rows). `realmlp.py` is the notebook's model
code (unchanged); `train.py` is its feature engineering and fold loop. Later versions of the notebook changed the model
(residual blocks, 256-wide layers); this folder is version 2.

- **Features (label-free, fitted once on all competition training rows):** Driver / Compound / Race factorized;
  LapNumber / RaceProgress, TyreLife / LapNumber, LapTime x Cumulative_Degradation (signed and absolute) and
  LapTime / |Cumulative_Degradation|; every numeric column and the two ratios floored and factorized; counts of Driver /
  Compound / Race / Year / PitStop; 200 quantile bins of RaceProgress and 7 of LapTime; Race x Compound and Race x Year
  combos. All 20 categoricals are integer codes.
- **Original rows:** the fold-matched 4/5 of the original dataset (the same `StratifiedKFold` on its own target) is
  appended to each fold's training rows.
- **Encodings per fold:** a scikit-learn `TargetEncoder` (cv=5, smooth 'auto', shuffle, seed 42) of the two combo
  columns, fitted on the training rows (competition + original).
- **Model:** RealMLP with 16 vectorized ensemble members: numeric features median-centered, robust-scaled and smoothly
  clipped, then PBLD periodic embeddings (20 hidden, 5 outputs, frequency scale 5, PReLU); categoricals one-hot (up to 4
  levels) or 6-d embeddings; a learnable front scaling layer; 512-64-128 SiLU NTP layers, dropout 0.05 (expm4t
  schedule); AdamW lr 0.008 (flat-cos, flat 30%) with per-group learning rates / weight decays, gradient clipping 1,
  label smoothing 0.04 (cos schedule), balanced class weights; 4 epochs, batch 256. The validation probabilities of the
  best epoch are the OOF; the restored best checkpoint predicts test. Test = mean of the 5 fold models.

**Changed to run it on our folds:** each fold re-seeds the global random generators with 43 + fold (the notebook seeds
42 once); the target encodings are stored as float32; the fold checkpoint is written to `cache/` and deleted after the
fold; OOF and test are clipped to [1e-7, 1 - 1e-7].

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/yekenot_realmlp_pytorch_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv,
f1_strategy_dataset_v4.csv); no other repo folder is needed (shared fold loop: `../lib/runner.py`). One GPU. Packages:
torch, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| yekenot_realmlp_pytorch | the notebook's PyTorch RealMLP (16 members) on our 5 folds | 0.953484 |
