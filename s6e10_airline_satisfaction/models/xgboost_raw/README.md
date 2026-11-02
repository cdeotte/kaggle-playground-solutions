# XGBoost on the raw features

The 21 raw features only (no feature engineering), tuned with a fixed number of rounds on the 5 folds, averaged over 8 seeds.
`max_bin 4096` is the key setting: Flight Distance has ~3.5k distinct values and acts as a route id.

`python train.py` -> pair `xgboost_raw`, OOF AUC 0.959852.
