# public_models

Models from other Kagglers' public S6E4 notebooks (thank you!) that are part of our stack B (stack A and stack B are
the two stacks of `stack/`). Each subfolder holds our run of one notebook's model on our 5 folds
(`StratifiedKFold(5, shuffle=True, random_state=42)` on the target), with a README that credits the author, links the
notebook and says what we changed to run it on our folds. The changes are small:
- the logistic regression, the two utaazu notebooks and the RealMLP already use our split; we leave out the utaazu
  notebooks' bias tuning (per-class logit offsets tuned on the OOF);
- the three blamerx notebooks run 10 folds over the competition and original rows together; our runs use our 5 folds
  and append all original rows to every fold's training rows, without the notebooks' threshold tuning (class
  probability multipliers tuned on the OOF) and pseudo-labelling (the XGBoost and LightGBM notebooks retrain on
  confidently predicted test rows; the CatBoost version we ported has no such step); the LightGBM runs on the CPU;
- from the emanuellcs and aliafzal9323 notebooks we use only the CatBoost, run in a nested 5 x 5 cross-validation:
  their OOF has shape `(5, 630000, 3)`, one slice per outer fold (inner out-of-fold predictions for the outer training
  rows, the mean of the 5 inner models for the outer held-out rows), and the stacker averages the slices.

| Folder (= pair) | Notebook | Author | Stack | OOF balanced acc. |
|---|---|---|---|---:|
| [blamerx_xgb](blamerx_xgb) | [S6E4 🌾\| XGBoost Adv FE \| 0.979 CV](https://www.kaggle.com/code/blamerx/s6e4-xgboost-adv-fe-0-979-cv) | @blamerx | B | 0.978298 |
| [blamerx_catboost](blamerx_catboost) | [s6e4-catboost-robust-0-978-cv](https://www.kaggle.com/code/blamerx/s6e4-catboost-robust-0-978-cv) | @blamerx | B | 0.977986 |
| [aliafzal9323_catboost](aliafzal9323_catboost) | [s6e4-0-978-xgb-cat-pairwise-te-magic](https://www.kaggle.com/code/aliafzal9323/s6e4-0-978-xgb-cat-pairwise-te-magic) (its CatBoost, nested 5 x 5) | @aliafzal9323 | B | 0.977471 |
| [utaazu_catboost](utaazu_catboost) | [0.979 CV Single CAT: Pairwise TE + Bias Tuning](https://www.kaggle.com/code/utaazu/0-979-cv-single-cat-pairwise-te-bias-tuning) | @utaazu | B | 0.977055 |
| [mahoganybuttstrings_logreg](mahoganybuttstrings_logreg) | [PG S6E4 - LR - \[CV 0.97652 LB 0.97487\]](https://www.kaggle.com/code/mahoganybuttstrings/pg-s6e4-lr-cv-0-97652-lb-0-97487) | @mahoganybuttstrings | B | 0.976519 |
| [blamerx_lgbm](blamerx_lgbm) | [S6E4 🍀\| LightGBM High Balance \| 0.979 CV](https://www.kaggle.com/code/blamerx/s6e4-lightgbm-high-balance-0-979-cv) | @blamerx | B | 0.975711 |
| [utaazu_xgb](utaazu_xgb) | [0.979 CV Single XGB: Pairwise TE + Bias Tuning](https://www.kaggle.com/code/utaazu/0-979-cv-single-xgb-pairwise-te-bias-tuning) | @utaazu | B | 0.974203 |
| [yekenot_realmlp_pytabkit](yekenot_realmlp_pytabkit) | [PS\|S6\|E4: RealMLP · PyTabKit](https://www.kaggle.com/code/yekenot/ps-s6-e4-realmlp-pytabkit) | Vladimir Demidov (@yekenot) | B | 0.973410 |
| [emanuellcs_catboost](emanuellcs_catboost) | [Predicting Irrigation Need \| XGBoost + CatBoost](https://www.kaggle.com/code/emanuellcs/predicting-irrigation-need-xgboost-catboost) (its CatBoost, nested 5 x 5) | Emanuel Lázaro (@emanuellcs) | B | 0.970478 |

For the two nested models the score is that of the mean of the 5 OOF slices.

Run any of them with `python train.py` inside its folder; it writes `preds/<folder>_oof.npy` and
`preds/<folder>_test.npy` (probabilities in class order Low, Medium, High; train.csv / test.csv row order) and needs
only `data/` and `common/`. All scripts except yekenot_realmlp_pytabkit (whose notebook seeds once and runs its folds
in one loop) take `--folds 0,1,...` (outer folds for the two nested models): finished folds are kept in
`<folder>/parts/` and skipped, and the pair is written once all 5 are done. Seven folders need one GPU (the CatBoosts,
the XGBoosts and the RealMLP); mahoganybuttstrings_logreg and blamerx_lgbm run on the CPU. Shared code in
`lib/runner.py`: the fold loops (independent folds, one-process folds, nested 5 x 5). Packages: numpy, pandas 2.x (the
scripts find string columns by the object dtype), scikit-learn >= 1.4 (multiclass `TargetEncoder`; below 1.8 for the
logistic regression, which passes `multi_class`), catboost, xgboost, lightgbm, pytabkit and torch.

The stacker searches this folder and all its subfolders, so each subfolder's `preds/` pair is stacked like any other
model; every pair whose name does not end in `_a` belongs to stack B. Pairs that someone shared publicly (without code)
can be dropped into `preds/` here or into any subfolder's `preds/` as `<name>_oof.npy` (`(630000, 3)`, or
`(5, 630000, 3)` for a nested model; train.csv order) and `<name>_test.npy` (`(270000, 3)`, test.csv order),
probabilities in class order Low, Medium, High. Each pair must use the same folds as everything else
(`StratifiedKFold(5, shuffle=True, random_state=42)`); otherwise the stack CV is leaky.
