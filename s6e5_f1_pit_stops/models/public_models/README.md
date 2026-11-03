# public_models

Models from other Kagglers' public S6E5 notebooks (thank you!) that are part of our stack. Each subfolder holds our run
of one notebook's model on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)` on the target), with a
README that credits the author, links the notebook and says what we changed to run it on our folds. The changes are
small: every notebook uses our 5 folds (four already used the same split, also for the original rows they append; the
others used an 80/20 holdout with a full-data refit, a full-data refit after a 5-fold check, or 7 folds); the notebooks
that tuned their parameters with Optuna use the best parameters printed by their own run instead of searching again; the
label-free feature engineering of the yekenot pytabkit notebook is fitted inside each fold; only the RealMLP of the
3-model ensemble notebook is used; and some runs clip their final predictions to [1e-7, 1 - 1e-7].

| Folder | Notebook | Author | OOF AUC |
|---|---|---|---:|
| [yekenot_realmlp_pytabkit](yekenot_realmlp_pytabkit) | [PS\|S6\|E5: RealMLP · PyTabKit](https://www.kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytabkit) (version 16) | Vladimir Demidov (@yekenot) | 0.954088 |
| [yekenot_realmlp_pytorch](yekenot_realmlp_pytorch) | [PS\|S6\|E5: RealMLP · PyTorch](https://www.kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytorch) (version 2) | Vladimir Demidov (@yekenot) | 0.953484 |
| [amanatar_realmlp](amanatar_realmlp) | [3-Model Ensemble: RealMLP + CatBoost + XGBoost](https://www.kaggle.com/code/amanatar/3-model-ensemble-realmlp-catboost-xgboost) (version 1, the RealMLP) | Aman Atar (@amanatar) | 0.953442 |
| [yekenot_realmlp_pytabkit](yekenot_realmlp_pytabkit) | [PS\|S6\|E5: RealMLP · PyTabKit](https://www.kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytabkit) (version 11) | Vladimir Demidov (@yekenot) | 0.953258 |
| [rohit8527kmr7518_catboost](rohit8527kmr7518_catboost) | [PS S6E5: CatBoost 10-Fold CV](https://www.kaggle.com/code/rohit8527kmr7518/ps-s6e5-catboost-10-fold-cv) (version 6) | Rohit Kumar (@rohit8527kmr7518) | 0.952703 |
| [berkayzkan_xgb](berkayzkan_xgb) | [xgboost baseline](https://www.kaggle.com/code/berkayzkan/xgboost-baseline) (version 6) | Berkay Özkan (@berkayzkan) | 0.949626 |
| [tamerlanomralinov_hgb](tamerlanomralinov_hgb) | [S6E5 \| HistBoost Baseline](https://www.kaggle.com/code/tamerlanomralinov/s6e5-histboost-baseline) (version 2) | Tamerlan Omralinov (@tamerlanomralinov) | 0.947546 |
| [shivamgravity_lgbm](shivamgravity_lgbm) | [pgs-s6e5-lgbm-optuna](https://www.kaggle.com/code/shivamgravity/pgs-s6e5-lgbm-optuna) (version 15) | Shivam (@shivamgravity) | 0.945580 |

Run any of them with `python train.py` inside its folder (`python train.py --version 11` / `--version 16` for
yekenot_realmlp_pytabkit); it writes `preds/<pair>_oof.npy` and `preds/<pair>_test.npy` (probabilities, train.csv /
test.csv row order) and needs only `data/`. The pair is named after its folder (plus `_v11` / `_v16` for the two
yekenot_realmlp_pytabkit versions). All scripts except amanatar_realmlp (whose notebook seeds once and runs its folds in
one loop) take `--folds 0,1,...`: finished folds are kept in `<folder>/parts/` and skipped, and the pair is written once
all 5 folds are done. Five folders need one GPU (the RealMLPs, CatBoost, XGBoost); tamerlanomralinov_hgb and
shivamgravity_lgbm run on CPU. Shared code in `lib/runner.py`: the fold loop, the fold-matched subset of the original
rows, and the final clipping. Our runs used pytabkit 1.7.3, torch 2.11, catboost 1.2.10, xgboost 3.2.0, lightgbm 4.6.0
and scikit-learn 1.6.1.

The stacker searches this folder and all its subfolders, so each subfolder's `preds/` pairs are stacked like any other model.
Pairs that someone shared publicly (without code) can be dropped into `preds/` here or into any subfolder's `preds/`:
- `<name>_oof.npy` (one value per train row, train.csv order) and `<name>_test.npy` (one value per test row, test.csv order), or
- `<name>_oof.csv` and `<name>_test.csv`, each with an `id` column and one prediction column.

Values can be probabilities or logits. Each pair must use the same folds as everything else
(`StratifiedKFold(5, shuffle=True, random_state=42)`); otherwise the stack CV is leaky.
