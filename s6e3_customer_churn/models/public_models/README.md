# public_models

Models from other Kagglers' public S6E3 notebooks and discussion posts (thank you!) that are part of our stack. Each
subfolder holds our run of one notebook's (or post's) model on our 5 folds (`StratifiedKFold(5, shuffle=True,
random_state=42)` on the target), with a README that credits the author, links the source and says what we changed to
run it on our folds. The changes are small: every model uses our 5 folds (one notebook already uses the same split; the
others train 10 or 20 stratified folds or 10 folds by row index, and the two discussion posts only cross-validate with
another seed, so we add the test predictions of the fold models); the TabICL ensemble uses 8 voting members instead of
the comment's 4; the from-scratch RealMLP predicts its OOF with its restored best weights, like its test predictions.
johnnyhyland_survival_xgb is our own fold-local version of the notebook's survival-feature XGBoost (frailty model
refitted in every fold with another grouping and other predictors, extra survival-curve features); its README lists the
differences.

| Folder | Notebook / post | Author | OOF AUC |
|---|---|---|---:|
| [yekenot_realmlp_pytabkit](yekenot_realmlp_pytabkit) | [PS\|S6\|E3: RealMLP · PyTabKit](https://www.kaggle.com/code/yekenot/ps-s6-e3-realmlp-pytabkit) (later version, `silu`) | Vladimir Demidov (@yekenot) | 0.919118 |
| [yunsuxiaozi_realmlp_pytorch](yunsuxiaozi_realmlp_pytorch) | [RealMLP from scratchCV:0.91908](https://www.kaggle.com/code/yunsuxiaozi/realmlp-from-scratchcv-0-91908) | yunsuxiaozi (@yunsuxiaozi) | 0.918943 |
| [blamerx_catboost](blamerx_catboost) | [S6E3 \| 0.91902 Optimized CatBoost](https://www.kaggle.com/code/blamerx/s6e3-0-91902-optimized-catboost) | BlamerX (@blamerx) | 0.918818 |
| [blamerx_tabm](blamerx_tabm) | [S6E3 \| TabM + Advanced Features \| 0.91898 CV](https://www.kaggle.com/code/blamerx/s6e3-tabm-advanced-features-0-91898-cv) | BlamerX (@blamerx) | 0.918788 |
| [yekenot_realmlp_pytabkit](yekenot_realmlp_pytabkit) | [PS\|S6\|E3: RealMLP · PyTabKit](https://www.kaggle.com/code/yekenot/ps-s6-e3-realmlp-pytabkit) (earlier version, `mish`) | Vladimir Demidov (@yekenot) | 0.918679 |
| [nalgirayergn_xgb](nalgirayergn_xgb) | [XGB - Predict Customer Churn - CV: 0.91819](https://www.kaggle.com/code/nalgirayergn/xgb-predict-customer-churn-cv-0-91819) | Ünal Giray Ergün (@nalgirayergn) | 0.918210 |
| [siukeitin_ydf](siukeitin_ydf) | [YDF gives pretty good CV score off default parameters](https://www.kaggle.com/competitions/playground-series-s6e3/discussion/679983) (discussion) | broccoli beef (@siukeitin) | 0.917976 |
| [johnnyhyland_survival_xgb](johnnyhyland_survival_xgb) | [s6e3 - Reframing as Bayesian Survival - PyMC](https://www.kaggle.com/code/johnnyhyland/s6e3-reframing-as-bayesian-survival-pymc) (our fold-local version) | Johnny Hyland (@johnnyhyland) | 0.916741 |
| [siukeitin_tabicl](siukeitin_tabicl) | [comment](https://www.kaggle.com/competitions/playground-series-s6e3/discussion/681229#3422698) on [TabICL baseline CV:0.91385,LB:0.91067](https://www.kaggle.com/competitions/playground-series-s6e3/discussion/681229) (discussion by @yunsuxiaozi) | broccoli beef (@siukeitin) | 0.915195 |

Run any of them with `python train.py` inside its folder (`python train.py --preset mish` / `--preset silu` for
yekenot_realmlp_pytabkit); it writes `preds/<pair>_oof.npy` and `preds/<pair>_test.npy` (probabilities, train.csv /
test.csv row order) and needs only `data/`. The pair is named after its folder (plus `_mish` / `_silu` for the two
yekenot_realmlp_pytabkit versions). Five scripts train independent folds (nalgirayergn_xgb, siukeitin_ydf,
johnnyhyland_survival_xgb, siukeitin_tabicl, blamerx_catboost): finished folds are kept in `<folder>/parts/` and
skipped when the script is restarted. The notebooks of the other three folders (blamerx_tabm, yekenot_realmlp_pytabkit,
yunsuxiaozi_realmlp_pytorch) seed once and let the random state run on from fold to fold, so their scripts train the
folds in order in one process. Every folder needs one GPU except siukeitin_ydf (CPU); johnnyhyland_survival_xgb also
samples a PyMC model on the CPU in every fold. Shared code in `lib/runner.py`: the two fold loops and the pair writing.
Packages: numpy, pandas 2.x (some scripts find string columns by the object dtype), scikit-learn >= 1.4
(`TargetEncoder`), xgboost, catboost, ydf, pytabkit, torch, tabicl, pymc; our runs used pytabkit 1.7.3 and ydf 0.15.0.

The stacker (`stack/stack.py`) searches this folder and all its subfolders (`models/public_models/**/preds/`), so each
subfolder's pairs are stacked like any other model. Pairs that someone shared publicly (without code) can be dropped
into `preds/` here or into any subfolder's `preds/` as `<name>_oof.npy` (one probability per train row, train.csv
order) and `<name>_test.npy` (one probability per test row, test.csv order). Each pair must use the same folds as
everything else (`StratifiedKFold(5, shuffle=True, random_state=42)`); otherwise the stack CV is leaky.
