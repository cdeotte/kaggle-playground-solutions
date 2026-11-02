# public_models

Models that came from other Kagglers' public notebooks (thank you!). Each subfolder holds our run of one public notebook's
model on our 5 folds, with a README crediting the original author and linking the notebook:

| Folder | Original notebook |
|---|---|
| [heuljax_xgb](heuljax_xgb) | Paul Bryan Elefante (@heuljax): XGBoost |
| [paul_lr](paul_lr) | Paul Bryan Elefante (@heuljax): logistic regression |
| [lgbm_kirill0212](lgbm_kirill0212) | kirill0212: LightGBM |
| [lgbm_najiama](lgbm_najiama) | najiama: LightGBM |

The stacker searches this folder and all its subfolders, so each subfolder's `preds/` pairs are stacked like any other model.
Pairs that someone shared publicly (without code) can be dropped into `preds/` here or into any subfolder's `preds/`:
- `<name>_oof.npy` (one value per train row, train.csv order) and `<name>_test.npy` (one value per test row, test.csv order), or
- `<name>_oof.csv` and `<name>_test.csv`, each with an `id` column and one prediction column.

Values can be probabilities or logits. Each pair must use the same folds as everything else
(`StratifiedKFold(5, shuffle=True, random_state=42)` on the target); otherwise the stack CV is leaky.
