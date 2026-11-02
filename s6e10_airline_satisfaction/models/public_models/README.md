# public_models (placeholder)

During the competition we also stacked OOF / test prediction pairs that other Kagglers shared in public notebooks and datasets
(thank you!). We do not redistribute other people's work here, so this folder is empty.

To add pairs, put them in `preds/`, or in `<any subfolder>/preds/` (the stacker searches this folder and all its subfolders):
- `<name>_oof.npy` (one value per train row, train.csv order) and `<name>_test.npy` (one value per test row, test.csv order), or
- `<name>_oof.csv` and `<name>_test.csv`, each with an `id` column and one prediction column.

Values can be probabilities or logits. Each pair must use the same folds as everything else
(`StratifiedKFold(5, shuffle=True, random_state=42)` on the target); otherwise the stack CV is leaky.
