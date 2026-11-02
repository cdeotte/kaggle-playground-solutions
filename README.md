# Kaggle Playground Series solutions

Code for our solutions to Kaggle Playground Series competitions. Every solution follows the same pattern:

1. **Level-1 models.** Each model has its own folder under `models/` and is trained with the same stratified 5-fold split.
   It writes an **OOF / test prediction pair** to `models/<model>/preds/`:
   - `<pair>_oof.npy`: one prediction per train row, made by the fold model that did not train on that row;
   - `<pair>_test.npy`: one prediction per test row, the mean of the 5 fold models.
2. **Stack.** `stack/stack.py` finds every pair under `models/*/preds/` and fits one L2 logistic regression on their logits,
   with the same 5 folds.
3. **Public pairs.** `models/public_models/` is an empty placeholder. During the competitions we also stacked OOF / test pairs
   that other Kagglers shared publicly; we do not redistribute their work here. Put any pairs made with the same folds into
   `models/public_models/preds/` and the stacker includes them.

| Competition | Folder | Result | Writeup |
|---|---|---|---|
| Playground S6E10: Predicting Airline Satisfaction (Oct 2026) | [s6e10_airline_satisfaction](s6e10_airline_satisfaction) | 1st place (private LB 0.96318); our 27 models alone: 0.96314 | [1st place: TFMs and LLMs with tabular data](https://www.kaggle.com/competitions/playground-series-s6e10/writeups/1st-place-tfms-and-llms-with-tabular-data) |

The code is shared as is. It is the code that produced our predictions, cleaned up for reading, and it was not re-run after
the clean-up, so expect to adjust paths, package versions and memory settings for your own hardware.
