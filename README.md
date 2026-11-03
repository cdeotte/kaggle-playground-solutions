# Kaggle Playground Series solutions

Code for our solutions to Kaggle Playground Series competitions. Every solution follows the same pattern:

1. **Level-1 models.** Each model has its own folder under `models/` and is trained with the same stratified 5-fold split.
   It writes an **OOF / test prediction pair** to `models/<model>/preds/`:
   - `<pair>_oof.npy`: one prediction per train row, made by the fold model that did not train on that row;
   - `<pair>_test.npy`: one prediction per test row, the mean of the 5 fold models.
2. **Stack.** `stack/stack.py` finds every pair under `models/*/preds/` and under `models/public_models/` (its own `preds/`
   and every subfolder's `preds/`), and fits one L2 logistic regression on their logits, with the same 5 folds.
3. **Public models.** `models/public_models/` holds the models that came from other Kagglers' public work.
   - When our solution ran a public notebook's model on our own 5 folds, that code is in a subfolder of `public_models/`,
     with a README crediting the original author.
   - When we only stacked the OOF / test pairs that someone shared, we do not redistribute them. Put any pairs made with
     the same folds into `models/public_models/preds/` (or any subfolder's `preds/`) and the stacker includes them.

| Competition | Folder | Result | Writeup |
|---|---|---|---|
| Playground S6E10: Predicting Airline Satisfaction (Oct 2026) | [s6e10_airline_satisfaction](s6e10_airline_satisfaction) | 1st place (private LB 0.96318); our 27 models alone: 0.96314 | [1st place: TFMs and LLMs with tabular data](https://www.kaggle.com/competitions/playground-series-s6e10/writeups/1st-place-tfms-and-llms-with-tabular-data) |
| Playground S6E9: Predicting Electric Vehicle Purchases (Sep 2026) | [s6e9_ev_purchases](s6e9_ev_purchases) | 1st place (private LB 0.94602) | [1st Place - The AI Relay](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/1st-place-the-ai-relay) |
| Playground S6E8: Predicting Smartphone Addiction (Aug 2026) | [s6e8_smartphone_addiction](s6e8_smartphone_addiction) | 1st place (private LB 0.97176); our 102 models alone: 0.97171 | [1st Place - Distributed Intelligence - NVIDIA Inference Hub](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/1st-place-distributed-intelligence-nvidia-infe) |
| Playground S6E5: Predicting F1 Pit Stops (May 2026) | [s6e5_f1_pit_stops](s6e5_f1_pit_stops) | 2nd place (private LB 0.95502); our 100 models alone: 0.95503 | [2nd Place - Autonomous Codex Yolo!](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/2nd-place-autonomous-codex-yolo) |

The code is shared as is. It is the code that produced our predictions, cleaned up for reading, and it was not re-run after
the clean-up, so expect to adjust paths, package versions and memory settings for your own hardware.
