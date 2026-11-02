# Distillation from the full-context XGB

Two models that learn from the true labels of the outer-train rows plus soft labels on the query rows (fold-k rows and
test rows) from a same-fold teacher: the 16-seed `digits_diffs` XGB of `../income_mixture_xgb` (sigmoid of the mean
clipped seed logit, `teacher.py`).

- **mlp_distilled_composite**: a fixed same-fold composite of three student families, 8 seeds (42-49) each.
  Students are 4-member parallel MLPs (256-128-64) on the teacher's 259 columns + a 48-column predictor-only joint
  bank, offset by the nested LR margin, 8 epochs: `distill_joint_w0` (true labels only), `distill_joint_w1` (+ soft
  labels, weight 1) and `multitask_w0` (also reads a raw-field embedding encoder). Per fold, the 32 member margins of
  each distill family are combined as mean / sqrt(1 + pi var / 8); the w1 margin is moved towards w0 by a gated half of
  their difference, 0.5 d / (1 + (d / 0.25)^2); the result is mixed 0.9 / 0.1 with the multitask logit.
  The joint bank (`joint_bank.py`) comes from masked-field density networks cross-fitted over all train + test rows:
  the original source law integrated over the generated joint of 6 field pairs, against independent completion, and
  the observed pairs' joint likelihood lifts.
- **gpt2_native_kd4**: GPT-2 small, first trained as a causal LM on the 10,000 original rows (60 epochs,
  `gpt_surrogate.py`), then fine-tuned per fold as a native classifier: the prompt is the 13 predictors ending in
  `, Will_Buy_EV is` and the margin is the last hidden state . (W[' Yes'] - W[' No']) of the tied output embedding.
  Each update: 256 fit rows (true labels) + 256 query rows (soft labels), loss = (BCE_hard + 4 BCE_soft) / 5, lr 5e-5,
  4 epochs, seed 42. Prediction = sigmoid of the mean margin of two column orders.

## Steps
```
python joint_bank.py                               # density networks + joint bank (5 predictor folds)
python train_students.py --family distill_joint_w0 # 5 folds x 8 seeds -> runs/
python train_students.py --family distill_joint_w1
python train_students.py --family multitask_w0
python composite.py                                # -> preds/mlp_distilled_composite_{oof,test}.npy

python gpt_surrogate.py                            # -> checkpoints/original_gpt2_e60 (~25 min)
python gpt_prefix.py                               # token banks of all train + test rows, two column orders
python train_native_gpt.py                         # 5 folds (~1.6 h each on one GPU) -> preds/gpt2_native_kd4_{oof,test}.npy
```
`--folds` (and `--seeds` for the students) run a subset; finished folds are skipped.

## Dependencies
- `../income_mixture_xgb`: `python train_xgb.py --model digits_diffs` (the teacher runs in
  `runs/stage8/digits_diffs_s{42..57}_f{k}/`), and its caches through `xgb_inputs.py` (the student feature bank).
- `../neural_contrasts/cache/stage6/masked_inputs.npz` + `.json` (field codes, numerics, original priors, source-law
  effects and the predictor-only fold of every row): input of the joint bank and of the multitask encoder.
- torch, transformers (public `gpt2` weights and tokenizer).

## Pairs
| Pair | OOF AUC |
|---|---:|
| mlp_distilled_composite | 0.946238 |
| gpt2_native_kd4 | 0.946447 |

## Gaps
Ported from the competition code and not re-run; GPU training is not bit-exact, so expect small differences.
