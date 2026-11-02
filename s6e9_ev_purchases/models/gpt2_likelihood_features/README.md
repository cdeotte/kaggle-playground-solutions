# GPT-2 likelihood features

Frozen GReaT replicas (`../gpt2_replica`, trained on the original data only) score every train and test row without labels; small
supervised models fitted per fold turn those scores into predictions.

- **gpt2_conditional_llr**: replicas e55 / e60 / e65 / e70, each in 3 field orders (original, competition, a fixed random one),
  score every row after both target prompts (`Will_Buy_EV is No, ...` / `Will_Buy_EV is Yes, ...`, temperature 0.70, bf16 model,
  fp32 log-softmax) and with the inverse query (`..., Will_Buy_EV is` -> ` Yes` vs ` No`); e60 / e65 also give top-50 support
  statistics (fp32). 306 features (LLRs, joint likelihoods, their spread over replicas and orders). Per fold, an L2 logistic
  regression (cuML) whose feature family and C are picked by inner CV on the training folds.
- **gpt2_field_pseudolikelihood**: replicas e55 / e60; each of the 13 fields is held out and its value is scored after both target
  prompts and the other 12 fields (`Will_Buy_EV is Yes, <12 fields>, <field> is <value>`). 21 statistics per map, 1,330
  features. Per fold, a nested XGBoost: 4 candidates scored by inner CV with early stopping, the winner refit on the training folds.
- **gpt2_likelihood_xgb**: the e55 replica scores every row in 12 random field orders (fp32 / TF32, temperature 0.70): an
  order-marginal LLR and per-field evidence (23 features); the inverse query in 6 random orders (4 features); plus the 306
  features of gpt2_conditional_llr. Per fold, XGBoost with early stopping on a 10% split of the training folds.

```
# gpt2_conditional_llr
python score_conditional.py           # 12 maps (4 replicas x 3 orders); --replicas e55 --orders source runs one
python score_topk.py                  # 6 maps (e60, e65 x 3 orders)
python build_conditional_features.py
python fit_conditional_lr.py

# gpt2_field_pseudolikelihood
python score_fields.py                # 26 maps (e55, e60 x 13 fields); --replicas e55 --fields 0,1,2 runs a subset
python build_field_features.py
python fit_field_xgb.py

# gpt2_likelihood_xgb (also needs build_conditional_features.py)
python score_orders.py                # e55, 12 orders; --rows 0:159206 etc. scores a slice (slices can run on several GPUs)
python score_inverse.py               # e55, 6 orders
python build_order_features.py
python fit_combo_xgb.py
```

score_conditional / score_topk / score_fields skip maps that already exist in `features/`. Every map is one pass over the 955,236
train + test rows: about 50 minutes on one GPU per score_conditional / score_topk / score_fields map, about 6 GPU-hours for
score_orders.
`fit_conditional_lr.py` and `fit_field_xgb.py` run the folds given by `--folds` (finished folds are kept in `parts/`).

| Pair | OOF AUC |
|---|---:|
| gpt2_field_pseudolikelihood | 0.942393 |
| gpt2_likelihood_xgb | 0.942333 |
| gpt2_conditional_llr | 0.940956 |

Needs `../gpt2_replica/checkpoints/{e55,e60,e65,e70}.pt`. The score_conditional / score_topk / score_fields maps ran with
transformers 4.26.1 and torch 2.5.1 (the replica environment); score_orders / score_inverse with transformers 5.x and torch 2.11.
The fits need xgboost (GPU) and, for `fit_conditional_lr.py`, cuML.
