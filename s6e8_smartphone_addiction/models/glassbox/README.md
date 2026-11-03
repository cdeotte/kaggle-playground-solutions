# Glass-box models: RuleFit, naive Bayes + spline GAM, and an EBM

**rulefit_rf_et_te** (`train_rulefit.py`, `lib/rulefit.py`): RuleFit. A RandomForest
(seed 777 + fold) and an ExtraTrees (seed 778 + fold), 20 trees each, max_depth 3, min_samples_leaf 10, max_features
0.5, give the rules (every root-to-leaf path, about 320 distinct rules per fold); an SGD elastic-net logistic regression
(alpha 1e-4, l1_ratio 0.9, max_iter 500) is fit on [features, binary rule firings], standardized on the outer-train rows.
Features (`lib/features.py`, 86 columns): the 9 raw numerics + 23 label-free derived columns (budget slack
`daily - (social + gaming + work)`, component sum, utilization, their cent parts, exact-budget and at-20 flags,
`sleep + daily`, per-daily-hour ratios, `weekend - daily`, `weekend / daily`, 3 products, repair flag, ordinal
categoricals; median-imputed on train + test), 18 original-data per-value rates (label and 'Severe' rate of each numeric
in the 7,500-row original dataset) and 36 nested target encodings of exact value tokens (`%.2f`; smoothing 30 / 50 / 10 /
200 / 100; outer-train rows inner 5-fold cross-fitted with seed 777, held-out and test rows from all outer-train rows).
Test = mean of the 5 fold models.

**spline_gam_nb_poly2** (`train_nb_gam.py`): two additive parts and a small combiner.
- naive Bayes over the 12 exact value tokens: per column `P(token | y) = (count + 20 prior_y) / (n_y + 20)`, score = sum
  of log-likelihood ratios + prior log-odds;
- spline GAM: logistic regression (C 0.5) on cubic B-splines (15 knots) of 15 numerics (the 9 raw + slack,
  utilization, `sleep + daily`, `weekend - daily`, notifications x app opens, component sum; NaN -> 0), missing flags,
  one-hot categoricals, a repair flag and 4 nested target encodings (daily, social, weekend, slack; m 20);
- combiner: logistic regression on the degree-2 polynomial features of [NB score, GAM probability, original-data
  evidence], the evidence being the summed smoothed per-value log-odds of daily / social / weekend in the original data.
Both parts are fit per outer fold; their outer-train scores come from a 3-fold inner cross-fit (seed 777 + fold). The
score arrays are filled fold by fold, so the combiner of every fold is trained on the scores left by the last pass
(details in the script docstring). Test = mean of the 5 fold combiners.

**ebm_ga2m_te** (`train_ebm.py`): interpret's ExplainableBoostingClassifier with 25 pairwise interactions (max_bins 128,
max_interaction_bins 32, outer_bags 4, inner_bags 0, learning_rate 0.03, validation_size 0.15, early stopping 50 of at
most 3000 rounds, min_samples_leaf 10, max_leaves 3, random_state 42) on the 9 raw numerics, slack, utilization,
component sum, repair flag, 12 missing flags, one-hot categoricals (NaN -> 0) and 4 nested target encodings (daily,
social, weekend, slack; m 20; 3-fold inner cross-fit, seed 777 + fold). Each fold's EBM is fit on a random subsample of
50,000 outer-train rows (NumPy's global RNG without a seed, so reruns differ slightly), then a per-fold logistic
calibration maps the scores to probabilities (the score array is filled fold by fold as above). Test = mean of the
5 calibrated fold predictions.

```
python train_rulefit.py      # --folds 0,1 runs a subset; finished folds in parts/ are skipped
python train_nb_gam.py
python train_ebm.py
```
Each script writes `preds/<pair>_{oof,test}.npy` (probabilities).

No other repo folders are needed.

| Pair | Model | OOF AUC |
|---|---|---:|
| rulefit_rf_et_te | RuleFit: RF + ExtraTrees rules (depth 3) + elastic-net logistic regression, on raw + derived + original rates + nested TEs | 0.960875 |
| spline_gam_nb_poly2 | exact-token naive Bayes + spline GAM, combined by a degree-2 polynomial logistic regression with original-data evidence | 0.950482 |
| ebm_ga2m_te | EBM with 25 pairwise interactions on raw + derived columns and 4 nested TEs, logistic calibration | 0.958392 |

Packages: interpret (EBM), scikit-learn, pandas, numpy.
