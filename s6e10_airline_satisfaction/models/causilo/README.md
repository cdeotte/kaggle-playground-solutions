# Causilo

Zero-shot, full context, 8 estimators, raw features. Causilo's built-in KV cache keeps the caches of all 8 estimators on the
GPU (out of memory at full context); `stream_proba()` uses Causilo's own cache functions one estimator at a time, with the
same member order and logit averaging.

`python train.py` -> pair `causilo`, OOF AUC 0.958275. Needs the `causilo` package.
