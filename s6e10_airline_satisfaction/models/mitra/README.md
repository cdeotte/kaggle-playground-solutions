# Mitra-v2

Zero-shot (no fine-tuning), full context, 8 members, raw features. Stock Mitra re-runs its support set for every chunk of
1,024 queries; `mitra_cache.py` is an exact layer-by-layer version that processes the support once per member. fp16 autocast
with fp32 logits (Mitra's default bf16 logits leave many tied probabilities).

`python train.py` -> pair `mitra2`, OOF AUC 0.957075. Needs AutoGluon (`autogluon.tabular`, for Tab2D and its preprocessor).
