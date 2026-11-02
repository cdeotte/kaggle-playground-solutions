# BatchEnsemble MLP with source pretraining (field view)

A BatchEnsemble MLP (TabM-style: shared weights with per-member input / output scaling; 16 members) on the "field" view of
a row from `../transfer_features`: an embedding for each of 35 categorical ids (the raw values, GPT-2 tokens of the
numbers, fixed bins) and 53 numeric features with periodic (sin / cos) encodings.
1. **Source pretraining:** BCE on the 2.1M labelled source rows (original data + the two replica worlds of
   `../replica_worlds`), no competition labels; 20 epochs, done once and shared by all folds.
2. **Fine-tuning per fold** from the pretrained weights (lr 3e-4): the number of epochs is selected on an inner split of
   the outer-train rows (up to 20, patience 6), then the network is trained again from the pretrained weights on all
   outer-train rows for that many epochs (14, 12, 16, 11, 10 in our run).
3. **Alignment** (label-free): each fold model's logits are standardized by the mean / std of its own test logits and
   mapped through sigmoid(3 z). This removes the intercept / scale drift between the fold models (it changes the pooled
   OOF AUC, not the per-fold AUC). OOF = the aligned held-out folds; test = mean of the 5 aligned fold vectors.

```
python ../transfer_features/prepare.py   # once; needs ../replica_worlds
python train.py --pretrain-only          # source pretraining -> pretrained/source.pt (run it first if folds run in parallel)
python train.py                          # 5 folds -> preds/field_be_transfer_aligned_{oof,test}.npy
```

Pair `field_be_transfer_aligned` (aligned scores in [0, 1], not calibrated probabilities), OOF AUC 0.960886.
`be_transfer.py` has the model and all settings; `../pair_be` runs the same code on the pair view.
