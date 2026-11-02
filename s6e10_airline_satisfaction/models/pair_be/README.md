# BatchEnsemble MLP with source pretraining (pair view)

The model of `../field_be` (same code: `../field_be/be_transfer.py`) on the "pair" view: the field view plus an embedding
(dim 4) for each of 154 exact field-pair ids from `../transfer_features` (all pairs of the 13 ratings and 4 categoricals,
and Flight Distance (a route id) / its first GPT-2 token / the 5-year Age bin crossed with the categoricals, wifi and online
boarding), with 8 members instead of 16. Same source pretraining on the original data + replica worlds, fold fine-tuning
with inner epoch selection (6, 6, 7, 6, 6 epochs in our run) and label-free test-moment alignment.

```
python ../transfer_features/prepare.py   # once (shared with field_be); needs ../replica_worlds
python train.py --pretrain-only          # source pretraining -> pretrained/source.pt (run it first if folds run in parallel)
python train.py                          # 5 folds -> preds/pair_be_transfer_aligned_{oof,test}.npy
```

Pair `pair_be_transfer_aligned` (aligned scores in [0, 1], not calibrated probabilities), OOF AUC 0.960170.
