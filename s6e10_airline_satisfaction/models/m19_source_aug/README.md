# M19: replica-pretrained field transformer, source-supervised, then fine-tuned per fold

A transformer encoder that reads a row as 21 tokens, one per field; every distinct value of a column is its own token
(numbers are categorical identities). It is built in four steps:
1. **Masked-field pretraining** (no labels) on the features of train + test + original + the two replica worlds
   (3.1M rows): L8, d384, 12 heads, 20 epochs, 25% of the fields masked and predicted.
2. **Source pretraining:** encoder + classification head fit on the labels of the original rows and both worlds (no
   competition labels): 4 epochs, 5% field masking. From here on the encoder runs with 8 heads; the 12-head weights load
   unchanged because no weight shape depends on the head count. This is how our model was trained.
3. **Fold fine-tuning:** per fold, 6 epochs on the outer-train rows with 15% field masking and two seeds; the logits after
   epochs 4, 5 and 6 of both seeds (6 snapshots) are averaged, then sigmoid ("bag2").
4. **Alignment** (label-free): per fold model, p -> sigmoid(3 (logit(p) - mu) / sd), with mu and sd the mean and std of
   that model's own test logits. Test = mean of the 5 aligned fold vectors.

```
python pretrain.py   # steps 1-2, once for all folds -> checkpoints/mfm.pt, checkpoints/source.pt
python train.py      # steps 3-4
```
-> pair `m19_source_aug_bag2_aligned`, OOF AUC 0.961409.

Needs the replica worlds M32_1M and MB32_1M from [../replica_worlds](../replica_worlds)
(`worlds/<world>/train.csv`, `test.csv`, `y.npy`, `y_test.npy`): their features in step 1, their labels in step 2.
The token codes of all rows are cached in `cache/codes.npz`.
