# GReaT replicas (GPT-2 small)

GPT-2 small fine-tuned as a GReaT model (be-great 0.0.3) on the 10,000 original rows only, like the model that generated the
competition data. The GPT-2 classifiers and likelihood features in this repo start from, or score rows with, these checkpoints.

- **e55, e60, e65, e70**: legacy recipe. One run per replica, each with its own schedule of 55 / 60 / 65 / 70 epochs (learning
  rate 5e-5 decaying linearly to 0 at the last epoch); the replica is the model at the end of its run. Trainer seed and data seed
  42, 2 dataloader workers, TF32 on.
- **d60**: the organizer-default Trainer path (only GReaT's own arguments, so data seed None and 0 dataloader workers), TF32 off,
  60 epochs.

Both use batch 16, a fresh random field order every time a row is read, and the transformers 4.26.1 Trainer defaults.

```
python train_replica.py --replica e55
python train_replica.py --replica e60
python train_replica.py --replica e65
python train_replica.py --replica e70
python train_replica.py --replica d60
```

Writes `checkpoints/<replica>.pt` (GPT2LMHeadModel state dict), loaded by `common.gpt_rows.load_replica`. Needs its own
environment with old pins (`requirements.txt`). On one GPU one run takes about 1 hour (e55) to 2 hours (d60).
