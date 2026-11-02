# GReaT replicas (GPT-2 small / medium / large, GPT-Neo 125M)

A language model fine-tuned as a GReaT model on the original data only (`data.csv`), like the model that generated the
competition data. Every GPT classifier (`../gpt2_target_first`, `../gpt2_label_anywhere`, `../gpt_neo_target_first`, and
through them `../gpt_hybrid`, `../gpt_fold_be`) starts from one of these checkpoints.

```
python train_replica.py --size small    # gpt2,                    60 epochs
python train_replica.py --size medium   # gpt2-medium,             60 epochs
python train_replica.py --size large    # gpt2-large,              30 epochs
python train_replica.py --size neo      # EleutherAI/gpt-neo-125M, 60 epochs
```

Writes `checkpoints/<size>/model.pt`. Needs its own environment with old pins (`requirements.txt`: be-great 0.0.3,
transformers 4.26.1). The classifiers then load the checkpoint with any recent transformers.
