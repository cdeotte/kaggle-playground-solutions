# GPT-2 all-target-position chain

The label clause `Will_Buy_EV is No|Yes` can sit anywhere among the 13 field clauses (the generator's training text put it at a
random position). For a field order and a label position p, the class score uses the chain rule over everything that depends on
the label: `log p(' Yes' | fields before) + log p(values after | Yes)` minus the same with `No`; the fields before the label are
identical for both labels and cancel.

1. Start from a GReaT replica (`../gpt2_replica`).
2. Fine-tune per fold on the training rows (competition column order, label at a random position): two-way cross-entropy on the
   LM head's ` No` / ` Yes` logits plus 0.25 x the NLL of 4 random value tokens after the label. 4 epochs, batch 256, lr 5e-5.
3. Score every held-out and test row at all 14 label positions and average the chain LLRs.

- **gpt2_chain_e55**: e55 replica, seed 42, scored in 2 field orders (competition and original column order).
- **gpt2_chain_d60_s42 / gpt2_chain_d60_s43**: d60 replica, scored in the competition order, seeds 42 / 43. The test column of
  `gpt2_chain_d60_s42` is the mean LLR of the 25 fold models of seeds 42-46.

```
python train.py --config e55 --seed 42
python train.py --config d60 --seed 42
python train.py --config d60 --seed 43
python train.py --config d60 --seed 44
python train.py --config d60 --seed 45
python train.py --config d60 --seed 46
python assemble.py
```

Each `train.py` run trains the 5 folds (`--folds 0,1` runs a subset; finished folds are kept in `parts/` and skipped) and saves
raw LLRs; `assemble.py` writes the pairs.

| Pair | OOF AUC |
|---|---:|
| gpt2_chain_d60_s42 | 0.946107 |
| gpt2_chain_d60_s43 | 0.946107 |
| gpt2_chain_e55 | 0.945997 |

Needs `../gpt2_replica/checkpoints/e55.pt` and `d60.pt`. Ran with transformers 4.26.1 and torch 2.5.1 (the replica environment,
`../gpt2_replica/requirements.txt`). Per fold on one GPU: about 1 hour of training, then about 1.8 hours of scoring per field order
(14 label positions, held-out fold and test rows).
