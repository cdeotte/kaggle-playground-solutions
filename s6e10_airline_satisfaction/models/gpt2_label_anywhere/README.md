# GPT-2 label anywhere (and the target-last "Detective")

Like `../gpt2_target_first`, but during fine-tuning the label is just one of the 22 fields, at a random position in every
row, exactly as GReaT writes its training text. One fine-tuned model can then be read in two ways:
- **any:** score the whole row with label True and with label False over 30 random orders (label anywhere), and compare;
- **last:** put the label last, so the model reads all 21 fields and then predicts the label token (target last).

| Command | Pair | Readout | OOF AUC |
|---|---|---|---:|
| `python train.py --config small_24ep` | gpt2_small_anywhere_24ep | any, mean of per-order LLRs | 0.962452 |
| `python train.py --config medium_8ep` | gpt2_medium_anywhere_8ep | any, mean of per-order LLRs | 0.962407 |
| `python train.py --config small_8ep` | gpt2_small_anywhere_8ep_any | any, log-sum-exp | 0.962032 |
| (same run) | gpt2_small_anywhere_8ep_last | last (Detective) | 0.960961 |

Needs the replicas from `../gpt2_replica`. Pairs hold log-likelihood ratios (logits).
