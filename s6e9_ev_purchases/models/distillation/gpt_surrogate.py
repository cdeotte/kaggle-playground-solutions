#!/usr/bin/env python
"""GPT-2 small surrogate trained as a causal LM on the 10,000 original rows only (the start of the native GPT KD model).

Each row (Buyer_ID dropped, the original label included as a field) is written as '<field> is <value>' joined by ', '
in a new random field order every epoch, empty cells as 'None', EOS appended; the loss is the mean over non-padding
predicted tokens. Public GPT-2 small, 60 epochs, effective batch 128 (micro-batches of 32), AdamW lr 5e-5 wd 0.01,
100 warmup updates then constant, gradient clip 1, bf16 autocast with fp32 parameters, seed 42 (~25 min on one GPU).
Writes checkpoints/original_gpt2_e60 (save_pretrained).
Usage: python gpt_surrogate.py"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import DATA_DIR, ORIGINAL_FILE  # noqa: E402

SURROGATE = HERE/'checkpoints'/'original_gpt2_e60'
CFG = dict(seed=42, epochs=60, microbatch=32, effective_batch=128, lr=5e-5, warmup_updates=100, wd=.01,
           adam_betas=[.9, .999], adam_eps=1e-8, gradient_clip=1.)


def original_rows():
    with (DATA_DIR/ORIGINAL_FILE).open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields = [key for key in reader.fieldnames if key != 'Buyer_ID']
        assert len(fields) == 14 and fields[-1] == 'Will_Buy_EV'
        rows = [[row[key].strip() if row[key].strip() else 'None' for key in fields] for row in reader]
    assert len(rows) == 10000
    return fields, np.asarray(rows, dtype=object)


def render_rows(fields, rows, seed, epoch):
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 12901]))
    docs = []
    for row in rows:
        order = rng.permutation(len(fields))
        docs.append(', '.join(f'{fields[j]} is {row[j]}' for j in order))
    return docs


def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    return tok


def encode_rows(tok, docs):
    result = tok(docs, add_special_tokens=False, truncation=False)['input_ids']
    result = [ids+[tok.eos_token_id] for ids in result]
    assert max(map(len, result)) <= 1024
    return result


def collate(encoded, indexes, pad_id, device):
    import torch
    selected = [encoded[int(i)] for i in indexes]
    length = ((max(map(len, selected))+7)//8)*8
    ids = np.full((len(selected), length), pad_id, dtype=np.int64)
    mask = np.zeros_like(ids)
    for i, values in enumerate(selected):
        ids[i, :len(values)] = values
        mask[i, :len(values)] = 1
    ids = torch.as_tensor(ids, device=device)
    mask = torch.as_tensor(mask, device=device)
    labels = ids.clone()
    labels[mask == 0] = -100
    return dict(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)


def main():
    import torch
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(CFG['seed'])
    torch.cuda.manual_seed_all(CFG['seed'])
    fields, rows = original_rows()
    tok = tokenizer()
    model = AutoModelForCausalLM.from_pretrained('gpt2', attn_implementation='sdpa', use_safetensors=True).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'], betas=tuple(CFG['adam_betas']),
                                  eps=CFG['adam_eps'], fused=True)
    step = 0
    for epoch in range(1, CFG['epochs']+1):
        started = time.time()
        encoded = encode_rows(tok, render_rows(fields, rows, CFG['seed'], epoch))
        order = np.random.default_rng(np.random.SeedSequence([CFG['seed'], epoch, 48721])).permutation(len(rows))
        model.train()
        total_loss, total_tokens = 0., 0
        for j in range(0, len(rows), CFG['effective_batch']):
            ids = order[j:j+CFG['effective_batch']]
            # each micro-batch mean loss is weighted by its number of predicted tokens
            group_tokens = sum(len(encoded[int(i)])-1 for i in ids)
            optimizer.zero_grad(set_to_none=True)
            rate = CFG['lr']*min((step+1)/CFG['warmup_updates'], 1.)
            for group in optimizer.param_groups:
                group['lr'] = rate
            for offset in range(0, len(ids), CFG['microbatch']):
                batch = collate(encoded, ids[offset:offset+CFG['microbatch']], tok.pad_token_id, 'cuda')
                n_tokens = int((batch['labels'][:, 1:] != -100).sum())
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(**batch).loss
                (loss*(n_tokens/group_tokens)).backward()
                total_loss += float(loss.detach())*n_tokens
                total_tokens += n_tokens
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['gradient_clip'])
            optimizer.step()
            step += 1
        print(f'epoch {epoch}: token NLL {total_loss/total_tokens:.5f} ({time.time()-started:.0f} s)', flush=True)
    SURROGATE.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(SURROGATE, safe_serialization=True)
    print(f'saved {SURROGATE}', flush=True)


if __name__ == '__main__':
    main()
