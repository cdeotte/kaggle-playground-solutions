#!/usr/bin/env python
"""Sample raw rows from a world's generator the way GReaT samples: a label-first prompt 'satisfaction is True,' or
'satisfaction is False,' drawn with the generator's TRAINING prior (recipe.WORLDS[world]['prior']), then free generation:
temperature 0.7, top-k 50, no top-p, at most 200 tokens including the prompt, stop at EOS. Seed 7, batches of 1000 rows,
fp16 weights, 1M rows. Runs with any recent transformers.
Writes raw/<world>/raw.jsonl.gz: one {"y": prompt label, "text": decoded row} per line (parsed by make_world.py).
Usage: python sample_world.py --world M32_1M|MB32_1M"""
import argparse
import gzip
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from recipe import LLM, MILESTONE, WORLDS

HERE = Path(__file__).resolve().parent
SEED, TEMPERATURE, TOP_K, MAX_LENGTH = 7, 0.7, 50, 200


def load_generator(world):
    model = GPT2LMHeadModel(GPT2Config.from_pretrained(LLM))
    state = torch.load(HERE / 'checkpoints' / world / f'epoch_{MILESTONE:03d}.pt', map_location='cpu', weights_only=True)
    res = model.load_state_dict(state, strict=False)  # transformers 4.26 state dicts carry old attention-mask buffers
    assert not [k for k in res.missing_keys if k != 'lm_head.weight'], res.missing_keys
    assert not [k for k in res.unexpected_keys if not k.endswith(('attn.bias', 'masked_bias'))], res.unexpected_keys
    model.tie_weights()
    return model.cuda().eval().half()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', required=True, choices=list(WORLDS))
    ap.add_argument('--n', type=int, default=1_000_000)
    ap.add_argument('--batch', type=int, default=1000, help='lower it if 1000 rows do not fit (the draws then differ)')
    a = ap.parse_args()
    prior = WORLDS[a.world]['prior']
    out = HERE / 'raw' / a.world
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained('gpt2')  # all GPT-2 sizes share this tokenizer
    tok.pad_token = tok.eos_token
    eos = tok.eos_token_id
    model = load_generator(a.world)
    prompt_true, prompt_false = tok('satisfaction is True,')['input_ids'], tok('satisfaction is False,')['input_ids']
    assert len(prompt_true) == len(prompt_false)

    t0, done = time.time(), 0
    with gzip.open(out / 'raw.jsonl.gz', 'wt', encoding='utf-8') as fh:
        while done < a.n:
            m = min(a.batch, a.n - done)
            ys = (np.random.random(m) < prior).astype(int)
            ids = torch.tensor([prompt_true if y else prompt_false for y in ys], dtype=torch.long, device='cuda')
            with torch.inference_mode():
                gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_length=MAX_LENGTH,
                                     do_sample=True, temperature=TEMPERATURE, top_k=TOP_K, top_p=1.0,
                                     pad_token_id=eos, eos_token_id=eos)
            for y, row in zip(ys, gen.tolist()):
                text = tok.decode(row).replace('<|endoftext|>', '').replace('\n', ' ').replace('\r', '')
                fh.write(json.dumps({'y': int(y), 'text': text}, ensure_ascii=False) + '\n')
            done += m
            if (done // a.batch) % 20 == 0 or done >= a.n:
                print(f'{done}/{a.n} rows, {time.time() - t0:.0f}s', flush=True)
    print(f'wrote {out / "raw.jsonl.gz"}', flush=True)


if __name__ == '__main__':
    main()
