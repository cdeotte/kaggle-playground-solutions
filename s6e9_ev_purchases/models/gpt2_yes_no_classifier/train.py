#!/usr/bin/env python
"""GPT-2 next-token Yes / No classifier, one model per fold.

Every row is written as GReaT text with the 13 fields first and the label prompt last:
    "Age is 36, City_Type is Urban, ..., Range_Anxiety_Level is Low, Will_Buy_EV is"
and the native (tied) LM head scores the next token: score = logit(' Yes') - logit(' No'). No classifier head is added.
1. Start from the e55 GReaT replica (../gpt2_replica).
2. Fine-tune on the outer-train rows of the fold with a two-way cross-entropy on the (' No', ' Yes') logits. Field orders: a bank of
   64 fixed orders (competition column order, original column order, 62 random); each epoch every row gets the competition order
   with probability 1/2, otherwise a random order from the bank. 4 epochs, batch 256, AdamW lr 5e-5 (weight decay 0.01 except
   biases / layer norms), 5% warmup then linear decay, grad clip 1, bf16 autocast, TF32.
3. Score the held-out fold and the test rows in the first 4 orders of the bank and average the 4 scores.
Pair gpt2_yes_no: OOF = sigmoid(score); test = sigmoid(mean of the 5 fold models' scores).
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds, target  # noqa: E402
from common.gpt_rows import assemble_pair, load_replica, load_tokenizer, order_bank, read_frame_text, seed_everything  # noqa: E402
from common.pairs import fold_list, have_part, save_part  # noqa: E402

NAME = 'gpt2_yes_no'
REPLICA, SEED = 'e55', 42
EPOCHS, BATCH, EVAL_BATCH, LR, WEIGHT_DECAY, WARMUP = 4, 256, 1024, 5e-5, 0.01, 0.05
ORDER_BANK, ORDER_SEED, CANONICAL_FRACTION, TTA_VIEWS = 64, 131300, 0.5, [0, 1, 2, 3]
WORKERS, MAX_TOKENS = 4, 192


@dataclass
class TokenStore:
    codes: np.ndarray
    first: list    # per column, per distinct value: token ids of '<c> is <v>'
    later: list    # ... of ', <c> is <v>'
    suffix: list   # token ids of ', Will_Buy_EV is'
    pad_id: int

    def encode(self, row, order):
        out = list(self.first[order[0]][int(self.codes[row, order[0]])])
        for c in order[1:]:
            out.extend(self.later[c][int(self.codes[row, c])])
        out.extend(self.suffix)
        assert len(out) <= MAX_TOKENS
        return out


def build_token_store(frame, columns, tok, orders):
    codes = np.empty((len(frame), len(columns)), np.int32)
    first, later = [], []
    for j, c in enumerate(columns):
        code, uniq = pd.factorize(frame[c].astype(str), sort=True)
        codes[:, j] = code
        uniq = [str(v) for v in uniq.tolist()]
        first.append(tok([f'{c} is {v}' for v in uniq], add_special_tokens=False)['input_ids'])
        later.append(tok([f', {c} is {v}' for v in uniq], add_special_tokens=False)['input_ids'])
    store = TokenStore(codes, first, later, tok.encode(f', {TARGET} is', add_special_tokens=False), int(tok.pad_token_id))
    rng = np.random.default_rng(130013)                                 # clause cache == tokenizing the whole text
    for _ in range(256):
        row, view = int(rng.integers(len(frame))), int(rng.integers(len(orders)))
        text = ', '.join([*(f'{columns[c]} is {frame.iloc[row][columns[c]]}' for c in orders[view]), f'{TARGET} is'])
        assert store.encode(row, orders[view]) == tok.encode(text, add_special_tokens=False)
    return store


class RowViewDataset(Dataset):
    def __init__(self, rows, labels, views):
        self.rows, self.labels, self.views = np.asarray(rows, np.int64), labels, np.asarray(views, np.int16)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = int(self.rows[i])
        return row, (-1 if self.labels is None else int(self.labels[row])), int(self.views[i])


class ClauseCollator:
    def __init__(self, store, orders):
        self.store, self.orders = store, orders

    def __call__(self, batch):
        rows, labels, views = zip(*batch)
        seqs = [self.store.encode(r, self.orders[v]) for r, v in zip(rows, views)]
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        ids = torch.full((len(seqs), int(lengths.max())), self.store.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        return ids, mask, lengths, torch.tensor(labels, dtype=torch.long)


def make_loader(dataset, collator, batch_size, shuffle, seed):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=WORKERS, pin_memory=True, collate_fn=collator,
                      generator=torch.Generator().manual_seed(seed), prefetch_factor=2, persistent_workers=False)


def class_logits(model, ids, mask, lengths, class_ids):
    hidden = model.transformer(input_ids=ids, attention_mask=mask).last_hidden_state
    return model.lm_head(hidden[torch.arange(len(ids), device=ids.device), lengths - 1])[:, class_ids]


def epoch_views(n, epoch, seed):
    rng = np.random.default_rng(seed + 1009 * epoch)
    views = rng.integers(0, ORDER_BANK, size=n, dtype=np.int16)
    views[rng.random(n) < CANONICAL_FRACTION] = 0
    return views


def finetune(model, store, orders, fit_rows, y, class_ids, run_seed, k, device):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if name.endswith('bias') or 'ln_' in name or 'layer_norm' in name else decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': WEIGHT_DECAY}, {'params': no_decay, 'weight_decay': 0.0}], lr=LR)
    total = math.ceil(len(fit_rows) / BATCH) * EPOCHS
    warm = round(total * WARMUP)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: max(s, 1) / warm if warm and s < warm else max(0.0, (total - s) / max(total - warm, 1)))
    collator = ClauseCollator(store, orders)
    t0 = time.time()
    model.train()
    for epoch in range(EPOCHS):
        loader = make_loader(RowViewDataset(fit_rows, y, epoch_views(len(fit_rows), epoch, run_seed + 3000)), collator, BATCH, True,
                             run_seed + epoch)
        model.train()
        for step, (ids, mask, lengths, labels) in enumerate(loader):
            ids, mask = ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            lengths, labels = lengths.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = class_logits(model, ids, mask, lengths, class_ids)
            loss = F.cross_entropy(logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step == 0 or (step + 1) % 500 == 0 or step + 1 == len(loader):
                print(f'fold {k} epoch {epoch + 1}/{EPOCHS} step {step + 1}/{len(loader)} loss {float(loss):.5f} '
                      f'[{time.time() - t0:.0f}s]', flush=True)


@torch.inference_mode()
def score(model, rows, store, orders, class_ids, device):
    """logit(' Yes') - logit(' No') per row and TTA view: float32 [n, 4]."""
    out = np.empty((len(rows), len(TTA_VIEWS)), np.float32)
    model.eval()
    for vi, view in enumerate(TTA_VIEWS):
        ds = RowViewDataset(rows, None, np.full(len(rows), view, np.int16))
        cursor = 0
        for ids, mask, lengths, _ in make_loader(ds, ClauseCollator(store, orders), EVAL_BATCH, False, 91000 + view):
            ids, mask, lengths = ids.to(device, non_blocking=True), mask.to(device, non_blocking=True), lengths.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pair = class_logits(model, ids, mask, lengths, class_ids)
            out[cursor:cursor + len(ids), vi] = (pair[:, 1].float() - pair[:, 0].float()).cpu().numpy()
            cursor += len(ids)
        assert cursor == len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    y = target().astype(np.int64)
    f = folds(y)
    frame, n_train = read_frame_text()
    columns = list(frame.columns)
    tok = load_tokenizer()
    no, yes = tok.encode(' No', add_special_tokens=False), tok.encode(' Yes', add_special_tokens=False)
    assert len(no) == 1 and len(yes) == 1
    class_ids = torch.tensor([no[0], yes[0]], device=device)
    orders = order_bank(columns, ORDER_BANK, ORDER_SEED)
    store = build_token_store(frame, columns, tok, orders)
    test_rows = np.arange(n_train, len(frame))
    for k in fold_list(a.folds):
        if have_part(HERE, NAME, k):
            continue
        run_seed = SEED + 100 * k
        seed_everything(run_seed)
        fit_rows, valid_rows = np.flatnonzero(f != k), np.flatnonzero(f == k)
        model = load_replica(REPLICA)
        model.config.pad_token_id = tok.pad_token_id
        model.config.use_cache = False
        model.to(device)
        finetune(model, store, orders, fit_rows, y, class_ids, run_seed, k, device)
        logit = {split: score(model, rows, store, orders, class_ids, device).mean(axis=1, dtype=np.float64)
                 for split, rows in (('valid', valid_rows), ('test', test_rows))}
        save_part(HERE, NAME, k, logit['valid'], logit['test'])
        print(f'fold {k}: AUC {roc_auc_score(y[valid_rows], logit["valid"]):.6f}', flush=True)
        del model
        torch.cuda.empty_cache()
    assemble_pair(HERE, NAME, f, y, sigmoid=True)


if __name__ == '__main__':
    main()
