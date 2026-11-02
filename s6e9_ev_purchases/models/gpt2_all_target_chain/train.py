#!/usr/bin/env python
"""GPT-2 all-target-position chain-rule classifier, one model per fold.

The label clause "Will_Buy_EV is No|Yes" is inserted at position p (0..13) among the 13 field clauses of a fixed field order:
    "Age is 36, ..., Will_Buy_EV is Yes, City_Type is Urban, ..."
For a given order and position, the class score is the chain rule over everything that depends on the label:
    chain(p) = [log p(' Yes' | prefix) + log p(suffix values | Yes)] - [log p(' No' | prefix) + log p(suffix values | No)]
(the fields before the label are identical for both labels and cancel; only value tokens of the later fields are scored).
1. Start from a GReaT replica (../gpt2_replica): e55 or d60.
2. Fine-tune on the outer-train rows of the fold, competition column order, label at a uniformly random position (fresh per row
   per epoch): two-way cross-entropy on the LM head's ' No' / ' Yes' logits at the label slot, plus 0.25 x the mean NLL of 4 random
   value tokens after the label (with the true label written). 4 epochs, batch 256, AdamW lr 5e-5 (weight decay 0.01 except
   biases / layer norms), 5% warmup then linear decay, grad clip 1, bf16 autocast, TF32.
3. Score every held-out and test row at all 14 positions in each scored field order; raw = mean of chain(p) over orders and positions.
Configs: e55 = replica e55, scored in 2 orders (competition, original); d60 = replica d60, scored in the competition order.
Writes parts/chain_<config>_s<seed>_f<k>_{valid,test}.npy (raw LLRs); assemble.py turns the runs into the pairs.
Usage: python train.py --config e55|d60 --seed 42 [--folds 0,1,2,3,4]"""
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
from common.gpt_rows import load_replica, load_tokenizer, order_bank, read_frame_text, seed_everything  # noqa: E402
from common.pairs import fold_list, have_part, save_part  # noqa: E402

CONFIGS = {'e55': dict(replica='e55', views=[0, 1]), 'd60': dict(replica='d60', views=[0])}
EPOCHS, BATCH, EVAL_BATCH, LR, WEIGHT_DECAY, WARMUP = 4, 256, 512, 5e-5, 0.01, 0.05
PAIR_WEIGHT, SUFFIX_WEIGHT, SUFFIX_TOKENS = 1.0, 0.25, 4
ORDER_BANK, ORDER_SEED, POSITIONS, TEMPERATURE = 64, 151513, list(range(14)), 1.0
WORKERS, MAX_TOKENS, HEAD_CHUNK = 4, 192, 1024


@dataclass
class EncodedRow:
    ids: list
    answer_position: int
    suffix_value_positions: list


@dataclass
class TokenStore:
    codes: np.ndarray
    first: list           # per column, per distinct value: token ids of '<c> is <v>'
    later: list           # ... of ', <c> is <v>'
    first_offsets: list   # number of tokens before the value in each clause form
    later_offsets: list
    target_first: list    # per label (No, Yes): token ids of 'Will_Buy_EV is No|Yes' and ', Will_Buy_EV is No|Yes'
    target_later: list
    pad_id: int

    def encode(self, row, order, target_position, label):
        ids, suffix_values, answer_position = [], [], -1
        for slot in range(len(order) + 1):
            if slot == target_position:
                clause = self.target_first[label] if not ids else self.target_later[label]
                ids.extend(clause)
                answer_position = len(ids) - 1                         # the ' No' / ' Yes' token
            if slot == len(order):
                continue
            c = order[slot]
            first = not ids
            clause = (self.first if first else self.later)[c][int(self.codes[row, c])]
            begin = len(ids)
            ids.extend(clause)
            if slot >= target_position:
                suffix_values.extend(range(begin + (self.first_offsets if first else self.later_offsets)[c], begin + len(clause)))
        assert answer_position >= 1 and len(ids) <= MAX_TOKENS
        return EncodedRow(ids, answer_position, suffix_values)


def build_token_store(frame, columns, tok, orders):
    codes = np.empty((len(frame), len(columns)), np.int32)
    first, later, first_off, later_off = [], [], [], []
    for j, c in enumerate(columns):
        code, uniq = pd.factorize(frame[c].astype(str), sort=True)
        codes[:, j] = code
        uniq = [str(v) for v in uniq.tolist()]
        fh, lh = tok.encode(f'{c} is', add_special_tokens=False), tok.encode(f', {c} is', add_special_tokens=False)
        first.append(tok([f'{c} is {v}' for v in uniq], add_special_tokens=False)['input_ids'])
        later.append(tok([f', {c} is {v}' for v in uniq], add_special_tokens=False)['input_ids'])
        assert all(x[:len(fh)] == fh for x in first[-1]) and all(x[:len(lh)] == lh for x in later[-1]), c
        first_off.append(len(fh))
        later_off.append(len(lh))
    target_first = [tok.encode(f'{TARGET} is {v}', add_special_tokens=False) for v in ('No', 'Yes')]
    target_later = [tok.encode(f', {TARGET} is {v}', add_special_tokens=False) for v in ('No', 'Yes')]
    for lab, v in enumerate(('No', 'Yes')):                            # the label is one token after the clause head
        assert target_first[lab] == tok.encode(f'{TARGET} is', add_special_tokens=False) + tok.encode(f' {v}', add_special_tokens=False)
        assert target_later[lab] == tok.encode(f', {TARGET} is', add_special_tokens=False) + tok.encode(f' {v}', add_special_tokens=False)
    store = TokenStore(codes, first, later, first_off, later_off, target_first, target_later, int(tok.pad_token_id))
    rng = np.random.default_rng(151500)                                 # clause cache == tokenizing the whole text
    for _ in range(512):
        row, view, pos, lab = int(rng.integers(len(frame))), int(rng.integers(len(orders))), int(rng.integers(14)), int(rng.integers(2))
        clauses = [f'{columns[c]} is {frame.iloc[row][columns[c]]}' for c in orders[view]]
        clauses.insert(pos, f'{TARGET} is {"Yes" if lab else "No"}')
        assert store.encode(row, orders[view], pos, lab).ids == tok.encode(', '.join(clauses), add_special_tokens=False)
    return store


class RowSpecDataset(Dataset):
    def __init__(self, rows, labels, views, positions):
        self.rows, self.labels = np.asarray(rows, np.int64), labels
        self.views, self.positions = np.asarray(views, np.int16), np.asarray(positions, np.int16)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = int(self.rows[i])
        return row, (-1 if self.labels is None else int(self.labels[row])), int(self.views[i]), int(self.positions[i])


def deterministic_sample(positions, count, seed):
    if count <= 0 or len(positions) <= count:
        return positions
    choice = np.sort(np.random.default_rng(seed).choice(len(positions), size=count, replace=False))
    return [positions[i] for i in choice]


class TrainCollator:
    def __init__(self, store, orders, epoch, seed):
        self.store, self.orders, self.epoch, self.seed = store, orders, epoch, seed

    def __call__(self, batch):
        rows, labels, views, tpos = zip(*batch)
        enc = [self.store.encode(r, self.orders[v], p, lab) for r, lab, v, p in zip(rows, labels, views, tpos)]
        sel = [deterministic_sample(e.suffix_value_positions, SUFFIX_TOKENS,
                                    self.seed + 1000003 * self.epoch + 1009 * r + 17 * v + 7919 * p)
               for r, v, p, e in zip(rows, views, tpos, enc)]
        width = max(len(e.ids) for e in enc)
        sw = max(1, max(len(s) for s in sel))
        ids = torch.full((len(enc), width), self.store.pad_id, dtype=torch.long)
        att = torch.zeros((len(enc), width), dtype=torch.long)
        answer = torch.empty(len(enc), dtype=torch.long)
        spos = torch.ones((len(enc), sw), dtype=torch.long)
        skeep = torch.zeros((len(enc), sw), dtype=torch.bool)
        for i, (e, s) in enumerate(zip(enc, sel)):
            ids[i, :len(e.ids)] = torch.tensor(e.ids)
            att[i, :len(e.ids)] = 1
            answer[i] = e.answer_position
            if s:
                spos[i, :len(s)] = torch.tensor(s)
                skeep[i, :len(s)] = True
        return ids, att, answer, spos, skeep, torch.tensor(labels, dtype=torch.long)


class EvalCollator:
    """Both labels for every row: the first half of the batch is written with No, the second with Yes."""
    def __init__(self, store, order, target_position):
        self.store, self.order, self.target_position = store, order, target_position

    def __call__(self, batch):
        rows = [b[0] for b in batch]
        enc = [self.store.encode(r, self.order, self.target_position, lab) for lab in (0, 1) for r in rows]
        width = max(len(e.ids) for e in enc)
        sw = max(1, max(1 + len(e.suffix_value_positions) for e in enc))
        ids = torch.full((len(enc), width), self.store.pad_id, dtype=torch.long)
        att = torch.zeros((len(enc), width), dtype=torch.long)
        pos = torch.ones((len(enc), sw), dtype=torch.long)
        keep = torch.zeros((len(enc), sw), dtype=torch.bool)
        for i, e in enumerate(enc):
            ids[i, :len(e.ids)] = torch.tensor(e.ids)
            att[i, :len(e.ids)] = 1
            s = [e.answer_position, *e.suffix_value_positions]
            pos[i, :len(s)] = torch.tensor(s)
            keep[i, :len(s)] = True
        return ids, att, pos, keep


def make_loader(dataset, collator, batch_size, shuffle, seed):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=WORKERS, pin_memory=True, collate_fn=collator,
                      generator=torch.Generator().manual_seed(seed), prefetch_factor=2, persistent_workers=False)


def selected_log_prob(model, hidden, ids, positions, keep, temperature):
    """log p(token at `positions`) from the hidden state one step earlier; LM head in bf16, log-softmax in fp32."""
    b, n = positions.shape
    bi = torch.arange(b, device=hidden.device)[:, None].expand(b, n)
    flat_hidden = hidden[bi, positions - 1].reshape(-1, hidden.shape[-1])
    flat_target = ids[bi, positions].reshape(-1)
    out = torch.zeros(len(flat_target), dtype=torch.float32, device=hidden.device)
    active = torch.nonzero(keep.reshape(-1), as_tuple=False).flatten()
    for s in range(0, len(active), HEAD_CHUNK):
        idx = active[s:s + HEAD_CHUNK]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model.lm_head(flat_hidden[idx])
        logits = logits.float() / temperature
        out[idx] = logits.gather(1, flat_target[idx, None]).squeeze(1) - torch.logsumexp(logits, dim=1)
    return out.reshape(b, n)


def finetune(model, store, orders, fit_rows, y, class_ids, run_seed, k, device):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if name.endswith('bias') or 'ln_' in name or 'layer_norm' in name else decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': WEIGHT_DECAY}, {'params': no_decay, 'weight_decay': 0.0}], lr=LR)
    total = math.ceil(len(fit_rows) / BATCH) * EPOCHS
    warm = round(total * WARMUP)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: max(s, 1) / warm if warm and s < warm else max(0.0, (total - s) / max(total - warm, 1)))
    t0 = time.time()
    for epoch in range(EPOCHS):
        views = np.zeros(len(fit_rows), np.int16)                       # training always uses the competition column order
        positions = np.random.default_rng(run_seed + 5000 + 2029 * epoch).integers(0, 14, size=len(fit_rows), dtype=np.int16)
        loader = make_loader(RowSpecDataset(fit_rows, y, views, positions), TrainCollator(store, orders, epoch, run_seed + 7000),
                             BATCH, True, run_seed + epoch)
        model.train()
        for step, (ids, att, answer, spos, skeep, labels) in enumerate(loader):
            ids, att, answer = ids.to(device, non_blocking=True), att.to(device, non_blocking=True), answer.to(device, non_blocking=True)
            spos, skeep, labels = spos.to(device, non_blocking=True), skeep.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hidden = model.transformer(input_ids=ids, attention_mask=att).last_hidden_state
                pair_logits = model.lm_head(hidden[torch.arange(len(ids), device=device), answer - 1])[:, class_ids]
            pair_loss = F.cross_entropy(pair_logits.float(), labels, reduction='none').sum() / len(ids)
            has_suffix = skeep.any(1)
            if has_suffix.any():
                logp = selected_log_prob(model, hidden, ids, spos, skeep, 1.0)
                suffix_rows = -(logp * skeep).sum(1) / skeep.sum(1).clamp_min(1)
                w = has_suffix.float()
                suffix_loss = (suffix_rows * w).sum() / w.sum()
            else:
                suffix_loss = pair_loss.new_zeros(())
            loss = PAIR_WEIGHT * pair_loss + SUFFIX_WEIGHT * suffix_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step == 0 or (step + 1) % 500 == 0 or step + 1 == len(loader):
                print(f'fold {k} epoch {epoch + 1}/{EPOCHS} step {step + 1}/{len(loader)} loss {float(loss):.5f} '
                      f'[{time.time() - t0:.0f}s]', flush=True)


@torch.inference_mode()
def score(model, rows, store, orders, views, device):
    """chain(p) for every row x scored order x position: float32 [n, len(views), 14]."""
    out = np.empty((len(rows), len(views), len(POSITIONS)), np.float32)
    model.eval()
    for vi, view in enumerate(views):
        for pi, p in enumerate(POSITIONS):
            ds = RowSpecDataset(rows, None, np.full(len(rows), view, np.int16), np.full(len(rows), p, np.int16))
            cursor = 0
            for ids, att, pos, keep in make_loader(ds, EvalCollator(store, orders[view], p), EVAL_BATCH, False, 150000 + 100 * view + p):
                ids, att = ids.to(device, non_blocking=True), att.to(device, non_blocking=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    hidden = model.transformer(input_ids=ids, attention_mask=att).last_hidden_state
                logp = selected_log_prob(model, hidden, ids, pos.to(device), keep.to(device), TEMPERATURE).cpu().numpy().astype(np.float64)
                keep = keep.numpy()
                n = len(ids) // 2
                suffix_keep = keep[:n].copy()
                suffix_keep[:, 0] = False                                    # column 0 = the label token
                no_chain = logp[:n, 0] + (logp[:n] * suffix_keep).sum(1)
                yes_chain = logp[n:, 0] + (logp[n:] * suffix_keep).sum(1)
                out[cursor:cursor + n, vi, pi] = yes_chain - no_chain
                cursor += n
            assert cursor == len(rows)
        print(f'scored view {view} [{len(rows)} rows]', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=list(CONFIGS))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    cfg = CONFIGS[a.config]
    run = f'chain_{a.config}_s{a.seed}'
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
    orders = order_bank(columns, ORDER_BANK, ORDER_SEED)
    store = build_token_store(frame, columns, tok, orders)
    class_ids = torch.tensor([tok.encode(' No', add_special_tokens=False)[0], tok.encode(' Yes', add_special_tokens=False)[0]],
                             dtype=torch.long, device=device)
    test_rows = np.arange(n_train, len(frame))
    for k in fold_list(a.folds):
        if have_part(HERE, run, k):
            continue
        run_seed = a.seed + 100 * k
        seed_everything(run_seed)
        fit_rows, valid_rows = np.flatnonzero(f != k), np.flatnonzero(f == k)
        model = load_replica(cfg['replica'])
        model.config.pad_token_id = tok.pad_token_id
        model.config.use_cache = False
        model.to(device)
        finetune(model, store, orders, fit_rows, y, class_ids, run_seed, k, device)
        raw = {split: score(model, rows, store, orders, cfg['views'], device).astype(np.float64).mean((1, 2))
               for split, rows in (('valid', valid_rows), ('test', test_rows))}
        save_part(HERE, run, k, raw['valid'], raw['test'])
        print(f'{run} fold {k}: AUC {roc_auc_score(y[valid_rows], raw["valid"]):.6f}', flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
