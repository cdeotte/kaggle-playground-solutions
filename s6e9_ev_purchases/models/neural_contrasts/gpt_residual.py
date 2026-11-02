#!/usr/bin/env python
"""N of copula_gpt2_contrast: GReaT replica e55 (GPT-2 small trained on the original data only), fine-tuned per outer fold to
learn a correction to the nested response-logit margin R.

Each row is the GReaT text of its 13 predictors, "<field> is <value>" clauses (exact csv spelling) joined by ", ", followed by
", Will_Buy_EV is". Score = R + head(hidden state of the last token), head = Linear(768, 1) initialized at 0; backbone dropout
off; loss = BCE + 0.001 mean(correction^2). Field orders: a bank of 16 (competition order, original order, 14 random); each
epoch every training row draws its order (competition order with probability 1/2, else uniform over the bank).
AdamW (backbone lr 5e-5, wd 0.01; head lr 1e-3, wd 0.1), linear warmup over 5% of a 2-epoch schedule then linear decay,
batch 256, gradient clip 1, bf16 autocast, deterministic algorithms, seed 42 + 100 k. The model after epoch 1 of that
schedule is used. Prediction = sigmoid(mean over 2 views (competition order, original order) of R + correction).
Writes cache/gpt_residual/gpt_runs/lr5e5/fold_<k>/epoch_1/{model.pt, valid.npy, raw_valid.npy, test.npy, raw_test.npy}.
Needs: response_margin.py --link logit, ../gpt2_replica/checkpoints/e55.pt. Ran in the replica environment
(../gpt2_replica/requirements.txt, transformers 4.26.1): the causal SDPA patch replaces GPT2Attention._attn.
Usage: python gpt_residual.py [--folds 0,1,2,3,4]"""
import argparse
import math
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.special import expit  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET  # noqa: E402
from common.gpt_rows import load_replica, load_tokenizer, order_bank, read_frame_text, seed_everything  # noqa: E402
from contrast_data import CONTRAST, GPT_RESIDUAL, fold_ids, ints, labels  # noqa: E402

LR, SCHEDULE_EPOCHS, STOP_EPOCH = 5e-5, 2, 1
N_ORDERS, ORDER_SEED, MAX_TOKENS = 16, 194213, 192
RUN = GPT_RESIDUAL / 'gpt_runs' / 'lr5e5'


@dataclass
class TokenStore:
    codes: np.ndarray
    first: list
    later: list
    suffix: list
    pad_id: int
    maximum_tokens: int

    def encode(self, row, order):
        output = list(self.first[order[0]][int(self.codes[row, order[0]])])
        for column in order[1:]:
            output.extend(self.later[column][int(self.codes[row, column])])
        output.extend(self.suffix)
        assert len(output) <= self.maximum_tokens, len(output)
        return output


def build_token_store(frame, columns, tokenizer, orders, maximum_tokens):
    """Tokens of every distinct clause, so a row's text is the concatenation of its clauses' tokens."""
    codes = np.empty((len(frame), len(columns)), dtype=np.int32)
    first, later = [], []
    for j, column in enumerate(columns):
        code, unique = pd.factorize(frame[column].astype(str), sort=True)
        assert (code >= 0).all()
        codes[:, j] = code
        unique = [str(v) for v in unique.tolist()]
        first.append(tokenizer([f'{column} is {v}' for v in unique], add_special_tokens=False)['input_ids'])
        later.append(tokenizer([f', {column} is {v}' for v in unique], add_special_tokens=False)['input_ids'])
    store = TokenStore(codes, first, later, tokenizer.encode(f', {TARGET} is', add_special_tokens=False),
                       int(tokenizer.pad_token_id), maximum_tokens)
    # the clause cache must equal whole-text BPE
    rng = np.random.default_rng(130013)
    for _ in range(256):
        row, view = int(rng.integers(len(frame))), int(rng.integers(len(orders)))
        text = ', '.join([*(f'{columns[c]} is {frame.iloc[row][columns[c]]}' for c in orders[view]), f'{TARGET} is'])
        assert tokenizer.encode(text, add_special_tokens=False) == store.encode(row, orders[view]), (row, view)
    return store


class RowViewDataset(Dataset):
    def __init__(self, rows, labels, views):
        self.rows, self.labels, self.views = np.asarray(rows, np.int64), labels, np.asarray(views, np.int16)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = int(self.rows[index])
        return row, -1 if self.labels is None else int(self.labels[row]), int(self.views[index])


class ClauseCollator:
    def __init__(self, store, orders):
        self.store, self.orders = store, orders

    def __call__(self, batch):
        rows, labels, views = zip(*batch)
        sequences = [self.store.encode(row, self.orders[view]) for row, view in zip(rows, views)]
        lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
        ids = torch.full((len(batch), int(lengths.max())), self.store.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(sequences):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1
        return ids, mask, lengths, torch.tensor(labels, dtype=torch.long), torch.tensor(rows, dtype=torch.long)


def make_epoch_views(rows, epoch, count, seed, canonical_fraction):
    rng = np.random.default_rng(seed + 1009 * epoch)
    views = rng.integers(0, count, size=len(rows), dtype=np.int16)
    views[rng.random(len(rows)) < canonical_fraction] = 0
    return views


def make_loader(dataset, collator, batch_size, workers, shuffle, seed):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True, collate_fn=collator,
                      generator=torch.Generator().manual_seed(seed), prefetch_factor=2, persistent_workers=False)


def flash_attn(self, query, key, value, attention_mask=None, head_mask=None):
    # with right padding, real positions never attend to padding under a causal mask
    assert attention_mask is None and head_mask is None
    assert self.scale_attn_weights and not self.scale_attn_by_inverse_layer_idx
    out = F.scaled_dot_product_attention(query, key, value, dropout_p=self.attn_dropout.p if self.training else 0., is_causal=True)
    return out, None


def last_hidden(model, batch, fast=True):
    ids, mask, lengths, _, _ = batch
    ids, lengths = ids.cuda(non_blocking=True), lengths.cuda(non_blocking=True)
    kw = {} if fast else {'attention_mask': mask.cuda(non_blocking=True)}
    hidden = model.transformer(input_ids=ids, **kw).last_hidden_state
    return hidden[torch.arange(len(ids), device='cuda'), lengths - 1]


def run_fold(k, tokenizer, collator, n_rows):
    ep = RUN / f'fold_{k}' / f'epoch_{STOP_EPOCH}'
    if (ep / 'test.npy').exists():
        return
    started = time.time()
    f, y = fold_ids(), labels()
    a, b = np.flatnonzero(f != k), np.flatnonzero(f == k)
    test = np.arange(len(f), n_rows)
    q = np.r_[b, test]
    folder = CONTRAST / 'margins' / 'response_logit' / f'fold_{k}'
    margins = np.empty(n_rows, np.float32)
    margins[a] = np.load(folder / 'train.npy').ravel()
    margins[q] = np.load(folder / 'query.npy').ravel()
    all_margins = torch.tensor(margins, device='cuda')
    train_y = np.full(n_rows, -1, np.int8)   # labels exist only for the training rows
    train_y[a] = y[a]
    model = load_replica('e55')
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    seed = 42 + 100 * k
    seed_everything(seed)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    model.cuda().eval()
    check = collator([(int(i), -1, j % 2) for j, i in enumerate(q[:4])])
    with torch.no_grad():
        eager = last_hidden(model, check, fast=False).float()
    for block in model.transformer.h:
        block.attn._attn = types.MethodType(flash_attn, block.attn)
    with torch.no_grad():
        fast = last_hidden(model, check).float()
    assert torch.allclose(eager, fast, atol=2e-4, rtol=2e-4), float((eager - fast).abs().max())
    head = torch.nn.Linear(model.config.n_embd, 1).cuda()
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)

    def loader(rows, views, training=False):
        return make_loader(RowViewDataset(rows, train_y if training else None, views), collator, 256 if training else 512, 2,
                           training, seed)

    def predict(rows):
        model.eval()
        head.eval()
        views_out = []
        for view in (0, 1):
            pred = []
            with torch.no_grad():
                for batch in loader(rows, np.full(len(rows), view, np.int16)):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        correction = head(last_hidden(model, batch)).squeeze(-1).float()
                    pred.append((all_margins[batch[-1].cuda(non_blocking=True)] + correction).cpu().numpy())
            views_out.append(np.concatenate(pred).astype(np.float64))
        return np.mean(views_out, axis=0), np.asarray(views_out)

    if (ep / 'model.pt').exists():
        state = torch.load(ep / 'model.pt', map_location='cpu', weights_only=True)
        model.transformer.load_state_dict(state['transformer'], strict=True)
        head.load_state_dict(state['head'], strict=True)
    else:
        optimizer = torch.optim.AdamW([{'params': model.transformer.parameters(), 'lr': LR, 'weight_decay': .01},
                                       {'params': head.parameters(), 'lr': 1e-3, 'weight_decay': .1}], betas=(.9, .999), eps=1e-8)
        total = math.ceil(len(a) / 256) * SCHEDULE_EPOCHS
        warmup = max(1, int(.05 * total))
        step = 0
        for epoch in range(1, STOP_EPOCH + 1):
            model.train()
            head.train()
            losssum = seen = 0.
            for batch in loader(a, make_epoch_views(a, epoch, N_ORDERS, seed, .5), True):
                factor = min((step + 1) / warmup, max(0., (total - step) / (total - warmup)))
                optimizer.param_groups[0]['lr'] = LR * factor
                optimizer.param_groups[1]['lr'] = 1e-3 * factor
                target = batch[3].cuda(non_blocking=True).float()
                assert bool((target >= 0).all())
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    correction = head(last_hidden(model, batch)).squeeze(-1).float()
                margin = all_margins[batch[-1].cuda(non_blocking=True)]
                loss = F.binary_cross_entropy_with_logits(margin + correction, target) + .001 * correction.square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.transformer.parameters()) + list(head.parameters()), 1.)
                optimizer.step()
                step += 1
                seen += len(target)
                losssum += float(loss.detach()) * len(target)
                if step % 200 == 0 or step == 1:
                    print(f'  fold {k} epoch {epoch} step {step}: loss {losssum / seen:.6f} ({time.time() - started:.0f}s)', flush=True)
        ep.mkdir(parents=True, exist_ok=True)
        torch.save({'transformer': {n: t.detach().cpu() for n, t in model.transformer.state_dict().items()},
                    'head': {n: t.detach().cpu() for n, t in head.state_dict().items()}}, ep / 'model.pt')
    for role, rows in (('valid', b), ('test', test)):
        p, raw = predict(rows)
        np.save(ep / f'raw_{role}.npy', raw)
        np.save(ep / f'{role}.npy', expit(p))
    print(f'GPT-2 residual fold {k} done ({time.time() - started:.0f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    frame, _ = read_frame_text()
    columns = list(frame.columns)
    tokenizer = load_tokenizer()
    orders = order_bank(columns, N_ORDERS, ORDER_SEED)
    collator = ClauseCollator(build_token_store(frame, columns, tokenizer, orders, MAX_TOKENS), orders)
    n_rows = len(frame)
    del frame
    for k in ints(args.folds):
        run_fold(k, tokenizer, collator, n_rows)


if __name__ == '__main__':
    main()
