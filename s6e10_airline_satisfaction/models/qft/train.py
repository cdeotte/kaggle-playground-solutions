#!/usr/bin/env python
"""Q-FT: a Qwen transformer that reads table rows as embeddings (discriminative), one model per fold.

Row -> 21 feature tokens (qtab.FeatureTokenizer over feats.py codes), in a fresh random order per row during training, + a
learned CLS token at the END -> Qwen3-0.6B-Base via inputs_embeds (causal attention, so CLS sees every feature) -> final
hidden state at CLS -> linear head -> logit; binary cross-entropy.
LoRA rank 16 (alpha 32) on every attention / MLP projection; AdamW, lr 2e-4 for the adapters and 1e-3 for tokenizer / CLS /
head, 3% linear warmup then linear decay to 0, batch 512, 3 epochs, bf16 autocast with fp32 master weights, grad clip 1.
Prediction = mean logit over the data.csv field order + 4 random orders.
Pair: OOF and test logits (test = mean of the 5 fold models).
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from feats import feature_codes  # noqa: E402
from qtab import FeatureTokenizer, column_name_init, load_qwen_lora  # noqa: E402

NAME = 'qft'
EPOCHS, BATCH, LR_LORA, LR_HEAD, TTA_ORDERS = 3, 512, 2e-4, 1e-3, 4


def run_fold(k, codes, nums, meta, y, f):
    torch.manual_seed(1000 + k)
    tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
    te = np.arange(meta['ntrain'], meta['ntrain'] + meta['ntest'])
    yt = torch.as_tensor(y.astype(np.float32)).cuda()
    bb, emb, tok = load_qwen_lora()
    d = emb.shape[1]
    gain = float(emb.float().pow(2).mean().sqrt())
    tokz = FeatureTokenizer(meta['vocab'], meta['numeric'], d, gain, col_init=column_name_init(tok, emb, meta['cols'], gain))
    head = torch.nn.Linear(d, 1)
    bb, tokz, head = bb.cuda(), tokz.cuda(), head.cuda()
    cls = torch.nn.Parameter((torch.randn(d) * gain).cuda())
    nf = len(meta['cols'])
    groups = [dict(params=list(tokz.parameters()) + [cls] + list(head.parameters()), lr=LR_HEAD, weight_decay=0.0),
              dict(params=[p for p in bb.parameters() if p.requires_grad], lr=LR_LORA, weight_decay=0.0)]
    opt = torch.optim.AdamW(groups, fused=True)
    spe = len(tr) // BATCH
    n_steps = EPOCHS * spe
    warm = max(1, int(0.03 * n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / n_steps))

    def forward(rows, perm=None):
        r = torch.as_tensor(rows, device='cuda')
        x = tokz(codes[r], nums[r])                                               # [B, nf, d]
        if perm is not None:
            x = torch.gather(x, 1, perm[:, :, None].expand(-1, -1, d))
        x = torch.cat([x, cls[None, None].expand(x.shape[0], 1, d)], 1)
        h = bb(inputs_embeds=x, use_cache=False).last_hidden_state[:, -1]
        return head(h).squeeze(-1)

    @torch.inference_mode()
    def predict(rows, bs=2048):
        bb.eval()
        g = torch.Generator(device='cuda')
        g.manual_seed(777)
        out = np.zeros(len(rows))
        orders = [None] + [torch.argsort(torch.rand(1, nf, device='cuda', generator=g), 1) for _ in range(TTA_ORDERS)]
        for o in orders:
            for s0 in range(0, len(rows), bs):
                rr = rows[s0:s0 + bs]
                perm = None if o is None else o.expand(len(rr), -1)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out[s0:s0 + len(rr)] += forward(rr, perm).float().cpu().numpy() / len(orders)
        bb.train()
        return out

    rng = np.random.default_rng(2000 + k)
    step = 0
    bb.train()
    while step < n_steps:
        rows_perm = tr[rng.permutation(len(tr))]
        for s0 in range(0, len(rows_perm) - BATCH + 1, BATCH):
            rr = rows_perm[s0:s0 + BATCH]
            perm = torch.argsort(torch.rand(len(rr), nf, device='cuda'), 1)        # a fresh random field order per row
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logit = forward(rr, perm)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), yt[torch.as_tensor(rr, device='cuda')])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g_ in groups for p in g_['params']], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f'fold {k}: step {step}/{n_steps} loss {loss.item():.4f}', flush=True)
            if step >= n_steps:
                break
    return predict(va), predict(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    codes, nums, meta = feature_codes()
    codes, nums = torch.as_tensor(codes).cuda(), torch.as_tensor(nums).cuda()
    y = target()
    f = folds(y)
    for k in fold_list(a.folds):
        if have_part(HERE, NAME, k):
            continue
        pv, pt = run_fold(k, codes, nums, meta, y, f)
        save_part(HERE, NAME, k, pv, pt)
        print(f'fold {k}: AUC {roc_auc_score(y[f == k], pv):.6f}', flush=True)
        torch.cuda.empty_cache()
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
