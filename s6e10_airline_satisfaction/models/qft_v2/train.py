#!/usr/bin/env python
"""Q-FT v2: an order-free ModernBERT that reads a table row as a set (+1e-3 AUC over Q-FT, ~5x cheaper).

Changes from Q-FT (../qft):
  1. Order-free encoder: ModernBERT-base (bidirectional) with every token at position 0, so the row is a set; no shuffling and
     a single forward pass at prediction time.
  2. Side features (side_features.py): extra numbers on every feature token + 2 summary tokens.
  3. Pretraining on the table itself, without labels: replaced-token detection (RTD). For a random 15% of the cells of a row,
     the cell (code, number and side features) is swapped in from another random row; the model predicts which cells were
     replaced. Rows = train + test + original features; 3 epochs; done ONCE and shared by all folds (it reads no labels).
Per fold: start from the pretrained weights, fine-tune all 149M weights for 2 epochs with BCE (AdamW, lr 5e-5 backbone /
1e-3 tokenizer + CLS + head, 3% warmup then linear decay, batch 512, bf16 autocast, grad clip 1).
Sequence = CLS + 21 feature tokens + 2 summary tokens; head = mean of the 23 non-CLS output vectors.
Pair: OOF and test logits (test = mean of the 5 fold models).
Usage: python side_features.py; python train.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / 'qft'))
sys.path.insert(0, str(HERE))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from feats import feature_codes  # noqa: E402  (../qft/feats.py)
from model import MeanHead, SideTokenizer, load_modernbert, run_backbone  # noqa: E402
from qtab import column_name_init  # noqa: E402  (../qft/qtab.py)

NAME = 'qft_v2'
BATCH, LR_BB, LR_HEAD = 512, 5e-5, 1e-3
RTD_EPOCHS, RTD_RATE, EPOCHS = 3, 0.15, 2
PRETRAIN = HERE / 'features' / 'pretrain_rtd.pt'


class Net:
    """Tokenizer + CLS + ModernBERT + head, and the data tensors (all rows: train, test, original)."""

    def __init__(self, D):
        self.D = D
        self.bb, emb, tok = load_modernbert()
        d = emb.shape[1]
        gain = float(emb.float().pow(2).mean().sqrt())
        m = D['meta']
        self.tokz = SideTokenizer(m['vocab'], m['numeric'], d, gain, D['side'].shape[2], [g.shape[1] for g in D['globs']],
                                  col_init=column_name_init(tok, emb, m['cols'], gain))
        self.head = MeanHead(d)
        self.bb, self.tokz, self.head = self.bb.cuda(), self.tokz.cuda(), self.head.cuda()
        self.cls = torch.nn.Parameter((torch.randn(d) * gain).cuda())
        self.nf = len(m['cols'])

    def groups(self, extra=()):
        return [dict(params=list(self.tokz.parameters()) + [self.cls] + list(self.head.parameters()) + list(extra), lr=LR_HEAD,
                     weight_decay=0.0),
                dict(params=list(self.bb.parameters()), lr=LR_BB, weight_decay=0.0)]

    def hidden(self, codes, nums, side, globs):
        x = self.tokz(codes, nums, side, globs)                                    # [B, 21 + 2, d]
        x = torch.cat([self.cls[None, None].expand(x.shape[0], 1, x.shape[2]), x], 1)
        return run_backbone(self.bb, x)[:, 1:]                                      # drop CLS: feature + summary tokens

    def rows(self, r):
        D = self.D
        return D['codes'][r], D['nums'][r], D['side'][r], [g[r] for g in D['globs']]

    def logit(self, r):
        return self.head(self.hidden(*self.rows(r)))


def schedule(opt, n_steps):
    warm = max(1, int(0.03 * n_steps))
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / max(n_steps, 1)))


def pretrain_rtd(D):
    """Replaced-token detection on all rows' features (train + test + original). Label-free; saved once for all folds."""
    torch.manual_seed(3000)
    net = Net(D)
    rtd_head = torch.nn.Linear(net.cls.shape[0], 1).cuda()
    groups = net.groups(rtd_head.parameters())
    opt = torch.optim.AdamW(groups, fused=True)
    N = D['codes'].shape[0]
    n_steps = int(round(RTD_EPOCHS * N / BATCH))
    sched = schedule(opt, n_steps)
    rng = np.random.default_rng(3000)
    col = torch.arange(net.nf, device='cuda')[None]
    net.bb.train()
    for st in range(n_steps):
        r = torch.as_tensor(rng.integers(0, N, BATCH), device='cuda')
        M = torch.rand(BATCH, net.nf, device='cuda') < RTD_RATE                    # cells to replace
        donor = torch.randint(N, (BATCH, net.nf), device='cuda')                   # a random other row per cell
        cb = torch.where(M, D['codes'][donor, col], D['codes'][r])
        nb = torch.where(M, D['nums'][donor, col], D['nums'][r])
        sb = torch.where(M[..., None], D['side'][donor, col], D['side'][r])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = net.hidden(cb, nb, sb, [g[r] for g in D['globs']])
            out = rtd_head(h[:, :net.nf]).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), M.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 1.0)
        opt.step()
        sched.step()
        if (st + 1) % 200 == 0:
            print(f'pretrain RTD: step {st + 1}/{n_steps} loss {loss.item():.4f}', flush=True)
    PRETRAIN.parent.mkdir(exist_ok=True)
    torch.save(dict(tokz=net.tokz.state_dict(), bb=net.bb.state_dict(), cls=net.cls.detach()), PRETRAIN)


def run_fold(k, D, y, f):
    torch.manual_seed(1000 + k)
    net = Net(D)
    sv = torch.load(PRETRAIN, map_location='cuda')
    net.tokz.load_state_dict(sv['tokz'])
    net.bb.load_state_dict(sv['bb'])
    with torch.no_grad():
        net.cls.copy_(sv['cls'])
    tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
    te = np.arange(D['meta']['ntrain'], D['meta']['ntrain'] + D['meta']['ntest'])
    yt = torch.as_tensor(y.astype(np.float32)).cuda()
    groups = net.groups()
    opt = torch.optim.AdamW(groups, fused=True)
    n_steps = EPOCHS * (len(tr) // BATCH)
    sched = schedule(opt, n_steps)
    rng = np.random.default_rng(2000 + k)
    step = 0
    net.bb.train()
    while step < n_steps:
        perm = tr[rng.permutation(len(tr))]
        for s0 in range(0, len(perm) - BATCH + 1, BATCH):
            r = torch.as_tensor(perm[s0:s0 + BATCH], device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logit = net.logit(r)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), yt[r])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f'fold {k}: step {step}/{n_steps} loss {loss.item():.4f}', flush=True)
            if step >= n_steps:
                break

    @torch.inference_mode()
    def predict(rows, bs=2048):
        net.bb.eval()
        out = np.zeros(len(rows))
        for s0 in range(0, len(rows), bs):
            r = torch.as_tensor(rows[s0:s0 + bs], device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out[s0:s0 + len(r)] = net.logit(r).float().cpu().numpy()
        return out
    return predict(va), predict(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    codes, nums, meta = feature_codes(include_original=True)
    z = np.load(HERE / 'features' / 'side.npz')
    D = dict(meta=meta, codes=torch.as_tensor(codes).cuda(), nums=torch.as_tensor(nums).cuda(),
             side=torch.as_tensor(np.concatenate([z['side_lf'], z['side_ol']], 2)).cuda(),
             globs=[torch.as_tensor(z['g_eng']).cuda(), torch.as_tensor(z['g_orig']).cuda()])
    assert D['side'].shape[0] == codes.shape[0], 'side.npz rows must be train, test, original (run side_features.py)'
    if not PRETRAIN.exists():
        pretrain_rtd(D)
    y = target()
    f = folds(y)
    for k in fold_list(a.folds):
        if have_part(HERE, NAME, k):
            continue
        pv, pt = run_fold(k, D, y, f)
        save_part(HERE, NAME, k, pv, pt)
        print(f'fold {k}: AUC {roc_auc_score(y[f == k], pv):.6f}', flush=True)
        torch.cuda.empty_cache()
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
