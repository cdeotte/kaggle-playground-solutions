"""BatchEnsemble MLP with supervised source pretraining. Shared by ../field_be (field view) and ../pair_be (pair view).

Input: the feature cache of ../transfer_features (competition train + test rows, then the labelled source rows: the
original data.csv and the replica worlds M32_1M / MB32_1M).
Network: embeddings of the 35 categorical ids (dim 12) [+ the 154 pair ids (dim 4), pair view] and the 53 numeric
features, each as x, sin(2 pi f x), cos(2 pi f x) with 4 learned frequencies f -> LayerNorm -> 3 BatchEnsemble ReLU layers
(width 384, dropout 0.1, k members: shared weights, per-member input / output scaling and bias) -> one logit per member.
Prediction = mean of the k member sigmoids. Training: BCE on every member logit, AdamW (weight decay 0.01), batch 1024,
bf16 autocast, grad clip 1.
1. Source pretraining (no competition labels), done once and shared by all folds: 20 epochs on the source rows, lr 1e-3
   with a cosine decay to 5e-5, seed 911. The numeric scaler (median / IQR, clipped to +-20) is fitted on the source rows
   and kept for fine-tuning.
2. Per outer fold k, starting from the pretrained weights, lr 3e-4 x (0.1 + 0.9 x cosine) planned over 20 epochs:
   - epoch selection: train on the first split of StratifiedKFold(5, shuffle=True, random_state=42) over the outer-train
     rows, score the inner holdout after every epoch, stop after 6 epochs without improvement (seed 100 + 1000k);
   - retrain on all outer-train rows for the selected number of epochs (seed 200 + 1000k).
3. Alignment (label-free): z = logit(p) of the fold model; mu, sd = mean and std of z over the test rows; valid and test
   become sigmoid(3 (z - mu) / sd). OOF = the aligned held-out folds; test = mean of the 5 aligned fold vectors.
Runs are not bit-deterministic (bf16, TF32, CUDA randperm)."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

CACHE = HERE.parent / 'transfer_features' / 'cache'
DEV = 'cuda'
BASE = dict(emb=12, pemb=4, width=384, depth=3, dropout=0.10, batch=1024,
            pre_lr=1e-3, pre_epochs=20, pre_seed=911,     # source pretraining
            lr=3e-4, epochs=20, patience=6, seed=100)     # fold fine-tuning
CONFIGS = {'field': dict(BASE, pair=False, k=16), 'pair': dict(BASE, pair=True, k=8)}
ALIGN_SCALE = 3.0


class BELinear(nn.Module):
    """BatchEnsemble linear layer: shared W, per-member input scale r, output scale s and bias b."""

    def __init__(self, din, dout, k, first=False):
        super().__init__()
        self.w = nn.Linear(din, dout, bias=False)
        self.r = nn.Parameter(torch.ones(k, din))
        self.s = nn.Parameter(torch.ones(k, dout))
        self.b = nn.Parameter(torch.zeros(k, dout))
        with torch.no_grad():
            if first:
                self.r.normal_(0, 0.5)
            else:
                self.r.copy_(torch.randint(0, 2, self.r.shape) * 2 - 1)
            self.s.copy_(torch.randint(0, 2, self.s.shape) * 2 - 1)

    def forward(self, x):  # [B, 1 or k, din] -> [B, k, dout]
        return self.w(x * self.r) * self.s + self.b


class Net(nn.Module):
    def __init__(self, meta, cfg):
        super().__init__()
        self.dropout = cfg['dropout']
        self.cat = nn.Embedding(meta['vocab'], cfg['emb'])
        nn.init.normal_(self.cat.weight, std=0.1)
        self.pair = nn.Embedding(meta['pair_vocab'], cfg['pemb']) if cfg['pair'] else None
        if self.pair is not None:
            nn.init.normal_(self.pair.weight, std=0.03)
        nf = len(meta['num_names'])
        self.freq = nn.Parameter(torch.randn(nf, 4) * 0.15)
        din = len(meta['cat_names']) * cfg['emb'] + nf * 9
        if cfg['pair']:
            din += len(meta['pairs']) * cfg['pemb']
        self.norm = nn.LayerNorm(din)
        w, k = cfg['width'], cfg['k']
        self.blocks = nn.ModuleList([BELinear(din if j == 0 else w, w, k, j == 0) for j in range(cfg['depth'])])
        self.head_w = nn.Parameter(torch.randn(k, w) / math.sqrt(w))
        self.head_b = nn.Parameter(torch.zeros(k))

    def forward(self, cats, nums, pairs=None):
        z = nums.unsqueeze(-1) * self.freq * (2 * math.pi)
        pieces = [self.cat(cats).flatten(1), nums, z.sin().flatten(1), z.cos().flatten(1)]
        if self.pair is not None:
            pieces.append(self.pair(pairs).flatten(1))
        x = self.norm(torch.cat(pieces, dim=1))[:, None, :]
        for layer in self.blocks:
            x = F.dropout(F.relu(layer(x)), self.dropout, self.training)
        return (x * self.head_w).sum(-1) + self.head_b  # [B, k] member logits


class Data:
    """The transfer_features cache on the GPU (all rows)."""

    def __init__(self, with_pairs):
        self.meta = json.loads((CACHE / 'meta.json').read_text())
        self.cat = torch.as_tensor(np.load(CACHE / 'cat.npy').astype(np.int64), device=DEV)
        self.pair = torch.as_tensor(np.load(CACHE / 'pair.npy').astype(np.int64), device=DEV) if with_pairs else None
        self.raw = np.load(CACHE / 'num.npy')
        self.nums = None

    def set_scaler(self, mu, sd):
        self.nums = torch.as_tensor(np.clip((self.raw - mu) / sd, -20, 20).astype(np.float32), device=DEV)

    def logits(self, model, idx):
        return model(self.cat[idx], self.nums[idx], self.pair[idx] if self.pair is not None else None)


def source_scaler(raw, rows):
    """Median / IQR per numeric feature (std, at least 1, where the IQR is ~0)."""
    x = raw[rows]
    mu = np.median(x, axis=0)
    q = np.quantile(x, [0.25, 0.75], axis=0)
    sd = q[1] - q[0]
    sd = np.where(sd > 1e-6, sd, np.maximum(x.std(0), 1.0))
    return mu, sd


def set_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr


def run_epoch(model, opt, D, rows, targets, batch):
    """One shuffled epoch of BCE on all member logits. rows: cache row ids, targets: labels of those rows (GPU tensors)."""
    model.train()
    perm = torch.randperm(len(rows), device=DEV)
    rows, targets = rows[perm], targets[perm]
    total = torch.zeros((), device=DEV)
    for j in range(0, len(rows), batch):
        idx = rows[j:j + batch]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = D.logits(model, idx)
        loss = F.binary_cross_entropy_with_logits(z.float(), targets[j:j + batch, None].expand_as(z))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.detach() * len(idx)
    return float(total) / len(rows)


@torch.inference_mode()
def predict(model, D, rows):
    """Mean of the k member probabilities."""
    model.eval()
    out = []
    for j in range(0, len(rows), 4096):
        idx = torch.as_tensor(rows[j:j + 4096], device=DEV)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(D.logits(model, idx).float().sigmoid().mean(1).cpu().numpy())
    return np.concatenate(out)


def pretrain(cfg, D, path):
    """Supervised pretraining on the source rows (original + replica worlds) only; no competition labels."""
    m = D.meta
    rows = np.arange(m['pre_start'], m['n_all'])
    pre_y = np.load(CACHE / 'pre_y.npy').astype(np.float32)
    assert len(pre_y) == len(rows)
    torch.manual_seed(cfg['pre_seed'])
    model = Net(m, cfg).to(DEV)
    mu, sd = source_scaler(D.raw, rows)
    D.set_scaler(mu, sd)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['pre_lr'], weight_decay=0.01)
    rows_t, y_t = torch.as_tensor(rows, device=DEV), torch.as_tensor(pre_y, device=DEV)
    for ep in range(cfg['pre_epochs']):
        set_lr(opt, cfg['pre_lr'] * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * ep / cfg['pre_epochs']))))
        loss = run_epoch(model, opt, D, rows_t, y_t, cfg['batch'])
        print(f'source pretraining: epoch {ep + 1}/{cfg["pre_epochs"]} loss {loss:.5f}', flush=True)
    path.parent.mkdir(exist_ok=True)
    torch.save(dict(state_dict={n: v.detach().cpu() for n, v in model.state_dict().items()},
                    mu=torch.as_tensor(mu), sd=torch.as_tensor(sd)), path)
    print(f'wrote {path}', flush=True)


def fit(cfg, D, state, labels, y, k, rows, epochs, holdout=None):
    """Fine-tune from the pretrained weights on `rows` (train row ids).
    With a holdout: epoch selection, returns the best epoch. Without: trains `epochs` epochs and returns the model."""
    inner = holdout is not None
    seed = cfg['seed'] + 1000 * k + (0 if inner else 100)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Net(D.meta, cfg).to(DEV)
    model.load_state_dict(state)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01)
    rows_t = torch.as_tensor(rows, device=DEV)
    y_t = labels[rows_t]
    best_auc, best_ep = -1.0, 0
    for ep in range(epochs):
        # the same planned schedule (over cfg['epochs']) in selection and retraining
        set_lr(opt, cfg['lr'] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * ep / cfg['epochs']))))
        loss = run_epoch(model, opt, D, rows_t, y_t, cfg['batch'])
        if not inner:
            print(f'fold {k} train: epoch {ep + 1}/{epochs} loss {loss:.5f}', flush=True)
            continue
        auc = roc_auc_score(y[holdout], predict(model, D, holdout))
        if auc > best_auc + 1e-7:
            best_auc, best_ep = auc, ep + 1
        print(f'fold {k} selection: epoch {ep + 1} loss {loss:.5f} inner AUC {auc:.6f}', flush=True)
        if ep + 1 - best_ep >= cfg['patience']:
            break
    if inner:
        print(f'fold {k}: selected {best_ep} epochs (inner AUC {best_auc:.6f})', flush=True)
        return best_ep
    return model


def align(valid, test):
    """Label-free readout: standardize the fold model's logits by their mean / std over its test predictions, then
    sigmoid(3 z). Removes the intercept / scale drift between independently fine-tuned fold models."""
    zv, zt = (logit(np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)) for p in (valid, test))
    mu, sd = zt.mean(), zt.std()
    return expit(ALIGN_SCALE * (zv - mu) / sd), expit(ALIGN_SCALE * (zt - mu) / sd)


def main(model_dir, view, name):
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--pretrain-only', action='store_true')
    a = ap.parse_args()
    cfg = CONFIGS[view]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    D = Data(cfg['pair'])
    pre_path = Path(model_dir) / 'pretrained' / 'source.pt'
    if not pre_path.exists():
        pretrain(cfg, D, pre_path)
        torch.cuda.empty_cache()
    if a.pretrain_only:
        return
    pre = torch.load(pre_path, map_location='cpu', weights_only=True)
    D.set_scaler(pre['mu'].numpy(), pre['sd'].numpy())
    y = target()
    f = folds(y)
    assert len(y) == D.meta['n_train']
    labels = torch.as_tensor(y.astype(np.float32), device=DEV)
    test_rows = np.arange(D.meta['n_train'], D.meta['n_train'] + D.meta['n_test'])
    for k in fold_list(a.folds):
        if have_part(model_dir, name, k):
            continue
        outer, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
        i, j = next(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(outer)), y[outer]))
        epochs = fit(cfg, D, pre['state_dict'], labels, y, k, outer[i], cfg['epochs'], holdout=outer[j])
        model = fit(cfg, D, pre['state_dict'], labels, y, k, outer, epochs)
        pv, pt = predict(model, D, va), predict(model, D, test_rows)
        print(f'fold {k}: AUC {roc_auc_score(y[va], pv):.6f}', flush=True)
        save_part(model_dir, name, k, *align(pv, pt))
        del model
        torch.cuda.empty_cache()
    assemble(model_dir, name, f, y)
