"""Masked-token density model: a transformer masked-language model over the 12 exact value token fields (the token
space of lib/tokens.py), trained on train + test predictors only (label-free).

Training: each row gets 1 or 2 of its observed cells replaced by a [MASK] id; cross-entropy on the masked cells;
AdamW lr 1e-3 (weight decay 1e-4), OneCycle schedule, batch 8192, seed 2026. The model used here has d 192, 6 pre-norm
GELU layers, 8 heads, FFN 512, dropout 0.1, 30 epochs (TAG / CONFIG below).
Row channels (each cell masked in turn, its conditional distribution given the rest of the row): pseudo-log-likelihood
of the observed cells, their mean predictive entropy, the minimum log-probability over observed cells, the mean entropy
over missing cells, the numbers of observed and missing cells, then 4 more aggregates and the 24 per-cell
log-probabilities / entropies; all standardized over train + test. The AFN model uses the first 6."""
import numpy as np
import torch
import torch.nn as nn

from common.data import cache_dir
from . import tokens as T

SEED = 2026
TAG = 'd192L6h8e30'
CONFIG = dict(d=192, n_layers=6, n_heads=8, epochs=30)


def _cache():
    return cache_dir(T.FOLDER)


class MLMDensity(nn.Module):
    def __init__(self, vocab_sizes, d=128, n_layers=4, n_heads=8, ff=512, dropout=0.1):
        super().__init__()
        self.vocab_sizes = [int(v) for v in vocab_sizes]
        self.n_fields = len(self.vocab_sizes)
        self.d = d
        # per-field token embedding with one extra row = [MASK]
        self.tok_emb = nn.ModuleList([nn.Embedding(v + 1, d) for v in self.vocab_sizes])
        self.mask_id = {c: self.vocab_sizes[c] for c in range(self.n_fields)}
        self.field_emb = nn.Embedding(self.n_fields, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, ff, dropout=dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.ModuleList([nn.Linear(d, v) for v in self.vocab_sizes])
        for e in self.tok_emb:
            nn.init.normal_(e.weight, 0.0, 0.05)
        nn.init.normal_(self.field_emb.weight, 0.0, 0.05)

    def embed(self, x):
        # x (B, F) long -> (B, F, d)
        B, F_ = x.shape
        h = torch.stack([self.tok_emb[c](x[:, c]) for c in range(F_)], dim=1)
        fpos = self.field_emb(torch.arange(F_, device=x.device))
        return h + fpos.unsqueeze(0)

    def forward_hidden(self, x):
        return self.norm(self.enc(self.embed(x)))  # (B, F, d)

    def forward_logits(self, x):
        h = self.forward_hidden(x)
        return [self.head[c](h[:, c]) for c in range(self.n_fields)]  # list of (B, V_c)


def _na_ids(train, test):
    """Token ids (n, 12) int64 over train + test, the observed mask (n, 12) (True = observed), the vocabulary sizes and
    the id of '__NA__' per field (-1 if the field has no missing values)."""
    ids_tr, ids_te, vsz, vocabs = T.build_token_ids(train, test)
    ids = np.concatenate([ids_tr, ids_te], axis=0).astype(np.int64)
    fields = list(T.FIELDS)
    na_id = {c: (vocabs[fields[c]].index(T.NA_TOKEN) if T.NA_TOKEN in vocabs[fields[c]] else -1)
             for c in range(len(vsz))}
    obs = np.ones_like(ids, dtype=bool)
    for c in range(len(vsz)):
        if na_id[c] >= 0:
            obs[:, c] = ids[:, c] != na_id[c]
    return ids, obs, vsz, na_id


def load_density(vocab_sizes, cache_tag, device="cpu"):
    """Load a cached density model; d / n_layers / ff are read from the checkpoint."""
    sd = torch.load(_cache() / f"density_{cache_tag}.pt", map_location="cpu")
    d = sd["field_emb.weight"].shape[1]
    ff = sd["enc.layers.0.linear1.weight"].shape[0]
    n_layers = max(int(k.split(".")[2]) for k in sd if k.startswith("enc.layers.")) + 1
    model = MLMDensity(vocab_sizes, d=d, n_layers=n_layers, n_heads=8, ff=ff)
    model.load_state_dict(sd)
    return model.to(device)


def train_density(train, test, d=128, n_layers=4, n_heads=8, ff=512, dropout=0.1, epochs=15, batch=8192, lr=1e-3,
                  device="cuda", cache_tag=None):
    """Train the masked-token model on train + test predictors (label-free). Caches the weights."""
    cache_tag = cache_tag or f"d{d}L{n_layers}h{n_heads}e{epochs}"
    wpath = _cache() / f"density_{cache_tag}.pt"
    ids, obs, vsz, na_id = _na_ids(train, test)
    model = MLMDensity(vsz, d, n_layers, n_heads, ff, dropout)
    if wpath.exists():
        model.load_state_dict(torch.load(wpath, map_location=device))
        print(f"loaded density model {cache_tag}", flush=True)
        return model.to(device), ids, obs, vsz
    device = torch.device(device)
    model = model.to(device)
    ids_t = torch.as_tensor(ids, device=device)
    obs_t = torch.as_tensor(obs, device=device)
    n = len(ids_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((n + batch - 1) // batch))
    torch.manual_seed(SEED)
    n_batches = (n + batch - 1) // batch
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = cnt = 0
        for b in range(n_batches):
            idx = perm[b * batch:(b + 1) * batch]
            x = ids_t[idx]                       # (B, F)
            ob = obs_t[idx]                      # (B, F) observed mask
            # mask 1 or 2 observed cells per row
            xm = x.clone()
            B, F_ = x.shape
            rand = torch.rand(B, F_, device=device)
            rand = rand * ob.float() + (~ob).float() * 1e9  # only observed cells are eligible
            order = rand.argsort(dim=1)
            k_mask = torch.randint(1, 3, (B,), device=device)
            mask_pos = torch.zeros(B, F_, dtype=torch.bool, device=device)
            for j in range(2):
                pos = order[:, j]
                mask_pos[torch.arange(B, device=device), pos] = k_mask > j
            for c in range(F_):
                xm[mask_pos[:, c], c] = model.mask_id[c]
            logits = model.forward_logits(xm)
            loss = 0.0
            for c in range(F_):
                mc = mask_pos[:, c]
                if mc.any():
                    loss = loss + nn.functional.cross_entropy(logits[c][mc], x[mc, c])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            cnt += len(idx)
        print(f"  density ep {ep:2d} loss {tot / max(cnt, 1):.5f}", flush=True)
    torch.save(model.state_dict(), wpath)
    print(f"trained + cached density model {cache_tag}", flush=True)
    return model, ids, obs, vsz


@torch.no_grad()
def density_features(train, test, model=None, device="cuda", batch=16384, cache_tag=TAG):
    """(side_tr, side_te): the 34 standardized density channels. Cached. Without a cached model, the model is trained
    first with CONFIG."""
    cpath = _cache() / f"density_feats_{cache_tag}.npz"
    if cpath.exists():
        z = np.load(cpath)
        return z["side_train"], z["side_test"]
    if model is None:
        ids, obs, vsz, _ = _na_ids(train, test)
        if (_cache() / f"density_{cache_tag}.pt").exists():
            model = load_density(vsz, cache_tag, device=device)
        else:
            model, ids, obs, vsz = train_density(train, test, cache_tag=cache_tag, device=device, **CONFIG)
    else:
        ids, obs, vsz, _ = _na_ids(train, test)
    device = torch.device(device)
    model = model.to(device).eval()
    ids_t = torch.as_tensor(ids, device=device)
    obs_t = torch.as_tensor(obs, device=device)
    n, F_ = ids_t.shape
    pll = np.zeros(n)
    ent_sum = np.zeros(n)
    min_logp = np.full(n, 1e9)
    miss_ent = np.zeros(n)
    n_miss = np.zeros(n)
    n_obs = np.zeros(n)
    cell_logp = np.zeros((n, F_), dtype=np.float32)   # per-cell log p (observed)
    cell_ent = np.zeros((n, F_), dtype=np.float32)    # per-cell entropy
    for s in range(0, n, batch):
        x = ids_t[s:s + batch]
        ob = obs_t[s:s + batch]
        B = x.shape[0]
        logp_obs = torch.zeros(B, F_, device=device)
        ent_obs = torch.zeros(B, F_, device=device)
        # per-cell conditional: mask cell c only
        for c in range(F_):
            xm = x.clone()
            xm[:, c] = model.mask_id[c]
            h = model.forward_hidden(xm)
            lp = torch.log_softmax(model.head[c](h[:, c]), dim=-1)  # (B, V_c)
            true_lp = lp.gather(1, x[:, c:c + 1]).squeeze(1)
            p = lp.exp()
            logp_obs[:, c] = true_lp
            ent_obs[:, c] = -(p * lp).sum(-1)
        logp_obs = logp_obs.cpu().numpy()
        ent_obs = ent_obs.cpu().numpy()
        obn = ob.cpu().numpy()
        cell_logp[s:s + B] = logp_obs
        cell_ent[s:s + B] = ent_obs
        mn = ~obn
        pll[s:s + B] = (logp_obs * obn).sum(1)
        ent_sum[s:s + B] = (ent_obs * obn).sum(1)
        n_obs[s:s + B] = obn.sum(1)
        min_logp[s:s + B] = np.where(obn, logp_obs, 1e9).min(1)
        miss_ent[s:s + B] = (ent_obs * mn).sum(1)
        n_miss[s:s + B] = mn.sum(1)
    mean_ent = ent_sum / np.maximum(n_obs, 1)
    mean_miss_ent = miss_ent / np.maximum(n_miss, 1)
    min_logp = np.where(n_obs > 0, min_logp, 0.0)
    mean_miss_ent = np.where(n_miss > 0, mean_miss_ent, 0.0)
    cell_logp = np.where(np.isfinite(cell_logp), cell_logp, 0.0)
    # more aggregates over the observed cells
    big = np.where(obs, cell_logp, np.nan)
    with np.errstate(invalid="ignore"):
        std_logp = np.nanstd(big, axis=1)
        srt = np.sort(big, axis=1)                     # NaN sort to the end
        min2_logp = srt[:, 1] if srt.shape[1] > 1 else srt[:, 0]
        mean_logp = np.nanmean(big, axis=1)
        max_ent = np.nanmax(np.where(obs, cell_ent, np.nan), axis=1)
    for arr in (std_logp, min2_logp, mean_logp, max_ent):
        np.nan_to_num(arr, copy=False)
    agg = np.column_stack([pll, mean_ent, min_logp, mean_miss_ent, n_obs, n_miss, std_logp, min2_logp, mean_logp,
                           max_ent])
    feats = np.hstack([agg, cell_logp, cell_ent]).astype(np.float32)
    # standardize (label-free, all rows)
    mu, sd = feats.mean(0), feats.std(0)
    sd = np.where(sd == 0, 1, sd)
    feats = (feats - mu) / sd
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    ntr = len(train)
    out_tr, out_te = feats[:ntr], feats[ntr:]
    np.savez_compressed(cpath, side_train=out_tr, side_test=out_te)
    print(f"built density features {feats.shape}", flush=True)
    return out_tr, out_te
