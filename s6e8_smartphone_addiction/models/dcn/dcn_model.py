"""DCNv2 with field-space crossing, LayerScale-gated cross layers, PLR numeric embeddings and K jointly trained members.

Inputs: the 12 exact-value token codes and the dense engineered channels.
  * deep tower: token embeddings + all channels (scaled by a learned per-channel front scale) + their PLR embeddings
    (periodic sin/cos features with learned frequencies, a per-channel linear map and ReLU), MLP with SiLU and dropout.
  * cross tower: crossing in FIELD space - the 12 token embeddings plus one field per engineered channel family (a linear
    projection of that family's channels to d_tok) - not over the raw channels. Each cross layer
    x_{l+1} = x_l + g * (x0 * (W x_l + b)) has a zero-initialized gate g, so the tower starts as the identity.
  * tok_drop drops whole token embeddings during training.
  * K members (independent weights, leading K dimension, batched matmuls) are trained in one fit; the prediction is the
    mean of their probabilities."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLinear(nn.Module):
    """K independent linear maps applied to a (K, B, in) tensor."""

    def __init__(self, K, d_in, d_out, bias=True, zero=False):
        super().__init__()
        w = torch.empty(K, d_in, d_out)
        if zero:
            w.zero_()
        else:
            for k in range(K):   # independent init per member
                nn.init.kaiming_uniform_(w[k].T, a=math.sqrt(5))
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(K, 1, d_out)) if bias else None

    def forward(self, x):   # (K, B, in) -> (K, B, out)
        y = torch.bmm(x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y


class PLREmbed(nn.Module):
    """Periodic-linear-ReLU embedding of every numeric channel: [sin(2 pi f x), cos(2 pi f x)] with k learned frequencies,
    then a per-channel linear map to d_out and ReLU."""

    def __init__(self, d_num, k=8, d_out=4, sigma=0.94):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(d_num, k) * sigma)
        w = torch.empty(d_num, 2 * k, d_out)
        nn.init.normal_(w, std=(2 * k) ** -0.5)
        self.lin = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(d_num, d_out))

    def forward(self, x):   # (B, D) -> (B, D * d_out)
        z = 2 * math.pi * x.unsqueeze(-1) * self.freq
        z = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        e = torch.einsum("bdk,dkr->bdr", z, self.lin) + self.bias
        return F.relu(e).flatten(1)


class CrossLayerV2(nn.Module):
    """x_{l+1} = x_l + g * (x0 * (W x_l + b)), LayerScale gate g (zero init)."""

    def __init__(self, K, d, p_drop=0.0, gate_init=0.0, ln=False):
        super().__init__()
        self.W = KLinear(K, d, d, bias=True)
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()
        self.gate = nn.Parameter(torch.full((K, 1, d), float(gate_init)))
        self.ln = nn.LayerNorm(d) if ln else nn.Identity()

    def forward(self, x0, x):
        return self.ln(x + self.gate * self.drop(x0 * self.W(x)))


class DCN(nn.Module):
    def __init__(self, vocab_sizes, d_num, cfg, groups):
        """groups: list of column-index lists, one cross field per engineered channel family."""
        super().__init__()
        K, d_tok = cfg["K"], cfg["d_tok"]
        self.K = K
        self.embs = nn.ModuleList([nn.Embedding(v, d_tok) for v in vocab_sizes])
        for e in self.embs:
            nn.init.normal_(e.weight, std=cfg["emb_std"])
        self.n_tok = len(vocab_sizes)
        # learnable per-channel front scale (trained at a higher learning rate)
        self.fscale = nn.Parameter(torch.ones(d_num)) if cfg["front_scale"] else None
        self.tok_drop = float(cfg["tok_drop"])
        self.plr = PLREmbed(d_num, cfg["plr_k"], cfg["plr_d"], cfg["plr_sigma"])
        d_deep_in = self.n_tok * d_tok + d_num + d_num * cfg["plr_d"]

        self.n_groups = len(groups)
        for i, g in enumerate(groups):
            self.register_buffer(f"gidx{i}", torch.as_tensor(g, dtype=torch.long))
        self.g_proj = nn.ModuleList([nn.Linear(len(g), d_tok) for g in groups])
        d_cross = (self.n_tok + len(groups)) * d_tok
        self.x0_ln = nn.LayerNorm(d_cross) if cfg["x0_ln"] else nn.Identity()
        self.cross = nn.ModuleList([CrossLayerV2(K, d_cross, cfg["cross_drop"], cfg["cross_gate"], bool(cfg["cross_ln"]))
                                    for _ in range(cfg["n_cross"])])

        self.deep, prev = nn.ModuleList(), d_deep_in
        for h in cfg["hidden"]:
            self.deep.append(KLinear(K, prev, h))
            prev = h
        self.deep_drop = nn.Dropout(cfg["p_drop"])
        self.act = {"relu": F.relu, "silu": F.silu, "gelu": F.gelu, "mish": F.mish}[cfg["act"]]
        self.head = KLinear(K, prev + d_cross, 1, zero=True)

    def _tok_emb(self, tok):
        e = torch.stack([emb(tok[:, j]) for j, emb in enumerate(self.embs)], dim=1)
        if self.training and self.tok_drop > 0:
            # drop a whole token's embedding, so the net falls back on the engineered channels
            keep = (torch.rand(e.shape[0], e.shape[1], 1, device=e.device) >= self.tok_drop).float()
            e = e * keep / (1 - self.tok_drop)
        return e.flatten(1)

    def forward(self, tok, xnum):
        if self.fscale is not None:
            xnum = xnum * self.fscale
        e = self._tok_emb(tok)
        x_deep = torch.cat([e, xnum, self.plr(xnum)], dim=-1).unsqueeze(0).expand(self.K, -1, -1)

        fields = [e] + [p(xnum[:, getattr(self, f"gidx{i}")]) for i, p in enumerate(self.g_proj)]
        x0 = self.x0_ln(torch.cat(fields, dim=-1)).unsqueeze(0).expand(self.K, -1, -1)
        x = x0
        for c in self.cross:
            x = c(x0, x)

        h = x_deep
        for lin in self.deep:
            h = self.deep_drop(self.act(lin(h)))
        return self.head(torch.cat([x, h], dim=-1)).squeeze(-1)   # (K, B) logits

    def param_groups(self, wd, emb_wd):
        """[token embeddings, front scale, cross tower, everything else]; each gets its own learning rate."""
        emb_p = [p for e in self.embs for p in e.parameters()]
        scale_p = [self.fscale] if self.fscale is not None else []
        cross_p = [p for c in self.cross for p in c.parameters()]
        taken = {id(p) for p in emb_p + scale_p + cross_p}
        rest = [p for p in self.parameters() if id(p) not in taken]
        return [{"params": emb_p, "weight_decay": emb_wd},
                {"params": scale_p, "weight_decay": 0.0},
                {"params": cross_p, "weight_decay": wd},
                {"params": rest, "weight_decay": wd}]
