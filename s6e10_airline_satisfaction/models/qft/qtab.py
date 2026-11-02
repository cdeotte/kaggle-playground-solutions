"""Feature-token encoder + Qwen3 backbone + minimal LoRA (Q-FT).

FeatureTokenizer: each of the 21 features becomes ONE embedding vector
    token_c = RMSNorm( col_emb[c] + cat_emb_c[code] + numeric_c(value) ) * gain
  - col_emb: one vector per column, initialized from the mean input embedding of the column-name tokens
  - cat_emb_c: per-column embedding table over the column's codes (feats.py)
  - numeric_c (Age, Flight Distance, 2 delays): periodic embedding [sin, cos](2 pi f x) with learned frequencies -> Linear -> ReLU
    -> Linear (per column; like RealMLP's PLR numeric embeddings)
  - RMSNorm + a gain initialized to the RMS of the backbone's token embeddings, so the tokens enter at the native scale.
Backbone: Qwen3Model (no LM head) fed with inputs_embeds; LoRA = rank-r adapters on q/k/v/o/gate/up/down, base weights frozen."""
import math

import torch
import torch.nn as nn

QWEN = 'Qwen/Qwen3-0.6B-Base'


class FeatureTokenizer(nn.Module):
    def __init__(self, vocab, numeric, d, gain, col_init=None, n_freq=32, sigma=1.0, hidden=64):
        super().__init__()
        nf = len(vocab)
        self.nf = nf
        self.col = nn.Parameter(col_init.clone() if col_init is not None else torch.randn(nf, d) * gain)
        self.cat = nn.ModuleList([nn.Embedding(v, d) for v in vocab])
        for e in self.cat:
            nn.init.normal_(e.weight, std=gain)
        self.num_idx = [j for j, f in enumerate(numeric) if f]
        m = len(self.num_idx)
        self.freq = nn.Parameter(torch.randn(m, n_freq) * sigma)
        self.w1 = nn.Parameter(torch.randn(m, 2 * n_freq, hidden) / math.sqrt(2 * n_freq))
        self.b1 = nn.Parameter(torch.zeros(m, hidden))
        self.w2 = nn.Parameter(torch.randn(m, hidden, d) * gain / math.sqrt(hidden))
        self.b2 = nn.Parameter(torch.zeros(m, d))
        self.gain = nn.Parameter(torch.full((d,), float(gain)))

    def embed(self, codes, nums):
        """Un-normalized tokens [B, nf, d]."""
        t = self.col[None].expand(codes.shape[0], -1, -1)
        t = t + torch.stack([e(codes[:, j]) for j, e in enumerate(self.cat)], 1)
        if self.num_idx:
            x = nums[:, self.num_idx]                                              # [B, m]
            ang = 2 * math.pi * x[..., None] * self.freq[None]                    # [B, m, k]
            p = torch.cat([torch.sin(ang), torch.cos(ang)], -1)                  # [B, m, 2k]
            h = torch.relu(torch.einsum('bmk,mkh->bmh', p, self.w1) + self.b1)
            v = torch.einsum('bmh,mhd->bmd', h, self.w2) + self.b2
            t = t.index_add(1, torch.tensor(self.num_idx, device=t.device), v)
        return t

    def norm(self, t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * self.gain

    def forward(self, codes, nums):
        """codes int64 [B, nf], nums float [B, nf] -> tokens [B, nf, d] (column order = data.csv order)."""
        return self.norm(self.embed(codes, nums))


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def load_qwen_lora(lora_r=16, lora_alpha=32):
    """Pretrained Qwen3-0.6B-Base (fp32) with LoRA adapters; returns (backbone, input embedding matrix, tokenizer)."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN)
    bb = AutoModel.from_pretrained(QWEN, dtype=torch.float32)
    emb = bb.get_input_embeddings().weight.detach().clone()
    for p in bb.parameters():
        p.requires_grad_(False)
    for layer in bb.layers:
        for parent in (layer.self_attn, layer.mlp):
            for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'):
                if hasattr(parent, n):
                    setattr(parent, n, LoRALinear(getattr(parent, n), lora_r, lora_alpha))
    return bb, emb, tok


def column_name_init(tok, emb, cols, gain):
    """Mean input embedding of each column name's tokens, rescaled to the token scale."""
    rows = [emb[tok(c, add_special_tokens=False)['input_ids']].mean(0) for c in cols]
    x = torch.stack(rows)
    return x / x.pow(2).mean(-1, keepdim=True).sqrt() * gain
