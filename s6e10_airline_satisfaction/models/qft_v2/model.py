"""Q-FT v2 model pieces: feature tokens + side features -> ModernBERT-base (order-free) -> mean head.

SideTokenizer = Q-FT's FeatureTokenizer (../qft/qtab.py) + per-column side features + global summary tokens:
    token_c  = RMSNorm(col_c + cat_c(code) + numeric_c(x) + side[c] @ Ws_c) * gain
    global_i = RMSNorm(gcol_i + Wg_i g_i) * gain
Ws / Wg start small, so the model starts close to plain feature tokens.
Order-free: ModernBERT is a bidirectional encoder with rotary positions driven by position_ids. With position_ids = 0 for every
token, all tokens look position-less and the model reads the row as a SET (no shuffling or order averaging needed).
Head: mean of the output vectors of the 21 feature tokens and the 2 summary tokens -> linear -> logit."""
import math

import torch
import torch.nn as nn

from qtab import FeatureTokenizer

MODERNBERT = 'answerdotai/ModernBERT-base'


def load_modernbert():
    """Pretrained ModernBERT-base (fp32, SDPA attention), all weights trainable; returns (backbone, input embeddings, tokenizer)."""
    from transformers import AutoModel, AutoTokenizer
    bb = AutoModel.from_pretrained(MODERNBERT, dtype=torch.float32, attn_implementation='sdpa')
    tok = AutoTokenizer.from_pretrained(MODERNBERT)
    return bb, bb.get_input_embeddings().weight.detach().clone(), tok


def run_backbone(bb, x):
    """x [B, L, d] -> last hidden states [B, L, d], every token at position 0 (order-free)."""
    B, L, _ = x.shape
    pid = torch.zeros(B, L, dtype=torch.long, device=x.device)
    am = torch.ones(B, L, dtype=torch.long, device=x.device)
    return bb(inputs_embeds=x, attention_mask=am, position_ids=pid).last_hidden_state


class SideTokenizer(FeatureTokenizer):
    def __init__(self, vocab, numeric, d, gain, side_k, glob_sizes, col_init=None):
        super().__init__(vocab, numeric, d, gain, col_init=col_init)
        self.side_w = nn.Parameter(torch.randn(len(vocab), side_k, d) * gain / math.sqrt(max(side_k, 1)))
        self.gcol = nn.Parameter(torch.randn(len(glob_sizes), d) * gain)
        self.glin = nn.ModuleList([nn.Linear(n, d) for n in glob_sizes])
        for lin in self.glin:
            nn.init.normal_(lin.weight, std=gain / math.sqrt(lin.in_features))
            nn.init.zeros_(lin.bias)

    def forward(self, codes, nums, side, globs):
        t = self.embed(codes, nums) + torch.einsum('bck,ckd->bcd', side.to(self.side_w.dtype), self.side_w)
        out = [t] + [(self.gcol[i][None] + lin(g.to(t.dtype)))[:, None] for i, (g, lin) in enumerate(zip(globs, self.glin))]
        return self.norm(torch.cat(out, 1))


class MeanHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.out = nn.Linear(d, 1)

    def forward(self, h):
        return self.out(h.mean(1)).squeeze(-1)
