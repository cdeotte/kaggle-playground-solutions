"""FT-Transformer with the linear numerical embedding x_j * W_j + b_j.

Tokens: one per numeric input (linear embedding), one per categorical input (nn.Embedding) and a CLS token; pre-norm
transformer layers (multi-head attention + ReLU FFN); the head reads the CLS token
(LayerNorm -> Dropout -> Linear -> ReLU -> Linear)."""
import torch
import torch.nn as nn


class TransformerLayer(nn.Module):
    def __init__(self, d, n_heads, d_ffn_factor, attn_dropout, ffn_dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=attn_dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        ffn_w = int(d * d_ffn_factor)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_w),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_w, d),
        )

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.ln2(x))
        return x


class FTTransformer(nn.Module):
    def __init__(self, n_num, categories, d_token=128, n_layers=3, n_heads=8, d_ffn_factor=4 / 3, attn_dropout=0.1,
                 ffn_dropout=0.1, head_dropout=0.0):
        super().__init__()
        self.n_num = n_num
        self.d_token = d_token

        self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
        self.num_bias = nn.Parameter(torch.empty(n_num, d_token))
        nn.init.normal_(self.num_weight, std=0.01)
        nn.init.zeros_(self.num_bias)

        self.cat_embeds = nn.ModuleList([nn.Embedding(c, d_token) for c in categories])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.01)
        self.layers = nn.ModuleList([
            TransformerLayer(d_token, n_heads, d_ffn_factor, attn_dropout, ffn_dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Dropout(head_dropout),
            nn.Linear(d_token, d_token),
            nn.ReLU(),
            nn.Linear(d_token, 1),
        )

    def forward(self, x_num, x_cat):
        num_emb = x_num.unsqueeze(-1) * self.num_weight + self.num_bias  # (B, n_num, d)
        cat_emb = torch.stack([self.cat_embeds[i](x_cat[:, i]) for i in range(x_cat.shape[1])], dim=1)  # (B, n_cat, d)
        x = torch.cat([num_emb, cat_emb], dim=1)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for layer in self.layers:
            x = layer(x)
        return self.head(x[:, 0]).squeeze(-1)
