"""Factorization models over the 12 token fields (PyTorch).

FM:  logit = w0 + sum_c w_c[t_c] + sum_{c<c'} <v_c[t_c], v_c'[t_c']>; the pairwise term is computed as
     0.5 * ((sum_c v_c)^2 - sum_c v_c^2) summed over the k factors, with dropout on the factor embeddings.
AFN-style: logit = w0 + sum_c w_c[t_c] + MLP([v_c * v_c' for the 66 field pairs, flattened] (+ numeric side input)).
     The pair products are not sum-pooled, so the MLP sees every field combination separately. With n_heads > 1 the
     model has n_heads MLP heads on the same input, each with a learned per-head input scaling (initialised uniformly
     in [0.5, 1.5]); training uses the per-head logits, inference the mean of the head probabilities."""
import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self, vocab_sizes, k=16, emb_dropout=0.0, lin_dropout=0.0):
        super().__init__()
        self.n_fields = len(vocab_sizes)
        self.lin = nn.ModuleList([nn.Embedding(int(v), 1) for v in vocab_sizes])
        self.fac = nn.ModuleList([nn.Embedding(int(v), k) for v in vocab_sizes])
        self.bias = nn.Parameter(torch.zeros(1))
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.lin_dropout = nn.Dropout(lin_dropout)
        for e in self.lin:
            nn.init.normal_(e.weight, 0.0, 0.01)
        for e in self.fac:
            nn.init.normal_(e.weight, 0.0, 0.05)

    def forward(self, x):
        # x: (B, F) long
        L = torch.cat([e(x[:, i]) for i, e in enumerate(self.lin)], dim=1)
        lin = self.bias + self.lin_dropout(L).sum(dim=1)
        V = torch.stack([e(x[:, i]) for i, e in enumerate(self.fac)], dim=1)
        V = self.emb_dropout(V)                       # (B, F, k)
        s = V.sum(dim=1)                              # (B, k)
        fac = 0.5 * ((s * s) - (V * V).sum(dim=1)).sum(dim=-1)
        return lin + fac


class AFN(nn.Module):
    def __init__(self, vocab_sizes, k=16, hidden=256, emb_dropout=0.0, mlp_dropout=0.0, n_side=0, hidden2=0,
                 n_heads=1):
        super().__init__()
        F = len(vocab_sizes)
        self.n_fields = F
        self.k = k
        self.n_heads = n_heads
        self.lin = nn.ModuleList([nn.Embedding(int(v), 1) for v in vocab_sizes])
        self.fac = nn.ModuleList([nn.Embedding(int(v), k) for v in vocab_sizes])
        self.bias = nn.Parameter(torch.zeros(1))
        self.emb_dropout = nn.Dropout(emb_dropout)
        pairs = [(c, cp) for c in range(F) for cp in range(c + 1, F)]
        self.n_pairs = len(pairs)
        self.register_buffer("pi", torch.tensor([p[0] for p in pairs], dtype=torch.long))
        self.register_buffer("pj", torch.tensor([p[1] for p in pairs], dtype=torch.long))
        D = self.n_pairs * k + n_side

        def _make_head():
            if hidden2 > 0:
                return nn.Sequential(
                    nn.Linear(D, hidden), nn.ReLU(), nn.Dropout(mlp_dropout),
                    nn.Linear(hidden, hidden2), nn.ReLU(),
                    nn.Dropout(mlp_dropout), nn.Linear(hidden2, 1))
            return nn.Sequential(
                nn.Linear(D, hidden), nn.ReLU(), nn.Dropout(mlp_dropout),
                nn.Linear(hidden, 1))

        if n_heads > 1:
            # internal ensemble trained in one fit: shared token embeddings and pair products, independent heads
            self.heads = nn.ModuleList([_make_head() for _ in range(n_heads)])
            self.head_mult = nn.Parameter(1.0 + (torch.rand(n_heads, D) - 0.5))
            self.mlp = None
        else:
            self.mlp = _make_head()
        for e in self.lin:
            nn.init.normal_(e.weight, 0.0, 0.01)
        for e in self.fac:
            nn.init.normal_(e.weight, 0.0, 0.05)

    def forward(self, x, x_num=None, return_heads=False):
        lin = self.bias + torch.cat([e(x[:, i]) for i, e in enumerate(self.lin)], dim=1).sum(dim=1)
        V = torch.stack([e(x[:, i]) for i, e in enumerate(self.fac)], dim=1)
        V = self.emb_dropout(V)                       # (B, F, k)
        z = (V[:, self.pi] * V[:, self.pj]).flatten(1)  # (B, P * k) pairwise products
        if x_num is not None:
            z = torch.cat([z, x_num], dim=1)
        if self.n_heads == 1:
            return lin + self.mlp(z).squeeze(-1)
        outs = [head(z * self.head_mult[h]).squeeze(-1) for h, head in enumerate(self.heads)]
        all_logits = torch.stack(outs, dim=1) + lin.unsqueeze(1)  # (B, H)
        if return_heads:
            return all_logits
        return all_logits.mean(dim=1)
