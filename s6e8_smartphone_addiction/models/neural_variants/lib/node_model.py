"""NODE (Neural Oblivious Decision Ensembles): differentiable oblivious trees.

- entmax15 (sparse softmax, alpha = 1.5) for sparse differentiable feature selection;
- entmax15 over the distances to the sorted thresholds for the (sparse) leaf region;
- stacked layers with residual connections (proj(x) + ODT(x)) -> linear head.
entmax15 is computed by bisection (forward, without autograd) with the closed-form backward."""
import torch
import torch.nn as nn


class Entmax15Function(torch.autograd.Function):
    """entmax with alpha = 1.5: p_i = [0.5 (z_i - tau)]_+^2 with sum p = 1."""

    @staticmethod
    def forward(ctx, z):
        with torch.no_grad():
            z_max = z.max(dim=-1, keepdim=True).values
            tau_lo = z_max - 2.0
            tau_hi = z_max
            for _ in range(12):
                tau = (tau_lo + tau_hi) / 2
                p = torch.clamp(0.5 * (z - tau), min=0) ** 2
                s = p.sum(dim=-1, keepdim=True)
                tau_lo = torch.where(s > 1, tau, tau_lo)
                tau_hi = torch.where(s > 1, tau_hi, tau)
            tau = (tau_lo + tau_hi) / 2
            p = torch.clamp(0.5 * (z - tau), min=0) ** 2
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, g):
        (p,) = ctx.saved_tensors
        sqrt_p = p ** 0.5
        num = (g * sqrt_p).sum(dim=-1, keepdim=True)
        den = sqrt_p.sum(dim=-1, keepdim=True)
        return sqrt_p * (g - num / den)


def entmax15(z):
    return Entmax15Function.apply(z)


class ODT(nn.Module):
    """One layer of differentiable oblivious decision trees: (B, in_features) -> (B, n_trees). Each tree
      1. selects a sparse combination of features via entmax15,
      2. compares the selected value to its sorted thresholds via entmax15 (sparse region indicator, never saturates),
      3. returns a linear response of that sparse region."""

    def __init__(self, in_features, n_trees, depth):
        super().__init__()
        self.n_trees = n_trees
        self.n_thresholds = 2 ** depth - 1

        self.feature_weights = nn.Parameter(torch.empty(n_trees, in_features))
        self.feature_bias = nn.Parameter(torch.zeros(n_trees, in_features))
        nn.init.normal_(self.feature_weights, std=1.0)

        self.thresholds = nn.Parameter(torch.empty(n_trees, self.n_thresholds))
        nn.init.uniform_(self.thresholds, -1, 1)

        self.response_matrix = nn.Parameter(torch.empty(n_trees, self.n_thresholds))
        self.response_bias = nn.Parameter(torch.zeros(n_trees))
        nn.init.xavier_uniform_(self.response_matrix)

    def forward(self, x):
        # sparse feature selection
        logits = x.unsqueeze(1) * self.feature_weights + self.feature_bias  # (B, T, d)
        alpha = entmax15(logits)  # (B, T, d) sparse over features
        f = (alpha * x.unsqueeze(1)).sum(-1)  # (B, T) selected feature value

        # sparse threshold region
        tau = torch.sort(self.thresholds, dim=-1).values  # ascending
        c = entmax15(f.unsqueeze(-1) - tau.unsqueeze(0))  # (B, T, n_thr) sparse
        response = (c * self.response_matrix.unsqueeze(0)).sum(-1) + self.response_bias
        return response  # (B, T)


class NODE(nn.Module):
    def __init__(self, in_features, n_layers=3, n_trees=256, depth=6, head_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.projs = nn.ModuleList()
        d = in_features
        for _ in range(n_layers):
            self.layers.append(ODT(d, n_trees, depth))
            self.projs.append(nn.Linear(d, n_trees))
            d = n_trees
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(d, 1),
        )

    def forward(self, x):
        for layer, proj in zip(self.layers, self.projs):
            x = layer(x) + proj(x)  # residual connection preserves the signal
        return self.head(x).squeeze(-1)
