"""SAINT-style binary transformer (a local PyTorch model, not TALENT's SAINT).

Tokens: a CLS token, one d_model embedding per categorical key (index 0 = value not seen in training) and `num_tokens`
numeric tokens from BatchNorm + one Linear layer over all numerics. A pre-norm transformer encoder (`depth` layers,
`heads` heads, GELU, FFN 4 x d_model) mixes them; the CLS output is concatenated with a numeric MLP branch (BatchNorm,
Linear mlp_hidden, SiLU, dropout, Linear d_model, SiLU) and a LayerNorm + MLP head gives one logit.

Training: BCE with pos_weight = ((1 - p) / p) ** pos_weight_power (p = positive rate of the training rows), AdamW,
cosine schedule from lr to 0.1 lr over max_epoch epochs, gradient-norm clipping. The held-out AUC is measured after every
epoch; the best epoch's weights predict, and training stops `patience` epochs after the best epoch.
"""
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .tensor_data import encode_fold, make_tensor_loader, predict_proba


class BinarySAINT(torch.nn.Module):
    def __init__(self, n_num, cat_cardinalities, d_model=96, num_tokens=8, depth=3, heads=4, dropout=0.12, mlp_hidden=512):
        super().__init__()
        self.n_num = int(n_num)
        self.cat_cardinalities = list(cat_cardinalities)
        self.d_model = int(d_model)
        self.num_tokens = int(num_tokens)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.cat_embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(int(card), self.d_model) for card in self.cat_cardinalities
        ])
        self.num_projector = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.n_num),
            torch.nn.Linear(self.n_num, self.num_tokens * self.d_model),
        )
        layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(heads),
            dim_feedforward=self.d_model * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=int(depth))
        self.num_branch = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.n_num),
            torch.nn.Linear(self.n_num, int(mlp_hidden)),
            torch.nn.SiLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(mlp_hidden), self.d_model),
            torch.nn.SiLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.d_model * 2),
            torch.nn.Linear(self.d_model * 2, int(mlp_hidden)),
            torch.nn.SiLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(mlp_hidden), 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        for emb in self.cat_embeddings:
            torch.nn.init.normal_(emb.weight, std=0.02)

    def forward(self, x_num, x_cat):
        bs = x_num.shape[0]
        tokens = [self.cls_token.expand(bs, -1, -1)]
        if len(self.cat_embeddings):
            cat_tokens = [emb(x_cat[:, j]).unsqueeze(1) for j, emb in enumerate(self.cat_embeddings)]
            tokens.append(torch.cat(cat_tokens, dim=1))
        num_tokens = self.num_projector(x_num).view(bs, self.num_tokens, self.d_model)
        tokens.append(num_tokens)
        x = torch.cat(tokens, dim=1)
        z = self.transformer(x)
        cls = z[:, 0, :]
        num_repr = self.num_branch(x_num)
        return self.head(torch.cat([cls, num_repr], dim=1)).squeeze(-1)


def fit_saint_fold(config, batch_size, max_epoch, Xtr_num, Xtr_cat, y_tr, Xva_num, Xva_cat, y_va, Xte_num, Xte_cat,
                   fold_dir, seed):
    """Trains on the training rows, keeps the best held-out-AUC epoch; returns (held-out, test) probabilities."""
    cfg_model = config["model"]
    cfg_train = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    (Xtr_num_s, Xtr_cat_i), (Xva_num_s, Xva_cat_i), (Xte_num_s, Xte_cat_i), cat_cards = encode_fold(
        Xtr_num, Xtr_cat, Xva_num, Xva_cat, Xte_num, Xte_cat)

    model = BinarySAINT(
        n_num=Xtr_num_s.shape[1],
        cat_cardinalities=cat_cards,
        d_model=cfg_model["d_model"],
        num_tokens=cfg_model["num_tokens"],
        depth=cfg_model["depth"],
        heads=cfg_model["heads"],
        dropout=cfg_model["dropout"],
        mlp_hidden=cfg_model["mlp_hidden"],
    ).to(device)

    train_loader = make_tensor_loader(Xtr_num_s, Xtr_cat_i, y_tr, batch_size, shuffle=True, seed=seed)
    pos_rate = float(np.mean(y_tr))
    raw_pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-6)
    pos_weight = raw_pos_weight ** float(cfg_train.get("pos_weight_power", 0.5))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_train["lr"], weight_decay=cfg_train["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=cfg_train["lr"] * 0.1)

    best_auc = -np.inf
    best_epoch = -1
    best_state = None
    patience = int(cfg_model.get("patience", 5))

    for epoch in range(1, max_epoch + 1):
        model.train()
        losses = []
        for xb_num, xb_cat, yb in train_loader:
            xb_num = xb_num.to(device, non_blocking=True)
            xb_cat = xb_cat.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb_num, xb_cat)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg_train.get("grad_clip", 5.0)))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_prob = predict_proba(model, Xva_num_s, Xva_cat_i, batch_size=max(batch_size, 8192))
        val_auc = roc_auc_score(y_va, val_prob)
        print(f"Epoch {epoch:02d}/{max_epoch} loss={np.mean(losses):.5f} val_auc={val_auc:.6f}", flush=True)

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, fold_dir / "saint_style_best.pt")
        elif epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} AUC {best_auc:.6f}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    val_prob = np.clip(predict_proba(model, Xva_num_s, Xva_cat_i, batch_size=max(batch_size, 8192)), 1e-7, 1 - 1e-7)
    test_prob = np.clip(predict_proba(model, Xte_num_s, Xte_cat_i, batch_size=max(batch_size, 8192)), 1e-7, 1 - 1e-7)
    return val_prob, test_prob
