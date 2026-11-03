"""Inputs of the SAINT-style and GrowNet-style networks: categorical keys as integer indices, standardized numerics,
tensor loaders and batched probability prediction."""
import numpy as np
import pandas as pd
import torch


def fit_cat_maps(X_cat):
    """Per categorical column: value -> 1..n in order of first appearance in the training rows (0 = unseen value)."""
    if X_cat is None or X_cat.shape[1] == 0:
        return []
    maps = []
    for j in range(X_cat.shape[1]):
        vals = pd.Series(X_cat[:, j].astype(str))
        uniq = vals.dropna().unique().tolist()
        maps.append({v: i + 1 for i, v in enumerate(uniq)})
    return maps


def transform_cats(X_cat, maps):
    if not maps:
        return np.zeros((len(X_cat), 0), dtype=np.int64)
    out = np.zeros((X_cat.shape[0], len(maps)), dtype=np.int64)
    for j, mp in enumerate(maps):
        out[:, j] = pd.Series(X_cat[:, j].astype(str)).map(mp).fillna(0).astype(np.int64).to_numpy()
    return out


def fit_num_scaler(X_num):
    mean = np.nanmean(X_num, axis=0).astype(np.float32)
    std = np.nanstd(X_num, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def transform_nums(X_num, mean, std):
    X = ((X_num.astype(np.float32) - mean) / std).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=8.0, neginf=-8.0)
    return np.clip(X, -8.0, 8.0).astype(np.float32)


def encode_fold(Xtr_num, Xtr_cat, Xva_num, Xva_cat, Xte_num, Xte_cat):
    """Index maps and scaler fitted on the training rows, applied to training / held-out / test rows.
    Returns (train, held-out, test) as (numeric, index) pairs and the embedding sizes (n values + 1)."""
    cat_maps = fit_cat_maps(Xtr_cat)
    Xtr_cat_i = transform_cats(Xtr_cat, cat_maps)
    Xva_cat_i = transform_cats(Xva_cat, cat_maps)
    Xte_cat_i = transform_cats(Xte_cat, cat_maps)
    cat_cards = [len(mp) + 1 for mp in cat_maps]

    num_mean, num_std = fit_num_scaler(Xtr_num)
    Xtr_num_s = transform_nums(Xtr_num, num_mean, num_std)
    Xva_num_s = transform_nums(Xva_num, num_mean, num_std)
    Xte_num_s = transform_nums(Xte_num, num_mean, num_std)
    return (Xtr_num_s, Xtr_cat_i), (Xva_num_s, Xva_cat_i), (Xte_num_s, Xte_cat_i), cat_cards


def make_tensor_loader(X_num, X_cat, y=None, batch_size=2048, shuffle=False, seed=42):
    tensors = [torch.from_numpy(X_num.astype(np.float32)), torch.from_numpy(X_cat.astype(np.int64))]
    if y is not None:
        tensors.append(torch.from_numpy(y.astype(np.float32)))
    dataset = torch.utils.data.TensorDataset(*tensors)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=True,
    )


def predict_proba(model, X_num, X_cat, batch_size=8192):
    device = next(model.parameters()).device
    loader = make_tensor_loader(X_num, X_cat, y=None, batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb_num, xb_cat in loader:
            xb_num = xb_num.to(device, non_blocking=True)
            xb_cat = xb_cat.to(device, non_blocking=True)
            logits = model(xb_num, xb_cat)
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(preds).astype("float64")
