"""Per-fold channel matrix of the tree models: the base matrix (assemble.py) plus cached channel blocks, the completion
margin, the smooth fields and the artifact residuals.

Cached blocks (original file names; built by ../density_features and ../lift_banks, `{k}` = outer fold):
  cmp3      cmp3_f{k}       completion response from the d384 MLM: per-screen completion logits + aggregates (23 cols)
  cmpv4     cmp4_f{k}       the same from the v4 MLM (23 cols)
  den384v3  den384v3        MLM per-cell log p(x_c | rest) and entropy, pseudo-log-likelihood (26 cols, renamed dn2_)
  denv4     den_v4          the same from the v4 MLM (renamed dn4_)
  denv5     den_v5          the same from the v5 MLM (renamed dn5_)
  mfill     mlmfill         MLM expected-value fills of missing cells + recomputed budget geometry (27 cols)
  mfill2    mlmfill2        fills averaged over two MLMs (15 cols)
  mfill4    mlmfill4        fills from the v4 MLM (27 cols)
  qfill     qfill           q10-q90 width and skew of the MLM posterior per screen cell (14 cols)
  rrd       rrd             reconstruction residuals of observed cells (19 cols)
  tokemb    tokemb          3 PCA components of the MLM token embedding per column (36 cols)
  bp2       bp2_f{k}        budget-mixture posteriors (26 cols)
  plift     plift_f{k}      9 exact pair lifts over the log-odds independence base + mean (10 cols)
  cbank     cbank_f{k}      coarse-lattice pair lifts, pair smoothing 1 (9 cols)
  wbank     wbank_f{k}      utilization / weekend-minus-daily pair lifts (5 cols)
  wbank12   wbank12_f{k}    wbank at pair smoothings 0.5 / 1 / 2 (13 cols)
  sbank     sbank_f{k}      notif x share_s, app x share_s, social x notif lifts (5 cols)
  ebank     ebank_f{k}      weekend / daily / social (0.5 lattice) x notif / app lifts (6 cols)
  dsx       dsx_f{k}        exact-grid (daily, social) residual surfaces at sigma 1 / 3 / 8 (3 cols)
Computed here:
  cmpmrg    adds cmp_scale * sum over the 5 screens of (completion logit - logit(prior)) [0 where observed] to the margin
  sf        smooth response fields (smooth_fields.py)
  ares      artifact residuals: logit(te) - logit(tes) per column, logit(te) - sf1 per numeric, and their two row sums"""
import numpy as np
import pandas as pd

from common.data import cache_dir

from .assemble import FEATS, assemble
from .smooth_fields import add_smooth_fields

DF, LB = "density_features", "lift_banks"
BLOCKS = {   # flag: (folder that builds it, file name pattern, column rename)
    "cmp3": (DF, "cmp3_f{k}_{s}.parquet", None),
    "cmpv4": (DF, "cmp4_f{k}_{s}.parquet", None),
    "den384v3": (DF, "den384v3_{s}.parquet", ("den_", "dn2_")),
    "denv4": (DF, "den_v4_{s}.parquet", ("den_", "dn4_")),
    "denv5": (DF, "den_v5_{s}.parquet", ("den_", "dn5_")),
    "mfill": (DF, "mlmfill_{s}.parquet", None),
    "mfill2": (DF, "mlmfill2_{s}.parquet", None),
    "mfill4": (DF, "mlmfill4_{s}.parquet", None),
    "qfill": (DF, "qfill_{s}.parquet", None),
    "rrd": (DF, "rrd_{s}.parquet", None),
    "tokemb": (DF, "tokemb_{s}.parquet", None),
    "bp2": (DF, "bp2_f{k}_{s}.parquet", None),
    "plift": (LB, "plift_f{k}_{s}.parquet", None),
    "cbank": (LB, "cbank_f{k}_{s}.parquet", None),
    "wbank": (LB, "wbank_f{k}_{s}.parquet", None),
    "wbank12": (LB, "wbank12_f{k}_{s}.parquet", None),
    "sbank": (LB, "sbank_f{k}_{s}.parquet", None),
    "ebank": (LB, "ebank_f{k}_{s}.parquet", None),
    "dsx": (LB, "dsx_f{k}_{s}.parquet", None),
}
# blocks appended to the base matrix, in this order (column order matters for column subsampling)
ORDER = ("cmp3", "qfill", "denv5", "cmpv4", "den384v3", "cmpmrg", "mfill", "mfill2", "plift", "cbank", "wbank", "mfill4",
         "denv4", "sf", "wbank12", "sbank")


def load_block(flag, fold):
    owner, pattern, rename = BLOCKS[flag]
    d = cache_dir(owner)
    tr = pd.read_parquet(d / pattern.format(k=fold, s="tr"))
    te = pd.read_parquet(d / pattern.format(k=fold, s="te"))
    if rename:
        tr.columns = [c.replace(*rename) for c in tr.columns]
        te.columns = [c.replace(*rename) for c in te.columns]
    return tr, te


def logit(v):
    p = np.clip(v, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def completion_margin(A, B, margin_tr, margin_te, prior, scale):
    ccols = [c for c in A.columns if c.startswith("cmp_c_")]
    if margin_tr is None or not ccols:
        return margin_tr, margin_te
    lp = np.log(prior / (1 - prior))

    def add(fr):
        M = fr[ccols].values
        return ((M - lp) * (M != 0)).sum(axis=1)
    return (margin_tr + scale * add(A)).astype(np.float32), (margin_te + scale * add(B)).astype(np.float32)


def add_artifact_residuals(A, B):
    sf1cols = [c for c in A.columns if c.startswith("sf1_")]
    for fr in (A, B):
        s12 = np.zeros(len(fr))
        s9 = np.zeros(len(fr))
        for c in FEATS:
            d = logit(fr[f"te_{c}"].values) - logit(fr[f"tes_{c}"].values)
            fr[f"ar_{c}"] = np.nan_to_num(d).astype(np.float32)
            s12 += np.nan_to_num(d)
        for sc in sf1cols:
            base_c = sc[4:]
            if f"te_{base_c}" in fr.columns:
                d = logit(fr[f"te_{base_c}"].values) - fr[sc].values
                fr[f"arf_{base_c}"] = np.nan_to_num(d).astype(np.float32)
                s9 += np.nan_to_num(d)
        fr["ar_sum"] = s12.astype(np.float32)
        fr["arf_sum"] = s9.astype(np.float32)


def channel_matrix(fold, cfg, par, post=()):
    """assemble() + the ORDER blocks present in cfg, duplicate columns dropped, then the `post` blocks in their order
    (the XGBoost runner appends ares, tokemb, dsx; the CatBoost runner ebank). Returns the assemble() dict updated."""
    D = assemble(fold, cfg, par)
    A, B, y = D["A"], D["B"], D["y"]
    trn_idx, val_idx = D["trn_idx"], D["val_idx"]
    margin_tr, margin_te = D["margin_tr"], D["margin_te"]
    for flag in ORDER:
        if flag not in cfg:
            continue
        if flag == "cmpmrg":
            margin_tr, margin_te = completion_margin(A, B, margin_tr, margin_te, y[trn_idx].mean(),
                                                     float(par.get("cmp_scale", 0.4)))
        elif flag == "sf":
            A, B = add_smooth_fields(A, B, y, trn_idx, val_idx, D["train"], D["test"])
        else:
            tr, te = load_block(flag, fold)
            A = pd.concat([A, tr], axis=1)
            B = pd.concat([B, te], axis=1)
    dup = A.columns[A.columns.duplicated()].tolist()
    if dup:
        print(f"duplicated columns dropped: {sorted(set(dup))}", flush=True)
        A = A.loc[:, ~A.columns.duplicated()]
        B = B.loc[:, ~B.columns.duplicated()]
    for flag in post:
        if flag not in cfg:
            continue
        if flag == "ares":
            add_artifact_residuals(A, B)
        else:
            tr, te = load_block(flag, fold)
            A = pd.concat([A, tr], axis=1)
            B = pd.concat([B, te], axis=1)
    D.update(A=A, B=B, margin_tr=margin_tr, margin_te=margin_te)
    return D
