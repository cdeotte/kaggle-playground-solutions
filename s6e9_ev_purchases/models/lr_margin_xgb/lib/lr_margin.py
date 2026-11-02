"""In-fold logistic-regression base margin for XGBoost.

Per outer fold: StandardScaler + LogisticRegression(C=1) fitted on the fold's training rows with
  - the 6 inputs of the generator's purchase formula (income / 1000, env, subsidy, anxiety Medium, anxiety High, home charging),
  - the logits of the nested TEs and the 4 income TE differences (the same TE columns the trees get),
  - the champion matrix (raw + log counts + flags) with its <= 12-level columns one-hot encoded.
Its decision function on the fold's training rows, held-out rows and test rows is XGBoost's base_margin."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import BLOCKS, add_diffs, assemble, logit


def formula_frame(df):
    return pd.DataFrame({"income_k": df.Annual_Income_USD.fillna(85000) / 1000.0,
                         "env": df.Environmental_Concern_Level.fillna(3.0),
                         "subsidy": (df.Subsidy_Available == "Yes").astype(float),
                         "anx_med": (df.Range_Anxiety_Level == "Medium").astype(float),
                         "anx_high": (df.Range_Anxiety_Level == "High").astype(float),
                         "home": (df.Home_Charging_Possible == "Yes").astype(float)}).values


def build(ctx, te_specs, blocks=BLOCKS, max_card=12, C=1.0, max_iter=2000, log=print):
    """Champion matrix, nested TE frames (+ diffs) and LR margins for every fold.
    Returns dict(X, Xtr, Xte, te_folds=[(F, V, T)], margins=[(margin_fit, margin_valid, margin_test)])."""
    B, TE, y, folds = ctx.B, ctx.TE, ctx.y, ctx.folds
    X = assemble(B, blocks)
    Xtr, Xte = B.train_part(X), B.test_part(X)
    te_folds = [tuple(add_diffs(D) for D in TE.fold(a, b, te_specs)) for a, b in folds]
    FF = np.vstack([formula_frame(ctx.tr), formula_frame(ctx.te)])
    low = [c for c in X.columns if X[c].nunique() <= max_card]
    num = [c for c in X.columns if c not in low]
    XOH = np.column_stack([X[num].values.astype(np.float64),
                           pd.get_dummies(X[low].astype(int).astype(str), drop_first=True).values.astype(np.float64)])
    test_rows = np.arange(B.n_tr, B.n_tr + B.n_te)

    def lr_margin(F, V, T, a, b):
        tecols = [c for c in F.columns if c.startswith("te_")]
        dcols = [c for c in F.columns if c.startswith("d_")]

        def mat(D, rows):
            return np.column_stack([FF[rows], np.column_stack([logit(D[c].values) for c in tecols]), D[dcols].values, XOH[rows]])
        Xa, Xb, Xt = mat(F, a), mat(V, b), mat(T, test_rows)
        sc = StandardScaler().fit(Xa)
        lr = LogisticRegression(C=C, max_iter=max_iter).fit(sc.transform(Xa), y[a])
        return lr.decision_function(sc.transform(Xa)), lr.decision_function(sc.transform(Xb)), lr.decision_function(sc.transform(Xt))

    margins = []
    for f, (a, b) in enumerate(folds):
        margins.append(lr_margin(*te_folds[f], a, b))
        log(f"  LR margin fold {f} done")
    return dict(X=X, Xtr=Xtr, Xte=Xte, te_folds=te_folds, margins=margins)
