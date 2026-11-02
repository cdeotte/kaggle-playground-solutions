#!/usr/bin/env python
"""Single and pair target encodings of the "pools2" key set, used by train_pair_lr.py and train_pair_mlp.py.

21 single keys (as strings): the 13 raw columns, income // 10, // 100, // 1000, // 5000, // 10000, commute // 1, commute // 5 and
age // 5; 210 pair keys: every pair of them ("a|b"). NestedTE (../lr_margin_xgb/lib/features.py) inside each outer fold with
m = 1 for singles and m = 10 for pairs. The models use the keys with at most 50,000 levels over train + test (215 of 231).
The 5 folds' (fit, valid, test) frames are built once (~1-2 min) and cached in cache/pair_te_pools2.pkl (~4.4 GB on disk and in RAM).
Usage: python pair_te.py   (optional: the training scripts build the cache if it is missing)"""
import itertools
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "lr_margin_xgb"))
from lib.features import logit, setup  # noqa: E402

CACHE = HERE / "cache" / "pair_te_pools2.pkl"
M_SINGLE, M_PAIR, MAX_CARD = 1.0, 10.0, 50000


def key_columns(both):
    cols = {"age": both.Age, "gender": both.Gender, "inc": both.Annual_Income_USD, "city": both.City_Type,
            "cars": both.Number_of_Cars_Owned, "ctype": both.Current_Car_Type, "km": both.Daily_Commute_km,
            "sth": both.Charging_Stations_Near_Home, "stw": both.Charging_Stations_Near_Work, "home": both.Home_Charging_Possible,
            "env": both.Environmental_Concern_Level, "sub": both.Subsidy_Available, "anx": both.Range_Anxiety_Level}
    for d in (10, 100, 1000, 5000, 10000):
        cols[f"inc_d{d}"] = both.Annual_Income_USD // d
    cols["km_i"] = both.Daily_Commute_km // 1
    cols["km_5"] = both.Daily_Commute_km // 5
    cols["age_5"] = both.Age // 5
    return cols


def build(ctx):
    B = ctx.B
    cols = key_columns(ctx.both)
    for k, s in cols.items():
        B.register(f"t_{k}", s.astype(str))
    specs = [(f"t_{k}", M_SINGLE) for k in cols]
    for a_, b_ in itertools.combinations(cols, 2):
        name = f"p_{a_}_{b_}"
        B.register(name, cols[a_].astype(str) + "|" + cols[b_].astype(str))
        specs.append((name, M_PAIR))
    print(f"pair TEs: {len(cols)} singles (m={M_SINGLE:g}) + {len(specs) - len(cols)} pairs (m={M_PAIR:g})", flush=True)
    t0, out = time.time(), []
    for f, (a, b) in enumerate(ctx.folds):
        F, V, T = ctx.TE.fold(a, b, specs)
        out.append((F.astype(np.float32), V.astype(np.float32), T.astype(np.float32)))
        print(f"  fold {f} built ({time.time() - t0:.0f}s)", flush=True)
    return dict(specs=specs, folds=out, cards={k: B.card[k] for k, _ in specs})


def load_or_build(ctx):
    """dict(specs=[(key, m)], folds=[(F, V, T) DataFrames per fold], cards={key: levels over train + test})."""
    if CACHE.exists():
        print(f"loading {CACHE}", flush=True)
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    cache = build(ctx)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(cache, fh, protocol=4)
    tmp.replace(CACHE)
    return cache


def selected(cache, max_card=MAX_CARD):
    """Column indices of the keys with at most max_card levels."""
    return [i for i, (k, m) in enumerate(cache["specs"]) if cache["cards"][k] <= max_card]


def pair_logits(D, sel):
    return logit(D.values[:, sel].astype(np.float64)).astype(np.float32)


if __name__ == "__main__":
    c = load_or_build(setup())
    print(f"{len(selected(c))} of {len(c['specs'])} keys have <= {MAX_CARD} levels")
