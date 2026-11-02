"""Label-free feature blocks, nested target encodings and the rule / artifact / BPE features.

setup() loads the data once and returns the shared context:
  tr, te, og     train.csv, test.csv and the original dataset
  both           train rows (without the target) then test rows, index 0..n_train+n_test-1
  y, folds       0/1 target and the 5 folds as a list of (train_idx, valid_idx): StratifiedKFold(5, shuffle=True, random_state=42)
  B, TE          Blocks and NestedTE(B, y)

Blocks: label-free blocks over train + test rows. "raw" = the 13 columns (categoricals as fixed integer codes), "freq_log" =
log1p of each column's value count over train + test, "flags" = income == 30,000, commute == 5 km, income not in the original
data, number of original rows with this income. B.codes / B.card hold integer codes of the target-encoding keys.
NestedTE: target encoding inside an outer fold. The fold's training rows are encoded with an inner 5-fold split (a row never
sees its own label); the held-out and test rows with the statistics of all training rows of the fold.
Smoothing: (sum + m * prior) / (count + m); columns are named te_<key>_m<m>.
ResidTE: the same construction on the residual r = y - p_rule (columns rte_<key>_m<m>)."""
import functools
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold

from common.data import TARGET, folds as fold_ids, labels, read_original, read_test, read_train

RAW = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned", "Charging_Stations_Near_Home",
       "Charging_Stations_Near_Work", "Environmental_Concern_Level", "Gender", "City_Type", "Current_Car_Type",
       "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
CAT_MAPS = {"Gender": {"Male": 0, "Female": 1, "Other": 2}, "City_Type": {"Urban": 0, "Suburban": 1, "Rural": 2},
            "Current_Car_Type": {"Sedan": 0, "SUV": 1, "Hatchback": 2, "Truck": 3}, "Home_Charging_Possible": {"No": 0, "Yes": 1},
            "Subsidy_Available": {"No": 0, "Yes": 1}, "Range_Anxiety_Level": {"Low": 0, "Medium": 1, "High": 2}}

# the champion matrix: blocks + nested TEs of 11 keys (+ 4 income TE differences)
BLOCKS = ["raw", "freq_log", "flags"]
TE_BASE = [("inc", 1.0), ("inc_d10", 10.0), ("inc_d100", 10.0), ("inc_d1000", 10.0), ("km_d1", 10.0)]
TE_ALL = [("age", 1.0), ("km", 1.0), ("st_home", 1.0), ("st_work", 1.0), ("cell4", 1.0), ("city_block", 1.0)]
CHAMPION_TE = TE_BASE + TE_ALL
# residual TEs of the income keys
RTE_INC = [("inc", 1), ("inc_d100", 10), ("inc_d1000", 10)]


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class Blocks:
    """Label-free blocks and integer key codes over train + test (rows 0..n_tr-1 are train)."""

    def __init__(self, tr, te, og):
        self.n_tr, self.n_te = len(tr), len(te)
        both = pd.concat([tr[RAW], te[RAW]], ignore_index=True)
        self.both = both
        enc = both.copy()
        for c in CATS:
            enc[c] = enc[c].map(CAT_MAPS[c])
        self.raw = enc.astype(np.float32)

        # target-encoding keys
        inc = both.Annual_Income_USD.astype(np.int64)
        self.keys = pd.DataFrame({
            "inc": inc, "inc_d10": inc // 10, "inc_d100": inc // 100, "inc_d1000": inc // 1000, "inc_d5000": inc // 5000,
            "km": (both.Daily_Commute_km * 10).round().astype(np.int64), "km_d1": both.Daily_Commute_km.round(0).astype(np.int64),
            "km_d5": (both.Daily_Commute_km // 5).astype(np.int64),
            "age": both.Age.astype(np.int64), "st_home": both.Charging_Stations_Near_Home.astype(np.int64),
            "st_work": both.Charging_Stations_Near_Work.astype(np.int64),
            "env": both.Environmental_Concern_Level.astype(np.int64), "cars": both.Number_of_Cars_Owned.astype(np.int64),
        })
        for c in CATS:
            self.keys[c] = enc[c].astype(np.int64)
        self.keys["cell4"] = ((self.keys.env * 100 + self.keys.Subsidy_Available * 10 + self.keys.Range_Anxiety_Level) * 2
                              + self.keys.Home_Charging_Possible)
        self.keys["cell3"] = self.keys.env * 100 + self.keys.Subsidy_Available * 10 + self.keys.Range_Anxiety_Level
        self.keys["city_block"] = (self.keys.City_Type * 20 + self.keys.st_home) * 20 + self.keys.st_work
        self.codes, self.card = {}, {}
        for k in self.keys.columns:
            c, u = pd.factorize(self.keys[k])
            self.codes[k] = c.astype(np.int64)
            self.card[k] = len(u)

        # value counts over train + test
        freq = {}
        for c in RAW:
            vc = both[c].value_counts()
            freq[f"{c}_cnt"] = both[c].map(vc).astype(np.float32)
        self.freq = pd.DataFrame(freq)
        self.freq_log = np.log1p(self.freq).add_suffix("_log").astype(np.float32)

        # flags, incl. lookups in the original data
        so = set(og.Annual_Income_USD.dropna().unique())
        ogy = (og.Will_Buy_EV == "Yes").astype(int)
        g = og.assign(y=ogy).groupby("Annual_Income_USD").y.agg(["mean", "size"])
        par = both[["Annual_Income_USD"]].join(g, on="Annual_Income_USD")
        og_n = par["size"].fillna(0).astype(np.float32).values
        self.flags = pd.DataFrame({
            "inc_is_30k": (both.Annual_Income_USD == 30000).astype(np.float32),
            "km_is_5": (both.Daily_Commute_km == 5.0).astype(np.float32),
            "inc_novel": (~both.Annual_Income_USD.isin(so)).astype(np.float32),
            "og_n": og_n,
        })

    def register(self, name, values):
        """Add a TE key: one value per train + test row (any hashable values) -> integer codes."""
        if name in self.codes:
            return
        c, u = pd.factorize(pd.Series(values))
        self.codes[name] = c.astype(np.int64)
        self.card[name] = len(u)

    def train_part(self, df):
        return df.iloc[:self.n_tr].reset_index(drop=True)

    def test_part(self, df):
        return df.iloc[self.n_tr:].reset_index(drop=True)


def assemble(B, blocks=BLOCKS):
    parts = {"raw": B.raw, "freq_log": B.freq_log, "flags": B.flags}
    return pd.concat([parts[b] for b in blocks], axis=1)


class NestedTE:
    """specs = list of (key, m). fold() returns (fit, valid, test) DataFrames."""

    def __init__(self, blocks, y, inner_splits=5, seed=0):
        self.b = blocks
        self.y = np.asarray(y).astype(np.float64)
        self.inner = inner_splits
        self.seed = seed

    def fold(self, fit_idx, val_idx, specs):
        b = self.b
        yf = self.y[fit_idx]
        prior = yf.mean()
        inner = list(StratifiedKFold(self.inner, shuffle=True, random_state=self.seed).split(np.zeros(len(fit_idx)), yf.astype(int)))
        F = np.empty((len(fit_idx), len(specs)), np.float32)
        V = np.empty((len(val_idx), len(specs)), np.float32)
        T = np.empty((b.n_te, len(specs)), np.float32)
        test_pos = np.arange(b.n_tr, b.n_tr + b.n_te)
        for j, (k, m) in enumerate(specs):
            code = b.codes[k]
            card = b.card[k]
            cf = code[fit_idx]
            tot_n = np.bincount(cf, minlength=card).astype(np.float64)
            tot_s = np.bincount(cf, weights=yf, minlength=card)
            for _, hold in inner:
                hn = np.bincount(cf[hold], minlength=card).astype(np.float64)
                hs = np.bincount(cf[hold], weights=yf[hold], minlength=card)
                F[hold, j] = ((tot_s - hs + m * prior) / (tot_n - hn + m))[cf[hold]]
            full = (tot_s + m * prior) / (tot_n + m)
            V[:, j] = full[code[val_idx]]
            T[:, j] = full[code[test_pos]]
        cols = [f"te_{k}_m{m:g}" for k, m in specs]
        return pd.DataFrame(F, columns=cols), pd.DataFrame(V, columns=cols), pd.DataFrame(T, columns=cols)


class ResidTE:
    """NestedTE of r = y - p_rule. Inner splits are stratified on the label, so they are the same as NestedTE's."""

    def __init__(self, blocks, y, r, inner_splits=5, seed=0):
        self.b = blocks
        self.y = np.asarray(y).astype(int)
        self.r = np.asarray(r).astype(np.float64)
        self.inner = inner_splits
        self.seed = seed

    def fold(self, fit_idx, val_idx, specs):
        b = self.b
        rf = self.r[fit_idx]
        yf = self.y[fit_idx]
        prior = rf.mean()
        inner = list(StratifiedKFold(self.inner, shuffle=True, random_state=self.seed).split(np.zeros(len(fit_idx)), yf))
        F = np.empty((len(fit_idx), len(specs)), np.float32)
        V = np.empty((len(val_idx), len(specs)), np.float32)
        T = np.empty((b.n_te, len(specs)), np.float32)
        test_pos = np.arange(b.n_tr, b.n_tr + b.n_te)
        for j, (k, m) in enumerate(specs):
            code = b.codes[k]
            card = b.card[k]
            cf = code[fit_idx]
            tot_n = np.bincount(cf, minlength=card).astype(np.float64)
            tot_s = np.bincount(cf, weights=rf, minlength=card)
            F[:, j] = 0
            for _, hold in inner:
                hn = np.bincount(cf[hold], minlength=card).astype(np.float64)
                hs = np.bincount(cf[hold], weights=rf[hold], minlength=card)
                F[hold, j] = ((tot_s - hs + m * prior) / (tot_n - hn + m))[cf[hold]]
            full = (tot_s + m * prior) / (tot_n + m)
            V[:, j] = full[code[val_idx]]
            T[:, j] = full[code[test_pos]]
        cols = [f"rte_{k}_m{m:g}" for k, m in specs]
        return pd.DataFrame(F, columns=cols), pd.DataFrame(V, columns=cols), pd.DataFrame(T, columns=cols)


def add_diffs(D):
    """4 differences of income TE logits (needs te_inc_m1, te_inc_d10_m10, te_inc_d100_m10, te_inc_d1000_m10)."""
    D = D.copy()
    L = lambda c: logit(D[c].values)  # noqa: E731
    D["d_inc_d100"] = L("te_inc_m1") - L("te_inc_d100_m10")
    D["d_inc_d1000"] = L("te_inc_m1") - L("te_inc_d1000_m10")
    D["d_d100_d1000"] = L("te_inc_d100_m10") - L("te_inc_d1000_m10")
    D["d_inc_d10"] = L("te_inc_m1") - L("te_inc_d10_m10")
    return D


def rule_prob(both):
    """p_rule = Phi((s - 5.6092) / 0.8345), s = 1.2 income / 1e5 + 0.6 env + 2 subsidy - 1 [anxiety Medium] - 3 [anxiety High]
    (the purchase score of the script that generated the original data). float32-rounded, as float64, one value per row."""
    inc = both.Annual_Income_USD.values.astype(float)
    env = both.Environmental_Concern_Level.values.astype(float)
    sub = (both.Subsidy_Available == "Yes").astype(float).values
    anx = both.Range_Anxiety_Level.values
    med = (anx == "Medium").astype(float)
    high = (anx == "High").astype(float)
    ps_noanx = 1.2 * inc / 1e5 + 0.6 * env + 2.0 * sub
    ps = ps_noanx - 1.0 * med - 3.0 * high
    return norm.cdf((ps - 5.6092) / 0.8345).astype(np.float32).astype(np.float64)


ARTIFACT_COLS = ["art_inc_mod10e1", "art_inc_mod10e2", "art_inc_mod10e3", "art_inc_mod10e4", "art_inc_digit0", "art_inc_digit1",
                 "art_inc_digit2", "art_inc_digit3", "art_inc_digit4", "art_km_frac", "art_km_int", "art_inc_round1000_dev"]


def artifact_features(both):
    """Artifact bundle (12 float32 columns, row-wise): income mod 10 / 100 / 1000 / 10000, its last 5 digits, the commute's
    first decimal and integer part, income minus the nearest multiple of 1000."""
    F = pd.DataFrame(index=both.index)
    for k in (1, 2, 3, 4):
        F[f"art_inc_mod10e{k}"] = (both.Annual_Income_USD % (10 ** k)).astype(np.float32)
    for k in range(0, 5):
        F[f"art_inc_digit{k}"] = ((both.Annual_Income_USD // (10 ** k)) % 10).astype(np.float32)
    F["art_km_frac"] = ((both.Daily_Commute_km * 10) % 10).astype(np.float32)
    F["art_km_int"] = np.floor(both.Daily_Commute_km).astype(np.float32)
    F["art_inc_round1000_dev"] = (both.Annual_Income_USD - 1000 * np.round(both.Annual_Income_USD / 1000)).astype(np.float32)
    return F[ARTIFACT_COLS]


def income_piece(both):
    """Leading GPT-2 BPE piece of every row's income string (common/bpe.py), e.g. 92887 -> '9'."""
    from common.bpe import income_bpe_table, leading_piece
    inc = both.Annual_Income_USD.values
    return leading_piece(inc, income_bpe_table(inc))


@functools.lru_cache(maxsize=None)
def setup():
    tr, te, og = read_train(), read_test(), read_original()
    y = labels(tr[TARGET]).astype(np.int64)
    f = fold_ids(y)
    folds = [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(5)]
    both = pd.concat([tr.drop(columns=[TARGET]), te], ignore_index=True)
    B = Blocks(tr, te, og)
    return SimpleNamespace(tr=tr, te=te, og=og, both=both, y=y, folds=folds, B=B, TE=NestedTE(B, y))
