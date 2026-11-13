"""FLASH on a NEW dataset: 15 years of daily NSE bars (20 stocks), same five model families as the minute pipeline.

Purpose: the 1-minute real-data run showed that *whether* a move happens is somewhat predictable but *direction* is
close to chance. This script re-runs the same course-unit models (Units 1-5) on a longer, lower-frequency dataset to
see whether direction becomes learnable when there is far more history per regime and far less microstructure noise.
No news (daily news is not available for this dataset); the news branch is out of scope here and is said so in the deck.

    .venv/bin/python scripts/train_daily.py            # train + score VALIDATION only
    .venv/bin/python scripts/train_daily.py --test     # same, plus ONE scoring of the test split (final)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.baseline_logistic import BaselineLogistic
from src.models.calibration import Calibrator
from src.models.fusion_lgbm import FusionBlend
from src.models.hmm_regime import HMMRegime
from src.models.regime_pca_gmm import RegimePCA

HORIZONS = [5, 10, 20]            # trading days
K = 1.0                           # barrier multiplier (sigma * sqrt(h))
CFG = {"lgbm": dict(n_estimators=3000, learning_rate=0.03, num_leaves=7, min_child_samples=300, reg_lambda=10,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.5, early_stopping_rounds=50),
       "blend_logit_weight": 0.25, "logit_C": 0.05}
PRICE = ["ret1", "ret5", "ret20", "mean20", "var20", "vol20", "range", "close_pos"]
MARKET = PRICE + ["cov20", "beta20", "mkt_ret1", "mkt_vol20", "vol_z", "dow"]
REGIME_F = ["mkt_ret5", "mkt_ret20", "mkt_mean20", "mkt_var20", "mkt_vol20"]


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    df["lr"] = np.log(df["close"]).groupby(df["symbol"]).diff()
    mret = df.groupby("date")["lr"].mean().rename("mret")                    # equal-weight universe return
    df = df.join(mret, on="date")
    g = df.groupby("symbol", sort=False)
    df["ret1"] = df["lr"]
    df["ret5"] = g["lr"].transform(lambda s: s.rolling(5).sum())
    df["ret20"] = g["lr"].transform(lambda s: s.rolling(20).sum())
    df["mean20"] = g["lr"].transform(lambda s: s.rolling(20).mean())          # Unit 1: rolling moments
    df["var20"] = g["lr"].transform(lambda s: s.rolling(20).var())
    df["vol20"] = np.sqrt(df["var20"])
    df["range"] = (df["high"] - df["low"]) / df["close"]
    df["close_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    lv = np.log1p(df["volume"])
    df["vol_z"] = (lv - lv.groupby(df["symbol"]).transform(lambda s: s.rolling(60).mean())) / \
        lv.groupby(df["symbol"]).transform(lambda s: s.rolling(60).std())
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["_m"] = df["mret"]
    df["cov20"] = df.groupby("symbol", sort=False).apply(
        lambda x: x["lr"].rolling(20).cov(x["_m"]), include_groups=False).reset_index(level=0, drop=True)
    mv = df.groupby("symbol", sort=False)["_m"].transform(lambda s: s.rolling(20).var())
    df["beta20"] = df["cov20"] / mv
    df["mkt_ret1"] = df["mret"]
    m = mret.to_frame()
    m["mkt_ret5"] = m["mret"].rolling(5).sum()
    m["mkt_ret20"] = m["mret"].rolling(20).sum()
    m["mkt_mean20"] = m["mret"].rolling(20).mean()
    m["mkt_var20"] = m["mret"].rolling(20).var()
    m["mkt_vol20"] = np.sqrt(m["mkt_var20"])
    df = df.drop(columns=["mkt_vol20"], errors="ignore").join(m[REGIME_F[:2] + REGIME_F[2:]], on="date")
    # triple-barrier labels, closes only, barrier = close * exp(+-k*sigma*sqrt(h)), sigma = vol20 at the signal day
    for h in HORIZONS:
        lab = np.full(len(df), np.nan)
        for _, idx in df.groupby("symbol", sort=False).indices.items():
            c = np.log(df["close"].values[idx])
            s = df["vol20"].values[idx]
            n = len(idx)
            out = np.full(n, np.nan)
            for t in range(n - h):
                if not np.isfinite(s[t]):
                    continue
                b = K * s[t] * np.sqrt(h)
                path = c[t + 1:t + h + 1] - c[t]
                up, dn = np.flatnonzero(path >= b), np.flatnonzero(path <= -b)
                u = up[0] if len(up) else 10 ** 9
                d = dn[0] if len(dn) else 10 ** 9
                out[t] = 2 if u < d else (0 if d < u else 1)
            lab[idx] = out
        df[f"y{h}"] = lab
    return df


def brier(P, y):
    return float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1)))


def score(P, y):
    pred = P.argmax(1)
    mv = y != 1
    out = dict(auc=float(roc_auc_score(y, P, multi_class="ovr", average="macro")),
               f1=float(f1_score(y, pred, average="macro")), brier=brier(P, y),
               pred_share=[float(np.mean(pred == c)) for c in range(3)])
    out["move_auc"] = float(roc_auc_score(mv, 1 - P[:, 1]))
    if mv.sum() > 50 and len(set(y[mv])) == 2:                           # UP vs DOWN on moving rows only
        pu = P[mv, 2] / (P[mv, 2] + P[mv, 0] + 1e-12)
        out["dir_auc"] = float(roc_auc_score(y[mv] == 2, pu))
        conf = np.abs(pu - 0.5) > np.quantile(np.abs(pu - 0.5), 0.8)       # most confident 20 %
        out["dir_acc_top20"] = float(np.mean((pu[conf] > 0.5) == (y[mv][conf] == 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/daily/daily_prices.parquet"))
    ap.add_argument("--out", default=str(ROOT / "reports/daily"))
    ap.add_argument("--test", action="store_true", help="also score the locked test split, once")
    a = ap.parse_args()
    t0 = time.time()
    raw = pd.read_parquet(a.data)
    raw = raw[~raw["symbol"].str.startswith("^")]
    df = build(raw)
    dates = np.array(sorted(df["date"].unique()))
    i1, i2 = int(len(dates) * 0.6), int(len(dates) * 0.8)
    split = {"train": (dates[0], dates[i1 - 1]), "val": (dates[i1], dates[i2 - 1]), "test": (dates[i2], dates[-1])}
    part = np.where(df["date"] <= split["train"][1], "train", np.where(df["date"] <= split["val"][1], "val", "test"))
    df["part"] = part
    # Unit 3 + 4: PCA -> GMM-initialised HMM on the MARKET-level series, fit on train only; filtered posteriors only
    mk = df.drop_duplicates("date").sort_values("date").dropna(subset=REGIME_F)
    mtr, mva = mk[mk["date"] <= split["train"][1]], mk[(mk["date"] > split["train"][1]) & (mk["date"] <= split["val"][1])]
    rp = RegimePCA(0.90).fit(mtr[REGIME_F].values, mva[REGIME_F].values)
    hmm = HMMRegime(4, seed=7).fit(rp.transform(mtr[REGIME_F].values), [len(mtr)])
    post = hmm.filter_posteriors(rp.transform(mk[REGIME_F].values), [len(mk)])
    rg = pd.DataFrame(post, columns=[f"reg{i}" for i in range(4)], index=mk["date"].values)
    df = df.join(rg, on="date")
    REG = [f"reg{i}" for i in range(4)]
    cols_D = MARKET + REG
    res = {"data": {"rows": int(len(df)), "symbols": int(df["symbol"].nunique()), "split": {k: [str(x) for x in v] for k, v in split.items()},
                    "days": {k: int((part[~df.duplicated("date").values] == k).sum()) for k in split}},
           "pca_components": int(rp.n_components_), "val_var_explained": rp.val_explained_, "horizons": {}}
    for h in HORIZONS:
        y = f"y{h}"
        ok = df[cols_D + [y]].notna().all(axis=1)
        sets = {p: df[ok & (df["part"] == p)] for p in ("train", "val", "test")}
        ytr, yva = sets["train"][y].astype(int).values, sets["val"][y].astype(int).values
        prior = np.bincount(ytr, minlength=3) / len(ytr)
        r = {"n": {p: int(len(v)) for p, v in sets.items()}, "class_share_train": prior.tolist(),
             "prior_brier_val": brier(np.tile(prior, (len(yva), 1)), yva)}
        A = BaselineLogistic(C=1.0).fit(sets["train"][PRICE].values, ytr)
        D = FusionBlend(CFG, 7).fit(sets["train"][cols_D].values, ytr, sets["val"][cols_D].values, yva)
        cal = Calibrator("temperature").fit(D.predict_proba(sets["val"][cols_D].values), yva)
        r["A_val"] = score(A.predict_proba(sets["val"][PRICE].values), yva)
        r["D_val"] = score(cal.transform(D.predict_proba(sets["val"][cols_D].values)), yva)
        r["D_val_uncal"] = score(D.predict_proba(sets["val"][cols_D].values), yva)
        r["lgbm_iters"] = D.best_iteration
        if a.test:                                                          # the one and only test scoring
            yte = sets["test"][y].astype(int).values
            r["A_test"] = score(A.predict_proba(sets["test"][PRICE].values), yte)
            r["D_test"] = score(cal.transform(D.predict_proba(sets["test"][cols_D].values)), yte)
            r["prior_brier_test"] = brier(np.tile(prior, (len(yte), 1)), yte)
        res["horizons"][str(h)] = r
        print(f"h={h}", json.dumps({k: r[k] for k in r if k.startswith(("A_val", "D_val"))}, indent=None)[:600], flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("daily_results_test.json" if a.test else "daily_results_val.json")).write_text(json.dumps(res, indent=2))
    print("done", res["seconds"], "s")


if __name__ == "__main__":
    main()
