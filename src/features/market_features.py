"""Market features: batch (training) and incremental (live) implementations of the SAME definitions.

Per (symbol, day) series -- the state resets at every session open, so bar index == minute of session.
A feature row is valid from the 31st bar of the day (30 returns are needed for the long window).

Unit 1 (probability): rolling mean / variance / covariance of returns (covariance is with the
equal-weight universe return, so beta = cov / var(market)).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

FEATURES = ["ret1", "ret5", "ret15", "mean20", "var20", "vol30", "cov20", "beta20",
            "mkt_ret1", "mkt_vol20", "range", "close_pos", "vol_z", "tod"]
F = len(FEATURES)
PRICE_FEATURES = ["ret1", "ret5", "ret15", "mean20", "var20", "vol30", "range", "close_pos"]
PRICE_IDX = [FEATURES.index(f) for f in PRICE_FEATURES]
IDX = {f: i for i, f in enumerate(FEATURES)}
# Slow-moving subset that feeds PCA -> HMM. Per-bar noise (ret1, close_pos, range, vol_z, mkt_ret1) is left out:
# fed to an HMM it makes the "regimes" flip every ~2 bars instead of describing market conditions.
REGIME_FEATURES = ["ret5", "ret15", "mean20", "var20", "vol30", "cov20", "beta20", "mkt_vol20"]
REGIME_IDX = [IDX[f] for f in REGIME_FEATURES]
W_MID, W_LONG = 20, 30
WARMUP = W_LONG          # first valid bar index within a day
EPS = 1e-12

DOWN, NEUTRAL, UP = 0, 1, 2
CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]


# ----------------------------------------------------------------------------- batch

def batch_day_features(o, h, l, c, v, mret, bars_per_day: int = 375) -> np.ndarray:
    """(n, F) features for ONE symbol-day; NaN before the warm-up."""
    n = len(c)
    X = np.full((n, F), np.nan)
    if n <= WARMUP:
        return X
    lc = np.log(c)
    ret = np.zeros(n)
    ret[1:] = np.diff(lc)
    mr = np.asarray(mret, dtype=float).copy()
    mr[0] = 0.0
    lv = np.log1p(v)
    end = np.arange(WARMUP, n)

    def win(a, w):
        return sliding_window_view(a, w)[end - w + 1]

    r20, r30, m20, v30 = win(ret, W_MID), win(ret, W_LONG), win(mr, W_MID), win(lv, W_LONG)
    mean20 = r20.mean(axis=1)
    var20 = r20.var(axis=1)
    mvar = m20.var(axis=1)
    cov20 = ((r20 - mean20[:, None]) * (m20 - m20.mean(axis=1)[:, None])).mean(axis=1)
    X[end, IDX["ret1"]] = ret[end]
    X[end, IDX["ret5"]] = lc[end] - lc[end - 5]
    X[end, IDX["ret15"]] = lc[end] - lc[end - 15]
    X[end, IDX["mean20"]] = mean20
    X[end, IDX["var20"]] = var20
    X[end, IDX["vol30"]] = r30.std(axis=1)
    X[end, IDX["cov20"]] = cov20
    X[end, IDX["beta20"]] = cov20 / (mvar + EPS)
    X[end, IDX["mkt_ret1"]] = mr[end]
    X[end, IDX["mkt_vol20"]] = np.sqrt(mvar)
    X[end, IDX["range"]] = (h[end] - l[end]) / c[end]
    X[end, IDX["close_pos"]] = (c[end] - l[end]) / (h[end] - l[end] + EPS)
    X[end, IDX["vol_z"]] = (lv[end] - v30.mean(axis=1)) / (v30.std(axis=1) + 1e-9)
    X[end, IDX["tod"]] = end / bars_per_day
    return X


def market_return(bars: pd.DataFrame) -> np.ndarray:
    """Equal-weight universe 1-bar log return per row of `bars` (sorted by symbol, timestamp)."""
    lc = np.log(bars["close"].to_numpy())
    day = bars["timestamp"].dt.normalize()
    ret = bars.assign(_lc=lc).groupby(["symbol", day])["_lc"].diff().fillna(0.0).to_numpy()
    m = pd.Series(ret).groupby(bars["timestamp"].to_numpy()).transform("mean").to_numpy().copy()
    first = (bars.groupby(["symbol", day]).cumcount() == 0).to_numpy()
    m[first] = 0.0
    return m


def batch_features(bars: pd.DataFrame, cfg: dict) -> np.ndarray:
    """(len(bars), F) for bars sorted by (symbol, timestamp)."""
    n = cfg["market"]["bars_per_day"]
    mret = market_return(bars)
    X = np.full((len(bars), F), np.nan)
    day = bars["timestamp"].dt.normalize()
    o, h, l, c, v = (bars[k].to_numpy() for k in ("open", "high", "low", "close", "volume"))
    for rows in bars.groupby(["symbol", day]).indices.values():
        X[rows] = batch_day_features(o[rows], h[rows], l[rows], c[rows], v[rows], mret[rows], n)
    return X


# ------------------------------------------------------------------------------ labels

def triple_barrier(close: np.ndarray, sigma: np.ndarray, horizon: int, k: float, scale_sqrt_h: bool) -> np.ndarray:
    """Labels for ONE symbol-day. Barriers are placed at close*exp(+-k*sigma*s) (s = sqrt(h) if scaled).

    UP if the upper barrier is hit first within `horizon` bars, DOWN if the lower is, else NEUTRAL.
    Bars whose horizon would run past the session end, or without a sigma, get -1.
    """
    n = len(close)
    y = np.full(n, -1, dtype=np.int8)
    if n <= horizon:
        return y
    lc = np.log(close)
    path = sliding_window_view(lc[1:], horizon) - lc[: n - horizon, None]      # (n-h, h)
    thr = k * sigma[: n - horizon] * (math.sqrt(horizon) if scale_sqrt_h else 1.0)
    up, dn = path >= thr[:, None], path <= -thr[:, None]
    big = horizon + 1
    t_up = np.where(up.any(axis=1), up.argmax(axis=1), big)
    t_dn = np.where(dn.any(axis=1), dn.argmax(axis=1), big)
    lab = np.where(t_up < t_dn, UP, np.where(t_dn < t_up, DOWN, NEUTRAL)).astype(np.int8)
    lab[~np.isfinite(thr)] = -1
    y[: n - horizon] = lab
    return y


def batch_labels(bars: pd.DataFrame, X: np.ndarray, cfg: dict) -> dict[int, np.ndarray]:
    lab = cfg["labels"]
    day = bars["timestamp"].dt.normalize()
    close = bars["close"].to_numpy()
    sigma = X[:, IDX["vol30"]]
    out = {h: np.full(len(bars), -1, dtype=np.int8) for h in lab["horizons"]}
    for rows in bars.groupby(["symbol", day]).indices.values():
        for h in lab["horizons"]:
            out[h][rows] = triple_barrier(close[rows], sigma[rows], h, lab["k"], lab["scale_sqrt_h"])
    return out


def build_table(bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Feature + label table, one row per bar (rows with invalid features/labels are KEPT, flagged)."""
    from src.data.align import Calendar, assign_split, split_days

    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    X = batch_features(bars, cfg)
    labels = batch_labels(bars, X, cfg)
    cal = Calendar.from_bars(bars, cfg)
    df = pd.DataFrame(X, columns=FEATURES)
    df.insert(0, "symbol", bars["symbol"].to_numpy())
    df.insert(1, "timestamp", bars["timestamp"])
    df["close"] = bars["close"].to_numpy()
    df["tmin"] = cal.bar_tmin(bars["timestamp"])
    df["day_idx"] = (df["tmin"] // cfg["market"]["bars_per_day"]).astype(int)
    for h, y in labels.items():
        df[f"y{h}"] = y
    df["valid_x"] = ~np.isnan(X).any(axis=1)
    df["split"] = assign_split(bars["timestamp"], split_days(bars["timestamp"].dt.normalize().unique(), cfg))
    return df
