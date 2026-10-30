"""Market data: CSV/Parquet loader, yfinance fallback, regularisation, synthetic generator.

Canonical schema: timestamp (tz-aware Asia/Kolkata, bar OPEN time), symbol, open, high, low, close, volume.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
_ALIASES = {
    "timestamp": ["timestamp", "datetime", "date", "time", "ts", "date_time"],
    "symbol": ["symbol", "ticker", "stock", "name", "instrument"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c", "last", "adj close"],
    "volume": ["volume", "vol", "v", "qty"],
}


def _normalise_columns(df: pd.DataFrame, default_symbol: str | None) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    out = {}
    for canon, alts in _ALIASES.items():
        for a in alts:
            if a in lower:
                out[canon] = df[lower[a]]
                break
    if "symbol" not in out:
        if default_symbol is None:
            raise ValueError("no symbol column and no default_symbol given")
        out["symbol"] = pd.Series(default_symbol, index=df.index)
    missing = [c for c in COLS if c not in out]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return pd.DataFrame(out)[COLS]


def validate_bars(df: pd.DataFrame, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    ts = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)
    df["timestamp"] = ts
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.replace(".NS", "", regex=False)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    df = df.drop_duplicates(["symbol", "timestamp"], keep="last")
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def regularize_bars(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Reindex every (symbol, day) onto the full session grid so that bar index == minute of day.

    Missing minutes are forward-filled (o=h=l=c=prev close, volume 0). Sparse days are dropped.
    """
    m = cfg["market"]
    n = m["bars_per_day"]
    open_h, open_m = map(int, m["session_open"].split(":"))
    ts = df["timestamp"]
    minute = (ts.dt.hour * 60 + ts.dt.minute) - (open_h * 60 + open_m)
    df = df.assign(_day=ts.dt.normalize(), _min=minute)
    df = df[(df["_min"] >= 0) & (df["_min"] < n)]
    parts = []
    for (sym, day), g in df.groupby(["symbol", "_day"], sort=True):
        if len(g) < m["min_day_coverage"] * n:
            continue
        g = g.set_index("_min").reindex(range(n))
        g["close"] = g["close"].ffill().bfill()
        for c in ["open", "high", "low"]:
            g[c] = g[c].fillna(g["close"])
        g["volume"] = g["volume"].fillna(0.0)
        g["symbol"] = sym
        g["timestamp"] = day + pd.to_timedelta(g.index * 60 + (open_h * 3600 + open_m * 60), unit="s")
        parts.append(g[COLS].reset_index(drop=True))
    if not parts:
        raise ValueError("no usable trading days after regularisation")
    return pd.concat(parts, ignore_index=True)


def load_market(path: str | Path, cfg: dict, default_symbol: str | None = None) -> pd.DataFrame:
    """Load a CSV/Parquet file (or a directory of them) of 1-minute OHLCV bars."""
    p = Path(path)
    files = sorted(p.glob("*.csv")) + sorted(p.glob("*.parquet")) if p.is_dir() else [p]
    frames = []
    for f in files:
        raw = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        frames.append(_normalise_columns(raw, default_symbol or f.stem.upper()))
    df = validate_bars(pd.concat(frames, ignore_index=True), cfg["market"]["tz"])
    return regularize_bars(df, cfg)


def load_yfinance(symbols: list[str], cfg: dict, period: str = "5d") -> pd.DataFrame:
    """Fallback for RECENT data (Yahoo only serves ~7 days of 1-minute bars)."""
    import yfinance as yf

    frames = []
    for s in symbols:
        h = yf.Ticker(f"{s}.NS").history(period=period, interval="1m", auto_adjust=False)
        if h.empty:
            continue
        h = h.reset_index().rename(columns={"Datetime": "timestamp"})
        frames.append(_normalise_columns(h, s))
    if not frames:
        raise RuntimeError("yfinance returned no data")
    df = validate_bars(pd.concat(frames, ignore_index=True), cfg["market"]["tz"])
    return regularize_bars(df, cfg)


# ------------------------------------------------------------------ synthetic data

def _intraday_vol_shape(n: int) -> np.ndarray:
    x = np.linspace(0, 1, n)
    return 1.0 + 1.2 * np.exp(-x / 0.08) + 0.6 * np.exp(-(1 - x) / 0.08)  # U shape


# latent market regimes: (drift per bar, vol per bar)
_REGIMES = {
    0: (0.0, 0.00035),        # calm
    1: (0.00012, 0.00055),    # bull
    2: (-0.00012, 0.00055),   # bear
    3: (0.0, 0.00110),        # volatile
}
_P_STAY = 0.995
_REACTION_SPAN = 20          # bars over which a news shock plays out
_REACTION_PER_BAR = 0.00018  # per-bar drift for magnitude 1 (synthetic assumption)


def make_synthetic(cfg: dict, news_events: list[dict] | None = None, seed: int | None = None):
    """Regime-switching 1-minute OHLCV for the configured symbols.

    news_events: dicts {symbol, tmin, direction (+1/-1), magnitude}. After the first bar that is
    strictly later than the news, the stock drifts for _REACTION_SPAN bars in `direction`.
    Returns (bars_df, truth) where truth holds the latent market regime per (day, minute).
    """
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    syn, m = cfg["synthetic"], cfg["market"]
    n = m["bars_per_day"]
    days = pd.bdate_range(syn["start"], periods=syn["n_days"], tz=m["tz"])
    symbols = cfg["symbols"]
    S, D = len(symbols), len(days)
    beta = rng.uniform(0.7, 1.3, S)
    idio = rng.uniform(0.8, 1.2, S)
    p0 = rng.uniform(300, 3500, S)
    shape = np.tile(_intraday_vol_shape(n), D)

    total = D * n
    reg = np.empty(total, dtype=int)
    state = 0
    flip = rng.random(total) > _P_STAY
    pick = rng.integers(0, 4, total)
    for t in range(total):
        if flip[t]:
            state = pick[t]
        reg[t] = state
    drift = np.array([_REGIMES[s][0] for s in reg])
    vol = np.array([_REGIMES[s][1] for s in reg]) * shape
    mkt = drift + vol * rng.standard_normal(total)
    idio_noise = rng.standard_normal((S, total)) * (0.00045 * shape)[None] * idio[:, None]
    rets = beta[:, None] * mkt[None] + idio_noise

    if news_events:
        sym_idx = {s: i for i, s in enumerate(symbols)}
        for ev in news_events:
            t0 = int(np.floor(ev["tmin"])) + 1
            if t0 >= total:
                continue
            span = min(_REACTION_SPAN, n - (t0 % n))  # never leaks past the session end
            rets[sym_idx[ev["symbol"]], t0:t0 + span] += ev["direction"] * ev["magnitude"] * _REACTION_PER_BAR

    rows = []
    for si, sym in enumerate(symbols):
        r = rets[si].reshape(D, n)
        gap = rng.standard_normal(D) * 0.003
        base = p0[si]
        for di, day in enumerate(days):
            base = base * np.exp(gap[di])
            close = base * np.exp(np.cumsum(r[di]))
            openp = np.concatenate([[base], close[:-1]])
            sh = shape[:n]
            high = np.maximum(openp, close) * (1 + np.abs(rng.standard_normal(n)) * 0.0003 * sh)
            low = np.minimum(openp, close) * (1 - np.abs(rng.standard_normal(n)) * 0.0003 * sh)
            volume = rng.lognormal(9.0, 0.5, n) * sh * (1 + 400 * np.abs(r[di]))
            ts = day + pd.to_timedelta(np.arange(n) * 60 + 9 * 3600 + 15 * 60, unit="s")
            rows.append(pd.DataFrame({"timestamp": ts, "symbol": sym, "open": openp, "high": high,
                                      "low": low, "close": close, "volume": volume.round()}))
            base = close[-1]
    bars = pd.concat(rows, ignore_index=True)
    truth = pd.DataFrame({"day": np.repeat(days, n), "minute": np.tile(np.arange(n), D), "regime": reg})
    return bars, truth
