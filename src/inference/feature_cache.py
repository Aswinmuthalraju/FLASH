"""Hot-path state: incremental features (ring buffers) and the in-memory news cache. No I/O, no pandas.

FeatureState.update() must reproduce src.features.market_features.batch_features exactly
(tests/test_feature_parity.py).
"""
from __future__ import annotations

import math

import numpy as np

from src.features.market_features import EPS, F, IDX, W_LONG, W_MID, WARMUP
from src.features.news_features import decay_table, decayed


class FeatureState:
    """Per-symbol incremental feature state. Mirror ring buffers (size 2W) give a contiguous,
    chronologically ordered window view with zero copies."""

    __slots__ = ("n", "prev_lc", "ret", "mret", "lc", "lv", "out", "bars_per_day")

    def __init__(self, bars_per_day: int = 375):
        W = W_LONG
        self.ret = np.zeros(2 * W)
        self.mret = np.zeros(2 * W)
        self.lc = np.zeros(2 * W)
        self.lv = np.zeros(2 * W)
        self.out = np.empty(F)
        self.n = 0
        self.prev_lc = 0.0
        self.bars_per_day = bars_per_day

    def reset(self) -> None:
        self.n = 0

    def update(self, h: float, l: float, c: float, v: float, mret: float) -> bool:
        """Push one bar (open is not used by any feature). Returns True when self.out is valid."""
        W = W_LONG
        i = self.n
        lc = math.log(c)
        if i == 0:
            r = 0.0
            mr = 0.0
        else:
            r = lc - self.prev_lc
            mr = mret
        p = i % W
        for buf, x in ((self.ret, r), (self.mret, mr), (self.lc, lc), (self.lv, math.log1p(v))):
            buf[p] = x
            buf[p + W] = x
        self.prev_lc = lc
        self.n = i + 1
        if i < WARMUP:
            return False
        s = p + 1
        r30 = self.ret[s:s + W]
        m30 = self.mret[s:s + W]
        lcw = self.lc[s:s + W]
        v30 = self.lv[s:s + W]
        r20, m20 = r30[-W_MID:], m30[-W_MID:]
        mean20 = r20.mean()
        mvar = m20.var()
        cov20 = ((r20 - mean20) * (m20 - m20.mean())).mean()
        o = self.out
        o[IDX["ret1"]] = r
        o[IDX["ret5"]] = lc - lcw[-6]
        o[IDX["ret15"]] = lc - lcw[-16]
        o[IDX["mean20"]] = mean20
        o[IDX["var20"]] = r20.var()
        o[IDX["vol30"]] = r30.std()
        o[IDX["cov20"]] = cov20
        o[IDX["beta20"]] = cov20 / (mvar + EPS)
        o[IDX["mkt_ret1"]] = mr
        o[IDX["mkt_vol20"]] = math.sqrt(mvar)
        o[IDX["range"]] = (h - l) / c
        o[IDX["close_pos"]] = (c - l) / (h - l + EPS)
        o[IDX["vol_z"]] = (self.lv[p] - v30.mean()) / (v30.std() + 1e-9)
        o[IDX["tod"]] = i / self.bars_per_day
        return True


class NewsCache:
    """In-memory dict cache: symbol -> (raw news vector, event trading-minute). read() applies the decay."""

    def __init__(self, symbols: list[str], tau: float, max_age: float):
        self.raw: dict[str, np.ndarray | None] = {s: None for s in symbols}
        self.tmin: dict[str, float] = {s: 0.0 for s in symbols}
        self.max_age = max_age
        self.table = decay_table(tau, int(max_age))

    def set(self, symbol: str, raw: np.ndarray, event_tmin: float) -> None:
        if self.raw.get(symbol) is None or event_tmin >= self.tmin[symbol]:      # latest event wins
            self.raw[symbol] = raw
            self.tmin[symbol] = event_tmin

    def read(self, symbol: str, bar_tmin: float, out: np.ndarray) -> np.ndarray:
        raw = self.raw[symbol]
        if raw is None:
            out[:] = 0.0
            return out
        return decayed(raw, bar_tmin - self.tmin[symbol], self.table, self.max_age, out)
