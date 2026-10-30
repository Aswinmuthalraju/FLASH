"""Time alignment and chronological splitting.

Convention: a bar's timestamp is its OPEN time. A news item is usable only at bars whose timestamp is
STRICTLY greater than the news timestamp (so a bar that was already running when the news arrived is
never allowed to see it).

`tmin` ("trading minute") is a monotone clock that skips nights and weekends:
    bar:  tmin = day_index * bars_per_day + minute_of_session
    news: in-session -> same formula (fractional); pre-open / after-close / weekend -> an epsilon before the
          open of the next trading day, so overnight news ages like it was released at the open and the
          mapping stays monotone in wall-clock time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_ns(ts: pd.Series) -> np.ndarray:
    """int64 nanoseconds since epoch, independent of the Series' datetime resolution (pandas 3 defaults to us)."""
    return pd.Series(pd.to_datetime(ts)).dt.as_unit("ns").astype("int64").to_numpy()


_EPS_T = 1e-6


class Calendar:
    def __init__(self, days: pd.DatetimeIndex, cfg: dict):
        self.days = pd.DatetimeIndex(sorted(set(days.normalize())))
        self.n = cfg["market"]["bars_per_day"]
        h, m = map(int, cfg["market"]["session_open"].split(":"))
        self.open_min = h * 60 + m
        self.tz = cfg["market"]["tz"]

    @classmethod
    def from_bars(cls, bars: pd.DataFrame, cfg: dict) -> "Calendar":
        return cls(pd.DatetimeIndex(bars["timestamp"].dt.normalize().unique()), cfg)

    def _day_and_offset(self, ts: pd.Series):
        ts = pd.Series(pd.to_datetime(ts))
        day = ts.dt.normalize()
        off = (ts - day).dt.total_seconds().to_numpy() / 60.0 - self.open_min
        j = self.days.searchsorted(pd.DatetimeIndex(day), side="left")
        jc = np.minimum(j, len(self.days) - 1)
        present = (j < len(self.days)) & (self.days[jc] == pd.DatetimeIndex(day))
        return j, off, present

    def bar_tmin(self, ts: pd.Series) -> np.ndarray:
        j, off, _ = self._day_and_offset(ts)
        return j * self.n + np.round(off)

    def news_tmin(self, ts: pd.Series) -> np.ndarray:
        j, off, present = self._day_and_offset(ts)
        n = self.n
        out = np.where(present & (off >= 0) & (off < n), j * n + off, np.nan)
        out = np.where(present & (off >= n), (j + 1) * n - _EPS_T, out)
        out = np.where(~present | (off < 0), j * n - _EPS_T, out)
        return out


def align_news_to_bars(bars: pd.DataFrame, events: pd.DataFrame, cal: Calendar, max_age_min: float):
    """For each bar row, index (into `events`) of the latest event of the same symbol with
    event.timestamp < bar.timestamp, and its age in trading minutes. -1 / 0 when none or stale.

    events needs columns: symbol, timestamp. Returns (event_idx, age_min) aligned to bars rows.
    """
    ev_idx = np.full(len(bars), -1, dtype=np.int64)
    age = np.zeros(len(bars))
    if len(events) == 0:
        return ev_idx, age
    ev_tmin = cal.news_tmin(events["timestamp"])
    bar_tmin = cal.bar_tmin(bars["timestamp"])
    ev_ns = to_ns(events["timestamp"])
    bar_ns = to_ns(bars["timestamp"])
    for sym, brows in bars.groupby("symbol").indices.items():
        erows = np.flatnonzero((events["symbol"] == sym).to_numpy())
        if len(erows) == 0:
            continue
        erows = erows[np.argsort(ev_ns[erows], kind="stable")]
        pos = np.searchsorted(ev_ns[erows], bar_ns[brows], side="left") - 1   # strictly before
        ok = pos >= 0
        sel = erows[np.where(ok, pos, 0)]
        dt = bar_tmin[brows] - ev_tmin[sel]
        ok &= (dt >= 0) & (dt <= max_age_min)
        ev_idx[brows] = np.where(ok, sel, -1)
        age[brows] = np.where(ok, dt, 0.0)
    return ev_idx, age


def check_no_lookahead(bars: pd.DataFrame, events: pd.DataFrame, ev_idx: np.ndarray) -> int:
    """Return the number of violating bars (event timestamp >= bar timestamp). Must be 0."""
    m = ev_idx >= 0
    ev_ts = to_ns(events["timestamp"])[ev_idx[m]]
    bar_ts = to_ns(bars["timestamp"])[m]
    return int((ev_ts >= bar_ts).sum())


# ------------------------------------------------------------------ chronological split

def split_days(days, cfg: dict) -> dict[str, pd.DatetimeIndex]:
    days = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(days).normalize())))
    n = len(days)
    n_tr = int(round(n * cfg["split"]["train"]))
    n_va = int(round(n * cfg["split"]["val"]))
    if n_tr < 1 or n_va < 1 or n_tr + n_va >= n:
        raise ValueError("not enough trading days for a train/val/test split")
    return {"train": days[:n_tr], "val": days[n_tr:n_tr + n_va], "test": days[n_tr + n_va:]}


def assign_split(ts: pd.Series, days_by_split: dict[str, pd.DatetimeIndex]) -> np.ndarray:
    day = pd.DatetimeIndex(pd.to_datetime(ts)).normalize()
    out = np.full(len(day), "", dtype=object)
    for name, d in days_by_split.items():
        out[day.isin(d)] = name
    if (out == "").any():
        raise ValueError("timestamps outside every split")
    return out


def chronological_split(df: pd.DataFrame, cfg: dict, ts_col: str = "timestamp") -> dict[str, pd.DataFrame]:
    sp = split_days(df[ts_col].dt.normalize().unique(), cfg)
    lab = assign_split(df[ts_col], sp)
    return {k: df[lab == k] for k in ("train", "val", "test")}
