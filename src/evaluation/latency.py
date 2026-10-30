"""Latency benchmark: single-bar predictions through the real hot loop, timed with time.perf_counter_ns."""
from __future__ import annotations

import gc
from time import perf_counter_ns

import numpy as np
import pandas as pd

from src.inference.predictor import STAGES, Predictor


def _stats(ns: np.ndarray) -> dict:
    us = ns / 1000.0
    return {"mean": us.mean(), "median": np.median(us), "p95": np.percentile(us, 95),
            "p99": np.percentile(us, 99), "max": us.max()}


def benchmark(pred: Predictor, bars_arrays: dict, n_measure: int = 10_000, n_warmup: int = 2_000,
              news_raw: np.ndarray | None = None) -> dict:
    """bars_arrays: dict of equal-length numpy arrays (si, high, low, close, volume, mret, tmin, newday),
    in replay order. Warm-up bars are run first (and discarded); then `n_measure` bars past the daily
    feature warm-up are timed. Wrap-around reuses the stream (state keeps evolving, which is fine for timing).
    """
    a = bars_arrays
    n = len(a["si"])
    pred.new_day()
    totals, stages = [], []
    i = k = 0
    gc.collect()
    gc_was = gc.isenabled()
    gc.disable()                      # avoid GC pauses polluting the tail; re-enabled below
    try:
        while len(totals) < n_measure:
            if a["newday"][i % n]:
                pred.new_day()
            if news_raw is not None and pred.mode == "D":      # keep a live news vector in the cache (not timed)
                pred.cache.set(pred.symbols[a["si"][i % n]], news_raw, a["tmin"][i % n] - 5.0)
            t0 = perf_counter_ns()
            p = pred.on_bar(a["si"][i % n], a["high"][i % n], a["low"][i % n], a["close"][i % n],
                            a["volume"][i % n], a["mret"][i % n], a["tmin"][i % n])
            t1 = perf_counter_ns()
            i += 1
            if p is None:
                continue
            k += 1
            if k <= n_warmup:
                continue
            totals.append(t1 - t0)
            stages.append(p.stage_ns)
    finally:
        if gc_was:
            gc.enable()
    totals = np.array(totals)
    stages = np.array(stages)
    res = {"n": len(totals), "mode": pred.mode, "total_incl_call": _stats(totals)}
    for j, s in enumerate(STAGES):
        res[s] = _stats(stages[:, j])
    res["total_stages"] = _stats(stages.sum(axis=1))
    return res


def latency_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        for stage in STAGES + ["total_stages", "total_incl_call"]:
            if r["mode"] == "A" and stage in ("pca", "hmm", "news"):
                continue
            rows.append({"model": name, "stage": stage, **{k: round(v, 2) for k, v in r[stage].items()}})
    return pd.DataFrame(rows)


def make_stream(bars: pd.DataFrame, cfg: dict, days) -> dict:
    """Arrays in replay order (minute-major, symbol-minor) for the given trading days."""
    from src.data.align import Calendar, to_ns
    from src.features.market_features import market_return

    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    mret = market_return(bars)
    tmin = Calendar.from_bars(bars, cfg).bar_tmin(bars["timestamp"])
    sym_idx = bars["symbol"].map({s: i for i, s in enumerate(cfg["symbols"])}).to_numpy()
    sel = np.flatnonzero(bars["timestamp"].dt.normalize().isin(days).to_numpy())
    order = sel[np.lexsort((sym_idx[sel], to_ns(bars["timestamp"])[sel]))]
    day = to_ns(bars["timestamp"].dt.normalize())[order]
    newday = np.r_[True, day[1:] != day[:-1]]
    g = lambda c: bars[c].to_numpy()[order]  # noqa: E731
    return {"si": sym_idx[order], "high": g("high"), "low": g("low"), "close": g("close"), "volume": g("volume"),
            "mret": mret[order], "tmin": tmin[order], "newday": newday}
