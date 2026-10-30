"""Replay one trading day of one stock through the real pipeline (slow path + hot loop), recording everything
the demo needs: per-bar probabilities, regime, latency, news markers, and the fusion vector for explanations.

The slow path (FinBERT / event memory) runs inline here only because this is a replay; in production it would run
in a worker and only write the news cache.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.align import Calendar
from src.features.market_features import market_return
from src.features.news_features import NewsProcessor
from src.inference.predictor import STAGES, Predictor


@dataclass
class DaySession:
    symbol: str
    timeline: pd.DataFrame          # one row per predicted bar
    news: pd.DataFrame              # headlines processed during the day window
    X: np.ndarray                   # fusion vectors per timeline row
    pred: Predictor
    horizons: list[int]


class ReplayContext:
    """Per-dataset precomputation (causal): sorted bars, market return, trading-minute clock."""

    def __init__(self, bars: pd.DataFrame, news: pd.DataFrame, cfg: dict):
        self.cfg = cfg
        self.bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        self.mret = market_return(self.bars)
        self.cal = Calendar.from_bars(self.bars, cfg)
        self.tmin = self.cal.bar_tmin(self.bars["timestamp"])
        self.news = news.sort_values("timestamp").reset_index(drop=True)
        self.news_tmin = self.cal.news_tmin(self.news["timestamp"])

    def days(self) -> list[pd.Timestamp]:
        return list(self.cal.days)


def run_day(ctx: ReplayContext, bundle: dict, memory, encoder, symbol: str, day: pd.Timestamp) -> DaySession:
    cfg = ctx.cfg
    pred = Predictor(bundle, "D")
    cache_proc = NewsProcessor(encoder, memory, pred.cache, cfg)
    si = pred.sym_idx[symbol]
    b = ctx.bars
    rows = np.flatnonzero(((b["symbol"] == symbol) & (b["timestamp"].dt.normalize() == day)).to_numpy())
    ts = b["timestamp"]
    hi, lo, cl, vo = (b[k].to_numpy() for k in ("high", "low", "close", "volume"))
    first_ts = ts.iat[rows[0]]
    window_start = first_ts - pd.Timedelta(hours=40)              # carry-over from the previous session
    j = int(ctx.news["timestamp"].searchsorted(window_start))
    nts = ctx.news["timestamp"]
    pred.new_day()
    recs, X, news_rows = [], [], []
    for r in rows:
        while j < len(ctx.news) and nts.iat[j] < ts.iat[r]:       # strictly before the bar
            out = cache_proc.process(ctx.news["headline"].iat[j], float(ctx.news_tmin[j]))
            news_rows.append({"timestamp": nts.iat[j], "headline": ctx.news["headline"].iat[j], "etype": out["etype"],
                              "sentiment": out["sentiment"], "relevance": out["linked"].get(symbol, 0.0),
                              "hist_mean": out["hist"][0], "hist_sim": out["hist"][2], "tmin": float(ctx.news_tmin[j])})
            j += 1
        p = pred.on_bar(si, hi[r], lo[r], cl[r], vo[r], ctx.mret[r], ctx.tmin[r])
        if p is None:
            continue
        rec = {"timestamp": ts.iat[r], "minute": int(ctx.tmin[r] % cfg["market"]["bars_per_day"]), "close": cl[r],
               "regime": p.regime, "latency_us": p.latency_us}
        for h, pr, lab in zip(p.horizons, p.probs, p.labels):
            rec.update({f"P_DOWN_{h}": pr[0], f"P_NEUTRAL_{h}": pr[1], f"P_UP_{h}": pr[2], f"signal_{h}": lab})
        for name, v in zip(STAGES, p.stage_ns):
            rec[f"us_{name}"] = v / 1000.0
        for name, v in zip(pred.regime_names, p.regime_post):
            rec[f"regime_{name}"] = v
        recs.append(rec)
        X.append(pred.xf.copy())
    news_df = pd.DataFrame(news_rows)
    if len(news_df):
        news_df = news_df[news_df["relevance"] > 0].reset_index(drop=True)
    return DaySession(symbol, pd.DataFrame(recs), news_df, np.array(X), pred, pred.horizons)


def around_news(sess: DaySession, news_ts: pd.Timestamp, horizon: int, offsets=(-2, 0, 1, 2, 5, 10, 15)) -> pd.DataFrame:
    """Prediction snapshot at fixed offsets (minutes) relative to the first bar strictly after the news."""
    tl = sess.timeline
    after = np.flatnonzero((tl["timestamp"] > news_ts).to_numpy())
    if len(after) == 0:
        return pd.DataFrame()
    k0 = after[0]
    rows = []
    for o in offsets:
        k = k0 + o
        if 0 <= k < len(tl):
            r = tl.iloc[k]
            rows.append({"offset_min": o, "time": r["timestamp"].strftime("%H:%M"), "P_UP": r[f"P_UP_{horizon}"],
                         "P_NEUTRAL": r[f"P_NEUTRAL_{horizon}"], "P_DOWN": r[f"P_DOWN_{horizon}"],
                         "signal": r[f"signal_{horizon}"], "regime": r["regime"], "latency_us": r["latency_us"], "_k": k})
    return pd.DataFrame(rows)
