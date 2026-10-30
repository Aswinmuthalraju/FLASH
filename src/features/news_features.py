"""News feature vectors.

raw   (13) = [sentiment, onehot(8 event types), relevance, hist_mean, hist_std, hist_similarity]
model (14) = [sentiment, onehot(8), relevance, minutes_since_news, hist_mean, hist_std, hist_similarity]
              every raw component is multiplied by exp(-age / tau); older than max_age => all zeros.
Zeros mean "no news". `decay_table` is shared by the batch and live (cache) code so they agree exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.align import Calendar, align_news_to_bars
from src.news.entities import EntityLinker
from src.news.event_memory import EventMemory, history_features
from src.news.event_rules import EVENT_IDX, EVENT_TYPES, classify_event
from src.news.sentiment import encode_cached

RAW_DIM = 13
NEWS_FEATURES = (["n_sent"] + [f"n_ev_{e}" for e in EVENT_TYPES] + ["n_rel", "n_minutes"]
                 + ["h_mean", "h_std", "h_sim"])
NEWS_DIM = len(NEWS_FEATURES)
HIST_IDX = [NEWS_FEATURES.index(f) for f in ("h_mean", "h_std", "h_sim")]
NEWS_ONLY_IDX = [i for i in range(NEWS_DIM) if i not in HIST_IDX]


def decay_table(tau: float, max_age: int) -> np.ndarray:
    return np.exp(-np.arange(max_age + 2) / tau)


def raw_vector(sentiment: float, etype: str, relevance: float, hist: tuple[float, float, float]) -> np.ndarray:
    v = np.zeros(RAW_DIM)
    v[0] = sentiment
    v[1 + EVENT_IDX[etype]] = 1.0
    v[9] = relevance
    v[10:13] = hist
    return v


def decayed(raw: np.ndarray, age_min: float, table: np.ndarray, max_age: float, out: np.ndarray | None = None) -> np.ndarray:
    """Single-vector version (live)."""
    out = np.zeros(NEWS_DIM) if out is None else out
    if age_min > max_age or age_min < 0:
        out[:] = 0.0
        return out
    k = int(age_min)
    d = table[k]
    out[0:10] = raw[0:10] * d
    out[10] = k
    out[11:14] = raw[10:13] * d
    return out


def decayed_matrix(raw: np.ndarray, age: np.ndarray, has: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Batch version: raw (n,13), age (n,), has (n,) bool -> (n,14); identical arithmetic to `decayed`."""
    k = np.floor(age).astype(int)
    d = table[np.minimum(k, len(table) - 1)]
    out = np.zeros((len(raw), NEWS_DIM))
    out[:, 0:10] = raw[:, 0:10] * d[:, None]
    out[:, 10] = k
    out[:, 11:14] = raw[:, 10:13] * d[:, None]
    out[~has] = 0.0
    return out


# ------------------------------------------------------------------ events (slow path, offline build)

def build_events(news: pd.DataFrame, table: pd.DataFrame, cal: Calendar, cfg: dict, encoder, cache_path=None):
    """One row per (headline, linked symbol). Returns (events_df, embeddings (n, 768)).

    Forward return = log(close[b0+H] / close[b0]) / (sigma*sqrt(H)), b0 = first bar strictly after the news,
    H = news.impact_horizon. Event i may use event j as a neighbour only if j's outcome was already
    observable when i arrived (known_at_j < tmin_i).
    """
    nc = cfg["news"]
    H = nc["impact_horizon"]
    n_bars = cfg["market"]["bars_per_day"]
    linker = EntityLinker(market_wide_relevance=nc["market_wide_relevance"])
    heads = news["headline"].tolist()
    sent, emb = encode_cached(encoder, heads, cache_path)
    rows, emb_idx = [], []
    for i, (ts, h) in enumerate(zip(news["timestamp"], heads)):
        et = classify_event(h)
        for sym, rel in linker.link(h, et, cfg["symbols"]).items():
            rows.append({"symbol": sym, "timestamp": ts, "headline": h, "etype": et, "sentiment": float(sent[i]),
                         "relevance": rel})
            emb_idx.append(i)
    ev = pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["symbol", "timestamp", "headline", "etype", "sentiment", "relevance"])
    order = pd.DataFrame({"i": emb_idx, "ts": [r["timestamp"] for r in rows]}).sort_values("ts", kind="stable")
    E = emb[order["i"].to_numpy()] if rows else np.zeros((0, emb.shape[1]))
    ev["tmin"] = cal.news_tmin(ev["timestamp"]) if len(ev) else []

    # forward returns on the bars table
    fwd = np.full(len(ev), np.nan)
    known = np.full(len(ev), np.nan)
    sig_fallback = table[table.split == "train"].groupby("symbol")["vol30"].median()
    for sym, g in table.groupby("symbol"):
        t_bar, c_bar, s_bar = g["tmin"].to_numpy(), g["close"].to_numpy(), g["vol30"].to_numpy()
        for r in np.flatnonzero((ev["symbol"] == sym).to_numpy()):
            T0 = np.floor(ev["tmin"].iat[r]) + 1
            a = np.searchsorted(t_bar, T0)
            if a >= len(t_bar) or t_bar[a] != T0 or (T0 + H) // n_bars != T0 // n_bars:
                continue
            b = np.searchsorted(t_bar, T0 + H)
            if b >= len(t_bar) or t_bar[b] != T0 + H:
                continue
            sig = s_bar[a] if np.isfinite(s_bar[a]) else sig_fallback[sym]
            fwd[r] = np.log(c_bar[b] / c_bar[a]) / (sig * np.sqrt(H))
            known[r] = T0 + H + 1                      # the bar closes one minute after its open label
    ev["fwd"], ev["known_at"] = fwd, known
    hist = history_features(E, ev["tmin"].to_numpy(), fwd, known, nc["memory_k"]) if len(ev) else np.zeros((0, 3))
    ev["h_mean"], ev["h_std"], ev["h_sim"] = hist[:, 0], hist[:, 1], hist[:, 2]
    return ev, E


def raw_matrix(ev: pd.DataFrame) -> np.ndarray:
    R = np.zeros((len(ev), RAW_DIM))
    R[:, 0] = ev["sentiment"].to_numpy()
    R[np.arange(len(ev)), 1 + ev["etype"].map(EVENT_IDX).to_numpy()] = 1.0
    R[:, 9] = ev["relevance"].to_numpy()
    R[:, 10] = ev["h_mean"].to_numpy()
    R[:, 11] = ev["h_std"].to_numpy()
    R[:, 12] = ev["h_sim"].to_numpy()
    return R


def news_matrix(table: pd.DataFrame, ev: pd.DataFrame, cal: Calendar, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(n_bars_rows, 14) decayed news vectors aligned to the (symbol, timestamp)-sorted table, and the age in minutes
    (np.nan when no news). Uses align_news_to_bars => strictly-after rule."""
    nc = cfg["news"]
    idx, age = align_news_to_bars(table, ev, cal, nc["max_age_min"])
    has = idx >= 0
    raw = raw_matrix(ev)[np.where(has, idx, 0)] if len(ev) else np.zeros((len(table), RAW_DIM))
    tbl = decay_table(nc["decay_tau_min"], nc["max_age_min"])
    return decayed_matrix(raw, age, has, tbl), np.where(has, age, np.nan)


def memory_from_events(ev: pd.DataFrame, E: np.ndarray, k: int) -> EventMemory:
    mem = EventMemory(k, E.shape[1] if E.size else 768)
    ok = np.isfinite(ev["known_at"].to_numpy())
    if ok.any():
        mem.add_many(E[ok], ev["fwd"].to_numpy()[ok], ev["known_at"].to_numpy()[ok])
    return mem


class NewsProcessor:
    """SLOW PATH (run in a worker / between bars, never inside the per-bar loop): headline -> cache update."""

    def __init__(self, encoder, memory: EventMemory, cache, cfg: dict):
        self.enc, self.mem, self.cache, self.cfg = encoder, memory, cache, cfg
        self.linker = EntityLinker(market_wide_relevance=cfg["news"]["market_wide_relevance"])

    def process(self, headline: str, event_tmin: float) -> dict:
        """Returns {"etype", "sentiment", "hist": (mean, std, sim), "linked": {symbol: relevance}}."""
        s, e = self.enc.encode([headline])
        et = classify_event(headline)
        h = self.mem.query(e[0], event_tmin)
        linked = self.linker.link(headline, et, self.cfg["symbols"])
        for sym, rel in linked.items():
            self.cache.set(sym, raw_vector(float(s[0]), et, rel, h), event_tmin)
        return {"etype": et, "sentiment": float(s[0]), "hist": h, "linked": linked}
