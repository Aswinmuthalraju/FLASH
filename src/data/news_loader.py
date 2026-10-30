"""News loader (CSV: timestamp, headline, source) and a synthetic news generator with known reactions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.align import Calendar
from src.data.company_universe import UNIVERSE

NEWS_COLS = ["timestamp", "headline", "source"]


def load_news(path: str | Path, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}
    missing = [c for c in NEWS_COLS if c not in lower]
    if missing:
        raise ValueError(f"news csv missing columns: {missing}")
    df = df.rename(columns={lower[c]: c for c in NEWS_COLS})[NEWS_COLS]
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df.dropna(subset=["timestamp"])
    df = df[df["headline"].str.len() > 0]
    df = df.drop_duplicates(["timestamp", "headline"])
    return df.sort_values("timestamp").reset_index(drop=True)


# (type, direction, magnitude, market_wide, template) ; {c}=company, {x}=number
_TEMPLATES = [
    ("Earnings", +1, 2.5, False, "{c} Q{q} net profit jumps {x}% on strong demand, beats estimates"),
    ("Earnings", +1, 2.5, False, "{c} posts record quarterly earnings, revenue up {x}% year on year"),
    ("Earnings", -1, 2.5, False, "{c} Q{q} profit falls {x}%, misses estimates on weak margins"),
    ("Earnings", -1, 2.5, False, "{c} reports disappointing results as revenue declines {x}%"),
    ("M&A", +1, 2.0, False, "{c} to acquire rival in ${x} million deal, analysts see upside"),
    ("M&A", -1, 2.0, False, "{c} abandons planned acquisition after regulatory pushback, shares slide"),
    ("Regulation", +1, 2.0, False, "SEBI approves relief for {c}, eases compliance norms"),
    ("Regulation", -1, 2.0, False, "SEBI issues show-cause notice to {c}; regulatory probe widens"),
    ("RBI/Rates", +1, 1.0, True, "RBI cuts repo rate by {y} bps, rate-sensitive stocks rally"),
    ("RBI/Rates", -1, 1.0, True, "RBI hikes repo rate by {y} bps and tightens liquidity, markets fall"),
    ("Geopolitical", +1, 1.5, True, "Ceasefire talks ease tensions, global markets rally"),
    ("Geopolitical", -1, 1.5, True, "Escalating conflict sparks global selloff, Sensex tumbles"),
    ("Legal/Fraud", -1, 3.0, False, "{c} faces fraud probe by ED, shares plunge"),
    ("Legal/Fraud", +1, 2.0, False, "Court clears {c} of fraud allegations, stock surges"),
    ("Management", +1, 1.5, False, "{c} appoints veteran as new CEO, investors cheer"),
    ("Management", -1, 1.5, False, "{c} CFO resigns abruptly, governance concerns mount"),
    ("Other", 0, 0.0, False, "{c} announces dividend and schedules analyst meet on Friday"),
    ("Other", 0, 0.0, False, "{c} to hold annual general meeting next month"),
]
_SOURCES = ["Moneycontrol", "ET Markets", "Mint"]
_FLIP_PROB = 0.2   # reaction opposes the headline


def make_synthetic_news(cfg: dict, cal: Calendar, seed: int | None = None):
    """Returns (news_df, events) where events = per-stock reaction specs for make_synthetic()."""
    rng = np.random.default_rng((cfg["seed"] if seed is None else seed) + 101)
    syn = cfg["synthetic"]
    symbols = cfg["symbols"]
    firm = [t for t in _TEMPLATES if not t[3]]
    mkt = [t for t in _TEMPLATES if t[3]]
    rows, events = [], []
    tz = cfg["market"]["tz"]
    for day in cal.days:
        n_firm = rng.poisson(syn["news_per_stock_day"] * len(symbols))
        n_mkt = rng.poisson(syn["market_news_per_day"])
        specs = [(False, rng.choice(len(firm))) for _ in range(n_firm)] + [(True, rng.choice(len(mkt))) for _ in range(n_mkt)]
        for is_mkt, ti in specs:
            etype, direction, mag, _, tpl = (mkt if is_mkt else firm)[ti]
            sym = symbols[rng.integers(len(symbols))]
            comp = UNIVERSE[sym][0]
            head = tpl.format(c=comp, q=rng.integers(1, 5), x=rng.integers(3, 40), y=rng.choice([25, 35, 50]))
            u = rng.random()
            if u < 0.88:                                   # in session
                sec = rng.uniform(0, cal.n * 60)
                ts = day + pd.Timedelta(seconds=cal.open_min * 60 + sec)
            elif u < 0.95:                                 # evening
                ts = day + pd.Timedelta(seconds=rng.uniform(16 * 3600, 23 * 3600))
            else:                                          # pre-open
                ts = day + pd.Timedelta(seconds=rng.uniform(6 * 3600, cal.open_min * 60 - 60))
            rows.append({"timestamp": ts, "headline": head, "source": _SOURCES[rng.integers(3)]})
            d = direction * (-1 if rng.random() < _FLIP_PROB else 1)
            targets = symbols if is_mkt else [sym]
            for s in targets:
                events.append({"symbol": s, "ts": ts, "direction": d, "magnitude": mag * rng.uniform(0.7, 1.3),
                               "market_wide": is_mkt, "etype": etype})
    news = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    news["timestamp"] = pd.to_datetime(news["timestamp"]).dt.tz_convert(tz)
    ev = pd.DataFrame(events)
    if len(ev):
        ev["tmin"] = cal.news_tmin(ev["ts"])
    return news, ev.to_dict("records")
