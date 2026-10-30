"""News may only be attached to bars STRICTLY after the news timestamp."""
import numpy as np
import pandas as pd

from src.data.align import Calendar, align_news_to_bars, check_no_lookahead, to_ns


def _expand(news, symbols):
    rows = [dict(symbol=s, timestamp=t, headline=h) for t, h in zip(news["timestamp"], news["headline"]) for s in symbols]
    return pd.DataFrame(rows)


def test_no_lookahead_on_synthetic(cfg, synth):
    bars, news, _ = synth
    cal = Calendar.from_bars(bars, cfg)
    ev = _expand(news, cfg["symbols"])
    idx, age = align_news_to_bars(bars, ev, cal, max_age_min=240)
    assert (idx >= 0).sum() > 0, "alignment attached nothing"
    assert check_no_lookahead(bars, ev, idx) == 0
    # and for every attached bar, the event really is the LATEST earlier one for that symbol
    ev_ns, bar_ns = to_ns(ev["timestamp"]), to_ns(bars["timestamp"])
    rng = np.random.default_rng(0)
    for r in rng.choice(np.flatnonzero(idx >= 0), 300, replace=False):
        sym = bars["symbol"].iat[r]
        earlier = (ev["symbol"].to_numpy() == sym) & (ev_ns < bar_ns[r])
        assert ev_ns[idx[r]] == ev_ns[earlier].max()
        assert age[r] >= 0


def test_news_on_exact_bar_timestamp_is_not_usable_until_next_bar(cfg):
    day = pd.Timestamp("2024-01-02", tz=cfg["market"]["tz"])
    ts = day + pd.to_timedelta(np.arange(375) * 60 + 9 * 3600 + 15 * 60, unit="s")
    bars = pd.DataFrame({"timestamp": ts, "symbol": "TCS"})
    cal = Calendar(pd.DatetimeIndex([day]), cfg)
    t_news = ts[10]                                   # exactly on bar 10's timestamp
    ev = pd.DataFrame({"symbol": ["TCS"], "timestamp": [t_news]})
    idx, _ = align_news_to_bars(bars, ev, cal, 1e9)
    assert (idx[:11] == -1).all() and (idx[11:] == 0).all()


def test_overnight_news_attaches_to_next_open_and_ages_from_there(cfg):
    tz = cfg["market"]["tz"]
    days = pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz=tz), pd.Timestamp("2024-01-03", tz=tz)])
    ts = pd.DatetimeIndex([d + pd.Timedelta(seconds=m * 60 + 9 * 3600 + 15 * 60) for d in days for m in range(375)])
    bars = pd.DataFrame({"timestamp": ts, "symbol": "TCS"})
    cal = Calendar(days, cfg)
    ev = pd.DataFrame({"symbol": ["TCS"], "timestamp": [days[0] + pd.Timedelta(hours=20)]})
    idx, age = align_news_to_bars(bars, ev, cal, 240)
    assert (idx[:375] == -1).all()                    # nothing during the session before the news
    assert idx[375] == 0 and 0 <= age[375] < 1e-3     # first bar next day, ~0 minutes old


def test_checker_catches_a_leaky_aligner(cfg, synth):
    """The test must be able to FAIL: an aligner using <= (side='right') is detected."""
    bars, news, _ = synth
    ev = _expand(news, cfg["symbols"])
    ev_ns, bar_ns = to_ns(ev["timestamp"]), to_ns(bars["timestamp"])
    leaky = np.full(len(bars), -1)
    for sym, rows in bars.groupby("symbol").indices.items():
        er = np.flatnonzero((ev["symbol"] == sym).to_numpy())
        er = er[np.argsort(ev_ns[er], kind="stable")]
        # LEAK: also allows events at the same timestamp / uses bar CLOSE time (+60 s)
        pos = np.searchsorted(ev_ns[er], bar_ns[rows] + 60 * 10**9, side="right") - 1
        leaky[rows] = np.where(pos >= 0, er[np.maximum(pos, 0)], -1)
    assert check_no_lookahead(bars, ev, leaky) > 0
