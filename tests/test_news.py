import numpy as np
import pandas as pd

from src.data.align import Calendar
from src.data.news_loader import _TEMPLATES
from src.features.market_features import build_table
from src.features.news_features import (NEWS_DIM, NewsProcessor, build_events, memory_from_events, news_matrix, raw_matrix)
from src.inference.feature_cache import NewsCache
from src.news.entities import EntityLinker
from src.news.event_memory import EventMemory, history_features
from src.news.event_rules import classify_event
from src.news.sentiment import HashEncoder


def test_event_rules_recover_template_types():
    bad = []
    for etype, _, _, _, tpl in _TEMPLATES:
        h = tpl.format(c="Infosys", q=3, x=12, y=25)
        if classify_event(h) != etype:
            bad.append((etype, classify_event(h), h))
    assert not bad, bad


def test_entities():
    L = EntityLinker(market_wide_relevance=0.3)
    syms = ["RELIANCE", "TCS", "HDFCBANK", "ITC"]
    assert L.link("Reliance Industries Q3 net profit jumps", "Earnings", syms) == {"RELIANCE": 1.0}
    assert L.link("RIL to acquire rival", "M&A", syms) == {"RELIANCE": 0.8}
    assert L.link("TCS and HDFC Bank rally", "Other", syms) == {"TCS": 1.0, "HDFCBANK": 1.0}
    assert L.link("RBI hikes repo rate by 25 bps", "RBI/Rates", syms) == {s: 0.3 for s in syms}   # market-wide
    assert L.link("Quarterly results due next week", "Earnings", syms) == {}                         # no entity
    assert L.link("ritc is not a ticker", "Other", syms) == {}                                       # word boundary


def test_memory_only_sees_outcomes_that_were_already_known():
    rng = np.random.default_rng(0)
    E = rng.standard_normal((5, 16)).astype(np.float32)
    mem = EventMemory(k=1, dim=16)
    for i in range(5):
        mem.add(E[i], fwd_ret=float(i), known_at=100.0 + 10 * i)
    q = E[4]                                              # identical to entry 4 (known at 140)
    assert mem.query(q, now=140.0)[2] < 0.999            # known_at must be STRICTLY before now
    assert mem.query(q, now=140.5)[2] > 0.99             # now visible
    assert mem.query(q, now=100.0) == (0.0, 0.0, 0.0)    # nothing known yet => zeros


def test_batch_history_equals_query_and_never_uses_future():
    rng = np.random.default_rng(1)
    n, d = 60, 16
    E = rng.standard_normal((n, d)).astype(np.float32)
    tmin = np.sort(rng.uniform(0, 1000, n))
    known = tmin + 16
    fwd = rng.standard_normal(n)
    H = history_features(E, tmin, fwd, known, k=4)
    mem = EventMemory(4, d)
    mem.add_many(E, fwd, known)
    for i in range(n):
        assert np.allclose(H[i], mem.query(E[i], tmin[i]))
    # corrupt the forward returns of every event that is not yet known at event 30 => event 30 unchanged
    fwd2 = fwd.copy()
    fwd2[known >= tmin[30]] = 99.0
    assert np.allclose(history_features(E, tmin, fwd2, known, 4)[30], H[30])


def test_events_pipeline_news_matrix_and_live_cache_agree(cfg, synth):
    bars, news, _ = synth
    table = build_table(bars, cfg)
    bars_sorted = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    cal = Calendar.from_bars(bars_sorted, cfg)
    enc = HashEncoder()
    ev, E = build_events(news, table, cal, cfg, enc)
    assert len(ev) > len(news) * 0.5 and E.shape == (len(ev), 768)
    assert ev["etype"].nunique() >= 5
    N, age = news_matrix(table, ev, cal, cfg)
    assert N.shape == (len(table), NEWS_DIM) and np.isfinite(N).all()
    assert (N[np.isnan(age)] == 0).all()                       # zeros mean "no news"

    # replay one symbol through the live cache and compare with the batch matrix
    sym = cfg["symbols"][1]
    rows = np.flatnonzero((table.symbol == sym).to_numpy())
    raw = raw_matrix(ev)
    cache = NewsCache([sym], cfg["news"]["decay_tau_min"], cfg["news"]["max_age_min"])
    e_sym = ev[ev.symbol == sym].sort_values("timestamp", kind="stable")
    ets = e_sym["timestamp"].tolist()
    j, out = 0, np.zeros(NEWS_DIM)
    for r in rows:
        t = table["timestamp"].iat[r]
        while j < len(e_sym) and ets[j] < t:                   # strictly before the bar
            cache.set(sym, raw[e_sym.index[j]], e_sym["tmin"].iat[j])
            j += 1
        cache.read(sym, table["tmin"].iat[r], out)
        assert np.allclose(out, N[r]), r


def test_memory_from_events_matches_event_table_history(cfg, synth):
    bars, news, _ = synth
    table = build_table(bars, cfg)
    cal = Calendar.from_bars(bars.sort_values(["symbol", "timestamp"]), cfg)
    ev, E = build_events(news, table, cal, cfg, HashEncoder())
    mem = memory_from_events(ev, E, cfg["news"]["memory_k"])
    for i in np.random.default_rng(0).choice(len(ev), 25, replace=False):
        got = mem.query(E[i], ev["tmin"].iat[i])
        assert np.allclose(got, ev[["h_mean", "h_std", "h_sim"]].iloc[i].to_numpy(), atol=1e-5)


def test_news_processor_updates_cache(cfg, synth):
    bars, news, _ = synth
    table = build_table(bars, cfg)
    cal = Calendar.from_bars(bars.sort_values(["symbol", "timestamp"]), cfg)
    enc = HashEncoder()
    ev, E = build_events(news, table, cal, cfg, enc)
    mem = memory_from_events(ev, E, 5)
    cache = NewsCache(cfg["symbols"], cfg["news"]["decay_tau_min"], cfg["news"]["max_age_min"])
    hit = NewsProcessor(enc, mem, cache, cfg).process("Infosys Q3 net profit jumps 12% on strong demand", 5000.0)
    assert list(hit["linked"]) == ["INFY"] and hit["etype"] == "Earnings"
    v = cache.read("INFY", 5003.0, np.zeros(NEWS_DIM))
    assert v[0] > 0 and v[10] == 3 and cache.read("TCS", 5003.0, np.zeros(NEWS_DIM)).sum() == 0
