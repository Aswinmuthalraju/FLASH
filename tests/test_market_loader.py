import numpy as np
import pandas as pd
import pytest

from src.data.market_loader import load_market, make_synthetic, regularize_bars, validate_bars
from src.data.rss_collector import parse_feed


def test_synthetic_is_reproducible_and_well_formed(cfg, synth):
    bars, _, _ = synth
    n = cfg["market"]["bars_per_day"]
    assert len(bars) == len(cfg["symbols"]) * cfg["synthetic"]["n_days"] * n
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    b2, _ = make_synthetic(cfg, None)
    b3, _ = make_synthetic(cfg, None)
    assert np.allclose(b2["close"], b3["close"])


def test_csv_roundtrip_and_regularisation(cfg, tmp_path):
    bars, _ = make_synthetic(cfg, None)
    one = bars[bars["symbol"] == cfg["symbols"][0]].copy()
    one["timestamp"] = one["timestamp"].dt.tz_localize(None)
    dropped = one.drop(one.index[100:110])                      # 10 missing minutes
    dropped = pd.concat([dropped, dropped.iloc[[5]]])            # duplicate row
    dropped = dropped.rename(columns={"timestamp": "Datetime", "close": "Close"}).drop(columns="symbol")
    f = tmp_path / "TCS.csv"
    dropped.to_csv(f, index=False)
    out = load_market(f, cfg)
    assert (out["symbol"] == "TCS").all()
    assert len(out) == one.shape[0]                              # gap re-filled, duplicate removed
    assert out["timestamp"].dt.tz is not None
    gap = out.iloc[100:110]
    assert (gap["volume"] == 0).all() and (gap["open"] == gap["close"]).all()


def test_missing_columns_rejected(cfg, tmp_path):
    f = tmp_path / "x.csv"
    pd.DataFrame({"timestamp": ["2024-01-01 09:15"], "close": [1.0]}).to_csv(f, index=False)
    with pytest.raises(ValueError):
        load_market(f, cfg)


def test_validate_drops_bad_prices(cfg):
    df = pd.DataFrame({"timestamp": ["2024-01-01 09:15", "2024-01-01 09:16"], "symbol": ["tcs.ns"] * 2,
                       "open": [1, -1.0], "high": [2, 2], "low": [1, 1], "close": [1.5, 1.5], "volume": [10, 10]})
    out = validate_bars(df)
    assert len(out) == 1 and out["symbol"].iat[0] == "TCS"


def test_rss_parser_and_xml_hardening():
    xml = b"<rss><channel><item><title>HDFC Bank profit up</title><pubDate>Tue, 02 Jan 2024 10:00:00 +0530</pubDate></item></channel></rss>"
    items = parse_feed(xml)
    assert items[0][0] == "HDFC Bank profit up" and items[0][1] is not None
    with pytest.raises(ValueError):
        parse_feed(b'<!DOCTYPE x [<!ENTITY a "b">]><rss></rss>')
