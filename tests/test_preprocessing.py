"""Adapters for the Hugging Face sources (preprocessing/)."""
import pandas as pd

from preprocessing.prepare_market import adapt as adapt_market
from preprocessing.prepare_news import adapt as adapt_news

SYMS = ["RELIANCE", "TCS"]


def test_market_utc_to_ist_and_filters():
    raw = pd.DataFrame({"symbol": ["RELIANCE", "TCS", "TCS", "20MICRONS"],
                        "timestamp": pd.to_datetime(["2024-06-03 03:45", "2024-06-03 03:45", "2024-06-03 09:59",
                                                     "2024-06-03 03:45"], utc=True),
                        "open": [1.0] * 4, "high": [1.0] * 4, "low": [1.0] * 4, "close": [1.0] * 4,
                        "volume": [1] * 4, "oi": [0] * 4})
    df, log = adapt_market(raw.astype({"open": "float32"}), SYMS, "2024-06-01", "2024-06-30", "Asia/Kolkata")
    assert list(df.columns) == ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    assert set(df.symbol) == set(SYMS)
    r = df[df.symbol == "RELIANCE"].timestamp.iloc[0]
    assert (r.hour, r.minute) == (9, 15)                     # 03:45 UTC == 09:15 IST == session open
    last = df[df.symbol == "TCS"].timestamp.iloc[-1]
    assert (last.hour, last.minute) == (15, 29)              # 09:59 UTC == 15:29 IST == last session bar
    assert df["open"].dtype == "float64"


def _news(rows):
    cols = ["URL", "Article url", "title", "subtitle", "content", "article datetime", "article date", "article time"]
    return pd.DataFrame(rows, columns=cols)


def test_news_layouts_mojibake_dedup_sections():
    u = "https://www.moneycontrol.com/news/business/x-{}.html"
    raw = _news([
        ["mc", u.format(1001), "Infosys beats estimates", "", "", "2024-06-18 13:15", "2024-06-18", "13:15"],
        ["mc", u.format(1002), "TCS isnâ€™t worried", "", "", None, None, "2024-06-18 13:20"],   # date in time col
        ["mc", u.format(1003), "No date here", "", "", None, None, "11:36"],                    # unrecoverable
        ["mc", u.format(1001), "Infosys beats estimates", "", "", "2024-06-18 13:15", "2024-06-18", "13:15"],  # dup
        ["mc", "https://www.moneycontrol.com/entertainment/x-1004.html", "Film", "", "", "2024-06-18 14:00", "", ""],
    ])
    out, undated, log = adapt_news(raw, ["entertainment"])
    assert len(undated) == 1
    assert log["timestamp_layouts"] == {"datetime": 3, "date_in_time_col": 1, "no_date": 1}
    assert log["undated_set_aside"] == 1 and log["dropped_duplicates"] == 1 and log["dropped_excluded_sections"] == 1
    assert list(out.headline) == ["Infosys beats estimates", "TCS isn't worried"]   # mojibake repaired
    assert str(out.timestamp.iloc[1]) == "2024-06-18 13:20:00"


def test_undated_rows_set_aside_unchanged_never_inferred():
    u = "https://www.moneycontrol.com/news/business/x-{}.html"
    raw = _news([
        ["mc", u.format(12750100), "a", "", "", "2024-06-18 10:00", "", ""],
        ["mc", u.format(12750105), "b", "", "", None, None, "11:00"],      # neighbours share a date: still NOT inferred
        ["mc", u.format(12750110), "c", "", "", "2024-06-18 12:00", "", ""],
    ])
    out, undated, log = adapt_news(raw, [])
    assert list(out.headline) == ["a", "c"]
    assert log["undated_set_aside"] == 1 and list(undated["title"]) == ["b"]
    assert list(undated.columns) == list(raw.columns)                   # original schema, untouched
    assert undated["article time"].iloc[0] == "11:00"


def test_market_late_open_day_dropped():
    ts = pd.to_datetime(["2024-06-03 03:45", "2024-06-03 03:46",             # RELIANCE: opens 09:15
                         "2024-06-03 04:20", "2024-06-03 04:21"], utc=True)  # TCS: first trade 09:50
    raw = pd.DataFrame({"symbol": ["RELIANCE", "RELIANCE", "TCS", "TCS"], "timestamp": ts,
                        "open": [1.0] * 4, "high": [1.0] * 4, "low": [1.0] * 4, "close": [1.0] * 4,
                        "volume": [1] * 4, "oi": [0] * 4})
    df, log = adapt_market(raw, SYMS, "2024-06-01", "2024-06-30", "Asia/Kolkata")
    assert set(df.symbol) == {"RELIANCE"} and log["late_open_symbol_days_dropped"] == ["TCS 2024-06-03"]
