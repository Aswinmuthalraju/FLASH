"""Phase-4 dataset audit. Uses FLASH's OWN loaders and pipeline, so "it loads here" means "it loads in training".

    python preprocessing/verify_dataset.py --market data/raw/market --news data/raw/news.csv [--stub-encoder]

Writes reports/data_prep/quality.json and reports/data_prep/QUALITY.md. Exit code 1 on any CRITICAL issue.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402
from src.data.market_loader import _normalise_columns, validate_bars, load_market  # noqa: E402
from src.data.news_loader import load_news  # noqa: E402
from src.news.entities import EntityLinker  # noqa: E402
from src.news.event_rules import classify_event, MARKET_WIDE  # noqa: E402

MOJIBAKE = ("â€", "Ã©", "Â ", "Ĺ", "Ă˘")


def raw_market_stats(path: Path, cfg: dict) -> dict:
    """Stats BEFORE FLASH's regularisation (what the forward-fill will paper over)."""
    files = sorted(path.glob("*.parquet")) + sorted(path.glob("*.csv"))
    frames = [_normalise_columns(pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f), f.stem.upper())
              for f in files]
    raw = pd.concat(frames, ignore_index=True)
    out = {"rows": int(len(raw)), "duplicate_symbol_ts": int(raw.duplicated(["symbol", "timestamp"]).sum()),
           "missing_values": raw.isna().sum().astype(int).to_dict(),
           "ohlc_violations": int(((raw["high"] < raw[["open", "close"]].max(axis=1)) |
                                   (raw["low"] > raw[["open", "close"]].min(axis=1))).sum()),
           "nonpositive_prices": int((raw[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())}
    v = validate_bars(raw, cfg["market"]["tz"])
    m = cfg["market"]
    oh, om = map(int, m["session_open"].split(":"))
    minute = v["timestamp"].dt.hour * 60 + v["timestamp"].dt.minute - (oh * 60 + om)
    v = v.assign(day=v["timestamp"].dt.date, in_sess=(minute >= 0) & (minute < m["bars_per_day"]))
    out["rows_outside_session"] = int((~v["in_sess"]).sum())
    cov = v[v["in_sess"]].groupby(["symbol", "day"]).size() / m["bars_per_day"]
    out["day_coverage"] = {"mean": float(cov.mean()), "min": float(cov.min()),
                           "days_below_min_coverage": int((cov < m["min_day_coverage"]).sum())}
    out["per_symbol_days"] = cov.groupby(level=0).size().astype(int).to_dict()
    ins = v[v["in_sess"]]
    g = ins.groupby("symbol")
    out["per_symbol"] = {s: {"records_in_session": int(len(d)),
                             "trading_days": int(d["day"].nunique()),
                             "first_day": str(d["day"].min()), "last_day": str(d["day"].max()),
                             "missing_minutes_frac": float(1 - len(d) / (d["day"].nunique() * m["bars_per_day"])),
                             "zero_volume_frac": float((d["volume"] == 0).mean())}
                         for s, d in g}
    raw_ts = pd.to_datetime(raw["timestamp"])
    out["timestamp_tz_as_loaded"] = str(raw_ts.dt.tz) if raw_ts.dt.tz is not None else "naive"
    out["chronological_within_symbol"] = bool(all(d["timestamp"].is_monotonic_increasing for _, d in v.groupby("symbol")))
    return out


def bar_stats(bars: pd.DataFrame) -> dict:
    r = np.log(bars.groupby("symbol")["close"].shift(-1) / bars["close"])
    z = (r - r.mean()) / r.std()
    flat = (bars["volume"] == 0)
    return {"rows": int(len(bars)), "symbols": int(bars.symbol.nunique()),
            "trading_days": int(bars["timestamp"].dt.date.nunique()),
            "first": str(bars.timestamp.min()), "last": str(bars.timestamp.max()),
            "zero_volume_bars_frac": float(flat.mean()),
            "return_outliers_gt10sd": int((z.abs() > 10).sum()),
            "max_abs_1min_return": float(r.abs().max())}


def news_stats(news: pd.DataFrame, bars: pd.DataFrame, cfg: dict) -> dict:
    lo, hi = bars.timestamp.min(), bars.timestamp.max()
    inwin = news[(news.timestamp >= lo - pd.Timedelta(hours=18)) & (news.timestamp <= hi)]
    linker = EntityLinker(market_wide_relevance=cfg["news"]["market_wide_relevance"])
    et = inwin["headline"].map(classify_event)
    linked = [linker.link(h, e, cfg["symbols"]) for h, e in zip(inwin["headline"], et)]
    named = np.array([bool(d) and e not in MARKET_WIDE for d, e in zip(linked, et)])
    mw = np.array([bool(d) and e in MARKET_WIDE and all(v == linker.mw for v in d.values()) for d, e in zip(linked, et)])
    per_sym = pd.Series([s for d in linked for s, v in d.items() if v > linker.mw]).value_counts().to_dict()
    days_mkt = pd.Index(sorted(bars["timestamp"].dt.date.unique()))
    used = inwin[[bool(d) for d in linked]]
    per_day_all = inwin["timestamp"].dt.date.value_counts()
    per_day_used = used["timestamp"].dt.date.value_counts()
    coverage = {"market_trading_days": int(len(days_mkt)),
                "trading_days_with_any_article": int(sum(d in per_day_all.index for d in days_mkt)),
                "trading_days_with_flash_usable_article": int(sum(d in per_day_used.index for d in days_mkt)),
                "articles_per_trading_day_median": float(per_day_all.reindex(days_mkt, fill_value=0).median()),
                "longest_run_of_trading_days_without_articles": _longest_zero_run(per_day_all.reindex(days_mkt, fill_value=0))}
    head = news["headline"].astype(str)
    enc = {"non_ascii_rows": int(head.map(lambda x: any(ord(c) > 127 for c in x)).sum()),
           "low_latin_letter_ratio_rows(<0.5)": int(head.map(lambda x: sum(c.isascii() and c.isalpha() for c in x) /
                                                         max(1, sum(c.isalpha() for c in x)) < 0.5).sum()),
           "empty_headlines": int((head.str.strip() == "").sum())}
    cats = news["section"].value_counts().to_dict() if "section" in news else "n/a (column not loaded)"
    return {"rows": int(len(news)), "rows_in_market_window": int(len(inwin)),
            "date_range": [str(news.timestamp.min()), str(news.timestamp.max())],
            "coverage": coverage, "encoding_language": enc, "categories": cats,
            "used_by_flash_frac": float(np.mean([bool(d) for d in linked])) if len(inwin) else 0.0,
            "stock_specific": int(named.sum()), "market_wide_broadcast": int(mw.sum()),
            "stock_specific_per_symbol": per_sym,
            "event_types_of_used": et[[bool(d) for d in linked]].value_counts().to_dict(),
            "residual_mojibake_rows": int(news["headline"].str.contains("|".join(MOJIBAKE), regex=True).sum()),
            "duplicate_ts_headline": int(news.duplicated(["timestamp", "headline"]).sum()),
            "in_session_frac": float(((inwin.timestamp.dt.hour * 60 + inwin.timestamp.dt.minute).between(555, 929)).mean())
            if len(inwin) else 0.0}


def _longest_zero_run(counts: pd.Series) -> int:
    best = cur = 0
    for c in counts.to_numpy():
        cur = cur + 1 if c == 0 else 0
        best = max(best, cur)
    return int(best)


def pipeline_check(cfg, bars, news, stub: bool) -> dict:
    """The data-loading test the brief asks for: FLASH's real prepare() + one tiny fit."""
    from src.evaluation.experiments import make_encoder, prepare
    enc = make_encoder(cfg, finbert=not stub)
    from src.config import resolve
    # cache lives in the CONFIG's processed dir (resolve() creates it); stub and FinBERT caches never mix
    cache = resolve(cfg, "processed") / ("hash_cache.npz" if stub else "finbert_cache.npz")
    ds = prepare(cfg, bars, news, enc, cache)
    t = ds.table
    out = {"feature_matrix_shape": list(ds.XF.shape), "n_features": len(ds.col_names),
           "nan_in_valid_rows": int(np.isnan(ds.XF[t["valid_x"].to_numpy()]).sum()),
           "inf_in_valid_rows": int(np.isinf(ds.XF[t["valid_x"].to_numpy()]).sum()),
           "labelled_rows_per_horizon": {h: int((t[f"y{h}"] >= 0).sum()) for h in cfg["labels"]["horizons"]},
           "per_symbol_valid_labelled_h15": t[t["valid_x"] & (t["y15"] >= 0)].groupby("symbol").size().astype(int).to_dict(),
           "pca_components": int(ds.regime.pca.n_components_), "hmm_states": ds.regime.names, "splits": {}}
    days = t.groupby("split")["day_idx"].agg(["min", "max", "nunique"])
    out["split_days"] = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in days.T.to_dict().items()}
    order = [days.loc[s, "max"] for s in ("train", "val", "test")] + [None]
    out["splits_chronological_and_disjoint"] = bool(days.loc["train", "max"] < days.loc["val", "min"]
                                                    and days.loc["val", "max"] < days.loc["test", "min"])
    for h in cfg["labels"]["horizons"]:
        for s in ("train", "val", "test"):
            y = ds.y(ds.rows(s, h), h)
            out["splits"][f"h{h}_{s}"] = {"n": int(len(y)), "DOWN": float((y == 0).mean()) if len(y) else 0,
                                          "NEUTRAL": float((y == 1).mean()) if len(y) else 0,
                                          "UP": float((y == 2).mean()) if len(y) else 0}
    out["test_rows_with_news_<=30min"] = int(ds.shock(ds.rows("test", cfg["labels"]["primary_horizon"])).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default=str(ROOT / "data/raw/market"))
    ap.add_argument("--news", default=str(ROOT / "data/raw/news.csv"))
    ap.add_argument("--stub-encoder", action="store_true", help="hash encoder: tests plumbing only, NOT for training")
    ap.add_argument("--config")
    ap.add_argument("--out-dir", default=str(ROOT / "reports/data_prep"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    rep = {"raw_market": raw_market_stats(Path(a.market), cfg)}
    bars = load_market(a.market, cfg)
    news = load_news(a.news, cfg["market"]["tz"])
    rep["bars_after_flash_loader"] = bar_stats(bars)
    rep["news"] = news_stats(news, bars, cfg)
    raw_news = pd.read_csv(a.news)
    if "section" in raw_news:                       # FLASH's loader keeps only timestamp/headline/source
        rep["news"]["categories"] = raw_news["section"].value_counts().astype(int).to_dict()
    rep["pipeline"] = pipeline_check(cfg, bars, news, a.stub_encoder)

    crit = []
    if rep["bars_after_flash_loader"]["symbols"] < len(cfg["symbols"]):
        crit.append("symbols missing after loading")
    if not rep["pipeline"]["splits_chronological_and_disjoint"]:
        crit.append("train/val/test overlap")
    if rep["pipeline"]["nan_in_valid_rows"] or rep["pipeline"]["inf_in_valid_rows"]:
        crit.append("NaN/inf in model-ready feature rows")
    if any(v == 0 for v in rep["pipeline"]["labelled_rows_per_horizon"].values()):
        crit.append("a configured horizon has no labelled rows")
    if rep["pipeline"]["test_rows_with_news_<=30min"] == 0:
        crit.append("no news-shock rows in test: experiment D vs B is untestable")
    if rep["news"]["stock_specific"] < 50:
        crit.append("fewer than 50 stock-specific headlines in the market window")
    rep["critical_issues"] = crit
    od = Path(a.out_dir)
    rep["data_paths"] = {"market": a.market, "news": a.news, "stub_encoder": a.stub_encoder}
    od.mkdir(parents=True, exist_ok=True)
    (od / "quality.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    sys.exit(1 if crit else 0)


if __name__ == "__main__":
    main()
