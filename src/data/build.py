"""Build the synthetic dataset (prices + news with injected reactions) and store it as Parquet/DuckDB."""
from __future__ import annotations

import pandas as pd

from src.data.align import Calendar
from src.data.market_loader import make_synthetic
from src.data.news_loader import make_synthetic_news
from src.data.store import register_duckdb, save_parquet


def build_synthetic(cfg: dict, seed: int | None = None, save: bool = True):
    days = pd.bdate_range(cfg["synthetic"]["start"], periods=cfg["synthetic"]["n_days"], tz=cfg["market"]["tz"])
    cal = Calendar(days, cfg)
    news, events = make_synthetic_news(cfg, cal, seed)
    bars, truth = make_synthetic(cfg, events, seed)
    if save:
        save_parquet(bars, cfg, "bars")
        save_parquet(news, cfg, "news")
        save_parquet(truth, cfg, "truth")
        register_duckdb(cfg, ["bars", "news", "truth"])
    return bars, news, truth


def get_data(cfg: dict, market: str | None = None, news: str | None = None, rebuild: bool = False):
    """Real data if paths are given, otherwise the synthetic dataset (cached in data/processed)."""
    from src.data.market_loader import load_market
    from src.data.news_loader import load_news
    from src.data.store import load_parquet
    from src.config import resolve

    if market:
        if not news:
            raise ValueError("pass --news together with --market")
        return load_market(market, cfg), load_news(news, cfg["market"]["tz"])
    proc = resolve(cfg, "processed")
    if not rebuild and (proc / "bars.parquet").exists() and (proc / "news.parquet").exists():
        return load_parquet(cfg, "bars"), load_parquet(cfg, "news")
    bars, news_df, _ = build_synthetic(cfg)
    return bars, news_df
