"""Write SMOKE-TEST files in the exact Hugging Face schemas, with the defects observed in the HF viewer.

Used only to validate the adapters + FLASH plumbing when huggingface.co is unreachable. Never train on this.
Prices come from FLASH's own synthetic generator; news reuses its templates plus off-topic rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402
from src.data.build import build_synthetic  # noqa: E402


def main(out_dir: Path = ROOT / "data/raw/hf_mock", n_days: int = 20, seed: int = 3):
    cfg = load_config()
    cfg["synthetic"]["n_days"] = n_days
    bars, news, _ = build_synthetic(cfg, seed=seed, save=False)
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- market: HF schema = UTC timestamps, float32, oi column, extra symbol, gaps, a duplicate, pre-open rows
    m = bars.sample(frac=0.97, random_state=seed).sort_values(["symbol", "timestamp"])   # 3 % missing minutes
    m = pd.concat([m, m.iloc[[10]]])                                                         # duplicate
    pre = m.iloc[:5].copy()
    pre["timestamp"] -= pd.Timedelta(minutes=30)                                           # pre-open prints
    other = m[m.symbol == "TCS"].iloc[:500].assign(symbol="20MICRONS")                     # not in universe
    m = pd.concat([m, pre, other])
    m["timestamp"] = m["timestamp"].dt.tz_convert("UTC")
    for c in ["open", "high", "low", "close"]:
        m[c] = m[c].astype("float32")
    m["volume"] = m["volume"].astype("int64")
    m["oi"] = np.int64(0)
    m[["symbol", "timestamp", "open", "high", "low", "close", "volume", "oi"]].to_parquet(out_dir / "minute.parquet")

    # ---- news: Moneycontrol schema, 3 timestamp layouts, mojibake, off-topic sections, duplicate URLs
    off = ["Vivo to launch its first mixed-reality headset", "Bollywood celebs celebrating birthday in January",
           "Flights delayed as dense fog drapes Bengaluru airport"]
    n = news.copy()
    n = pd.concat([n, pd.DataFrame({"timestamp": n.timestamp.sample(60, random_state=1).to_numpy(),
                                    "headline": rng.choice(off, 60), "source": "x"})], ignore_index=True)
    ids = 12_800_000 + np.argsort(np.argsort(n.timestamp.to_numpy())) * 7
    sect = np.where(n.headline.isin(off), rng.choice(["entertainment", "technology", "india"], len(n)), "business")
    url = [f"https://www.moneycontrol.com/news/{s}/story-{i}.html" if s != "entertainment"
           else f"https://www.moneycontrol.com/{s}/story-{i}.html" for s, i in zip(sect, ids)]
    local = n.timestamp.dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d %H:%M")
    layout = rng.choice(3, len(n), p=[0.8, 0.15, 0.05])
    dt = np.where(layout == 0, local, None)
    date = np.where(layout == 0, local.str[:10], None)
    tm = np.where(layout == 0, local.str[11:], np.where(layout == 1, local, local.str[11:]))
    title = n.headline.str.replace("'", "â€™", regex=False)
    title = np.where(rng.random(len(n)) < 0.1, title + " â€” analysts", title)
    df = pd.DataFrame({"URL": "moneycontrol.com", "Article url": url, "title": title, "subtitle": "",
                       "content": "", "article datetime": dt, "article date": date, "article time": tm})
    df = pd.concat([df, df.iloc[:15]])                                                     # duplicate URLs
    df.to_parquet(out_dir / "money_control_news.parquet")
    print(f"mock written to {out_dir}: {len(m):,} bar rows, {len(df):,} news rows, "
          f"{bars.timestamp.dt.date.min()} .. {bars.timestamp.dt.date.max()}")


if __name__ == "__main__":
    main()
