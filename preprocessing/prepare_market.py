"""Adapt xxparthparekhxx/indian-stock-market-minute-data (HF, MIT tag) to FLASH's market input.

Source schema : symbol, timestamp (UTC, tz-aware), open, high, low, close (float32), volume, oi (int64)
FLASH schema  : timestamp (Asia/Kolkata, bar OPEN time), symbol, open, high, low, close, volume
                one file per symbol in data/raw/market/  (read by src.data.market_loader.load_market)

Transformations (each one justified in reports/data_prep/REAL_DATA_AUDIT.md):
  1. keep only cfg["symbols"]                       - FLASH's universe / entity dictionary is fixed to these 10
  2. UTC -> Asia/Kolkata                            - FLASH's session grid (09:15, 375 bars) is defined in IST
  3. restrict to [start, end] (the news window)     - news features are meaningless outside the news coverage
  4. drop `oi`                                      - always 0 for cash equities; not a FLASH input
  5. float32 -> float64                             - avoid float32 rounding in log-returns of ~1e-4
  6. drop symbol-days whose first in-session trade is later than the feature warm-up (09:15 + 30 min).
     LEAKAGE FIX: src.data.market_loader.regularize_bars back-fills leading gaps (`close.ffill().bfill()`),
     so on such a day the bars between 09:45 and the first trade - which FLASH treats as VALID feature
     rows - would carry a FUTURE close. Dropping the day avoids editing FLASH's loader. Count is logged.
NOT done here (FLASH's own loader already does it, so we don't duplicate it): OHLC repair, dedup,
regularisation onto the 375-bar grid, sparse-day removal. We only REPORT those counts (verify_dataset.py).

    python preprocessing/prepare_market.py --start 2024-06-01 --end 2025-01-31
    python preprocessing/prepare_market.py --source data/raw/hf/minute_subset.parquet ...   (offline / mock)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402

HF_REPO = "xxparthparekhxx/indian-stock-market-minute-data"


def fetch_hf_subset(symbols: list[str], start: str, end: str, out: Path) -> Path:
    """Stream only the needed rows out of the ~10.5 GB minute split (pyarrow predicate push-down)."""
    import pyarrow.dataset as ds
    from huggingface_hub import HfFileSystem

    from preprocessing.hf_manifest import record_local, resolve_revision
    rev = resolve_revision(HF_REPO)["revision_sha"]          # pin: exactly this commit, recorded in MANIFEST
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{HF_REPO}@{rev}/minute/*.parquet"))
    if not files:
        raise RuntimeError("no minute shards found - check repo layout / network access to huggingface.co")
    dset = ds.dataset(files, filesystem=fs, format="parquet")
    t0 = pd.Timestamp(start, tz="Asia/Kolkata").tz_convert("UTC")
    t1 = (pd.Timestamp(end, tz="Asia/Kolkata") + pd.Timedelta(days=1)).tz_convert("UTC")
    flt = ds.field("symbol").isin(symbols) & (ds.field("timestamp") >= t0) & (ds.field("timestamp") < t1)
    table = dset.to_table(filter=flt)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_pandas().to_parquet(out, index=False)          # untouched copy of the source rows = data/raw
    record_local(HF_REPO, out, f"rows for {len(symbols)} symbols, {start}..{end} IST, revision {rev}, "
                               f"read from {len(files)} minute shards via predicate push-down")
    return out


def drop_late_open_days(df: pd.DataFrame, warmup_min: int, session_open: str) -> tuple[pd.DataFrame, list[str]]:
    """Drop (symbol, day) whose first in-session bar is after minute `warmup_min` of the session."""
    oh, om = map(int, session_open.split(":"))
    minute = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute - (oh * 60 + om)
    day = df["timestamp"].dt.date
    first = minute.where(minute >= 0).groupby([df["symbol"], day]).min()
    late = first[first > warmup_min]
    if late.empty:
        return df, []
    key = pd.MultiIndex.from_arrays([df["symbol"], day])
    return df[~key.isin(late.index)], [f"{s} {d}" for s, d in late.index]


def adapt(raw: pd.DataFrame, symbols: list[str], start: str, end: str, tz: str,
          warmup_min: int = 30, session_open: str = "09:15") -> tuple[pd.DataFrame, dict]:
    log = {"rows_in": int(len(raw))}
    df = raw[raw["symbol"].isin(symbols)].copy()
    log["rows_after_symbol_filter"] = int(len(df))
    ts = pd.to_datetime(df["timestamp"], utc=True)          # source is documented as UTC
    df["timestamp"] = ts.dt.tz_convert(tz)
    lo = pd.Timestamp(start, tz=tz)
    hi = pd.Timestamp(end, tz=tz) + pd.Timedelta(days=1)
    df = df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)]
    log["rows_after_date_filter"] = int(len(df))
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype("float64")
    df["volume"] = df["volume"].astype("float64")
    df = df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
    df, log["late_open_symbol_days_dropped"] = drop_late_open_days(df, warmup_min, session_open)
    log["rows_after_late_open_drop"] = int(len(df))
    log["symbols_missing"] = sorted(set(symbols) - set(df["symbol"].unique()))
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--source", help="local parquet already in the HF schema (skip the download)")
    ap.add_argument("--out", default=str(ROOT / "data/raw/market"))
    ap.add_argument("--log-dir", default=str(ROOT / "reports/data_prep"))
    a = ap.parse_args()
    cfg = load_config()
    syms, tz = cfg["symbols"], cfg["market"]["tz"]
    src = Path(a.source) if a.source else fetch_hf_subset(syms, a.start, a.end, ROOT / "data/raw/hf/minute_subset.parquet")
    from src.features.market_features import WARMUP
    df, log = adapt(pd.read_parquet(src), syms, a.start, a.end, tz, WARMUP, cfg["market"]["session_open"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s, g in df.groupby("symbol"):
        g.to_parquet(out / f"{s}.parquet", index=False)
    log["files_written"] = int(df["symbol"].nunique())
    Path(a.log_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.log_dir) / "market_adapt_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
