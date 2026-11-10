"""Adapt LogeshChandran/money_control_news (HF, MIT tag, ~29.3k rows) to FLASH's news CSV.

Source columns : URL, Article url, title, subtitle, content, article datetime, article date, article time
FLASH columns  : timestamp (naive IST -> localised by src.data.news_loader.load_news), headline, source

Observed source defects (seen in the HF viewer, verified again by verify_dataset.py on the real download):
  (a) three timestamp layouts:  datetime filled | datetime+date null with "YYYY-MM-DD HH:MM" in `article time`
      | datetime+date null with only "HH:MM" in `article time` (no date at all)
  (b) mojibake (UTF-8 read as cp1252: "â€™" for an apostrophe)
  (c) many non-business sections (entertainment, gadgets, politics)

Transformations:
  1. coalesce (a) into one timestamp. Rows with no DATE are removed from the model-ready file and written,
     unchanged, to data/interim/news_undated.csv. Dates are never inferred (e.g. from article ids): a wrong
     date silently mis-aligns news with prices, so it is a leakage/noise risk with no way to verify it.
  2. ftfy repair of (b)                    - FinBERT tokenises "â€™" as junk sub-words
  3. headline = title (FLASH and FinBERT are headline-level; content is not used by the model)
  4. dedup on article URL, then on (timestamp, headline)
  5. --exclude-sections drops URL sections that cannot move the 10 stocks (default: entertainment).
     We do NOT keyword-filter for the 10 companies: FLASH's own EntityLinker decides relevance, and
     pre-filtering would hide its false positives/negatives from the evaluation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HF_REPO = "LogeshChandran/money_control_news"
_FULL = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})[ T](\d{1,2}:\d{2})")


def fetch_hf(out: Path) -> Path:
    from datasets import load_dataset
    from preprocessing.hf_manifest import record_local, resolve_revision
    rev = resolve_revision(HF_REPO)["revision_sha"]          # pin: exactly this commit, recorded in MANIFEST
    df = load_dataset(HF_REPO, split="train", revision=rev).to_pandas()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)                # untouched copy = data/raw (never modified afterwards)
    record_local(HF_REPO, out, f"full train split at revision {rev}")
    return out


def _fix(s: pd.Series) -> pd.Series:
    try:
        import ftfy
    except ImportError:                            # degrade gracefully, the verifier will flag residual mojibake
        return s.fillna("").astype(str)
    return s.fillna("").astype(str).map(ftfy.fix_text)


def _full(s: pd.Series) -> pd.Series:
    """'YYYY-MM-DD HH:MM' where the string starts with a full date-time, else <NA>."""
    x = s.astype("string").str.extract(_FULL)
    return (x[0] + " " + x[1]).astype("string")


def coalesce_ts(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Returns (timestamp string 'YYYY-MM-DD HH:MM' or <NA>, layout label)."""
    a = _full(df["article datetime"]) if "article datetime" in df else pd.Series(pd.NA, index=df.index, dtype="string")
    b = _full(df["article time"])
    out = a.fillna(b)
    layout = pd.Series(np.where(a.notna(), "datetime", np.where(b.notna(), "date_in_time_col", "no_date")),
                       index=df.index)
    return out, layout


def adapt(raw: pd.DataFrame, exclude_sections: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (model-ready news, undated rows kept aside unchanged, log)."""
    log = {"rows_in": int(len(raw))}
    df = raw.reset_index(drop=True)
    ts, layout = coalesce_ts(df)
    log["timestamp_layouts"] = layout.value_counts().to_dict()
    df["timestamp"] = pd.to_datetime(ts, errors="coerce", format="%Y-%m-%d %H:%M")
    undated = raw.reset_index(drop=True)[df["timestamp"].isna().to_numpy()].copy()
    log["undated_set_aside"] = int(len(undated))
    df = df.dropna(subset=["timestamp"])
    df["headline"] = _fix(df["title"]).str.strip()
    log["dropped_empty_headline"] = int((df["headline"].str.len() == 0).sum())
    df = df[df["headline"].str.len() > 0]
    url = df["Article url"].astype(str)
    df["section"] = url.str.extract(r"moneycontrol\.com/(?:news/)?([a-z\-]+)/")[0].fillna("other")
    log["sections"] = df["section"].value_counts().to_dict()
    n = len(df)
    df = df[~df["section"].isin(exclude_sections)]
    log["dropped_excluded_sections"] = n - len(df)
    n = len(df)
    df = df.drop_duplicates("Article url").drop_duplicates(["timestamp", "headline"])
    log["dropped_duplicates"] = n - len(df)
    df["source"] = "Moneycontrol"
    df["url"] = url.loc[df.index]
    out = df[["timestamp", "headline", "source", "section", "url"]].sort_values("timestamp").reset_index(drop=True)
    log["rows_out"] = int(len(out))
    log["date_range"] = [str(out["timestamp"].min()), str(out["timestamp"].max())] if len(out) else None
    log["distinct_dates"] = int(out["timestamp"].dt.date.nunique()) if len(out) else 0
    return out, undated, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="local parquet in the HF schema (skip the download)")
    ap.add_argument("--out", default=str(ROOT / "data/raw/news.csv"))
    ap.add_argument("--exclude-sections", default="entertainment")
    ap.add_argument("--undated-out", default=str(ROOT / "data/interim/news_undated.csv"))
    ap.add_argument("--log-dir", default=str(ROOT / "reports/data_prep"))
    a = ap.parse_args()
    src = Path(a.source) if a.source else fetch_hf(ROOT / "data/raw/hf/money_control_news.parquet")
    out, undated, log = adapt(pd.read_parquet(src), [s for s in a.exclude_sections.split(",") if s])
    out.to_csv(a.out, index=False)
    Path(a.undated_out).parent.mkdir(parents=True, exist_ok=True)
    undated.to_csv(a.undated_out, index=False)
    Path(a.log_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.log_dir) / "news_adapt_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
