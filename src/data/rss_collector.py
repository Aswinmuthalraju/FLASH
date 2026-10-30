"""RSS collector (Moneycontrol / ET Markets) -> appends to data/raw/rss_news.csv.

    python -m src.data.rss_collector --once
    python -m src.data.rss_collector --loop --interval 300

`timestamp` is max(published, first_seen): we can never act on an item before we have actually fetched it.
Standard library only (urllib + ElementTree).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from src.config import ROOT, load_config

DEFAULT_FEEDS = {
    "Moneycontrol": [
        "https://www.moneycontrol.com/rss/latestnews.xml",
        "https://www.moneycontrol.com/rss/marketreports.xml",
    ],
    "ET Markets": [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    ],
}
FIELDS = ["timestamp", "headline", "source", "published", "first_seen", "key"]


def parse_feed(xml_bytes: bytes) -> list[tuple[str, datetime | None]]:
    # stdlib parser: refuse DTDs/entities outright (billion-laughs / XXE) instead of adding defusedxml
    head = xml_bytes[:4096].lower()
    if b"<!doctype" in head or b"<!entity" in xml_bytes.lower():
        raise ValueError("feed contains a DTD/entity declaration; refusing to parse")
    root = ET.fromstring(xml_bytes)
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        pub = item.findtext("pubDate")
        try:
            pub_dt = parsedate_to_datetime(pub) if pub else None
        except (TypeError, ValueError):
            pub_dt = None
        if title:
            out.append((title, pub_dt))
    return out


def collect_once(out_path: Path, feeds: dict[str, list[str]] | None = None, timeout: float = 10.0) -> int:
    feeds = feeds or DEFAULT_FEEDS
    seen = set()
    if out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            seen = {r["key"] for r in csv.DictReader(f)}
    new_rows = []
    now = datetime.now(timezone.utc)
    for source, urls in feeds.items():
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 flash-market"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    items = parse_feed(r.read())
            except Exception as e:  # network/feed errors must not kill the collector
                print(f"[rss] {source} {url}: {e}")
                continue
            for title, pub in items:
                key = hashlib.sha1(title.lower().encode()).hexdigest()[:16]
                if key in seen:
                    continue
                seen.add(key)
                pub_utc = pub.astimezone(timezone.utc) if pub and pub.tzinfo else None
                usable = max(pub_utc, now) if pub_utc else now
                new_rows.append({"timestamp": usable.isoformat(), "headline": title, "source": source,
                                 "published": pub_utc.isoformat() if pub_utc else "",
                                 "first_seen": now.isoformat(), "key": key})
    if new_rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out_path.exists()
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            w.writerows(new_rows)
    return len(new_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "raw" / "rss_news.csv"))
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    a = ap.parse_args()
    load_config()
    while True:
        n = collect_once(Path(a.out))
        print(f"[rss] {datetime.now():%H:%M:%S} +{n} new headlines")
        if not a.loop:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
