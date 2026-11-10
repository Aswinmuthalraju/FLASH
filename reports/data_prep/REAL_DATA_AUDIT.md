# REAL_DATA_AUDIT — status: NOT EXECUTED (Hugging Face unreachable)

**No real record was downloaded, so this file contains no real-data statistics.** Nothing was substituted,
inferred or synthesised to fill the gap.

## Download attempt (executed)

| Dataset | Mechanism | Result |
|---|---|---|
| `LogeshChandran/money_control_news` | `huggingface_hub.HfApi().dataset_info` → `datasets.load_dataset` | `ConnectError: [Errno -2] Name or service not known` |
| `xxparthparekhxx/indian-stock-market-minute-data` | `HfApi().dataset_info` → `HfFileSystem` shard read | same |
| `curl https://huggingface.co` | — | `curl: (6) Could not resolve host: huggingface.co` |

Cause: this sandbox's network allow-list does not include huggingface.co (DNS does not resolve). It is an
environment restriction, not an HF permission problem; neither dataset is gated as far as was observed.
Errors are recorded with timestamps in `data/raw/hf/MANIFEST.json`. **Revision: not recorded** (unreachable).
On a machine with access, the download code now pins the commit SHA, downloads at that revision, and writes
file sizes + sha256 of every local file to the manifest.

## What is known — from the HF web pages viewed in the previous session (NOT downloaded, NOT verified)

| | Market (minute dataset) | News |
|---|---|---|
| Rows shown | 720,421,072 (minute split 713M, day split 6.97M) | 29,286 |
| Total size shown | 10.5 GB parquet | 44.9 MB |
| Range shown | minute timestamps 2022-01-03 03:45 → 2026-01-21 09:59 UTC | 115 distinct dates; pages showed June 2024 … Jan 2025 |
| Licence tag | MIT (uploader-set) | MIT (uploader-set) |
| Defects seen in viewer | — | 3 timestamp layouts incl. time-only rows; mojibake; off-topic sections |

Treat these as leads to verify, not as statistics.

## How each Phase-4 question will be answered once data is available

Run: `bash preprocessing/run_pipeline.sh data && bash preprocessing/run_pipeline.sh verify && bash preprocessing/run_pipeline.sh audit`

| Brief item | Where the answer appears |
|---|---|
| Symbol availability | `market_adapt_log.json → symbols_missing`; `quality.json → raw_market.per_symbol` |
| Date coverage / records per symbol | `quality.json → raw_market.per_symbol.{first_day,last_day,trading_days,records_in_session}` |
| Timestamp format / timezone | adapter converts UTC→IST (tested: 03:45Z → 09:15 IST); `raw_market.timestamp_tz_as_loaded` |
| Missing intervals | `per_symbol.missing_minutes_frac`, `day_coverage`, `days_below_min_coverage` |
| Duplicates / invalid OHLC / non-positive | `raw_market.duplicate_symbol_ts`, `ohlc_violations`, `nonpositive_prices` |
| Zero volume | `per_symbol.zero_volume_frac` |
| Session coverage | `rows_outside_session`, `day_coverage` |
| Late-open days (leakage) | `market_adapt_log.json → late_open_symbol_days_dropped` |
| 5/15/30-bar horizon support | `pipeline.labelled_rows_per_horizon`, `per_symbol_valid_labelled_h15` |
| Articles / date coverage / timestamps | `news_adapt_log.json → rows_in, timestamp_layouts, date_range, distinct_dates` |
| Undated (kept aside) | `undated_set_aside`; rows in `data/interim/news_undated.csv` |
| Missing headlines / duplicates | `dropped_empty_headline`, `dropped_duplicates` |
| Encoding / language | `news.encoding_language`, `news.residual_mojibake_rows` |
| Category distribution | `news.categories` (Moneycontrol URL section) |
| Association with the 10 stocks / per stock | `news.used_by_flash_frac`, `stock_specific`, `stock_specific_per_symbol` |
| Coverage gaps | `news.coverage.{trading_days_with_*, longest_run_of_trading_days_without_articles}` |

No company-keyword pre-filtering is applied; FLASH's own `EntityLinker` decides relevance, so its errors stay visible.

## Mock run (plumbing evidence only — NOT data quality)

On HF-shaped mock files (`reports/data_prep/smoke_mock/`): 700 news rows → 31 undated set aside, 19
entertainment excluded, 15 duplicates removed, 635 model-ready, 0 residual mojibake; 10/10 symbols, 0 NaN/inf
in model-ready rows, all 3 horizons labelled, 0 critical issues.

## Limitations expected for the real data (to confirm)

1. Few stock-specific headlines: Moneycontrol covers thousands of companies plus politics and gadgets.
2. Coverage gaps: 115 dates over ~7 months means many trading days without scraped news. On those days "no news" is a scraping artefact.
3. Linker false positives on real text (e.g. "SBI Mutual Fund" → SBIN; "Infosys founder …" → INFY), and the geopolitical keyword rule broadcasting political headlines to all 10 stocks.
4. Licence: uploader "MIT" tags cannot grant rights to Moneycontrol articles or broker-sourced prices. Use for research only, don't redistribute (enforced by `.gitignore`).
