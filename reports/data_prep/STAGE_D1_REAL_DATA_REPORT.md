# Stage D1 — Real-Data Download, Preparation and Verification

Date: 9 Nov 2026 · Session 3 · Branch `data-pipeline` (uncommitted) · Nothing committed or pushed.

Every value in this report was **observed in this run** unless it is explicitly marked *Estimate* or listed under
*Unresolved*. Commands were run with the explicit interpreter `.venv\Scripts\python.exe` (CPython 3.11.18).

---

## 1. Repository and branch verification

| Item | Observed |
|---|---|
| Repository root | `C:/Users/prsan/Projects/FLASH` |
| Remote | `origin` → `https://github.com/Aswinmuthalraju/FLASH.git` |
| Branch | `data-pipeline` |
| HEAD (before and after) | `ffa1cda72861d22ac9f2ec71e029858e582cc9d3` (V_1.2) |
| `backup/pre-data-pipeline-20261109` (before and after) | `ffa1cda72861d22ac9f2ec71e029858e582cc9d3` |
| Staged changes (before and after) | none |

## 2. Disk space

| | Bytes free on C: |
|---|---|
| Before Stage D1 | 420,828,876,800 (≈ 391.9 GiB) |
| After Stage D1 | 419,620,855,808 (≈ 390.8 GiB) |
| Consumed | ≈ 1.21 GB (FinBERT ≈ 418 MB in HF cache, news parquet in `datasets` + hub cache, repo data ≈ 160 MB) |

*Estimate used for planning:* 20 GB. Actual consumption was far lower because the market data was read with
predicate push-down (only the row groups containing the 10 symbols), not downloaded in full.
No files or caches were deleted.

Pre-existing state: `data/raw/hf/` held only the patch's `MANIFEST.json` (both repos `ACCESS_FAILED` from
Session 2); no real `data/raw/market/`, `data/raw/news.csv` or partial downloads; no FLASH-related entries
in `~/.cache/huggingface` (4.7 GB of unrelated caches left untouched).

## 3. Hugging Face connectivity

| Check | Result |
|---|---|
| DNS `huggingface.co` | resolves (54.230.27.69 / .75 / .119) |
| `GET https://huggingface.co/` | HTTP 200, 0.59 s |
| Hub API `GET /api/datasets/LogeshChandran/money_control_news` | HTTP 200, 0.57 s |
| `HfApi().dataset_info(..., files_metadata=True)` | OK for both repos |
| `HfFileSystem().glob(...@<sha>/**/*.parquet)` | OK — 1 file (news), 9 files (market: 8 minute + 1 day) |
| `HfFileSystem` parquet footer reads (market) | OK — all 8 minute shards |
| `datasets.load_dataset(news, revision=<sha>)` | OK (run by `prepare_news.py`) |
| FinBERT `ProsusAI/finbert` model download | OK |

The Session 2 failure (`ConnectError: [Errno -2] Name or service not known`) does not reproduce here.
Warning (harmless): `huggingface_hub` cannot create symlinks on this Windows machine (Developer Mode off), so the
cache stores copies.

## 4. Dataset repository metadata and revisions

| | News | Market |
|---|---|---|
| Repo | `LogeshChandran/money_control_news` | `xxparthparekhxx/indian-stock-market-minute-data` |
| Pinned revision (commit sha) | `033afe61ca4b3ebb8879ac9097461f36f99348e1` | `91fc0616910dc9d92c5e34f8470dd58817529cc8` |
| Last modified | 2025-01-02 10:25:56 UTC | 2026-01-25 15:44:42 UTC |
| License tag | MIT | MIT |
| Gated / private | no / no | no / no |
| Files | 3 (1 parquet, 44,945,153 B) | 11 (8 minute shards = 10,392,289,058 B; 1 day shard = 117,731,670 B) |
| Schema | `URL, Article url, title, subtitle, content, article datetime, article date, article time` (all string) | `symbol (large_string), timestamp (timestamp[us, tz=UTC]), open/high/low/close (float32), volume (int64), oi (int64)` |
| Rows | 29,286 (card `num_examples`, confirmed on load) | 713,452,456 minute rows (sum of shard footers) |
| Observed layout | single train split | shards sorted by symbol then timestamp; 100 row groups per shard (14 in the last); minute timestamps 2022-01-03 03:45 UTC … 2026-01-21 09:59 UTC |

FinBERT (used in §10): `ProsusAI/finbert`, cached snapshot `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`,
`model.safetensors` 437,965,908 B. Loaded on CPU.

## 5. Downloaded files, sizes and SHA-256

### Source copies (`data/raw/hf/`, git-ignored, preserved)

| File | Bytes | SHA-256 |
|---|---|---|
| `money_control_news.parquet` | 44,000,769 | `7213bcd4a42d8ea3ab95798461524bb0ca6479085ac1be2afe26e524fd42e65d` |
| `minute_subset.parquet` | 7,405,059 | `5d55a71e0791f6a3d59205cb767f9f401a3d528edc4dd20a8fc63f21763d3931` |

Provenance of the news copy: the original HF file in the hub cache
(`.../snapshots/033afe61.../data/train-00000-of-00001.parquet`, 44,945,153 B) has SHA-256
`a8fcc6e4c2dbc0b411da2ba8ef53d1a1e0bef1594637c8b7fd033794f2eb3a3c`, which **matches the LFS sha256 published by
Hugging Face**. `money_control_news.parquet` is a pandas re-serialisation of that file (hence the different
size/hash); its content is **identical** to the original (`DataFrame.equals` → True, 29,286 × 8).

Market shards were not downloaded in full (range reads only), so they cannot be hashed locally. Their published LFS
sha256 values at the pinned revision are:

| Shard | Bytes | LFS SHA-256 |
|---|---|---|
| minute/train-00000.parquet | 1,446,269,838 | `5ca18212d72ee23797464a794b0348a339ec5f30c7a94f38952f9dbdcce28173` |
| minute/train-00001.parquet | 1,468,492,209 | `56ac76a612136f00eecffeedfac133eaf842256001c81bea09cd0aa769157dae` |
| minute/train-00002.parquet | 1,432,730,390 | `4c78de4c828aca83e34bc9882667f4409d1229cd3ddc9b568036bfd4989a469e` |
| minute/train-00003.parquet | 1,405,964,246 | `c1c33fd47617ae6787bcf5ee12e6f790f6abaa7b496f59d455d02cc69bcbb34a` |
| minute/train-00004.parquet | 1,574,124,848 | `d5a0db25718519aae48c2ccad01ba40dd08750cf796e3e4251cefabcbef04ba4` |
| minute/train-00005.parquet | 1,427,834,477 | `a3e725b7c336bfd390580e9b5553068c62d01ef9ba978523eb6c6ae19a9d9397` |
| minute/train-00006.parquet | 1,443,474,644 | `16016573de56301df7ade203f0ad0de40ad85199ba36c894335199310e10903e` |
| minute/train-00007.parquet | 193,398,406 | `03ffbeae4d5a2e3730959c11c0389c028a049afb337397f06fcbcec79f808751` |

### Model-ready outputs (git-ignored)

| File | Bytes | SHA-256 |
|---|---|---|
| `data/raw/news.csv` | 7,742,119 | `87ea1c4b21f5b682189d0a6f3e7fe002347696bb51629012a0c64646ba3ba47e` |
| `data/interim/news_undated.csv` | 234,291 | `04a05ebad2a0be51feea0d9a00527aac9fc33bcbd474ccee2ae42b2c0094b682` |
| `data/raw/market/BHARTIARTL.parquet` | 1,085,386 | `21f1d41bc4650e1c82ae08759ecb3f806000aa62819efb0f7e52e8c943dd2d12` |
| `data/raw/market/HDFCBANK.parquet` | 979,660 | `27a11082006c6186e3a3c578eee4d5e31e91499b257b285d0cac95c5dd78a711` |
| `data/raw/market/HINDUNILVR.parquet` | 1,135,278 | `99e9126ee941c19061081aba9ca1f7fd46240b37891f76016ee05040e3d65b93` |
| `data/raw/market/ICICIBANK.parquet` | 1,067,582 | `c4da26cb625f0250073f4185398fc0899c943731e53c652c32c17a30055bb965` |
| `data/raw/market/INFY.parquet` | 1,107,623 | `9c160df18e1afa8d9168dafb6b4aabfd7a3b8e6477d14d56e42be16825e22175` |
| `data/raw/market/ITC.parquet` | 1,003,538 | `224ed69ffc703c808b15b34fd90aca2e9938c6b3b4b1f5524999e4fc68428340` |
| `data/raw/market/KOTAKBANK.parquet` | 861,766 | `1625bdb279676bef3d0543dbb4dfab103f44173f7bd5cc3ef2edef7767017f56` |
| `data/raw/market/RELIANCE.parquet` | 1,109,479 | `037ba39a7945c6962ef96065731a679a801175f9f3350f3efc569ba606d3c838` |
| `data/raw/market/SBIN.parquet` | 1,031,019 | `387d8a3778e966bc9a8d83955fe962e9335c98a16a75e23ddcaa36cc37c8c8ff` |
| `data/raw/market/TCS.parquet` | 1,180,128 | `6e2384b3c5c48bf81b60ac37cd10290e39f456e3100d0aaf222606108225e781` |
| `data/processed_real/finbert_cache.npz` | 93,000,144 | `43447937a34ba869603ca6552e222d31d2da35e26af8eba634c943e3a89f1373` |

## 6. News — date range and row counts

Command: `.venv\Scripts\python.exe preprocessing/prepare_news.py` (run once; 24 s). Log:
`reports/data_prep/news_adapt_log.json`.

| Item | Observed |
|---|---|
| Input rows | 29,286 |
| Timestamp layouts | `datetime` 24,832 · `date_in_time_col` 4,374 · `no_date` 80 |
| Undated rows set aside (unchanged, never date-inferred) | 80 → `data/interim/news_undated.csv` |
| Dropped: empty headline | 1 |
| Dropped: excluded sections (`entertainment`) | 70 |
| Dropped: duplicates (URL, then timestamp+headline) | 12 |
| **Output rows** | **29,123** (29,286 − 80 − 1 − 70 − 12 ✔) |
| **First / last adapted timestamp** | **2024-06-16 15:42 / 2025-01-02 15:48** (IST, naive in CSV) |
| Distinct publication dates | 197 |
| Sections kept | 28 (largest: business 17,776 · india 5,198 · world 2,693 · technology 2,333) |

*Estimate superseded:* the script's default window `START=2024-06-01`, `END=2025-01-31` was a placeholder; the
observed coverage is 2024-06-16 … 2025-01-02.

## 7. Market — date range and row counts

Command: `.venv\Scripts\python.exe preprocessing/prepare_market.py --start 2024-06-16 --end 2025-01-02`
(dates taken from §6; run once; 28 s). Log: `reports/data_prep/market_adapt_log.json`.

| Item | Observed |
|---|---|
| Minute shards scanned | 8 (predicate push-down on `symbol` ∈ 10 symbols and the IST window) |
| Input rows (subset) | 510,600 |
| After symbol filter | 510,600 |
| After date filter | 510,600 |
| **After late-open protection** | **510,000** |
| Files written | 10 (`data/raw/market/<SYMBOL>.parquet`) |
| Missing symbols | none |
| **Market coverage** | **2024-06-18 09:15 → 2025-01-02 15:29 IST, 136 trading days**, every symbol |
| Bars per symbol-day | exactly 375 for all 1,360 symbol-days (std 0) |
| `oi` non-zero | 0 rows (column dropped as designed) |

2024-06-16 was a Sunday and 2024-06-17 an NSE holiday, so the first trading day in the window is 2024-06-18.

## 8. Late-open symbol-day drops

10 symbol-days dropped, all on **2024-11-01**, one per symbol (BHARTIARTL, HDFCBANK, HINDUNILVR, ICICIBANK, INFY,
ITC, KOTAKBANK, RELIANCE, SBIN, TCS) — 600 rows.
Observed: on that date the source contains only bars from **18:00 to 18:59 IST** (60 bars × 10 symbols). This
matches the Diwali *Muhurat* special evening session (interpretation; the timestamps are the verified fact). Without
the drop, FLASH's loader would back-fill the 09:15–17:59 grid from the first evening trade (the L5 look-ahead leak),
so dropping is the intended behaviour.

## 9. Manifest verification (`data/raw/hf/MANIFEST.json`)

| Check | News | Market |
|---|---|---|
| `status` | `downloaded` | `downloaded` |
| `revision_sha` recorded | ✔ `033afe61…` | ✔ `91fc0616…` |
| `remote_files` with sizes | ✔ 3 files | ✔ 11 files (minute total 10,392,289,058 B) |
| `local_files` bytes + sha256 | ✔ 44,000,769 / `7213bcd4…` | ✔ 7,405,059 / `5d55a71e…` |
| Matches independent `sha256sum` | ✔ | ✔ |

**News source hash unchanged since Step 3:** `7213bcd4…e65d` re-checked after market preparation, after
verification, and at the end of the stage. `news.csv` and `news_undated.csv` hashes also unchanged.
`run_pipeline.sh data` was **not** run; news was resolved and downloaded exactly once.

Manifest defects (recorded, not fixed — code changes were out of scope):
1. Stale fields: each entry still carries `error_type: ConnectError` and `error: [Errno -2] Name or service not
   known` from Session 2, because `resolve_revision()` merges into the old record instead of replacing it. `status`
   is correct (`downloaded`).
2. `local_files` keys use Windows separators (`data\\raw\\hf\\...`), so the manifest is platform-dependent.
3. `checked_at_utc` reflects revision resolution time, not download completion time.

## 10. Real-data verification

Command: `.venv\Scripts\python.exe preprocessing/verify_dataset.py --config configs/real.yaml` — **no
`--stub-encoder`**. Exit 0, 1,959 s (≈ 33 min, dominated by CPU FinBERT encoding). Output:
`reports/data_prep/quality.json`.

**Critical issues: none (`[]`).**

| Area | Observed |
|---|---|
| Raw market rows | 510,000; duplicates 0; missing values 0 in every column; non-positive prices 0; rows outside session 0 |
| OHLC violations (high < max(open,close) or low > min(open,close)) | 3 rows (left to FLASH's loader, which repairs OHLC) |
| Timezone as loaded | Asia/Kolkata; chronological within every symbol |
| Day coverage | mean 1.0, min 1.0, days below `min_day_coverage` 0; missing-minute fraction 0.0 for all symbols |
| Zero-volume bars | 0.0043 % overall (2–4 bars per symbol over 51,000) |
| After FLASH loader | 510,000 bars, 10 symbols, 136 days, 2024-06-18 09:15 → 2025-01-02 15:29 IST |
| Returns | max abs 1-min log return 3.10 %; 278 bars beyond 10 σ (informational, no filter applied) |
| News rows / in market window | 29,123 / 29,113 |
| Trading days with any article / with FLASH-usable article | 136 / 127 of 136; longest run without articles 0; median 195.5 articles per trading day |
| Used by FLASH (entity-linked) | 5.99 % of in-window headlines; stock-specific 716; market-wide broadcast 1,009 |
| Stock-specific per symbol | RELIANCE 144, SBIN 122, INFY 95, KOTAKBANK 92, HDFCBANK 76, TCS 70, BHARTIARTL 57, ICICIBANK 38, ITC 35, HINDUNILVR 35 |
| Event types of used headlines | RBI/Rates 596, Other 525, Geopolitical 431, Earnings 73, M&A 42, Management 37, Regulation 30, Legal/Fraud 9 |
| Encoding | residual mojibake 0; non-ASCII rows 627; low-Latin-ratio rows 0; empty headlines 0; duplicate (ts, headline) 0 |
| In-session fraction of news | 55.4 % |
| **FinBERT** | real `ProsusAI/finbert` (snapshot `0bc2ac10…`) loaded on CPU; embeddings cached to `data/processed_real/finbert_cache.npz` |
| Feature matrix | 510,000 × 42; NaN in valid rows 0; inf in valid rows 0 |
| Labelled rows per horizon | h5 462,400 · h15 448,800 · h30 428,400; h15 valid+labelled 44,880 per symbol (balanced) |
| Regime (fit on **train split only**) | PCA 4 components; HMM states calm / bull / volatile / bear |
| **Split chronology** | train days 0–81 (82), val 82–108 (27), test 109–135 (27); **chronological and disjoint: true** |
| News-shock rows in test (≤ 30 min) | 24,746 (experiment D vs B is testable) |

Label class shares (structural counts only — no model, no predictions):

| Horizon | Train DOWN / NEUTRAL / UP | Val | Test |
|---|---|---|---|
| h5 | 0.188 / 0.625 / 0.187 (n 278,800) | 0.192 / 0.616 / 0.193 (n 91,800) | 0.179 / 0.636 / 0.185 (n 91,800) |
| h15 | 0.214 / 0.573 / 0.213 (n 270,600) | 0.221 / 0.556 / 0.223 (n 89,100) | 0.203 / 0.586 / 0.211 (n 89,100) |
| h30 | 0.229 / 0.542 / 0.229 (n 258,300) | 0.235 / 0.530 / 0.235 (n 85,050) | 0.219 / 0.556 / 0.226 (n 85,050) |

Scope note: `verify_dataset.py` calls FLASH's `prepare()` — news encoding, feature build, and the PCA/GMM/HMM
regime fit on the train split. It does **not** call `fit_variant()` (no LightGBM/logistic fit, no predictions,
no calibration, no scoring). The test split is used only for row counts and label shares.

Warnings during the run: `huggingface_hub` symlink warning (cosmetic); a FinBERT weight-loading progress bar.
No errors.

## 11. Errors, limitations and pending items

**Errors:** none in any Stage D1 command.

**Limitations / observations (verified):**
- Source market data is a complete 375-bar grid for every symbol-day with almost no zero-volume bars. Either the
  provider pre-fills gaps or these large-caps genuinely trade every minute; the dataset does not say which. If
  pre-filled, intraday gaps would be invisible to FLASH's own gap handling.
- 278 one-minute returns exceed 10 σ (max 3.1 %); not investigated further in this stage.
- 3 OHLC-violation rows in the source (repaired downstream by FLASH's loader).
- Only 5.99 % of headlines link to the 10 symbols; ITC and HINDUNILVR have 35 stock-specific headlines each.
- Manifest defects listed in §9.
- FinBERT revision is not pinned by code; the snapshot actually used is recorded above (`0bc2ac10…`).
- The patch-supplied `data/raw/hf/MANIFEST.json` was updated by design (+42/−4 lines vs. the patch version).

**Not done (out of scope, awaiting approval):** real leakage audit (`run_pipeline.sh audit` /
`leakage_audit.py --config configs/real.yaml`), validation baselines, any training, any test-split scoring.

**Unresolved / to decide:**
- Whether the source pre-fills missing minutes (needs provider documentation or a raw-vs-grid comparison).
- Whether to commit the real-data reports (`news_adapt_log.json`, `market_adapt_log.json`, `quality.json`,
  this report, `MANIFEST.json`) — they contain statistics and hashes, not third-party data. `quality.json` contains
  absolute Windows paths.
- Whether to fix the manifest's stale-field merge before the next stage.

## 12. Readiness for the next audit stage

**Ready.** Both real datasets are downloaded at pinned revisions with recorded hashes, prepared without errors,
and pass FLASH's own real-data verification with zero critical issues, real FinBERT embeddings, chronological and
disjoint splits, and 24,746 news-shock rows in the test split. The leakage audit on real data can run next, once
approved (`leakage_audit.py` uses the stub encoder by design, since its checks concern timestamps, labels and
splits).

## Final safety check

| Check | Result |
|---|---|
| News source hash unchanged | ✔ `7213bcd4a42d8ea3ab95798461524bb0ca6479085ac1be2afe26e524fd42e65d` |
| Mock files used as real data | ✘ none — real stages read only `data/raw/hf/*`, `data/raw/market/`, `data/raw/news.csv`; `stub_encoder: false` |
| Model training / baseline evaluation | ✘ none — `models_real/`, `reports/real/`, `reports/baselines/real_val_baselines.json`, `data/raw/market_light/` absent; `models/` unchanged |
| Final test split scored or unlocked | ✘ no |
| `run_pipeline.sh data` run | ✘ no |
| Commit / push | ✘ none; HEAD `ffa1cda…`, nothing staged |
| Backup branch | ✔ unchanged at `ffa1cda…` |
| Source code / dependencies changed | ✘ no |
