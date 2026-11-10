# FLASH — Real-data pipeline: progress report (pre-training review)

> **Superseded in part (9 Nov 2026, second session)** — see INTEGRATION_REPORT, REAL_DATA_AUDIT, LEAKAGE_AUDIT,
> TRAINING_READINESS. Changes: the `--infer-date-from-id` option was **removed** (undated rows now kept aside);
> `verify` no longer runs without real data (`verify-mock` added); a back-fill look-ahead was found and handled
> in `prepare_market.py`; baselines + bootstrap CIs are now implemented; test count is 58.


Branch: `data-pipeline` (original code untouched; all additions are new files plus a `.gitignore` append).
Status: Phases 1–7 done up to the stopping condition. **No full-scale training was run.**
Session date: 9 Nov 2026.

> **Biggest caveat first.** The sandbox this was built in could reach GitHub but **not huggingface.co**
> (blocked by the egress proxy). So the two datasets were researched and inspected through the HF web
> viewer, and the whole pipeline was validated on *mock files built to the exact HF schemas and defects*,
> not on the real downloads. Step 1 of the review is to run `bash preprocessing/run_pipeline.sh data` then
> `verify` on a machine with HF access and read the real numbers before authorising training.

---

## 1. Repository technical audit

| Item | Finding |
|---|---|
| Commits | 2 (`V_1.1`, `V_1.2`); 36 MB incl. committed artefacts |
| Size | ~3.1k lines across `src/`, `scripts/`, `autoresearch/` |
| Tests | **41/41 pass** on Python 3.12 (README says 39; two were added since) |
| Executable? | Yes. `torch`/`transformers` are imported lazily, only when FinBERT must encode a headline not in the cache |
| Committed artefacts | `models/bundle.joblib`, `models/event_memory.npz`, `data/processed/*.parquet` (225,000 synthetic bars, 60 days, 1,797 headlines), `finbert_cache.npz`, `reports/*` |
| Data actually used so far | **Synthetic only** (regime-switching random walk + templated headlines with injected reactions). The README says so explicitly |
| Real-data entry points | `scripts/train.py --market <dir|file> --news <csv>`; loaders in `src/data/market_loader.py`, `src/data/news_loader.py` |

**What FLASH is (and is not).** The brief assumed a neural network to be trained or fine-tuned. FLASH is a
*classical ML pipeline* built for a university ML course ("exactly five models, no LLM calls"):

- **Problem:** short-horizon direction of 10 NSE large caps (RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK,
  HINDUNILVR, ITC, SBIN, BHARTIARTL, KOTAKBANK).
- **Input:** 1-minute OHLCV bars (IST, 09:15–15:29, 375 bars/day) + timestamped news headlines.
- **Output:** per stock per bar, calibrated P(UP/NEUTRAL/DOWN) at 5, 15, 30 bars, the market regime, and the
  measured latency; a confidence gate emits "NO STRONG SIGNAL" when max p < 0.50.
- **Labels:** triple barrier, `close·exp(±k·σ·√h)`, k = 1, σ = rolling 30-bar std.
- **Hardware:** CPU-only inference (~150 µs median per bar for the full model). FinBERT is the only
  component that benefits from a GPU, and only offline.

## 2. Architecture

Slow path (offline/async): headline → keyword event-type rules (8 types) → dictionary entity linker → frozen
**FinBERT** (`ProsusAI/finbert`) sentiment ∈ [−1, 1] + [CLS] embedding → **historical-event memory** (top-5
cosine neighbours among past events whose outcomes were already known) → per-symbol news cache.

Fast path (per bar, NumPy): ring-buffer market features → **PCA** (components chosen on validation, ≥90%
variance) → incremental **HMM** forward step (4 states, initialised by a **GMM** via EM) → feature vector
[market | regime posteriors | decayed news | history | derived] → **LightGBM** (blended 75/25 with an L2
**logistic** leg) → **temperature scaling** → gate.

Experiments A–D: A logistic on price features; B LightGBM market; C B + HMM; D C + news + history.

**Implication for "training":** nothing is trained from scratch with epochs and an optimizer in the deep-
learning sense. FinBERT is frozen (feature extractor). Training = fitting PCA/GMM/HMM (EM) on train,
boosting LightGBM with early stopping on validation, fitting the logistic leg (L-BFGS) and the temperature
on validation. That changes what Phase 5 means, and the plan below reflects it.

## 3. Hugging Face dataset comparison

FLASH has a hard compatibility requirement most datasets fail: **minute-level** bars for **these 10 stocks**
*and* **minute-timestamped** headlines for the **same period**.

| Dataset | What it is | Granularity / period | Licence tag | Fit for FLASH |
|---|---|---|---|---|
| **xxparthparekhxx/indian-stock-market-minute-data** | ~720M rows, 2,500+ NSE stocks/indices; cols symbol, timestamp (UTC), OHLCV, oi; ~10.5 GB parquet shards sorted by symbol | 1-min, 2022-01-03 → 2026-01-21; daily 2000–2026 | MIT (uploader) | **Selected (market).** Exact schema match after UTC→IST; covers all 10 symbols |
| **LogeshChandran/money_control_news** | 29.3k Moneycontrol articles: url, title, subtitle, content, datetime | minute timestamps; 115 distinct dates, viewer shows ~Jun 2024 → Jan 2025 with gaps | MIT (uploader) | **Selected (news).** Only candidate found with minute timestamps in a window the bars cover |
| kdave/Indian_Financial_News | 26k Indian articles: URL, content, T5 summary, GPT sentiment label | **no timestamp column** | none stated | Rejected: cannot be aligned to bars at all |
| raeidsaqur/NIFTY (NIFTY-LM) | News-headline → market-direction dataset for LLM research | daily | — | Rejected: built around **US** headlines/index; daily; not NSE |
| thetrademarkk/india-index-options-1m | 1-min NIFTY/BANKNIFTY/SENSEX spot + options | 1-min, ~2021–2026 | not checked | Rejected for training (indices, not the 10 stocks). Possible future market-context feature |
| IITKGP/Nifty500_companies_5_years_data, vishnun0027/indian-market-historical-ohlcv | NSE daily OHLCV | daily | — | Rejected: wrong granularity (FLASH horizons are 5–30 minutes) |
| Traders-Lab/TroveLedger, paperswithbacktest/Stocks-1Min-Price | 1-min bars | US/EU | — | Rejected: wrong market |
| times_of_india_news_headlines | General Indian headlines | date only | — | Rejected: no intraday time |
| SEntFiN 1.0 (via tahp0604/finbert-sentfin) | 10.7k human-labelled Indian financial headlines (ET/Moneycontrol, 2002–2017) | no intraday time | not verified | **Supporting only** (see §4) — HF hosting of the dataset itself not verified |

## 4. Selected datasets and justification

| Role | Choice | Why |
|---|---|---|
| Primary training | minute bars (xxparthparekhxx) ⨝ Moneycontrol news, over the news window | The only combination that satisfies FLASH's actual input contract |
| Supporting | SEntFiN (optional) | FinBERT was trained on non-Indian text; SEntFiN can *measure* how well its sentiment transfers to Indian headlines. Swapping in a fine-tuned FinBERT would add a model, which the course spec forbids — so it's an ablation for the team to decide on, not a default |
| Validation | **chronological slice** (next 20% of trading days) of the same joined data | FLASH needs prices and news for the same minutes; a *different* dataset as "validation" would test distribution shift, not generalisation. Early stopping + calibration + any re-tuning use this only |
| Independent test | **final 20% of trading days**, touched once | Never used for tuning. Optional extra: a market-only out-of-period check (experiments A–C on 2025 bars with no news) for the regime/market legs |

Leakage controls inherited from FLASH and kept: day-level chronological splits, never shuffled; scalers/PCA/
GMM/HMM fit on train only; news attaches only to bars strictly after its timestamp; event memory uses only
already-observed outcomes. Hyper-parameters were found on *synthetic* data, so they carry no information
about the real test period.

**Licence caveat (needs a team decision).** Both "MIT" tags were set by the uploaders, but the content is
third-party (prices apparently from Upstox; articles are Moneycontrol's copyright). An uploader's tag cannot
grant rights they don't hold. For a course project this is normally acceptable as research use, but: don't
redistribute the raw data (already enforced via `.gitignore`), cite both HF pages, and use headlines only.

## 5. Preprocessing methodology

New files (nothing in the original repo was edited except a `.gitignore` append):

```
preprocessing/prepare_market.py   HF minute shards -> data/raw/market/<SYMBOL>.parquet
preprocessing/prepare_news.py     HF news -> data/raw/news.csv
preprocessing/verify_dataset.py   Phase-4 audit through FLASH's own loaders + prepare()
preprocessing/make_mock_hf.py     offline mock in the exact HF schemas (smoke tests only)
preprocessing/run_pipeline.sh     reproducible stages: smoke | data | verify | train-light | train | evaluate
training/run_with_config.py       runs the UNMODIFIED scripts/*.py against configs/*.yaml
configs/real.yaml, configs/smoke.yaml   same model settings; only output paths differ
tests/test_preprocessing.py       3 tests for the adapters
requirements-prep.txt             huggingface_hub, datasets, ftfy (runtime requirements unchanged)
```

Raw downloads are preserved untouched in `data/raw/hf/`; every adapter writes a JSON log of row counts
(`reports/data_prep/*_adapt_log.json`).

**Market — each step and why**

1. Keep the 10 configured symbols — FLASH's entity dictionary and universe are fixed to them.
2. UTC → Asia/Kolkata — the dataset is UTC; FLASH's session grid is IST (03:45 UTC = 09:15 IST, tested).
3. Restrict to the news window — outside it, news features would be zeros that mean "no data", not "no news".
4. Drop `oi` (always 0 for cash equities); float32 → float64 to avoid rounding noise in ~1e-4 log-returns.
5. **Deliberately not done here:** OHLC repair, dedup, forward-filling missing minutes, dropping sparse days.
   FLASH's own `load_market` already does these, and duplicating them would hide what it changes. The
   verifier reports their counts instead.

**News — each step and why**

1. Coalesce three timestamp layouts seen in the viewer: full `article datetime`; null datetime with the full
   date-time inside `article time`; only "HH:MM" with no date. The last kind is **dropped by default**.
   `--infer-date-from-id` can fill dates from the monotonic article id, but only when both neighbours agree,
   and is off by default because a wrong date silently mis-aligns news and prices.
2. `ftfy` repair of mojibake (e.g. "â€™" → "'"), which otherwise becomes junk FinBERT sub-tokens.
3. Headline = `title` (FLASH is headline-level; body text unused).
4. Dedup on article URL, then (timestamp, headline).
5. Exclude the `entertainment` URL section only. **No keyword filtering for the 10 companies** — FLASH's
   `EntityLinker` decides relevance, and pre-filtering would hide its errors from evaluation.
6. Timestamps written as naive IST; `load_news` localises them.

Class imbalance: **not resampled.** The triple-barrier labels are roughly balanced by construction (σ-scaled
barriers), LightGBM sees the natural distribution, and resampling a time series breaks its ordering. Imbalance
is reported per split and handled at evaluation time (macro-F1, per-class metrics).

## 6. Dataset quality verification

Validated on the HF-shaped mock (20 days × 10 symbols + 700 news rows with injected defects). Results in
`reports/data_prep/smoke_mock/`.

| Check | Result on mock |
|---|---|
| Injected defects caught | 31/31 undated rows, 15/15 duplicate URLs, 19/19 entertainment rows, 0 residual mojibake |
| Out-of-universe symbol / pre-open rows | removed (500) / reported (5) |
| Duplicate (symbol, ts), OHLC violations, NaNs | 1 / 0 / 0, all handled by FLASH's loader |
| Day coverage before forward-fill | mean 97%, min 93.6% (3% minutes deleted on purpose) |
| Feature matrix via FLASH `prepare()` | 75,000 × 42, **0 NaNs in model-ready rows** |
| PCA / HMM | 3 components (val variance 0.902) / 4 states named bull, calm, volatile, bear |
| Splits | train days 0–11, val 12–15, test 16–19: chronological and disjoint |
| h=15 label mix | train 25/44/31 %, test 21/39/40 % (DOWN/NEUTRAL/UP) |
| News-shock rows in test | 2,872 (so D vs B is testable) |
| `train.py` + `evaluate.py` end-to-end | ran (17 s train, stub encoder), wrote to `models_smoke/`, `reports/smoke/`; committed `models/` untouched |
| Test suite | **44/44 pass** |

`verify_dataset.py` exits non-zero on critical issues: missing symbols, split overlap, NaNs in feature rows,
zero news-shock rows in test, or < 50 stock-specific headlines.

**Risks the real-data run must answer** (cannot be measured on the mock):

- **News volume per stock.** Moneycontrol covers thousands of companies plus politics and gadgets. Of ~29k rows
  maybe only a few hundred to low thousands will name one of the 10 stocks. If `stock_specific` is small,
  experiment D has little to learn from.
- **Coverage gaps.** 115 distinct dates (weekends included) across ~7 months means many trading days have no
  scraped news. On those days "no news" is a scraping artefact. Recommended fix to review: restrict training
  to days whose article count is above a threshold. This is a justified filter, but it's a judgement call, so
  I've left it for the team.
- **Linker false positives on real text.** "SBI" matches "SBI Mutual Fund", "Infosys" matches "Infosys founder
  Narayana Murthy on coaching classes". The geopolitical rule ("border", "tariff", "war") will broadcast many
  political headlines to all 10 stocks. Plan: hand-audit 100 linked headlines and report precision.
- **Reactive headlines.** Many titles report a move that already happened ("shares rise 6%"). That's causal at
  the timestamp (no leakage) but means news features may mostly encode momentum.
- **Timestamp semantics.** If Moneycontrol shows *updated* rather than *published* time, news arrives late in
  our data. That's conservative (no look-ahead) but understates the news effect.

## 7. Final dataset statistics

Real-data counts are **pending the first HF download**. Expected scale, so the review can sanity-check:
10 symbols × ≈80–140 trading days × 375 bars ≈ 0.3–0.5 M bar rows (synthetic set: 225k rows, 60 days);
news 29.3k rows in, minus undated/duplicate/excluded rows, with the stock-specific subset the key number.
Exact figures will land in `reports/data_prep/news_adapt_log.json`, `market_adapt_log.json` and `quality.json`.

## 8. Training configuration

Keep `config/config.yaml` model settings **unchanged** for the first real run (copied verbatim into
`configs/real.yaml`, only paths differ). Rationale: changing them before seeing a real baseline would be
tuning blind, and every later change must be justified on validation, never test.

| Brief's item | What it means for FLASH | Setting |
|---|---|---|
| From scratch / fine-tune / transfer | Fit classical models from scratch; FinBERT used as frozen transfer features | — |
| Optimizer | LightGBM gradient boosting; EM for GMM/HMM; L-BFGS for logistic | library defaults |
| Learning rate | LightGBM shrinkage | 0.03 |
| Batch size | n/a for boosting; FinBERT inference batch | 32 |
| Epochs | Boosting rounds, capped, early-stopped | ≤ 3000, stop after 50 without val improvement |
| LR scheduler | n/a (constant shrinkage) | — |
| Loss | multiclass log-loss; calibration minimises val NLL | — |
| Regularisation | small trees, large leaves, L2, row/col subsampling, L2-logistic | leaves 7, min_child 300, λ 10, subsample 0.8, colsample 0.5, C 0.05 |
| Checkpointing | one bundle per run (`models_real/bundle.joblib` + event memory); FinBERT outputs cached by headline hash | — |
| Early stopping | val multi-logloss | 50 rounds |
| Validation frequency | every boosting round (built into LightGBM) | — |

**Lightweight config (first real experiment):** same settings, first ~6 weeks of the news window
(`run_pipeline.sh train-light`). Purpose: see real FinBERT throughput, real label balance, and whether D beats
B at all, in minutes.

**Scalable config (full run):** the full news window; then, only if validation supports it, a small re-tune of
`num_leaves`, `min_child_samples`, `k` (barrier width) and `gate_threshold` on validation. Extending the market
history beyond the news window is possible for A–C only.

## 9. Hardware requirements (estimates — confirm on the light run)

| Resource | Need |
|---|---|
| CPU | any modern 4+ core; all model fitting is CPU |
| GPU | optional; only speeds up the one-off FinBERT pass |
| GPU memory | ~1–2 GB if used (BERT-base, batch 32, max length 64) |
| RAM | 8 GB comfortable (≈0.5M rows × 42 float64 features ≈ 170 MB, plus HMM/LightGBM overhead) |
| Disk | ~20 GB free during download (minute shards are ~1.5 GB each; filtered reads still pull whole row groups), ~0.5 GB FinBERT weights, < 200 MB processed |
| Time | FinBERT on CPU for a few thousand to ~25k headlines: roughly 5–15 min once, then cached. Train + evaluate after caching: minutes (the 20-day mock trained in 17 s) |

## 10. Proposed training workflow

1. `pip install -r requirements.txt -r requirements-prep.txt`
2. `bash preprocessing/run_pipeline.sh smoke` — offline plumbing check.
3. `bash preprocessing/run_pipeline.sh data` — download + adapt; read the news log, set `START`/`END` to the
   actual coverage, rerun the market step if needed.
4. `bash preprocessing/run_pipeline.sh verify` — must exit 0. **Team review of `reports/data_prep/`.**
5. `train-light` → inspect A vs B vs D on validation only.
6. Decide the coverage-day filter and any validation-based re-tuning; record each decision in this report.
7. `train` (full) → `evaluate` once on test.

## 11. Evaluation methodology

Already implemented in the repo and reused unchanged: macro-F1, ROC-AUC (one-vs-rest), Brier score,
calibration plots, confusion matrices, the **news-shock subset** (bars ≤ 30 min after news), ablation
(drop news/history/HMM/PCA), latency benchmark, and a cost-aware backtest.

Additions recommended before results are reported (not implemented yet):

- **Trivial baselines:** majority class and class-prior probabilities (their Brier is the bar to beat).
  Real 1-minute markets are close to unpredictable; without these, an AUC of 0.55 is hard to interpret.
- **Uncertainty:** day-level block bootstrap CIs for each metric and for the D − B difference; with ~16–28
  test days, point estimates alone aren't defensible.
- **Error analysis:** per symbol, per event type, per HMM regime, and by time of day; hand-audited linker
  precision on 100 headlines.
- **Ablations:** existing set, plus FinBERT vs SEntFiN-tuned FinBERT if the team allows it.
- **Discipline:** the test split is evaluated once, after every decision is frozen.

## 12. Commands

```bash
git checkout data-pipeline
pip install -r requirements.txt -r requirements-prep.txt
python -m pytest -q                                   # 44 tests
bash preprocessing/run_pipeline.sh smoke              # offline, mock data
bash preprocessing/run_pipeline.sh data               # needs huggingface.co access
bash preprocessing/run_pipeline.sh verify
bash preprocessing/run_pipeline.sh train-light        # after review
bash preprocessing/run_pipeline.sh train              # after authorisation
bash preprocessing/run_pipeline.sh evaluate
```

## Decisions needed at the review

1. Accept the two datasets given the licence caveat (§4)?
2. Coverage-day filter for news gaps: yes/no, and threshold (§6)?
3. Keep the dated-only news default, or enable `--infer-date-from-id`?
4. Allow the SEntFiN FinBERT ablation, given the five-model constraint?
5. Implement the trivial baselines and bootstrap CIs before the full run (recommended)?
