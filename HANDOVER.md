# FLASH — Master Project Handover

**FLASH: Financial Language and Stock Analysis System.** The README calls it `flash-market`.

| | |
|---|---|
| Document date | 2026-11-10 |
| Repository (local) | `C:\Users\prsan\Projects\FLASH` |
| Repository (GitHub) | `https://github.com/Aswinmuthalraju/FLASH.git` |
| Working branch | `data-pipeline` (local only; it has never been pushed) |
| HEAD | `ffa1cda72861d22ac9f2ec71e029858e582cc9d3` ("V_1.2", 2026-11-05) |
| Document scope | Documentation only. No code, data, model, configuration or report was changed to produce it. |

This document should let a developer who has never seen FLASH understand:

- what was built and what was measured;
- what the results mean and what they do not mean;
- what remains to be done, and in which order;
- which rules protect the integrity of the evaluation.

It is written to stand alone. Wherever possible it points to the stage report that holds the full evidence.

> **Read this first: most of the real-data work is not on GitHub.**
> Everything done on real data so far is in the local working tree only. That covers stages D1–D5, D5.1 and E3, plus the
> code that supports them (`preprocessing/`, `evaluation/`, `training/`, `configs/`, three new test files).
> None of it has been committed, and the `data-pipeline` branch has never been pushed.
> A fresh `git clone` gives you only commit `ffa1cda`, which is the **original synthetic-data version** of FLASH.
> The real datasets, FinBERT embeddings and trained models are also git-ignored by design.
> See [§5](#5-current-technical-state) and [§10](#10-first-day-checklist-for-the-teammate).

---

## Contents

1. [Project status summary](#1-project-status-summary)
2. [What FLASH is](#2-what-flash-is)
3. [Architecture and code map](#3-architecture-and-code-map)
4. [Completed work, stage by stage](#4-completed-work-stage-by-stage)
5. [Current technical state](#5-current-technical-state)
6. [Remaining work](#6-remaining-work)
7. [Recommended order for the next developer](#7-recommended-order-for-the-next-developer)
8. [NON-NEGOTIABLE DEVELOPMENT RULES](#8-non-negotiable-development-rules)
9. [Known limitations and caveats](#9-known-limitations-and-caveats)
10. [First-day checklist for the teammate](#10-first-day-checklist-for-the-teammate)
11. [File and report directory guide](#11-file-and-report-directory-guide)
12. [Migration readiness: NOT STARTED](#12-migration-readiness-not-started)
13. [Glossary](#13-glossary)

---

## 1. Project status summary

**The project is not finished.** The real-data pipeline, the leakage audit, the baselines, the model training and the
validation-only evaluation are done. The core prediction problem is not solved in a useful form:

- The models can rank *whether* a price move will happen somewhat better than chance.
- They predict almost no *direction*.
- Under argmax they output NEUTRAL for 98.4–99.99 % of rows.

The final test evaluation has not been run.

### 1.1 Stage status

| Stage | Description | Status | Primary report |
|---|---|---|---|
| Original build | Synthetic-data FLASH: pipeline, five models, tests, demo, benchmark | COMPLETE (synthetic data only) | [README.md](README.md), [reports/REPORT.md](reports/REPORT.md) |
| Session 2 | Real-data pipeline code built and tested on mock files (Hugging Face was unreachable) | COMPLETE WITH CAVEATS | [reports/data_prep/INTEGRATION_REPORT.md](reports/data_prep/INTEGRATION_REPORT.md) |
| **D1** | Real data acquisition and preprocessing | COMPLETE WITH CAVEATS | [STAGE_D1_REAL_DATA_REPORT.md](reports/data_prep/STAGE_D1_REAL_DATA_REPORT.md) |
| **D2** | Real-data leakage audit (L1–L8) | COMPLETE WITH CAVEATS | [STAGE_D2_REAL_LEAKAGE_AUDIT_REPORT.md](reports/data_prep/stage_d2_real_audit/STAGE_D2_REAL_LEAKAGE_AUDIT_REPORT.md) |
| **D3** | Validation baselines (majority, prior, momentum) | COMPLETE | [STAGE_D3_VALIDATION_BASELINES_REPORT.md](reports/data_prep/stage_d3_validation_baselines/STAGE_D3_VALIDATION_BASELINES_REPORT.md) |
| **D4** | Real-data training of Model A and Model D | COMPLETE WITH CAVEATS | [STAGE_D4_MODEL_TRAINING_REPORT.md](reports/data_prep/stage_d4_model_training/STAGE_D4_MODEL_TRAINING_REPORT.md) |
| **D5** | Validation-only evaluation of A and D | COMPLETE WITH CAVEATS | [STAGE_D5_VALIDATION_EVALUATION_REPORT.md](reports/data_prep/stage_d5_validation_evaluation/STAGE_D5_VALIDATION_EVALUATION_REPORT.md) |
| **D5.1** | NEUTRAL-bias investigation (diagnostic) | COMPLETE WITH CAVEATS | [D5_1_NEUTRAL_BIAS_INVESTIGATION.md](reports/data_prep/stage_d5_validation_evaluation/D5_1_NEUTRAL_BIAS_INVESTIGATION.md) |
| **E3** | Two-stage (movement / direction) signal investigation | COMPLETE WITH CAVEATS | [E3_TWO_STAGE_SIGNAL_INVESTIGATION.md](reports/data_prep/stage_d5_validation_evaluation/E3_TWO_STAGE_SIGNAL_INVESTIGATION.md) |
| E8 | Late-session directional bias | NOT STARTED | — |
| E4 | Model component investigation | NOT STARTED (BLOCKED / REQUIRES APPROVAL: needs a data-isolation protocol) | — |
| E1 | Label sensitivity | NOT STARTED | — |
| E2 | Decision-rule investigation | NOT STARTED (BLOCKED / REQUIRES APPROVAL: needs a new evaluation protocol) | — |
| E5 | Calibration | NOT STARTED (BLOCKED / REQUIRES APPROVAL: needs an independent calibration window) | — |
| E6 | Retraining / class imbalance | NOT STARTED (BLOCKED / REQUIRES APPROVAL: training) | — |
| E7 | News ablation | NOT STARTED (BLOCKED / REQUIRES APPROVAL: training) | — |
| **D6** | Final test evaluation | BLOCKED / REQUIRES APPROVAL. The test split is locked. | — |
| Version control of the real-data work | Commit and push of the untracked work | PENDING (project owner's decision) | — |
| Migration readiness | Cleanup, packaging, portability, onboarding | NOT STARTED | [§12](#12-migration-readiness-not-started) |

The E-numbers were assigned in D5.1 §8 as experiment IDs. They are not an execution order; see [§7](#7-recommended-order-for-the-next-developer).

### 1.2 Results in one table (validation split, 27 trading days)

| | h5 (A / D) | h15 (A / D) | h30 (A / D) |
|---|---|---|---|
| 3-class ROC-AUC (OvR macro), D5 | 0.580 / 0.589 | 0.587 / 0.592 | 0.602 / 0.602 |
| 3-class Brier (lower is better), D5 | 0.538 / 0.536 | 0.581 / 0.579 | 0.592 / 0.590 |
| Prior-baseline Brier (the bar to beat), D3 | 0.547 | 0.593 | 0.609 |
| Share of argmax predictions that are NEUTRAL, D5 | 99.88 % / 99.99 % | 99.92 % / 99.87 % | 99.82 % / 98.40 % |
| Movement AUC (move vs. NEUTRAL), E3 | 0.594 / 0.605 | 0.606 / 0.616 | 0.629 / 0.633 |
| Direction AUC (UP vs. DOWN, moving rows only), E3 | 0.532 / 0.521 | 0.529 / 0.506 | 0.526 / 0.491 |

**In one sentence:** FLASH is modestly better than chance at telling *whether* a significant move will happen
within 5–30 minutes, and close to chance at telling *which way*. These are validation results, the validation set has
been reused several times, and the probabilities are uncalibrated. **Nothing here shows a profitable or tradable signal.**

---

## 2. What FLASH is

### 2.1 Objective and problem

FLASH is a low-latency, event-aware machine-learning system for short-horizon movement prediction on Indian equities
(NSE). For every stock at every one-minute bar it outputs:

- **P(UP), P(NEUTRAL) and P(DOWN)** for three horizons: **5, 15 and 30 bars** (one bar is one minute);
- the current **market regime** (an HMM state: calm / bull / volatile / bear);
- the **measured inference latency** of the hot loop (CPU only, microseconds).

**The question FLASH addresses:** can intraday price features, combined with financial news (sentiment, event type, and
the market's past reaction to similar events), forecast a stock's next 5/15/30 minutes better than simple baselines?
Can it do that fast enough to run on every bar?

### 2.2 Intended users and use cases

According to the README, FLASH is a **university ML course project**. It is designed around exactly five model families,
one per course unit (§3.2), with no extra models and no LLM calls.

Its intended uses are:

- research and teaching in probabilistic and sequential modelling;
- a demonstration of news-aware intraday prediction (console replay and Streamlit app);
- a latency benchmark of a CPU hot loop.

**It is not a trading system.** No profitability, transaction-cost or execution analysis has been done on real data.
The backtest numbers in the README come from **synthetic** data and the README itself says they mean nothing about
profitability.

### 2.3 Why three horizons

The 5-, 15- and 30-minute horizons cover very short, short and medium intraday reaction windows:

- 5 minutes captures immediate reactions.
- 30 minutes captures slower drift, for example after news.

Each horizon has its own model and its own label.

All label windows must end inside the same trading session. A bar whose horizon would cross the session close gets no
label. As a result, longer horizons have fewer labelled rows, and the last 30 minutes of each day are unlabelled at h30.
The primary horizon in the configuration is h15.

### 2.4 Overall prediction workflow

```
Raw 1-min bars (10 NSE stocks)      Moneycontrol headlines
        |                                    |
        v                                    v
market features (ring buffers)     entity linking -> event-type rules -> FinBERT (frozen)
  14 per-bar features                sentiment + [CLS] embedding -> event memory
        |                            (past events only: known_at < now)
        v                                    |
PCA (fit on train) -> HMM forward filter     |   news vector, exponentially decayed with age
  regime posteriors (4)                      |
        \____________________  ______________/
                             \/
               fusion vector (42 features)
                             |
          Model D: 0.75 * LightGBM + 0.25 * L2 logistic     (Model A: logistic on 8 price features)
                             |
                calibration (identity in the real-data bundle)
                             |
          argmax -> UP / NEUTRAL / DOWN, or "NO STRONG SIGNAL" if max p < 0.50 (live Predictor only)
```

**How market data and news are combined:**

- A headline is linked to a stock by dictionary/regex entity matching (`src/news/entities.py`). Headlines that match no
  configured stock but are judged market-wide are broadcast to all stocks at relevance 0.3.
- Each linked headline gets three things:
  - an event type from keyword rules (`src/news/event_rules.py`);
  - FinBERT sentiment and a 768-dimensional embedding (`src/news/sentiment.py`);
  - three history features from the event memory (`src/news/event_memory.py`): mean, std and similarity of the
    forward returns that followed the k = 5 most similar *past* events.
- A headline is attached only to bars whose open timestamp is **strictly after** its publication time.
- Its features decay as exp(−age / 30 trading minutes) and are zeroed after 240 minutes.
- All news features are concatenated with market features, regime posteriors and derived interaction features into one
  42-column fusion vector per bar.

**How UP / NEUTRAL / DOWN are produced:**

- **Labels (training targets)** come from a symmetric **triple barrier**, implemented in
  `src/features/market_features.py::triple_barrier`:
  - Barriers are set at `close · exp(±k·σ·√h)`, with k = 1.0 and σ = the std of the last 30 one-minute log returns
    (reset each day).
  - Only bar closes are checked.
  - If the upper barrier is hit first the label is UP; if the lower barrier is hit first it is DOWN; if neither is hit
    within h bars it is NEUTRAL.
  - Encoding: DOWN = 0, NEUTRAL = 1, UP = 2.
- **Predictions** are the class probabilities of the trained model.
  - The evaluation stages used plain argmax.
  - The live `Predictor` additionally applies a confidence gate: if max p < 0.50, it outputs `NO STRONG SIGNAL`.

### 2.5 Implementation status by capability

| Capability | Status | Notes |
|---|---|---|
| Synthetic data generator, feature pipeline, labels, chronological split | **Implemented** | Committed in `ffa1cda` |
| Five model families (logistic, PCA+GMM, HMM, LightGBM fusion, calibration) | **Implemented** | Committed; trained on synthetic data in `models/` |
| Hot-loop `Predictor` with latency instrumentation and confidence gate | **Implemented** | Benchmarked on synthetic data only |
| Real-data acquisition and adaptation (HF market + Moneycontrol news) | **Implemented** | Untracked code in `preprocessing/`; executed in D1 |
| Real-data leakage audit (L1–L8) | **Implemented** | `preprocessing/leakage_audit.py`; executed in D2 |
| Reference baselines and day-block bootstrap; test-split lock for baselines | **Implemented** | `evaluation/`; executed in D3 |
| Real-data trained models (A and D, three horizons) | **Implemented, with caveats** | `models_real/stage_d4/`; uncalibrated |
| Validation evaluation on real data | **Implemented** | D5 script in the reports folder |
| Demo, Streamlit app and benchmark on the real-data bundle | **Partially implemented** | Default loaders look in `models_real/`, not `models_real/stage_d4/`. These scripts also stream test days, so they must not be run without approval. |
| Probability calibration on real data | **Planned (E5)** | Identity calibrator used so far |
| Decision rules beyond argmax / gate | **Planned (E2)** | — |
| Two-stage (movement → direction) model | **Planned (E6 option)** | E3 was a diagnostic only |
| News ablation on real data | **Planned (E7)** | — |
| Final test evaluation | **Planned (D6)** | Locked |
| `autoresearch/` loop (recipe search on synthetic data) | **Experimental** | Produced the current Model D recipe; not used on real data |
| RSS live news collector (`src/data/rss_collector.py`) | **Experimental** | Feed URLs were never verified (README) |
| yfinance market loader fallback | **Experimental** | Not used on real data |
| Live trading or production deployment | **Not planned / not implemented** | — |

---

## 3. Architecture and code map

### 3.1 Directory responsibilities

| Area | Tracked in Git? | Responsibility |
|---|---|---|
| `src/data/` | yes | `market_loader.py`: CSV/Parquet loading into the canonical schema (IST, bar open time), a yfinance fallback and the synthetic generator. `news_loader.py`: news CSV loading and synthetic news. `align.py`: the trading-minute clock (`tmin`), strict news-to-bar alignment, and the chronological day split. `company_universe.py`: ticker aliases. `store.py`: Parquet/DuckDB helpers. `build.py`: synthetic dataset build. `rss_collector.py`: experimental live RSS. |
| `src/features/` | yes | `market_features.py`: batch and incremental definitions of the 14 market features (returns, rolling mean/var/cov, beta, vol30, range, close position, time of day…) and the **triple-barrier labels**. `news_features.py`: news vectors with age decay. `derived.py`: volatility-normalised returns and news×regime interactions (the `derived` and `derived_x` groups). |
| `src/news/` | yes | Slow path. `entities.py`: entity linking. `event_rules.py`: keyword event types (RBI/Rates, Earnings, M&A, Management, Regulation, Legal/Fraud, Geopolitical, Other). `sentiment.py`: `FinBertEncoder` (real) and `HashEncoder` (test stub only). `event_memory.py`: past-events-only cosine nearest-neighbour memory. |
| `src/models/` | yes | `baseline_logistic.py` (Unit 2), `regime_pca_gmm.py` (Unit 3: PCA and GMM initialisation), `hmm_regime.py` (Unit 4: GaussianHMM and the scaled forward filter), `fusion_lgbm.py` (Unit 5: LightGBM plus logistic blend), `calibration.py` (Unit 1: temperature / isotonic / Platt). |
| `src/inference/` | yes | Fast path. `feature_cache.py`: ring buffers plus the news cache (must equal the batch features exactly). `predictor.py`: single-bar `Predictor`, gate, `explain()`, plus `load_bundle()` / `load_memory()` (these read `cfg.paths.models`). `session.py`: one-day replay used by the demo. |
| `src/evaluation/` | yes | `experiments.py`: `prepare()` (builds the full dataset: features, labels, regime fit, news, history), `variant_cols`, `fit_variant` (fits **and scores the test split**). Also `metrics.py`, `ablation.py`, `backtest.py`, `latency.py`, `report.py`. |
| `src/config.py` | yes | `load_config()` (default `config/config.yaml`) and `resolve()` (repo-relative paths; **creates directories**). |
| `preprocessing/` | **no (untracked)** | Real-data pipeline. `prepare_market.py` and `prepare_news.py` adapt the two HF datasets. `hf_manifest.py` pins revisions and hashes. `verify_dataset.py` runs the data-quality audit through FLASH's own `prepare()`. `leakage_audit.py` holds checks L1–L8. `make_mock_hf.py` writes mock files. `run_pipeline.sh` is the stage runner (see [§8](#8-non-negotiable-development-rules) for dangerous stages). |
| `evaluation/` | **no** | Additions kept outside `src/evaluation/`. `baselines.py` (majority / prior / momentum), `metrics_ext.py` (extended metrics, `block_bootstrap` with B = 1000, seed 0, day blocks), and `run_baselines.py`, which contains the **only programmatic test-split lock** (`check_split_allowed`). |
| `training/` | **no** | `run_with_config.py` runs an unmodified repo script with a different config file. |
| `scripts/` | yes | Original entry points: `train.py`, `evaluate.py`, `benchmark.py`, `demo.py`. **All of them touch the test split** (see §8). |
| `configs/` | **no** | `real.yaml`: the same model settings as `config/config.yaml`, but paths redirected to `data/processed_real`, `models_real`, `reports/real`. `smoke.yaml`: mock / smoke runs. |
| `config/` | yes | `config.yaml`: original synthetic-data configuration (default for `load_config()`). |
| `tests/` | mixed | 11 original test files (tracked). `test_preprocessing.py`, `test_leakage_audit.py` and `test_baselines.py` are untracked. |
| `app/` | yes | `streamlit_demo.py`: replays one test day; needs a bundle in the configured models path. |
| `autoresearch/` | yes | Experimental recipe search on synthetic data, with a frozen harness (`prepare.py`) and one editable model file. |

### 3.2 The five model families (course-unit mapping)

| Unit | Model | Where | Role in the real-data stages |
|---|---|---|---|
| 1 Probability / Bayes | Rolling statistics; Bayes update in the HMM; calibration | `market_features.py`, `hmm_regime.py`, `calibration.py` | Features used. **Calibration not fitted** (identity). |
| 2 Linear / MLE | Multinomial L2 logistic regression | `baseline_logistic.py` | **Model A** (8 price features), and the 25 % leg of Model D |
| 3 Unsupervised | PCA plus GMM (EM) | `regime_pca_gmm.py` | PCA fit on train; component count chosen on validation (= 4) |
| 4 Sequential | Gaussian HMM (4 states), scaled forward filter | `hmm_regime.py` | Fit on train; filtered posteriors are 4 features of D |
| 5 Ensemble | LightGBM multiclass, blended 75/25 with logistic | `fusion_lgbm.py` | **Model D** (42 features) |

**FinBERT** (`ProsusAI/finbert`) is a *frozen feature extractor*, not a trained model. It must not be fine-tuned.

### 3.3 The 42-column fusion vector (Model D)

| Group | Columns | Content |
|---|---|---|
| market | 14 | `ret1, ret5, ret15, mean20, var20, vol30, cov20, beta20, mkt_ret1, mkt_vol20, range, close_pos, vol_z` (log-volume z-score), `tod` (time of day) |
| regime | 4 | HMM filtered posteriors |
| news | 11 | decayed sentiment, event-type one-hot (8), relevance, minutes since news |
| history | 3 | event-memory mean / std / similarity |
| derived | 4 | volatility-normalised returns, market-relative return (market only) |
| derived_x | 6 | news×regime / news×volatility interactions |

Model A uses 8 price features: `ret1, ret5, ret15, mean20, var20, vol30, range, close_pos`.

---

## 4. Completed work, stage by stage

Every number in this section is copied from the stage reports. For full detail and confidence intervals, read the
reports themselves.

### 4.0 Data split used by every real-data stage

| Split | Trading-day index | Dates (IST) | Days |
|---|---|---|---|
| Train | 0–81 | 2024-06-18 … 2024-10-14 | 82 |
| Validation | 82–108 | 2024-10-15 … 2024-11-25 | 27 |
| **Test (locked)** | 109–135 | 2024-11-26 … 2025-01-02 | 27 |

The split is chronological, by whole trading day, never shuffled and disjoint (D2, L1). 82/27/27 corresponds to the
configured 0.6/0.2/0.2.

Labelled rows per split and horizon (D1 §10):

| h | Train | Validation | Test | Validation DOWN / NEUTRAL / UP |
|---|---|---|---|---|
| 5 | 278,800 | 91,800 | 91,800 | 17,581 / 56,517 / 17,702 |
| 15 | 270,600 | 89,100 | 89,100 | 19,697 / 49,554 / 19,849 |
| 30 | 258,300 | 85,050 | 85,050 | 20,006 / 45,035 / 20,009 |

Only *structural* test counts (row counts and label shares) were ever computed, in D1. No test row has been predicted
or scored.

### 4.1 D1 — Real data acquisition and preprocessing · COMPLETE WITH CAVEATS

Report: [reports/data_prep/STAGE_D1_REAL_DATA_REPORT.md](reports/data_prep/STAGE_D1_REAL_DATA_REPORT.md)

**Datasets (Hugging Face, pinned revisions):**

| | Market | News |
|---|---|---|
| Repo | `xxparthparekhxx/indian-stock-market-minute-data` | `LogeshChandran/money_control_news` (Moneycontrol headlines) |
| Pinned revision | `91fc0616910dc9d92c5e34f8470dd58817529cc8` | `033afe61ca4b3ebb8879ac9097461f36f99348e1` |
| License tag on HF | MIT | MIT |
| Local source copy | `data/raw/hf/minute_subset.parquet` (sha `5d55a71e…3931`) | `data/raw/hf/money_control_news.parquet` (sha **`7213bcd4a42d8ea3ab95798461524bb0ca6479085ac1be2afe26e524fd42e65d`**) |

**The 10 configured NSE stocks:** RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, HINDUNILVR, ITC, SBIN, BHARTIARTL, KOTAKBANK.

**Market data:**

- **510,000 one-minute bars**: 10 stocks × 136 trading days × 375 bars.
- Coverage is **2024-06-18 09:15 → 2025-01-02 15:29 IST**.
- Every symbol-day has exactly 375 bars.
- The source was read with predicate push-down, so only the needed row groups were fetched. The full 10.4 GB of shards
  was not downloaded; their published LFS hashes are recorded in D1 §5.
- **10 symbol-days were dropped, all on 2024-11-01.** The source has only 18:00–18:59 bars that day (the Diwali Muhurat
  evening session). Without the drop, the loader would back-fill the 09:15–17:59 grid with a future price.

**News data:**

- 29,286 input rows. 80 undated rows were set aside, unchanged, in `data/interim/news_undated.csv`; 1 empty headline,
  70 entertainment rows and 12 duplicates were dropped. That leaves **29,123 rows**, timestamped
  2024-06-16 15:42 → 2025-01-02 15:48 (naive, treated as IST).
- The source has three timestamp layouts (`datetime` 24,832, date in the time column 4,374, no date 80).
- Mojibake was repaired with `ftfy`.

**Symbol linking:**

- Only **5.99 %** of in-window headlines link to the configured stocks: 716 stock-specific plus 1,009 market-wide
  broadcasts.
- Stock-specific counts per symbol range from RELIANCE 144 down to ITC and HINDUNILVR at 35 each.
- **127 of 136** trading days have at least one FLASH-usable article.

**FinBERT:**

- Real `ProsusAI/finbert`, snapshot `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`, run on CPU (≈ 33 min).
- Embeddings and sentiment are cached in `data/processed_real/finbert_cache.npz` (sha `43447937…f1373`).

**Verification:**

- `verify_dataset.py --config configs/real.yaml` found **0 critical issues**: no duplicates, no missing values, no
  out-of-session rows, and a 510,000 × 42 feature matrix with no NaN or inf in valid rows.

**Unresolved caveats:**

- The source is a complete 375-bar grid with almost no zero-volume bars. The provider may pre-fill gaps; this is
  unknown.
- 278 one-minute returns lie beyond 10σ (max 3.1 %); not investigated.
- 3 OHLC-violation rows (repaired by FLASH's loader).
- The FinBERT revision is not pinned in code or configuration; only the snapshot actually used is recorded.
- `MANIFEST.json` has three defects, recorded but deliberately not fixed:
  - stale `ConnectError` fields from Session 2;
  - Windows-style path keys;
  - `checked_at_utc` records the revision-resolution time, not the download time.
- Licensing: see [§9.1](#91-data).

### 4.2 D2 — Real-data leakage audit · COMPLETE WITH CAVEATS

Report: [reports/data_prep/stage_d2_real_audit/STAGE_D2_REAL_LEAKAGE_AUDIT_REPORT.md](reports/data_prep/stage_d2_real_audit/STAGE_D2_REAL_LEAKAGE_AUDIT_REPORT.md)

`preprocessing/leakage_audit.py` executes each check on the real data; none is asserted from reading code. It uses the
stub hash encoder on purpose, because the checks concern timestamps, labels and splits. **Result: 8 / 8 PASS.**

| Check | What it verifies | Result | Limitation |
|---|---|---|---|
| L1 Split integrity | Whole days, chronological, disjoint | PASS (82/27/27) | 60/20/20 proportion verified by hand |
| L2 Purge | No label window crosses its session | PASS (0 at h5/h15/h30) | Label windows only |
| L3 Label construction | Independent naive triple-barrier vs. FLASH labels | PASS (0 mismatches) | **Sample** of 9,000 labels (seed 0); σ taken from FLASH's table |
| L4 Feature causality | Features from a day prefix equal features from the full day | PASS (max diff 0.0) | **Sample** of 300 symbol-day cut-offs; market features only |
| L5 Back-fill look-ahead | No valid row carries a back-filled future close | PASS (0) | Relies on "real trade means volume > 0"; checked separately (no leading flat runs; every day starts 09:15) |
| L6 News timestamps | Every attached headline is strictly earlier than its bar | PASS (0 violations; 411,331 rows with attached news) | **Assumes** Moneycontrol times are IST first-publication times; unverifiable from the data |
| L7 Event memory | Corrupting not-yet-known outcomes cannot change earlier features | PASS (max change 0.0 before val and before test) | Ran with stub embeddings (same masking code path as FinBERT) |
| L8 Regime fit | PCA/HMM unchanged when test-split features are corrupted | PASS (identical) | **Validation chooses the PCA component count by design.** The code corrupts only test, although the docstring says "val/test". |

**Why the test split stays locked:** D2 shows that the test split is isolated from labels, features, event memory and
regime fitting. That isolation is only worth something if test results are produced **once**, after every decision is
frozen. Validation is already partly used for model selection (L8); test is the only clean window left.

### 4.3 D3 — Validation baselines · COMPLETE

Report: [reports/data_prep/stage_d3_validation_baselines/STAGE_D3_VALIDATION_BASELINES_REPORT.md](reports/data_prep/stage_d3_validation_baselines/STAGE_D3_VALIDATION_BASELINES_REPORT.md)

The baselines use market data only, and every parameter is estimated on train rows only:

- **majority:** a one-hot prediction of the most frequent train class.
- **prior:** constant train class frequencies.
- **momentum:** P(class | train tercile of z), where z = past h-bar return / (vol30·√h), with Laplace smoothing.

| h | Accuracy (all three) | Macro-F1 | Prior Brier | Momentum Brier | Momentum AUC | Majority Brier |
|---|---|---|---|---|---|---|
| 5 | 0.6157 | 0.2540 | 0.5472 | 0.5466 | 0.5187 | 0.7687 |
| 15 | 0.5562 | 0.2383 | 0.5926 | 0.5925 | 0.5079 | 0.8877 |
| 30 | 0.5295 | 0.2308 | 0.6092 | 0.6089 | 0.5118 | 0.9410 |

**Key finding:** all three baselines predict NEUTRAL for **every** row under argmax.

- NEUTRAL has the highest probability in every momentum tercile.
- Their accuracy therefore equals the NEUTRAL share, and DOWN and UP recall are 0.
- Momentum gives a small but real ranking improvement over the prior: the ΔAUC CIs exclude 0 at all horizons. Its
  Brier gain is separable only at h5.

**Why accuracy is not enough:** with 53–62 % NEUTRAL labels, a constant NEUTRAL prediction scores 0.53–0.62 accuracy
while carrying zero information. Models must be compared on proper scoring rules (Brier, log loss), ranking (AUC) and
calibration (ECE), against **prior** and **momentum**. The D3 report sets the "bars to beat": Brier below
0.5466 / 0.5925 / 0.6089.

### 4.4 D4 — Real-data model training · COMPLETE WITH CAVEATS

Report: [reports/data_prep/stage_d4_model_training/STAGE_D4_MODEL_TRAINING_REPORT.md](reports/data_prep/stage_d4_model_training/STAGE_D4_MODEL_TRAINING_REPORT.md)

**Model A** is the existing `BaselineLogistic`: L2, C = 1.0, lbfgs, multinomial, no class weights. It is fit on a
train-only `StandardScaler` over 8 price features and converged in 13–15 iterations. Validation is not used.

**Model D** is the existing `FusionBlend`: **P = 0.75 · LightGBM + 0.25 · L2 logistic (C = 0.05)** over all 42
features.

- LightGBM settings: multiclass, lr 0.03, 7 leaves, min_child_samples 300, reg_lambda 10, subsample 0.8,
  colsample 0.5, seed 7, no class weights.
- It is fit on train and early-stopped on the **validation** multi_logloss (patience 50).

| h | n_train | D LightGBM best iteration (of 3000) | Train class shares D / N / U |
|---|---|---|---|
| 5 | 278,800 | 174 | 0.188 / 0.625 / 0.187 |
| 15 | 270,600 | 121 | 0.214 / 0.573 / 0.213 |
| 30 | 258,300 | 107 | 0.229 / 0.542 / 0.229 |

**Artifacts** are in `models_real/stage_d4/` (git-ignored):

- `bundle.joblib`
  - Top-level keys: `A, D, cfg, col_names, groups, hmm, horizons, pca, stage_d4`.
  - `D[h]` contains `model` (LightGBM string), `cal`, `cols`, `logit`. `A[h]` contains `W`, `b`, `cal`.
- `event_memory.npz`: 7,923 of 10,854 linked events, keeping only outcomes known by the end of validation
  (`known_at ≤ 40874`).
- `lgbm_D_h5.txt`, `lgbm_D_h15.txt`, `lgbm_D_h30.txt`.

Hashes are in [§11.4](#114-key-artifacts-and-their-sha-256).

**Why a separate driver was needed:** `scripts/train.py` (through `fit_variant`) breaks D4's rules in three ways:

1. It **predicts and prints test-split metrics**.
2. It fits a temperature calibrator on validation.
3. It saves to `models_real/` directly.

An isolated driver, `reports/data_prep/stage_d4_model_training/d4_train_driver.py`, was approved and written instead. It
reuses the unmodified FLASH components and guards against any split other than train or val.

**Caveats:**

- **No calibration was fitted.** The identity temperature calibrator (`maps: [1.0]`) is stored, so all probabilities
  are **uncalibrated**.
- Validation was used for early stopping and for PCA component selection.
- `load_bundle()` and `load_memory()` read `models_real/`, not `models_real/stage_d4/`, so the bundle must be loaded
  directly (D4 §6).
- `prepare()` processes the **full chronology in memory, including test-period rows**: features, labels, HMM filtering
  and event forward returns. Test rows were not used for fitting, prediction, calibration or scoring.
- `n_jobs = -1` means results may not be bit-identical on another machine.
- LightGBM 4.7.0 raises an `eval_set` deprecation warning.

### 4.5 D5 — Validation-only model evaluation · COMPLETE WITH CAVEATS

Report: [reports/data_prep/stage_d5_validation_evaluation/STAGE_D5_VALIDATION_EVALUATION_REPORT.md](reports/data_prep/stage_d5_validation_evaluation/STAGE_D5_VALIDATION_EVALUATION_REPORT.md)
(supporting files: `D5_PREFLIGHT_REPORT.md`, `D5_SCRIPT_REVIEW.md`, the script `d5_validation_evaluation.py`)

**Method:**

- A reviewed, gated script loaded the D4 bundle, rebuilt the dataset with `prepare()`, and predicted **validation rows
  only**. Runtime guards raise on any other split.
- It computed accuracy, macro-F1, ROC-AUC (one-vs-rest macro), Brier, log loss and ECE (10 bins), plus per-class P/R/F1,
  confusion matrices and reliability bins.
- Uncertainty comes from a **paired day-block bootstrap**: B = 1000, seed 0, resampling the 27 validation trading days.
- Gates G1–G8 all passed. In particular:
  - G7: the baselines reproduce D3 exactly.
  - G8 (blocking): the streaming `Predictor` equals the batch pipeline to about 1e-16 on validation day 82.

**Results:**

| h | Model | Accuracy | Macro-F1 | AUC | Brier | Log loss | ECE | Argmax NEUTRAL share |
|---|---|---|---|---|---|---|---|---|
| 5 | A | 0.6156 | 0.2554 | 0.5798 | 0.5382 | 0.9198 | 0.0159 | 99.88 % |
| 5 | D | 0.6156 | 0.2541 | 0.5886 | 0.5356 | 0.9162 | 0.0158 | 99.99 % |
| 15 | A | 0.5561 | 0.2389 | 0.5867 | 0.5807 | 0.9786 | 0.0290 | 99.92 % |
| 15 | D | 0.5562 | 0.2394 | 0.5916 | 0.5787 | 0.9762 | 0.0283 | 99.87 % |
| 30 | A | 0.5296 | 0.2324 | 0.6023 | 0.5918 | 0.9942 | 0.0289 | 99.82 % |
| 30 | D | 0.5324 | 0.2477 | 0.6018 | 0.5904 | 0.9928 | 0.0259 | 98.40 % |

**Central result:**

- Both models beat every baseline on Brier, log loss and AUC at every horizon (all CIs exclude 0).
- The gains are small: Brier −0.008 to −0.019 vs. prior, and AUC rises from about 0.50–0.52 to about 0.58–0.60.
- Almost every argmax prediction is NEUTRAL. DOWN and UP recall is ≤ 0.004 everywhere, except **D at h30**: 1,084 DOWN
  calls, DOWN recall 0.021, precision 0.388.
- Both models have **worse ECE than the prior** (expected for uncalibrated probabilities).
- **D vs. A:**
  - D is better at h5 (Brier, log loss, AUC).
  - At h15 only the Brier difference is distinguishable from 0.
  - At h30 the proper-score and AUC differences cannot be distinguished from 0.

**Why this is not an independent generalisation estimate:**

- Validation already chose the PCA component count and D's early-stopping point.
- The D3 baselines were also evaluated on it.
- Only 27 day-blocks are available.

**D differs from A in model family, features and blending at once, so a D > A result does not show that news has
value.**

### 4.6 D5.1 — NEUTRAL-bias investigation · COMPLETE WITH CAVEATS

Report: [reports/data_prep/stage_d5_validation_evaluation/D5_1_NEUTRAL_BIAS_INVESTIGATION.md](reports/data_prep/stage_d5_validation_evaluation/D5_1_NEUTRAL_BIAS_INVESTIGATION.md)

This was a read-only diagnosis. It used the D5 outputs, source code, LightGBM gain importances, and a validation-only
rebuild of features and labels that reproduced D5's labels exactly.

**Findings** (each tagged [M] measured or [H] hypothesis in the report):

1. **NEUTRAL dominance is not a pipeline bug.**
   - NEUTRAL is the plurality label: 61.6 / 55.6 / 53.0 % at h5 / h15 / h30.
   - The models learned **whether** price moves (move-vs-neutral AUC 0.59–0.63) but almost nothing about **direction**
     (UP-vs-DOWN AUC 0.49–0.53).
   - Probability taken from NEUTRAL is split roughly evenly between UP and DOWN, so neither overtakes NEUTRAL.
2. **The label design makes NEUTRAL the plurality class.**
   - Under the symmetric ±1·σ₃₀·√h close-only barrier, a pure random walk already gives 46–60 % NEUTRAL (simulation).
   - Trailing σ₃₀ also overstates forward volatility: the median forward/trailing ratio is 0.87–0.97.
3. **The learned signal is volatility and time of day.**
   - `vol30` and `tod` carry 35–48 % of LightGBM gain.
   - All 14 market features have univariate UP-vs-DOWN AUC within 0.47–0.53.
4. **D's 1,084 DOWN calls at h30 are late-session move detections, not direction detections.**
   - 69 % fall in session minutes 300–345.
   - Their realised labels are DOWN 38.8 %, NEUTRAL 16.6 %, **UP 44.6 %**.
   - Precision varies widely by day (0.11–0.63).
5. **The probabilities are not sharp enough in the tails** (secondary).
   - In the lowest-P(N) decile at h30, the realised NEUTRAL share is 0.29–0.32, while the models predict 0.39–0.42.
   - Mean P(N) exceeds the validation NEUTRAL share by 1.5–2.9 pp.
6. **Class imbalance:**
   - Unweighted multinomial training is the proximate mechanism, not a defect (medium confidence).
   - With a direction signal this weak, re-weighting would mostly add false positives [H].
7. **Feature importance:**
   - LightGBM gain by group, h5 → h30: market 0.69–0.73, regime 0.03–0.07, news 0.02–0.04, history 0.01–0.04.
   - Several event-type one-hot features are never split on.
   - **This does not measure news value.**
8. **Not measurable from the stored artifacts:** the separate LightGBM and logistic components, because D5 stored only
   the blend.

**Key conclusion:** FLASH appears better at identifying *whether* a significant movement may occur than at identifying
its *direction*.

**What D5.1 does not claim:** it does **not** claim that predicting NEUTRAL is Bayes-optimal, because the probabilities
are uncalibrated.

**Limitations:**

- Validation reuse.
- 27 days.
- Gain importances are training-time measures and are not per class.
- Univariate AUCs ignore interactions.
- Non-market features were not measured univariately (that needs `prepare()`).

### 4.7 E3 — Two-stage movement and direction signal diagnostic · COMPLETE WITH CAVEATS

Report: [reports/data_prep/stage_d5_validation_evaluation/E3_TWO_STAGE_SIGNAL_INVESTIGATION.md](reports/data_prep/stage_d5_validation_evaluation/E3_TWO_STAGE_SIGNAL_INVESTIGATION.md)

**Method:** read-only. The only input was the stored D5 validation arrays (`d5_val_probabilities.npz`). There was no
model loading, no `prepare()`, no raw data and no new files except the report.

- **Stage 1 — movement:** (UP or DOWN) vs. NEUTRAL, scored with P(MOVE) = 1 − P(NEUTRAL).
- **Stage 2 — direction:** UP vs. DOWN on rows whose **actual** label was UP or DOWN, scored with
  P(UP | MOVE) = P_UP / (P_UP + P_DOWN). There were no zero-denominator rows.
- **Bootstrap:** paired day-block, B = 1000, seed 0. The same day draws were used for A and D, both stages and all
  horizons, matching the existing `block_bootstrap`. **All 1,000 replicates were valid for every metric.**

**Results:**

| | h5 (A / D) | h15 (A / D) | h30 (A / D) |
|---|---|---|---|
| Movement AUC | 0.594 / 0.605 | 0.606 / 0.616 | 0.629 / 0.633 |
| Direction AUC | 0.532 / 0.521 | 0.529 / 0.506 | 0.526 / 0.491 |
| Movement Brier skill vs. a constant validation-prevalence forecast | 2.4 % / 3.2 % | 3.0 % / 3.7 % | 4.5 % / 5.1 % |
| Days (of 27) with direction AUC > 0.5 | 24 / 19 | 18 / 13 | 16 / 10 |

**Interpretation:**

- **Movement is consistently stronger than direction.**
  - The movement-minus-direction AUC gap CI excludes 0 in all six model × horizon combinations (+0.05 to +0.17).
  - Movement AUC is above 0.5 on **all 27 validation days** for every model and horizon.
- **Movement skill is modest.** Its Brier score is only 2.4–5.1 % better than a constant forecast, and at the fixed 0.5
  threshold movement recall is low (0.05–0.38).
- **Direction is close to chance.**
  - Direction Brier is 0.249–0.252, against 0.25 for a constant 50/50 forecast.
  - A's direction CIs sit just above 0.5 (for example h5 [0.523, 0.540]).
  - D's CIs at h15 and h30 include 0.5.
- **Model D vs. Model A:**
  - D improves movement (AUC CIs exclude 0 at h5 and h15; h30 not distinguishable; Brier lower at all horizons, h30
    marginal).
  - D is **worse than A on direction at every horizon**: ΔAUC CIs h5 [−0.018, −0.002], h15 [−0.042, −0.006],
    h30 [−0.065, −0.008].
- **Horizons:** movement is strongest at **h30**; direction is strongest at **h5**, but its signal is weak.
- **Not driven by one day:** leave-one-day-out changes any pooled AUC by at most 0.014.

**Important limitations:**

- **Stage 2 conditions on the actual movement label. It is an oracle-conditioned diagnostic, not a deployable two-stage
  predictor.** A real system would have to predict movement first, and its direction quality on predicted-move rows
  could be worse.
- Validation has now been reused for PCA selection, early stopping, the D3 baselines, D5, D5.1 and E3. E3 is descriptive
  evidence, not an independent estimate.
- An AUC above 0.5 is not a trading signal. No cost or profitability analysis was done.

---

## 5. Current technical state

This state was verified on 2026-11-10 while writing this document.

| # | Item | Verified state |
|---|---|---|
| 1 | Current branch | `data-pipeline` |
| 2 | HEAD | `ffa1cda72861d22ac9f2ec71e029858e582cc9d3` ("V_1.2", 2026-11-05 18:34:51 +0530) |
| 3 | Backup branch | `backup/pre-data-pipeline-20261109` → `ffa1cda…` (same commit) |
| 4 | Tracked modifications | `.gitignore` only (+15 lines that exclude the real data, `models_real/`, smoke outputs and so on) |
| 5 | Untracked files | 63 (counted with `git status -uall`): `.gitattributes`, `configs/` (2), `data/raw/hf/MANIFEST.json`, `evaluation/` (4), `preprocessing/` (8), `training/` (2), 3 test files, `requirements-prep.txt`, `reports/baselines/SYNTHETIC_val_baselines.json`, and 40 files under `reports/data_prep/` (all stage reports and their JSON/log/npz outputs). **`HANDOVER.md` makes it 64.** |
| 6 | Staged files | none |
| 7 | New commits / pushes | none. The reflog shows only the clone (2026-11-09 15:08) and the checkout of `data-pipeline`. `origin` has only `main`, `V_1.1` and `V_1.2`; **`data-pipeline` does not exist on GitHub**. Local `main` = `origin/main` = `ffa1cda`. |
| 8 | Test-split unlock | `FLASH_TEST_UNLOCK` is **unset**. The test split has never been predicted or scored. |
| 9 | Real datasets | `data/raw/market/*.parquet` (10 files), `data/raw/news.csv`, `data/interim/news_undated.csv`, and source copies in `data/raw/hf/`. All git-ignored except `MANIFEST.json`. About 73 MB. |
| 10 | FinBERT | Embeddings: `data/processed_real/finbert_cache.npz` (93 MB, git-ignored). Stub cache from D2: `data/processed_real/hash_cache.npz`. Model weights: Hugging Face user cache (`~/.cache/huggingface/hub/models--ProsusAI--finbert`, snapshot `0bc2ac10…`). |
| 11 | Trained models | Real data: `models_real/stage_d4/` (git-ignored). Synthetic (committed): `models/bundle.joblib`, `models/event_memory.npz` (unchanged). |
| 12 | Evaluation outputs | `reports/data_prep/stage_d3_validation_baselines/` and `reports/data_prep/stage_d5_validation_evaluation/` (untracked) |
| 13 | Python and dependencies | CPython **3.11.18** in `.venv` (created with `uv`). numpy 2.4.6, pandas 3.1.1, scikit-learn 1.9.1, hmmlearn 0.3.3, lightgbm 4.7.0, torch 2.15.0+cpu, transformers 5.22.2, duckdb 2.0.0, pyarrow 26.0.1, streamlit 1.67.0, pytest 9.2.0, pyyaml 6.0.3, yfinance 1.8.0, matplotlib 3.11.2, huggingface_hub 2.4.0, datasets 5.1.0, ftfy 6.3.1, joblib 1.6.0, scipy 1.17.1. **`requirements.txt` and `requirements-prep.txt` are unpinned**, so a fresh install will get different versions. |
| 14 | Test suite | Last recorded run: **58 passed, 0 failed, 0 skipped** (41 original, 4 preprocessing, 8 leakage audit, 5 baselines), in [INTEGRATION_REPORT.md](reports/data_prep/INTEGRATION_REPORT.md) from Session 2, before any real data existed. **It has not been re-run since D1.** The README still says "39 tests" (out of date). |
| 15 | Artifact integrity | All recorded hashes still match: D1 data, news source, FinBERT cache, D4 artifacts, D5 outputs, and every stage report. See §11.4. |

**What a teammate who only clones GitHub will get:** commit `ffa1cda`. That is the synthetic-data FLASH with its
committed synthetic data (`data/processed/`, `data/flash.duckdb`) and synthetic models (`models/`). They will **not** get:

- the real-data pipeline code, configs and tests (untracked);
- any of the D1–E3 reports;
- the real datasets, FinBERT cache and trained real models (git-ignored);
- the `.venv`.

These must be transferred by an authorised method, or committed with the project owner's approval. Datasets need a
licensing check first; see §9.1.

**The repository is not currently reproducible from GitHub, and it is not migration-ready.**

---

## 6. Remaining work

None of the following has started. **Each stage needs its own scope definition, review and explicit approval before
any execution.** The descriptions below are starting points, not authorisations.

> **The validation-reuse problem shapes all remaining work.** The 27-day validation window has now informed PCA
> selection, early stopping, D3, D5, D5.1 and E3. Any experiment that *chooses* something (a label parameter, a
> threshold, a weight, a calibrator, a model variant) must make that choice on data that is not the existing validation
> set and not the test set. Two options:
>
> - a walk-forward protocol inside the 82 training days;
> - a newly reserved window, agreed in advance.
>
> The test split stays locked until D6.

### E1 — Label sensitivity investigation · NOT STARTED

- **Purpose:** find out how much of the high NEUTRAL share comes from the triple-barrier design rather than from the
  data.
- **Areas:**
  - the barrier multiplier k (for example 0.5 / 0.75 / 1.0);
  - the volatility estimator (trailing σ₃₀ that resets daily, versus EWMA or prior-day σ);
  - intrabar high/low touches versus close-only barriers;
  - asymmetric or alternative barrier configurations.
- **First step:** a descriptive study of class shares on train (and, cautiously, train+val), compared with the
  random-walk reference. It needs no training.
- **Caution:** **changing the label definition changes the prediction target.** Results under a new label are not
  comparable with D3–E3, and a new label means new baselines and new models. The label code must not be edited for
  exploration; use an in-memory analysis copy.

### E2 — Decision-rule investigation · NOT STARTED

- **Purpose:** test whether the existing probabilities support a more useful rule than argmax / max-p-gate.
- **Areas:**
  - a movement-confidence rule;
  - a direction-confidence rule;
  - combined "move and direction" rules;
  - cost-aware expected-utility rules;
  - abstention and NO SIGNAL behaviour.
- **Caution:** **thresholds must not be selected on the reused validation set.** Use a training-only walk-forward or a
  newly reserved window. E3 suggests that movement-only rules are the plausible direction; direction rules face a
  near-chance signal.

### E4 — Model component investigation · NOT STARTED · BLOCKED / REQUIRES APPROVAL

- **Purpose:** separate the contributions of the LightGBM leg, the logistic leg and their 0.75/0.25 blend.
- **Areas:**
  - component-level movement and direction AUC;
  - argmax disagreement between components;
  - whether the blend compresses probabilities;
  - feature-group univariate and multivariate contributions (all 42 features).
- **Caution:** D5 stored only the blended probabilities. Component outputs need a re-run of `prepare()` plus prediction
  with the existing D4 models (no fitting). **`prepare()` processes the full chronology, including test-period rows, in
  memory.** Agree a data-isolation protocol first (guards that refuse non-validation rows, as in D4 and D5).

### E5 — Calibration investigation · NOT STARTED · BLOCKED / REQUIRES APPROVAL

- **Purpose:** determine whether the probabilities can be calibrated reliably, and whether calibration changes any
  decision.
- **Approach:**
  - a separate, chronologically placed calibration window;
  - fit temperature or vector scaling there;
  - evaluate on a different, untouched window;
  - report reliability diagrams, ECE, Brier and log loss.
- **Caution:** **do not fit a calibrator on the existing validation set and then present results on that same set as
  independent performance.**

### E6 — Retraining and class-imbalance experiments · NOT STARTED · BLOCKED / REQUIRES APPROVAL

- **Purpose:** test whether different training improves directional discrimination without damaging movement
  detection.
- **Areas:**
  - class weighting or cost-sensitive loss;
  - an explicit two-stage architecture (movement model, then direction model);
  - model capacity (trees, leaves, regularisation);
  - walk-forward retraining.
- **Caution:** this involves model changes and retraining. It requires explicit approval, must keep the five-model
  architecture unless a change is approved, and must not overwrite `models_real/stage_d4/`.

### E7 — News ablation · NOT STARTED · BLOCKED / REQUIRES APPROVAL

- **Purpose:** measure the real contribution of news and FinBERT features.
- **Comparisons:** full D vs. D without news features, vs. D without event-memory (history) features, and possibly
  vs. D without the `derived_x` interactions. Hold everything else fixed, and evaluate on a window not used for
  selection.
- **Caution:** **do not infer news value from LightGBM feature importance, or from D vs. A.** Linkage is sparse
  (5.99 % of headlines), which limits the power of any ablation.

### E8 — Late-session directional bias investigation · NOT STARTED

- **Purpose:** understand why D's h30 DOWN calls concentrate in session minutes 300–345, and whether that pattern
  repeats.
- **Areas:**
  - late-session vs. earlier-session performance;
  - day-by-day stability of the late-session DOWN share in train vs. validation;
  - stock-wise consistency;
  - per-day call precision.
- **Feasibility:** mostly possible with existing stored outputs (`d5_val_probabilities.npz`) and training-period labels,
  read-only.

### D6 — Final test evaluation · BLOCKED / REQUIRES APPROVAL

The test split (day 109–135, 2024-11-26 … 2025-01-02) is **the only untouched, independent evaluation window**.

Before D6:

- [ ] Model choice, model configuration and features are frozen.
- [ ] The label definition is frozen (if E1 changes it, everything downstream is retrained and re-validated first).
- [ ] Calibration (or the decision not to calibrate) is frozen and was fitted without test data.
- [ ] The decision rule and thresholds are frozen and were chosen without test data.
- [ ] Metrics, baselines, the bootstrap protocol and the success criteria are agreed **in writing** before unlocking.
- [ ] The project owner explicitly authorises test access.

**The test set is used once.** If D6 results lead to changes, the test set is no longer an independent estimate, and
that must be stated in every later report.

---

## 7. Recommended order for the next developer

This is a **recommendation, not an automatic execution plan.** Every step needs its own scope, review and approval.
Do not run the stages back to back.

| Step | Stage | Why here | Depends on | When the order might change |
|---|---|---|---|---|
| 1 | Review and accept D1–D5, D5.1, E3 | Everything below builds on these findings and caveats | — | — |
| 2 | **E8** late-session behaviour | Cheapest: read-only on stored outputs. It tells you whether the only directional calls D makes come from a non-repeatable artifact. | 1 | Skip or shorten if the owner considers D5.1 §6.2 sufficient |
| 3 | **E4** components | Explains *why* D is worse than A on direction (blend? LightGBM leg?) before anyone retrains | 1; an agreed isolation protocol for `prepare()` | If E4 shows the logistic leg carries the direction signal, E6 should focus on blend weights or architecture |
| 4 | **E1** label sensitivity | Decides whether the prediction target itself should change; must precede any retraining | 1 | If the owner decides the label is fixed, E1 becomes descriptive only. If the label changes, E2/E5/E6/E7 all run on the new target. |
| 5 | **E2** decision rules | Uses the movement signal that exists (E3); needs a training-only or new-window protocol | 1, 4 (target fixed), ideally 6 (calibration) | E2 and E5 interact: thresholds on uncalibrated probabilities may need redoing after calibration. Consider designing them together. |
| 6 | **E5** calibration | Needs a calibration window and a separate evaluation window | 1, 4 | May come before E2 if the decision rule is probability-based |
| 7 | **E6** retraining | Only if E1/E4 justify it (for example a two-stage model or class weights) | 3, 4 | May be unnecessary if no direction signal exists to recover |
| 8 | **E7** news ablation | Measures news value on the final architecture | 7 (or the frozen current D) | Can run earlier on the current D if the owner wants the news answer first |
| 9 | Freeze, then **D6** | Final, one-time test evaluation | Everything above that is adopted | — |

Dependencies to keep in mind:

- Anything that changes the **label** (E1) or the **model** (E6) invalidates earlier E2/E5 results that were computed
  on the old target or model.
- Every stage that *selects* something consumes evaluation data. Plan the data budget (walk-forward folds or a reserved
  window) **before** starting E2, E5 or E6.

---

## 8. NON-NEGOTIABLE DEVELOPMENT RULES

These rules protect both the code and the credibility of every result. Breaking any of them can make the existing
evaluation, or the final test result, worthless.

1. **Preserve the existing classical-ML architecture** (the five course-unit models) unless a change is explicitly
   approved.
2. **Do not introduce, add or fine-tune any additional model** without approval. No LLM calls.
3. **Do not fine-tune FinBERT.** It is a frozen feature extractor. Its embeddings are cached; re-encoding also takes
   about 33 minutes on CPU.
4. **Do not modify the model architecture casually.** That covers the blend weights, the feature layout (the
   `Predictor` depends on the 42-column order) and the HMM/PCA design.
5. **Preserve chronological splits** by whole trading day.
6. **Never shuffle time-series data** across, or within, train, validation and test.
7. **Fit scalers, PCA, GMM and HMM only within the permitted training protocol** (train rows, plus validation only for
   the documented PCA component count).
8. **Prevent future information from entering features, labels or the event memory.**
   - News attaches only to bars strictly after its timestamp.
   - The event memory may use only outcomes with `known_at < now`.
   - Labels must never cross the session.
   - Re-run the leakage audit after any change to features or data.
9. **Keep the final test set locked until D6 is explicitly authorised.**
   - Before that approval, never set `FLASH_TEST_UNLOCK` and never pass `--unlock-test`.
   - Before that approval, no new test-label analysis, test predictions, test metrics or test scoring is permitted,
     and the test split must not be used for model selection or calibration.
   - The structural test row counts and label shares already recorded in D1 (§4.0) are historical records. They may
     stay in this document, but they do not authorise any further test analysis.
   - The purpose of this rule is to keep the test set an independent evaluation resource for D6.
10. **Do not tune on validation repeatedly and then claim independent generalisation.** Validation is already reused;
    say so in every report.
11. **Do not overwrite existing model artifacts or reports** (`models/`, `models_real/stage_d4/`,
    `reports/data_prep/**`) without approval. New work goes in new, separately named directories.
12. **Do not delete** datasets, caches, backups (including the `backup/pre-data-pipeline-20261109` branch) or model
    files without approval.
13. **Do not commit, push, rebase, reset or switch the shared branches** without the project owner's approval.
14. **Record SHA-256 hashes of important artifacts before and after every experimental stage**, and verify them.
15. **Create a separate report for every new experiment**, in its own directory, with exact commands, timings,
    versions, gates and integrity checks.
16. **Clearly distinguish measured results [M], interpretations [I] and hypotheses [H].**
17. **Never claim profitability from AUC or classification metrics.** Profitability needs a separately approved,
    cost-aware analysis.

### 8.1 Commands that are dangerous in the current state

Several existing entry points **predict or score the test split**, **overwrite artifacts** or **re-download data**.
Do not run any of them without explicit approval and a reviewed plan.

| Command | Why it is dangerous |
|---|---|
| `bash preprocessing/run_pipeline.sh data` | Re-downloads and **overwrites** `data/raw/news.csv`, `data/raw/market/`, and the manifest. The news source hash must stay `7213bcd4…e65d`. |
| `run_pipeline.sh train` / `train-light` / `python scripts/train.py` | `fit_variant` **predicts the test split and prints test metrics**. It fits a calibrator on validation and writes to `models_real/`. |
| `run_pipeline.sh evaluate` / `python scripts/evaluate.py` | Produces **test-set numbers only** (experiments, ablation, backtest) |
| `python scripts/benchmark.py`, `python scripts/demo.py`, `streamlit run app/streamlit_demo.py` | Stream or replay **test days**. Their default loaders also do not find the D4 bundle. |
| `run_pipeline.sh audit` | Also runs a new validation baseline evaluation, writing `reports/baselines/real_val_baselines.json` |
| `run_pipeline.sh verify` / `preprocessing/verify_dataset.py` | Rewrites `reports/data_prep/quality.json` (a D1 artifact) and takes about 33 minutes |
| `evaluation/run_baselines.py --split test --unlock-test` | Scores test. Forbidden until D6. |
| Anything calling `src.config.resolve()` | Silently **creates** missing directories |

**Note:** `evaluation/run_baselines.py` is the only script with a programmatic test lock. `scripts/train.py` and
`scripts/evaluate.py` have **none**.

---

## 9. Known limitations and caveats

### 9.1 Data

- **Sparse news linkage:**
  - only 5.99 % of in-window headlines link to the 10 stocks;
  - ITC and HINDUNILVR have only 35 stock-specific headlines each;
  - 9 of 136 days have no FLASH-usable article.
- **Moneycontrol licensing and redistribution:**
  - The HF dataset carries an MIT tag set by its uploader, but the content consists of Moneycontrol headlines. The
    uploader's tag does not establish redistribution rights for that content.
  - `.gitignore` deliberately prevents committing third-party data.
  - Treat both datasets as **not redistributable** until the owner confirms otherwise.
- **Market-data anomalies (D1):** 278 returns beyond 10σ, 3 OHLC violations, and the Muhurat session (2024-11-01)
  dropped.
- **Possibly pre-filled market grid:** every symbol-day has exactly 375 bars with almost no zero-volume bars. If the
  provider fills gaps, FLASH cannot see intraday gaps.
- **Timestamp semantics:**
  - Moneycontrol times are minute-resolution and naive, assumed IST, assumed first-publication. This is not verifiable.
  - A timezone error would shift every headline (D2 L6).
  - Bar timestamps are bar **open** times.
- **FinBERT revision not pinned** in `configs/real.yaml` (`ProsusAI/finbert` only). The snapshot used is recorded in D1
  (`0bc2ac10…`).
- **Manifest defects** (stale fields, Windows paths) are recorded and deliberately unfixed.

### 9.2 Evaluation

- Only **27 validation trading days** (one six-week window), so day-block CIs rest on few clusters.
- Validation has been **reused** for PCA selection, early stopping, D3, D5, D5.1 and E3.
- D4 probabilities are **uncalibrated**, and ECE is worse than the prior's.
- Almost all three-class argmax predictions are **NEUTRAL**.
- Directional discrimination is **close to chance** (AUC 0.49–0.53).
- Movement discrimination is **modest** (AUC 0.59–0.63; Brier skill 2–5 %).
- **No profitability or transaction-cost analysis** exists for real data. The README backtest is synthetic.
- No multiplicity correction across the many D5/E3 comparisons.
- The README's results and latency tables describe **synthetic** data only.

### 9.3 Technical

- D4 artifacts live in `models_real/stage_d4/`, but `load_bundle()` / `load_memory()` and the demo, benchmark and app
  read `models_real/`. Load D4 directly (D4 report §6). The bundle is a pickle: load only trusted local copies.
- **LightGBM CRLF issue:**
  - `lgb.Booster(model_file=…)` **fails** on `models_real/stage_d4/lgbm_D_h*.txt`, which have Windows CRLF line endings.
  - `lgb.Booster(model_str=open(path).read())` works, and the bundle already holds the model strings.
  - Do not rewrite the files.
- `reports/data_prep/stage_d4_model_training/d4_artifacts.sha256` has **CRLF** line endings. Verify it with
  `tr -d '\r' < reports/data_prep/stage_d4_model_training/d4_artifacts.sha256 | sha256sum -c -` from the repo root.
- `d5_outputs.sha256` uses repo-relative paths. Run `sha256sum -c` **from the repo root**.
- In `d5_val_probabilities.npz`, the `h*_timestamp_ns` arrays actually hold **microseconds** (pandas datetime64[us]).
  E3 interpreted them correctly; the key name was not changed.
- `prepare()` processes the **full chronology in memory, including test-period rows**. The authorised stages excluded
  test rows from fitting, prediction and scoring with runtime guards. Any new stage calling `prepare()` needs the same
  guards.
- LightGBM 4.7.0 `eval_set` deprecation warning: harmless now, could break on upgrade.
- Unpinned requirements; `n_jobs = -1` thread-count non-determinism.
- The `leakage_audit.py` L8 docstring says "val/test", but the code corrupts only test.

**None of these issues should be fixed casually.** Fixing them is separate, approved work.

---

## 10. First-day checklist for the teammate

1. **Get the correct repository state.**
   - Cloning GitHub gives only `ffa1cda` (synthetic FLASH).
   - The real-data code and reports exist only in the original working tree at `C:\Users\prsan\Projects\FLASH`, until
     the owner approves a commit/push or another transfer.
   - Ask the project owner which method applies.
2. **Read** [README.md](README.md) (remembering that its results are synthetic) and this `HANDOVER.md`.
3. **Verify branch and commit** (read-only):
   ```
   git branch --show-current            # expect data-pipeline
   git rev-parse HEAD                    # expect ffa1cda72861d22ac9f2ec71e029858e582cc9d3
   git rev-parse backup/pre-data-pipeline-20261109
   git status --porcelain -uall          # expect: M .gitignore, plus untracked files only
   git diff --cached --name-only         # expect: empty
   ```
4. **Review the uncommitted changes before doing anything else.** Read `git diff .gitignore` and the untracked code in
   `preprocessing/`, `evaluation/`, `training/`, `configs/` and `tests/`.
5. **Confirm the real datasets are available through an authorised transfer:**
   - `data/raw/market/` (10 parquet files)
   - `data/raw/news.csv`
   - `data/raw/hf/`
   - `data/interim/news_undated.csv`

   Respect the licensing caveat (§9.1). Do **not** re-download with `run_pipeline.sh data`.
6. **Confirm the FinBERT cache and models are present:**
   - `data/processed_real/finbert_cache.npz` (sha `43447937…f1373`)
   - `models_real/stage_d4/` (5 files)
   - the FinBERT snapshot `0bc2ac10…`, only if re-encoding is ever approved
7. **Understand the environment:**
   - Python 3.11 in `.venv`, created with `uv`.
   - Requirements are unpinned; §5 item 13 lists the versions that produced every result.
   - Do **not** install or upgrade packages in the existing `.venv`.
8. **Read the stage reports in order:** D1 → D2 → D3 → D4 → D5 → D5.1 → E3 (paths in §11.3).
9. **Run only safe, read-only integrity checks**, after reading what each command does. From the repo root:
   ```
   sha256sum data/raw/hf/money_control_news.parquet     # expect 7213bcd4a42d8ea3ab95798461524bb0ca6479085ac1be2afe26e524fd42e65d
   sha256sum -c reports/data_prep/stage_d5_validation_evaluation/d5_outputs.sha256
   tr -d '\r' < reports/data_prep/stage_d4_model_training/d4_artifacts.sha256 | sha256sum -c -
   echo "FLASH_TEST_UNLOCK=${FLASH_TEST_UNLOCK:-<unset>}"   # expect <unset>
   ```
   Compare the other files against the hashes in §11.4. Do not run the test suite, `scripts/*` or `run_pipeline.sh`
   until you have reviewed what each writes. The last recorded test run is in §5 item 14.
10. **Do not train, evaluate, calibrate or tune anything until the project owner approves the next stage** and its
    written scope.

---

## 11. File and report directory guide

### 11.1 Top level

| Path | Tracked | Purpose |
|---|---|---|
| `README.md` | yes | Original project description, architecture diagram, unit mapping, **synthetic** results |
| `HANDOVER.md` | no (new) | This document |
| `config/config.yaml` | yes | Original synthetic-data configuration (default for `load_config()`) |
| `configs/real.yaml` | no | Real-data configuration. Same model settings; paths `data/processed_real`, `models_real`, `reports/real`. |
| `configs/smoke.yaml` | no | Mock / smoke configuration |
| `requirements.txt` | yes | Runtime dependencies (unpinned) |
| `requirements-prep.txt` | no | Extra preprocessing dependencies: `huggingface_hub`, `datasets`, `ftfy` |
| `pytest.ini` | yes | Test configuration (warnings filtered) |
| `.gitignore` | modified | Excludes real data, `models_real/`, smoke outputs |
| `.gitattributes` | no | `*.sh text eol=lf` |
| `preprocessing/` | no | Real-data adapters, verification, leakage audit, stage runner |
| `training/` | no | `run_with_config.py` |
| `evaluation/` | no | Baselines, extended metrics, bootstrap, test lock |
| `src/` | yes | Core FLASH package (see §3.1) |
| `scripts/` | yes | `train.py`, `evaluate.py`, `benchmark.py`, `demo.py` (all touch test; see §8.1) |
| `tests/` | mixed | 14 test files (3 untracked) |
| `app/` | yes | `streamlit_demo.py` |
| `autoresearch/` | yes | Experimental recipe search on synthetic data |

### 11.2 Data and models

| Path | Tracked | Content |
|---|---|---|
| `data/raw/hf/` | `MANIFEST.json` only (untracked) | Source copies `money_control_news.parquet`, `minute_subset.parquet`, and `MANIFEST.json` (pinned revisions and hashes) |
| `data/raw/market/` | ignored | 10 per-symbol parquet files (real, FLASH schema) |
| `data/raw/news.csv` | ignored | 29,123 adapted headlines |
| `data/interim/news_undated.csv` | ignored | 80 undated headlines set aside |
| `data/raw/*_mock*`, `data/raw/hf_mock/`, `data/processed_smoke/` | ignored | Mock / smoke data. **Never train on these.** |
| `data/processed_real/` | ignored | `finbert_cache.npz` (real FinBERT), `hash_cache.npz` (stub, from D2) |
| `data/processed/`, `data/flash.duckdb` | yes | **Synthetic** data from the original build |
| `models/` | yes | **Synthetic** bundle and event memory (unchanged; do not overwrite) |
| `models_real/stage_d4/` | ignored | **Real-data D4 models**: `bundle.joblib`, `event_memory.npz`, `lgbm_D_h{5,15,30}.txt` |

### 11.3 Reports

The original synthetic-data reports are in `reports/` (`REPORT.md`, `results.json`, CSVs, PNGs, all tracked) and
`reports/baselines/SYNTHETIC_val_baselines.json`. The real-data work is in `reports/data_prep/` (untracked):

| Stage | Path |
|---|---|
| Session 2 (pre-real-data) | [PROGRESS_REPORT.md](reports/data_prep/PROGRESS_REPORT.md), [INTEGRATION_REPORT.md](reports/data_prep/INTEGRATION_REPORT.md), [REAL_DATA_AUDIT.md](reports/data_prep/REAL_DATA_AUDIT.md) (status: not executed; superseded by D1), [LEAKAGE_AUDIT.md](reports/data_prep/LEAKAGE_AUDIT.md) (mock; superseded by D2), [TRAINING_READINESS.md](reports/data_prep/TRAINING_READINESS.md) (written before data existed) |
| D1 | [reports/data_prep/STAGE_D1_REAL_DATA_REPORT.md](reports/data_prep/STAGE_D1_REAL_DATA_REPORT.md), plus `quality.json`, `market_adapt_log.json`, `news_adapt_log.json` |
| D2 | [reports/data_prep/stage_d2_real_audit/STAGE_D2_REAL_LEAKAGE_AUDIT_REPORT.md](reports/data_prep/stage_d2_real_audit/STAGE_D2_REAL_LEAKAGE_AUDIT_REPORT.md), plus `leakage_audit_real.json` / `.log` |
| D3 | [reports/data_prep/stage_d3_validation_baselines/STAGE_D3_VALIDATION_BASELINES_REPORT.md](reports/data_prep/stage_d3_validation_baselines/STAGE_D3_VALIDATION_BASELINES_REPORT.md), plus `validation_baselines_real.json` / `.log` |
| D4 | [reports/data_prep/stage_d4_model_training/STAGE_D4_MODEL_TRAINING_REPORT.md](reports/data_prep/stage_d4_model_training/STAGE_D4_MODEL_TRAINING_REPORT.md), plus `d4_train_driver.py`, `d4_training.log`, `d4_training_metadata.json`, `d4_artifact_inventory.json`, `d4_artifacts.sha256` |
| D5 | [reports/data_prep/stage_d5_validation_evaluation/STAGE_D5_VALIDATION_EVALUATION_REPORT.md](reports/data_prep/stage_d5_validation_evaluation/STAGE_D5_VALIDATION_EVALUATION_REPORT.md), plus [D5_PREFLIGHT_REPORT.md](reports/data_prep/stage_d5_validation_evaluation/D5_PREFLIGHT_REPORT.md), [D5_SCRIPT_REVIEW.md](reports/data_prep/stage_d5_validation_evaluation/D5_SCRIPT_REVIEW.md), `d5_validation_evaluation.py`, `.log`, `d5_validation_metrics.json`, `d5_run_metadata.json`, `d5_val_probabilities.npz`, `d5_outputs.sha256` |
| D5.1 | [reports/data_prep/stage_d5_validation_evaluation/D5_1_NEUTRAL_BIAS_INVESTIGATION.md](reports/data_prep/stage_d5_validation_evaluation/D5_1_NEUTRAL_BIAS_INVESTIGATION.md) |
| E3 | [reports/data_prep/stage_d5_validation_evaluation/E3_TWO_STAGE_SIGNAL_INVESTIGATION.md](reports/data_prep/stage_d5_validation_evaluation/E3_TWO_STAGE_SIGNAL_INVESTIGATION.md) |

**`d5_val_probabilities.npz`** is the main input for any read-only follow-up (E8, decision-rule exploration). It has
10 arrays per horizon h:

- `h{h}_row_index`, `_y`, `_day_idx`, `_symbol`
- `_timestamp_ns` (actually µs)
- `_P_A`, `_P_D`, `_P_majority`, `_P_prior`, `_P_momentum` (columns DOWN, NEUTRAL, UP)

It contains **validation rows only** (day_idx 82–108).

### 11.4 Key artifacts and their SHA-256

All values were verified on 2026-11-10.

| Artifact | SHA-256 |
|---|---|
| `data/raw/hf/money_control_news.parquet` | `7213bcd4a42d8ea3ab95798461524bb0ca6479085ac1be2afe26e524fd42e65d` |
| `data/raw/hf/minute_subset.parquet` | `5d55a71e0791f6a3d59205cb767f9f401a3d528edc4dd20a8fc63f21763d3931` |
| `data/raw/news.csv` | `87ea1c4b21f5b682189d0a6f3e7fe002347696bb51629012a0c64646ba3ba47e` |
| `data/processed_real/finbert_cache.npz` | `43447937a34ba869603ca6552e222d31d2da35e26af8eba634c943e3a89f1373` |
| `models_real/stage_d4/bundle.joblib` | `fd91c20e5823d20c469a3424249cee196912fa4ef2b5314c86936a9e077ad37a` |
| `models_real/stage_d4/event_memory.npz` | `9f2d2e397b428450d462247db5a1680eac8bdd4da502051ecb90f3b7e000f552` |
| `models_real/stage_d4/lgbm_D_h5.txt` | `57418265c0ee1fa2d9f0979104e84d40db1eefde23f3f2fc3ac63b963b9b43bf` |
| `models_real/stage_d4/lgbm_D_h15.txt` | `d286dc409d5f5bf068ac5008d634a5fefe65718867e77ae5ffbabcac2f82681a` |
| `models_real/stage_d4/lgbm_D_h30.txt` | `9d0cacd93472894a7e60efcf56a3f63af00358473a44ee20d17394b0183b16ea` |
| `…/stage_d5_validation_evaluation/d5_validation_metrics.json` | `29f3c9aa9ee4a8e2ebc71aac5dab4d2ca562fbc7527630fe37333f589c79bcb8` |
| `…/stage_d5_validation_evaluation/d5_run_metadata.json` | `4511a185edf8e908aef6cb9ec887e1a50d4441e5b99b23ad18ec104e96115f63` |
| `…/stage_d5_validation_evaluation/d5_val_probabilities.npz` | `ef8704cb32709837d2e610d88cb54abbc9885158d2f6cbaf144ba604d35f40fb` |
| `…/stage_d5_validation_evaluation/d5_validation_evaluation.py` | `0cf317ef82aaf92a3d13dee99d9d23d9687588d10029b57a5ae92d1ecaeec0db` |
| `…/stage_d4_model_training/d4_train_driver.py` | `3d496a72b42897498634cd16ba33b4997750afe4af77b225d80404d0cda233f6` |
| `STAGE_D1_REAL_DATA_REPORT.md` | `9f3598f2d358b290f4b9cb6b59b215bdd99fa6d20ace105daf46e04a5855a1ca` |
| `STAGE_D2_REAL_LEAKAGE_AUDIT_REPORT.md` | `85b146a26809fdd27ad4adb8fdebd8d6b44e63f7665d7c281ce0afd24fdb6fdd` |
| `STAGE_D3_VALIDATION_BASELINES_REPORT.md` | `3fcc3fae65ccdf968e59ac5be3fe91bafa4b7d9bf4fad187c10aa5520c1f9a78` |
| `STAGE_D4_MODEL_TRAINING_REPORT.md` | `4ff8beeafa2f3ea3b00265682dc4904ebf381e72895846a633361625eca84a21` |
| `STAGE_D5_VALIDATION_EVALUATION_REPORT.md` | `f047f1a031ea20d7032e2bfb846d22fea0014eb225e9b81489ab6f976003b90e` |
| `D5_1_NEUTRAL_BIAS_INVESTIGATION.md` | `6cf62b6763cdf6852abad9b51ad03aa69df4d70983f959b1227b5bf9119df8ad` |
| `E3_TWO_STAGE_SIGNAL_INVESTIGATION.md` | `4ebc6df208ce07a833d965c987b931eeb4debdc9aa235eec5f835300f7bdc28b` |

The per-file hashes of the 10 market parquet files are in D1 §5.

---

## 12. Migration readiness: NOT STARTED

**Migration readiness has not started. The current task is only to document the project and its continuation plan.**

Migration readiness is a separate, future phase. It will be planned only after the remaining technical work is complete
and approved, and it will cover:

- repository cleanup;
- Git history and branch strategy, including what to do with the untracked real-data work and `data-pipeline`;
- dependency and environment reproducibility (pinned versions, lock file);
- dataset transfer and licensing;
- model artifact packaging;
- configuration portability;
- removal of absolute local paths (for example in `quality.json` and the run metadata) and consistent relative path
  handling;
- secrets and environment-variable management;
- reproducible setup instructions;
- test and validation procedures;
- teammate onboarding;
- GitHub synchronisation;
- backup and recovery;
- final handover verification.

None of these activities has been performed. Nothing in this document should be read as saying the repository is
portable, reproducible or ready for migration.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| Bar | One-minute OHLCV record; its timestamp is the bar's **open** time (IST) |
| h5 / h15 / h30 | Prediction horizons of 5, 15 and 30 bars (minutes) |
| Triple barrier | Labelling rule: UP or DOWN if the ±k·σ·√h barrier is first hit within h bars, else NEUTRAL |
| `tmin` | Trading-minute clock: `day_index × 375 + minute_of_session`; skips nights and weekends |
| Model A | L2 logistic regression on 8 price features |
| Model D | 0.75 · LightGBM + 0.25 · L2 logistic over 42 fusion features |
| Movement AUC (E3 Stage 1) | AUC of 1 − P(NEUTRAL) for (UP or DOWN) vs. NEUTRAL |
| Direction AUC (E3 Stage 2) | AUC of P(UP) / (P(UP) + P(DOWN)) for UP vs. DOWN, on rows whose *true* label is UP or DOWN (oracle-conditioned) |
| Day-block bootstrap | Resampling whole trading days with replacement (B = 1000, seed 0) to get CIs that respect intraday correlation |
| Brier | Mean squared error of probabilities. The 3-class version is summed over classes; lower is better. |
| ECE | Expected calibration error (10 bins) |
| Gate | Live `Predictor` rule: output `NO STRONG SIGNAL` if max probability < 0.50 |
| [M] / [I] / [H] | Measured / interpretation / hypothesis tags used in the stage reports |
