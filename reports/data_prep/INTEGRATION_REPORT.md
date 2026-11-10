# INTEGRATION_REPORT — data-pipeline branch

Date: 9 Nov 2026 · Nothing committed or pushed (awaiting approval).

## Repository state before integration

| Item | Value |
|---|---|
| Fresh clone of | `https://github.com/Aswinmuthalraju/FLASH.git` |
| Branch / commit at clone | `main` @ `ffa1cda72861d22ac9f2ec71e029858e582cc9d3` (V_1.2), working tree clean |
| Upstream changed since previous session? | No — same base commit the patch was built on |
| Backup checkpoint | branch `backup/pre-data-pipeline-20261109` (= `ffa1cda`) |
| Work branch | `data-pipeline` (created from `ffa1cda`) |

## Source of truth

The uploads folder in this session was **empty** — the three files the brief names were not attached. I used
the previous session's own outputs, still present in the environment:
`0001-Real-data-pipeline-HF-adapters-dataset-verifier-conf.patch`, `FLASH-data-pipeline-additions.zip`,
`PROGRESS_REPORT.md`. Please confirm these are the same files you meant.

## Comparison and conflicts

- Patch and ZIP list the identical 17 paths. After `git apply`, every file was **byte-identical** to the ZIP (`diff -r`).
- `git apply --check`: clean. Only pre-existing path touched: `.gitignore` (append-only).
- Original code untouched: `git diff` shows no change under `src/ scripts/ app/ autoresearch/ models/ data/ config/`,
  `requirements.txt`, `README.md` or any original test.

## Changes made in this session (on top of the integrated patch), with reasons

| File | Change | Why |
|---|---|---|
| `preprocessing/verify_dataset.py` | cache path now from `resolve(cfg, "processed")` | **Bug fix.** Hard-coded `data/processed_smoke/` only existed by accident last session; on a fresh clone the smoke stage crashed (`FileNotFoundError`). It also ignored `--config`. |
| same | per-symbol stats, tz/chronology, inf check, labelled rows per horizon, news coverage gaps, encoding, categories, `--out-dir` | Phase-4 checklist items |
| `preprocessing/run_pipeline.sh` | `verify` refuses to run without real data (exit 3); new `verify-mock`, `audit`, `audit-mock` | Brief ran `verify` after `smoke`; the stage pointed at real paths. Deliberately **no silent fallback to mock** so mock results can never pass as real |
| `preprocessing/prepare_news.py` | **removed** date-from-article-id inference; undated rows written unchanged to `data/interim/news_undated.csv` | Brief: never infer dates; keep undated separate |
| `preprocessing/prepare_market.py` | drop symbol-days whose first trade is after 09:45 | **Leakage fix** (see LEAKAGE_AUDIT L5) without editing FLASH's loader |
| both adapters | `--log-dir`; revision pinning via new `preprocessing/hf_manifest.py` | mock logs no longer overwrite real ones; exact HF revision + file sizes + sha256 recorded |
| `preprocessing/leakage_audit.py` (new) | 8 executable checks | Phase 5 |
| `evaluation/` (new package) | baselines, extended metrics, day-block bootstrap, test-split lock | Phase 6 (kept out of `src/evaluation/`) |
| `tests/test_leakage_audit.py`, `tests/test_baselines.py` (new) | 13 tests incl. positive controls | |
| `tests/test_preprocessing.py` | date-inference test replaced by undated-set-aside test (feature removed on instruction); late-open test added; one fixture made realistic | the old fixture's single 15:29 bar is, correctly, now dropped as a late-open day |
| `.gitignore` | `data/raw/hf/*` with `!MANIFEST.json`, `data/interim/` | keep the tiny manifest, exclude data |

Diff vs. the previous patch: preprocessing 398+/53−, test_preprocessing 27+/14−; new: evaluation (195 lines),
leakage_audit (221), hf_manifest (55), 2 test files (180).

## Excluded from Git (verified with `git status --ignored`)

`data/raw/hf/*` (except manifest), `data/raw/market*/`, `data/raw/news*.csv`, `data/raw/hf_mock/`, `data/interim/`,
`data/processed_real|smoke/`, `models_real/`, `models_smoke/`, `reports/smoke/`, `__pycache__/`.
Committed synthetic artefacts in `models/` and `data/processed/` are unchanged (`git diff --quiet` passes).

## Test results (executed)

| Suite | Result |
|---|---|
| Original 41 tests | 41 pass |
| `test_preprocessing.py` | 4 pass |
| `test_leakage_audit.py` | 8 pass |
| `test_baselines.py` | 5 pass |
| **Total** | **58 passed, 0 failed, 0 skipped** |

Pipeline stages: `smoke` ✅ · `verify-mock` ✅ (0 critical issues) · `audit-mock` ✅ (all 8 checks pass) ·
`verify` ⛔ exit 3 (no real data, by design) · `data` ⛔ HF unreachable (see REAL_DATA_AUDIT).
