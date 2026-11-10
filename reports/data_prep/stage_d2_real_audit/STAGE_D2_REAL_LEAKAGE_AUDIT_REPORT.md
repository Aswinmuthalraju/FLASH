# Stage D2 — Real-Data Leakage Audit

Date: 9 Nov 2026 · Session 3 · Branch `data-pipeline` (uncommitted) · Nothing committed or pushed.

Audit-only stage. No predictive model was fitted, no predictions were made, no metric was computed, and the
test split was not scored. Values below were observed in this run unless marked *Interpretation* or listed under
*Unresolved*.

---

## 1. Preflight results

| Check | Result |
|---|---|
| Working directory | `C:\Users\prsan\Projects\FLASH` |
| Branch / HEAD | `data-pipeline` / `ffa1cda72861d22ac9f2ec71e029858e582cc9d3` |
| Working tree vs. end of Stage D1 | unchanged (same 41 status entries; nothing staged) |
| `backup/pre-data-pipeline-20261109` | `ffa1cda72861d22ac9f2ec71e029858e582cc9d3` (original base) |
| `data/raw/market/` | present, 10 parquet files (hashes = Stage D1) |
| `data/raw/news.csv` | present, 7,742,119 B, SHA-256 `87ea1c4b…` (= Stage D1) |
| `data/raw/hf/MANIFEST.json` | present (= Stage D1) |
| `configs/real.yaml` paths | `raw: data/raw`, `processed: data/processed_real`, `models: models_real`, `reports: reports/real`; the word "mock" does not occur in the file |
| Moneycontrol source SHA-256 | `7213bcd4a42d8ea3ab95798461524bb0ca6479085ac1be2afe26e524fd42e65d` ✔ expected value |

Before the audit, SHA-256 hashes of 24 files were recorded as an integrity baseline: market parquets, `news.csv`,
`news_undated.csv`, `minute_subset.parquet`, `MANIFEST.json`, `finbert_cache.npz`, and every JSON/MD report in
`reports/data_prep/`. The two files in `models/` were hashed too.

## 2. Interface inspection and side effects

- `run_pipeline.sh audit` was **not** used: after the audit it also runs `evaluation/run_baselines.py ... --split val`,
  a validation baseline evaluation, which is prohibited in this stage. The audit script was called directly.
- `leakage_audit.py --help`: `[--market] [--news] [--config] [--out]`. All four were passed explicitly, so no
  default path was used.
- Files the script writes, as determined from the code:
  1. `--out`: the JSON report. Pointed at the new directory; the default
     `reports/data_prep/leakage_audit.json` was not touched and does not exist.
  2. `prepare_base(...)` → `encode_cached(...)` writes the stub-encoder cache to
     `resolve(cfg,"processed")/hash_cache.npz` = `data/processed_real/hash_cache.npz`. That file did **not** exist
     before, so it was created, not overwritten. It is git-ignored and kept separate from `finbert_cache.npz`.
     The stub is `HashEncoder` (MD5 bag-of-words); FinBERT was not loaded and its cache was not modified.
- What the audit computes internally: FLASH's feature table and labels; news-event alignment; event-memory history
  features; and two unsupervised regime fits (PCA + GMM-initialised HMM) for L8. It does **not** fit LightGBM,
  logistic regression or the calibrator, and generates no predictions.
- It does not rerun news/market preparation, does not read `data/raw/hf/*`, and does not touch the network.

## 3. Exact command executed

```
.venv/Scripts/python.exe preprocessing/leakage_audit.py \
    --market data/raw/market --news data/raw/news.csv --config configs/real.yaml \
    --out reports/data_prep/stage_d2_real_audit/leakage_audit_real.json
```

Run from the repo root via Git Bash with `PYTHONIOENCODING=utf-8`. Exit code **0**, 892 s.
stdout and stderr were captured in `leakage_audit_real.log`, together with the command, start/finish time and exit
code.

Data used: **real** market data (`data/raw/market/`, 10 symbols, 510,000 bars, 2024-06-18 → 2025-01-02 IST) and
**real** news (`data/raw/news.csv`, 29,123 headlines), configured by `configs/real.yaml`. No mock path was referenced.

## 4. Individual results L1–L8

Summary: **8 / 8 PASS**, including 1 PASS with an assumption that was checked separately (L5) and 1 PASS with a
design caveat (L8). The interpretation of each check is below. `all_pass: true` alone is not treated as evidence.

### L1 — Chronological split integrity · **PASS** · real data
- **Tested:** for every calendar day in the real feature table, all rows carry one split label, splits occur in
  train → val → test order, and all three are present.
- **Result:** whole days ✔, chronological ✔. Train has 82 days (2024-06-18 … 2024-10-14), val 27 days
  (2024-10-15 … 2024-11-25), test 27 days (2024-11-26 … 2025-01-02). 82/27/27 of 136 days =
  60.3 / 19.9 / 19.9 %, matching the configured 0.6 / 0.2 / 0.2.
- **Limitations:** the split is by trading day; the code does not check that it is 60/20/20 (it was checked by hand
  above).

### L2 — Purge / separation between splits · **PASS** · real data
- **Tested:** for each horizon (5, 15, 30 bars), whether any labelled row's forward window `[t, t+h]` extends past
  its own session.
- **Result:** windows crossing a session: h5 = 0, h15 = 0, h30 = 0.
- **Interpretation:** label windows end inside their own session, and splits are whole days (L1), so no label can
  look into the next split. The purge is structural and no extra embargo gap is needed for labels.
- **Limitations:** covers label windows only. Information carried across the boundary by features (overnight state,
  event memory, regime) is covered by L4, L7 and L8.

### L3 — Label construction and future-target boundaries · **PASS** · real data (sampled)
- **Tested:** an independent, naive loop re-implementation of the triple-barrier label (threshold
  `k·σ·√h`, first barrier touched within `h` bars, otherwise NEUTRAL; `−1` where the window or σ is unavailable),
  compared with FLASH's labels.
- **Result:** 9,000 labels checked (3,000 random rows × 3 horizons, RNG seed 0) — **0 mismatches**.
- **Limitations:** this is a sample (≈ 2 % of 448–462k labelled rows per horizon), not exhaustive. σ (`vol30`) is
  read from FLASH's table, not recomputed independently, so σ's own causality relies on L4 and the unit tests
  (`test_labels_use_only_the_future_features_use_only_the_past`).

### L4 — Feature causality · **PASS** · real data (sampled)
- **Tested:** for a random symbol-day and a random cut-off bar after warm-up, market features at the cut-off are
  computed once from the full day and once from only the day's prefix up to the cut-off. Any difference would mean
  the feature uses later bars.
- **Result:** 300 symbol-day/cut-off pairs; max abs difference prefix vs. full = **0.0**.
- **Limitations:** sample-based (300 of 1,360 symbol-days). Covers the market features from `batch_day_features`;
  news features and regime posteriors are covered by L6/L7/L8. `mret`, the equal-weight universe return, uses all
  10 symbols' bars at the same minute; it is known at that bar's close, so it is contemporaneous, not look-ahead.

### L5 — Future-price backfill / look-ahead · **PASS (assumption verified separately)** · real data
- **Tested:** for each symbol-day, the first bar with `volume > 0` is located. If it is after the 30-bar warm-up,
  the feature rows marked valid before it would carry a back-filled future close (FLASH's loader does
  `close.ffill().bfill()`).
- **Result:** valid rows with a back-filled future price = **0**; symbol-days affected = **0**.
- **Assumption:** "real trades have volume > 0". Stage D1 found the source to be a complete 375-bar grid, so I
  checked whether a back-fill could exist that the volume test would miss. Read-only check on `data/raw/market/`:
  - symbol-days starting with a run of ≥ 2 flat (o=h=l=c) bars: **0**; every symbol-day's first bar is 09:15.
  - zero-volume bars: 22 in total, all mid-session (session minutes 52, 176, 253–254, 351, 372), none at the open.
  - flat bars equal to the previous close: 1,004 of 510,000, all intraday.
  - *Interpretation:* there is no sign of a leading-gap back-fill. Any provider filling is intraday and
    backward-looking (it repeats a past price), which is not look-ahead.
- **Related:** the only late-open days in the window (all 10 symbols on 2024-11-01, the 18:00–18:59 Muhurat
  session) were removed in Stage D1 by `prepare_market.py`. L5 confirms nothing similar remains.

### L6 — News publication timestamp alignment · **PASS** · real data
- **Tested:** after FLASH's own `align_news_to_bars` (max age 240 min), every headline attached to a bar must have
  a timestamp **strictly earlier** than the bar's timestamp.
- **Result:** bar rows with an attached news event: 411,331; events linked to the 10 symbols: 10,854;
  **violations: 0**.
- **Limitations / assumptions:**
  - The check is only as good as the source timestamps. Moneycontrol times are minute-resolution, naive, and
    treated as IST. Whether they are first-publication or last-update times cannot be verified from the dataset.
    If they were update times, news would appear *later* than it was really published — conservative, not a leak.
    A timezone error, however, would shift every headline by a fixed offset. This is unresolved.
  - 4,374 headlines take their timestamp from the `article time` column (one of three source layouts).
  - Entity linking does not depend on the encoder, so the stub encoder does not affect this check.

### L7 — Event-memory boundaries across splits · **PASS** · real data (stub embeddings)
- **Tested:** at the first bar of val and of test, every event outcome not yet known at that boundary is replaced
  with random noise. The event-memory history features of all events *before* the boundary are then recomputed and
  compared.
- **Result:** max abs change for events before val = **0.0**; before test = **0.0**.
- **Limitations:** neighbour similarity in the memory uses the stub hash embeddings, not FinBERT. The property
  tested — only outcomes with `known_at` before the query are visible — is a structural property of
  `history_features` and does not depend on which embedding is used. Neighbour sets with FinBERT would differ, but
  the masking logic is the same code path.

### L8 — PCA/HMM regime fitting restricted to training data · **PASS (design caveat)** · real data
- **Tested:** the regime (PCA + HMM) is fitted twice — on the real table and on a copy whose **test-split** regime
  features are replaced with random noise.
- **Result:** PCA components identical ✔; max abs difference of train-row posteriors = **0.0**.
- **Caveats:**
  1. Only the **test** split is corrupted. The module docstring says "val/test", but the code corrupts only test.
  2. By design, the **validation** split is used to choose the number of PCA components
     (`RegimePCA(...).fit(X[train], X[val])`). It is also used later for early stopping and calibration. So regime
     fitting is restricted to *train + validation for model selection*, not train alone. Consequence: validation
     metrics will be somewhat optimistic. The test split is untouched.
  3. HMM parameters are not compared directly; they are covered indirectly, because identical train posteriors
     require identical parameters.

## 5. Overall conclusion

All eight leakage checks pass on the real 10-symbol, 136-day dataset with zero violations. The test split is
isolated from labels, features, event memory and regime fitting. The one structural caveat is that validation data
informs PCA component selection by design (L8), which makes future validation metrics optimistic but does not
affect the test split. On the evidence of this audit, the real dataset is free of the leakage classes L1–L8.

## 6. Warnings, limitations and unresolved risks

- **Sampled checks:** L3 (9,000 labels) and L4 (300 symbol-day cut-offs) are random samples (seed 0), not
  exhaustive.
- **L5 depends on volume:** it relies on the volume heuristic, which the separate read-only check supports for this
  data.
- **L6 is only as good as the source timestamps:** IST and first-publication semantics are assumed and unverified.
- **L7 used stub embeddings.**
- **L8:** validation participates in model selection; the docstring/code mismatch is noted above.
- **New 93 MB file:** `data/processed_real/hash_cache.npz` was created (stub cache, git-ignored).
- **Previously reported defects are unchanged:** the manifest's stale `ConnectError` fields and Windows-style keys,
  and the possibly pre-filled 375-bar grid.
- **Out of scope here and not run:** validation baselines, headline ablation, entity-linking analysis, training,
  and test-split scoring.

## 7. Integrity checks (before → after)

| Check | Result |
|---|---|
| Moneycontrol source SHA-256 | `7213bcd4…e65d` before and after ✔ |
| 24 baselined files (real market, news, subset, manifest, FinBERT cache, all Stage D1 / Session 2 reports) | all hashes unchanged ✔ |
| `models/bundle.joblib`, `models/event_memory.npz` | hashes unchanged; `git status models/` empty ✔ |
| Mock data used | no — only `data/raw/market`, `data/raw/news.csv`, `configs/real.yaml` |
| Training / baselines / predictions / metrics | none — `models_real/`, `reports/real/`, `reports/baselines/real_val_baselines.json`, `data/flash_real.duckdb` absent |
| Test split locked | yes — `check_split_allowed` requires `--unlock-test` **and** `FLASH_TEST_UNLOCK=I_HAVE_APPROVAL`; the variable is unset and `run_baselines.py` was not invoked |
| Source code, tests, configs, dependencies | unchanged — reverse-applying the Session 2 patch matches all 31 code/test/config/requirements files byte-for-byte (`MANIFEST.json` excluded; it was updated by design in Stage D1); no tracked file changed except `.gitignore` (the patch's append); `.gitattributes` unchanged |
| Commit / push / branch / reset | none |

## 8. Git status

- HEAD: `ffa1cda72861d22ac9f2ec71e029858e582cc9d3` on `data-pipeline`; `origin/main` is also `ffa1cda…`;
  nothing staged.
- `backup/pre-data-pipeline-20261109`: `ffa1cda72861d22ac9f2ec71e029858e582cc9d3`, unchanged.
- `git status --short`: the 41 Stage D1 entries plus the new, untracked `reports/data_prep/stage_d2_real_audit/`.

## 9. Files created in Stage D2

| File | Bytes | Tracked by Git? |
|---|---|---|
| `reports/data_prep/stage_d2_real_audit/leakage_audit_real.json` | 1,959 | untracked |
| `reports/data_prep/stage_d2_real_audit/leakage_audit_real.log` | 2,312 | untracked |
| `reports/data_prep/stage_d2_real_audit/STAGE_D2_REAL_LEAKAGE_AUDIT_REPORT.md` | this file | untracked |
| `data/processed_real/hash_cache.npz` | 93,000,144 | git-ignored (side effect of the audit's stub encoder) |
