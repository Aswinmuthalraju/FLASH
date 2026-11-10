# Stage D4: Real-Data Predictive Model Training

- **Date:** 2026-11-09
- **Repo:** `C:\Users\prsan\Projects\FLASH`
- **Branch:** `data-pipeline`
- **HEAD:** `ffa1cda72861d22ac9f2ec71e029858e582cc9d3`, with the Session 2 patch applied but not committed.
- **Outcome:** training succeeded (exit code 0). Six models were trained: two families (A and D) × three horizons (5, 15 and 30).

**What this stage did not do:**
- No test rows were selected, predicted or scored.
- No calibration was fitted.
- FinBERT was not loaded or fine-tuned.
- No source code, tests, configuration or dependencies were modified.
- Nothing was committed.

The numbers below come from `d4_training_metadata.json`, `d4_training.log`, and verification commands run after training. **This report contains no predictive-performance metrics.** The only validation number reported is LightGBM's early-stopping value; it is the stopping criterion, not a performance estimate.

---

## 1. Preflight (before D4 execution)

| # | Check | Result |
|---|---|---|
| 1 | Repo path `C:\Users\prsan\Projects\FLASH` | PASS |
| 2 | Branch `data-pipeline` | PASS |
| 3 | HEAD `ffa1cda7…` | PASS |
| 4 | `backup/pre-data-pipeline-20261109` at the same commit | PASS |
| 5 | Real market (10 parquet) and news (`data/raw/news.csv`) present | PASS |
| 6 | `configs/real.yaml` present | PASS |
| 7 | FinBERT embeddings `data/processed_real/finbert_cache.npz` present | PASS |
| 8 | Stage D2 and D3 reports present | PASS |
| 9 | Test split locked (no unlock flag used; `train.py` not used) | PASS |
| 10 | `FLASH_TEST_UNLOCK` unset | PASS |
| 11 | `models/bundle.joblib`, `models/event_memory.npz` unchanged | PASS |
| 12 | No source or config modifications. Only tracked change: `.gitignore` (Session 2 patch). Reverse-patch check exact. 40-file baseline hash set matches. | PASS |

### Why a separate driver was used

The existing `scripts/train.py` / `fit_variant()` path breaks D4's rules in three ways:
1. It predicts on the test rows and prints test macro-F1, AUC and Brier (`experiments.py` `fit_variant`/`score`; `train.py`).
2. It fits a temperature calibrator on the validation split.
3. It saves to `models_real/` rather than `models_real/stage_d4/`.

With the user's approval (Option A), an isolated driver was written instead:
- **Driver:** `reports/data_prep/stage_d4_model_training/d4_train_driver.py`
- **Final sha256:** `3d496a72b42897498634cd16ba33b4997750afe4af77b225d80404d0cda233f6`

The driver reuses the unmodified FLASH components (`prepare`, `variant_cols`, `BaselineLogistic`, `FusionBlend`, `make_bundle`, `memory_from_events`).

Before execution:
- The driver passed a static compatibility review.
- One documentation-only correction was applied: a `data_scope` metadata field plus a matching docstring paragraph.

## 2. Exact command executed

```
cd C:\Users\prsan\Projects\FLASH
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe reports/data_prep/stage_d4_model_training/d4_train_driver.py \
    > reports/data_prep/stage_d4_model_training/d4_training.log 2>&1
```

| | |
|---|---|
| Started | 2026-11-09T21:34:47+05:30 |
| Finished | 2026-11-09T21:37:45+05:30 |
| Wall time (shell) | 178 s |
| Driver-measured total | 171.8 s (`prepare()` 143.9 s) |
| Exit code | 0 |
| Environment | Python 3.11.18, LightGBM 4.7.0, scikit-learn 1.9.1, NumPy 2.4.6. CPU only. |

## 3. Training methodology

### 3.1 Data scope

This scope was accepted by the user before execution.

- **`prepare()` processes the full chronological dataset in memory.** That includes test-period rows. It computes:
  - features and triple-barrier labels for all 510,000 bars
  - HMM forward filtering
  - event forward returns

  This is inherent to the existing pipeline, because the train/val/test day split is derived from all 109 trading days.
- **Predictive model fitting uses training rows only.**
- **Validation rows are used for exactly two things:**
  - LightGBM early stopping (model D)
  - the existing `RegimePCA` component-count selection inside `prepare()` (known D2 caveat L8)
- **No test rows are used for model fitting, prediction, calibration or scoring.**
  - The driver calls `ds.rows()` only for `"train"` and `"val"`.
  - A guard subclass raises an error on any other split.
  - No `predict_proba`, `score`, `evaluate`, `fit_variant` or `Calibrator` call exists in the driver.

### 3.2 Inputs and shared components

**Data:** `data/raw/market`, which contains 510,000 bars for 10 symbols, and `data/raw/news.csv` (29,123 headlines).

**FinBERT** (`ProsusAI/finbert`, frozen):
- Embeddings were read from `finbert_cache.npz` (sha `43447937…`) through a cache-only guard encoder.
- FinBERT was never loaded (`finbert_loaded: false`).
- There were no cache misses, and the cache file is byte-identical before and after the run.

**Regime model** (existing `fit_regime`):
- PCA has 4 components, with 0.915 of variance explained on val.
- The HMM has 4 states (`calm`, `bull`, `volatile`, `bear`) and was fitted on train rows only.

**Fusion matrix:** 42 columns. The groups are:
- market (14)
- regime posteriors (4)
- news (8)
- history (3)
- derived (4 + 9)

### 3.3 Model A: standalone L2 logistic regression

- **Implementation:** the existing `BaselineLogistic()` with the same defaults as `fit_variant`.
  - scikit-learn `LogisticRegression`, lbfgs, L2, C = 1.0, max_iter = 500, tol = 1e-4, multinomial.
  - No class weights.
- **Scaling:** a `StandardScaler` fitted on train rows only.
- **Features (8, price only):** `ret1, ret5, ret15, mean20, var20, vol30, range, close_pos`.
- **Data use:** fitted on train only. Validation is not used.

### 3.4 Model D: LightGBM fusion with logistic blend

- **Implementation:** the existing `FusionBlend(cfg["fusion"], seed=7)`, which outputs P = 0.75 · P_LightGBM + 0.25 · P_logit.
- **LightGBM:**
  - `LGBMClassifier(objective=multiclass, num_class=3, n_estimators=3000, learning_rate=0.03, num_leaves=7, min_child_samples=300, reg_lambda=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=7, n_jobs=-1)`
  - No class weights.
  - Fitted on train.
  - Early stopping uses the validation `eval_set` (multi_logloss, patience 50); validation is used for nothing else.
- **Logistic leg:**
  - `BaselineLogistic(C=0.05, max_iter=300)` on `nan_to_num`-ed inputs.
  - Scaler and model fitted on train only.
- **Features:** all 42 columns, the full layout required by `Predictor`.

### 3.5 Calibration

None is fitted. Each bundle calibrator slot holds an explicit identity calibrator, `{"method": "temperature", "maps": [1.0]}`, which leaves probabilities unchanged. It exists only so the existing `Predictor` can load the bundle. All stored model probabilities are therefore **uncalibrated**.

### 3.6 Event memory

The memory keeps only events whose outcome was observable by the end of the validation period:
- **Rule:** `known_at ≤ 40874`, the tmin of the last validation bar (day index 108, minute 374).
- **Kept:** 7,923 of 10,854 linked events.
- **Excluded:** 2,931 events whose outcome falls in the test period or was never observable.

This deliberately differs from `train.py`, which stores every event.

## 4. Results by model and horizon

`n_val` is the number of validation rows used **only** for early stopping.

| Model | h | n_train | n_val (ES) | Features | Train class dist. D / N / U | Seed | Convergence / best iteration | Train time |
|---|---|---|---|---|---|---|---|---|
| A | 5 | 278,800 | 0 | 8 | 0.1881 / 0.6248 / 0.1871 | n/a (lbfgs deterministic) | converged, 13 iterations | 0.42 s |
| A | 15 | 270,600 | 0 | 8 | 0.2140 / 0.5732 / 0.2128 | n/a | converged, 15 iterations | 0.44 s |
| A | 30 | 258,300 | 0 | 8 | 0.2292 / 0.5420 / 0.2289 | n/a | converged, 14 iterations | 0.41 s |
| D | 5 | 278,800 | 91,800 | 42 | 0.1881 / 0.6248 / 0.1871 | 7 | best_iter **174** / 3000 (stopped early); logit leg converged in 40 iterations | 12.79 s |
| D | 15 | 270,600 | 89,100 | 42 | 0.2140 / 0.5732 / 0.2128 | 7 | best_iter **121** / 3000 (stopped early); logit leg converged in 49 iterations | 6.44 s |
| D | 30 | 258,300 | 85,050 | 42 | 0.2292 / 0.5420 / 0.2289 | 7 | best_iter **107** / 3000 (stopped early); logit leg converged in 46 iterations | 6.86 s |

Train class counts (DOWN / NEUTRAL / UP):
- h = 5: 52,437 / 174,202 / 52,161
- h = 15: 57,897 / 155,113 / 57,590
- h = 30: 59,193 / 139,990 / 59,117

Validation class distribution of the early-stopping set (D / N / U):
- h = 5: 0.1915 / 0.6157 / 0.1928
- h = 15: 0.2211 / 0.5562 / 0.2228
- h = 30: 0.2352 / 0.5295 / 0.2353

**Early-stopping criterion values (not performance estimates).** These are val multi_logloss at the best iteration of the LightGBM component alone, before the blend and with no calibration:
- h = 5: 0.9164
- h = 15: 0.9767
- h = 30: 0.9942

They are recorded only to document the stopping rule. They are not comparable to D3 baseline metrics, and no model-selection or tuning decision was based on them.

## 5. Warnings and errors

- **Errors:** none. The log contains no errors or tracebacks, and the exit code is 0.
- **Warnings:** 3 in total, one per D fit:
  - `LGBMDeprecationWarning: The argument 'eval_set' is deprecated, use 'eval_X' and 'eval_y' instead.` (LightGBM `sklearn.py:1106`)
  - This is a deprecation notice from LightGBM 4.7.0 about the existing `FusionLGBM.fit` call signature. Early stopping still worked, as the best iterations show. It may need attention if LightGBM is upgraded; nothing was changed now.
- **Convergence:** no `ConvergenceWarning` was raised, and all six logistic fits (three A models and three D logit legs) converged well under `max_iter`.

## 6. Model artifacts

All files are in `models_real/stage_d4/`, a new directory. `models_real/` is gitignored.

| File | Bytes | sha256 |
|---|---|---|
| `bundle.joblib` | 1,152,594 | `fd91c20e5823d20c469a3424249cee196912fa4ef2b5314c86936a9e077ad37a` |
| `event_memory.npz` | 24,466,970 | `9f2d2e397b428450d462247db5a1680eac8bdd4da502051ecb90f3b7e000f552` |
| `lgbm_D_h5.txt` | 503,793 | `57418265c0ee1fa2d9f0979104e84d40db1eefde23f3f2fc3ac63b963b9b43bf` |
| `lgbm_D_h15.txt` | 351,419 | `d286dc409d5f5bf068ac5008d634a5fefe65718867e77ae5ffbabcac2f82681a` |
| `lgbm_D_h30.txt` | 311,710 | `9d0cacd93472894a7e60efcf56a3f63af00358473a44ee20d17394b0183b16ea` |

### Bundle structure (verified by loading)

- **Top-level keys:** `A, D, cfg, col_names (42), groups, hmm (4 states), horizons [5, 15, 30], pca (P 8×4), stage_d4`.
- **`D[h]`** for each h ∈ {5, 15, 30} has keys `model, cal, cols, logit`:
  - `cols == range(42)`
  - `logit.W` is 3×42, with `w = 0.25`
  - `cal` is the identity calibrator
  - `model` is byte-identical to `lgbm_D_h{h}.txt`
- **`A[h]`** for each h has keys `W (3×8), b, cal` (identity).
- **Loading check:** `Predictor(bundle, "D")` and `Predictor(bundle, "A")` both construct successfully for horizons [5, 15, 30]. Only the constructors were run; no predictions were made.
- **`event_memory.npz`:**
  - `emb` 7,923×768, `fwd` 7,923, `known_at` 7,923
  - all values finite
  - max `known_at` = 40853, which is ≤ the cutoff of 40874 and below the first test-period tmin of 40875

### How to load the new bundle

`load_bundle()` and `load_memory()` hardcode `models_real/`, so they cannot be used. Load the files directly instead:

```python
bundle = joblib.load("models_real/stage_d4/bundle.joblib")       # trusted local artifact only (pickle)
pred = Predictor(bundle, "D")                                     # or "A"
z = np.load("models_real/stage_d4/event_memory.npz")
mem = EventMemory(bundle["cfg"]["news"]["memory_k"], z["emb"].shape[1]); mem.emb, mem.fwd, mem.known_at = z["emb"], z["fwd"], z["known_at"]
```

### Report-side files

All are in `reports/data_prep/stage_d4_model_training/`.

| File | Bytes | sha256 |
|---|---|---|
| `d4_train_driver.py` | 14,772 | `3d496a72b42897498634cd16ba33b4997750afe4af77b225d80404d0cda233f6` |
| `d4_training.log` | 1,991 | `282474855a2a03c74a967ab7135fb59134c8403500eaffb89502f3bf9f40b95d` |
| `d4_training_metadata.json` | 17,094 | `d14eddf89bdc0e0a3850616358361991f7f08e200269519447c8c1a400c9ba3d` |
| `d4_artifact_inventory.json` | 841 | `2563cd1427713eabacc56a3201e9c6d03344dffd363434b7be48551e48cb6a9a` |
| `d4_artifacts.sha256` | 515 | `2a261bbdac34d6cdd213026cd70b44a81d945d9d14eaabef2f8afb09bf370c0a` |
| `STAGE_D4_MODEL_TRAINING_REPORT.md` | (this file) | — |

**Note on `d4_artifacts.sha256`.** Python's `write_text` on Windows wrote it with CRLF line endings. `sha256sum -c` run directly on the file therefore reports "No such file" for every entry, because each path ends in `\r`. The hashes themselves are correct: `tr -d '\r' < d4_artifacts.sha256 | sha256sum -c` returns OK for all five files, and an independent `sha256sum` matches. Per the no-overwrite rule, the file was left as written.

## 7. Limitations and caveats

1. **The probabilities are uncalibrated** (identity calibrator). The bundle's `Predictor` confidence gate (0.5) will apply to raw blended probabilities. Calibration is a separate, future decision.
2. **Validation has already been used twice:** for early stopping (D) and for PCA component selection. Any later validation evaluation of these models will be optimistically biased; D/LightGBM in particular was stopped on validation.
3. **Test-period data is processed in memory.** `prepare()` computed features, labels, HMM posteriors and event forward returns for test-period rows. None of these entered fitting, prediction, calibration or scoring. Training-row features are backward-looking (D2 L1–L7), so they do not depend on test-period data.
4. **The new bundle does not work with the existing loaders.** `load_bundle()`, `load_memory()`, `scripts/demo.py`, `scripts/benchmark.py` and `app/streamlit_demo.py` all read `models_real/`, not `models_real/stage_d4/`, so the bundle must be loaded directly (§6). `benchmark.py` also streams test days and must not be run without approval.
5. **The event memory differs from `train.py`'s.** It uses the validation cutoff, not every event, which affects only slow-path news history at inference.
6. **Reproducibility is limited.** The seed is fixed (7), but LightGBM ran with `n_jobs=-1` on 32 logical CPUs, so results may not be bit-identical with a different thread count.
7. **No class weighting is used** anywhere, as implemented. NEUTRAL is the majority class at 54–62%.
8. **The `eval_set` deprecation** in LightGBM 4.7.0 (§5) is harmless now but could break on a future LightGBM version.
9. **Data caveats carried over from D1 and D2:**
   - 278 returns beyond 10σ
   - 3 OHLC violations
   - the Muhurat session dropped
   - a possibly pre-filled 375-bar grid
   - the FinBERT revision is not pinned in code (the cache sha is recorded instead)
10. **The `d4_artifacts.sha256` CRLF format** needs `tr -d '\r'` before verification (§6).
11. **No predictive performance was measured in D4.** Comparison with the D3 baselines is left to a later, separately approved stage.

## 8. Post-training integrity checks

| # | Check | Result |
|---|---|---|
| 1 | Both authorised families trained: A (L2 logit) and D (LightGBM + logit blend) | PASS |
| 2 | All three horizons: six model/horizon combinations present in the bundle and metadata | PASS |
| 3 | No FinBERT fine-tuning; FinBERT not loaded; `finbert_cache.npz` sha `43447937…` unchanged (also checked inside the run) | PASS |
| 4 | No additional predictive models (variants B and C not trained; no other estimators) | PASS |
| 5 | Test split not selected, predicted or scored. No test metrics anywhere. Event memory excludes test-period outcomes. | PASS |
| 6 | `FLASH_TEST_UNLOCK` unset | PASS |
| 7 | Original `models/bundle.joblib` and `models/event_memory.npz` unchanged | PASS |
| 8 | Real source datasets unchanged. News sha `7213bcd4…`; all D1 data hashes match. | PASS |
| 9 | FinBERT embeddings unchanged | PASS |
| 10 | Source, tests, config and dependencies unchanged. Only tracked change is `.gitignore` (patch). Reverse-patch check exact. | PASS |
| 11 | D1/D2 reports unchanged (40-file and 24-file hash sets OK). D3 JSON and log hashes match the values recorded in D3 (`f74ad79f…`, `6ded7eb0…`). D3 report last modified 19:41, before D4. | PASS |
| 12 | No commits or pushes; nothing staged | PASS |
| 13 | Branch `data-pipeline`; HEAD and backup both `ffa1cda72861d22ac9f2ec71e029858e582cc9d3` | PASS |
| — | `models_real/` contains only `stage_d4/`. `models_real/bundle.joblib`, `reports/real/` and `reports/baselines/real_val_baselines.json` absent. | PASS |
| — | All five expected artifacts exist and are non-empty; hashes verified twice (inventory and independent) | PASS |

## 9. Final Git status

- **Branch:** `data-pipeline`
- **HEAD:** `ffa1cda72861d22ac9f2ec71e029858e582cc9d3`
- **Backup branch:** same commit
- **Tracked modifications:** `.gitignore` only. Nothing staged; no commits.
- **Untracked entries:** 52 before this report, 53 including it. The D4 files (the driver plus 5 outputs plus this report) are the only additions since D3. `models_real/` is gitignored, so it does not appear.

---

**Stage D4 is complete. Stopping here.** Model selection, calibration, validation or test evaluation, D5 and frontend integration have not started and are awaiting review and explicit approval.
