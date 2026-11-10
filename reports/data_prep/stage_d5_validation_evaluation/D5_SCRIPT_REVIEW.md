# D5 Script Review: Validation-Only Evaluation Driver

- **Date:** 2026-11-09
- **Script:** `reports/data_prep/stage_d5_validation_evaluation/d5_validation_evaluation.py` (536 lines)
- **Status:** created and statically reviewed. **NOT executed.** No predictions, metrics or probability files exist.

## 1. Files created in this stage

| File | Purpose |
|---|---|
| `d5_validation_evaluation.py` | evaluation driver |
| `D5_SCRIPT_REVIEW.md` | this review |

Already present from the D5 preflight: `D5_PREFLIGHT_REPORT.md`.

The script's sha256 is recorded in the chat summary. Any later edit must be re-reviewed.

## 2. Architecture and workflow

Gates G1–G8 each abort **before any metric is computed**.

**Failure handling** (revised in the final review): every failure after G1 is recorded through `_fail()` in `d5_gate_failure.json`. The record includes `status`, `failure_type`, the message, the full traceback, and all gate diagnostics collected so far. The exit codes are:

| Exit code | Meaning |
|---|---|
| 2 | Gate failure before metrics |
| 3 | Any other error before metrics |
| 4 | Error during metrics or bootstrap. Any `d5_val_probabilities.npz` present is then a partial-run output. |

G1 failures (unlock variable set, or an output already exists) stop the run with `SystemExit` before anything is written, so an existing file is never overwritten.

| Step | What happens | Code |
|---|---|---|
| G1 | Abort if `FLASH_TEST_UNLOCK` is set. Abort if any output file already exists (no overwrite). | `main` |
| G2 | sha256 of all 5 D4 artifacts must equal `d4_artifact_inventory.json`. Repeated at the end of the run. | `gate_artifacts` |
| — | `bundle = joblib.load("models_real/stage_d4/bundle.joblib")` directly; `load_bundle()` is not used. | `main` |
| G3 | Bundle structure and parameters checked against D4 metadata (detail below). | `gate_bundle` |
| — | `get_data` → unmodified `prepare()` with cache-only FinBERT guard; FinBERT cache sha checked before and after. `ds.__class__ = _NoTestDataset`. | `main` |
| G4 | Refitted PCA (`mean, scale, P, c, clip`) and HMM (`startprob, transmat, means, covars, names`) must equal the bundle (`np.allclose`, rtol 1e-9, atol 1e-12). `col_names` and `groups` must be identical. | `gate_regime` |
| G3b | Model-A feature list (`variant_cols(ds,"A")`) equals the D4 metadata list. | `main` |
| G5 | Per horizon: `va = ds.rows("val", h)` must give exactly 91,800 / 89,100 / 85,050 rows and the expected class supports. Every row must be `split=="val"`, `valid_x`, label ≥ 0, inside the val day range, over 27 days, with sorted unique indices. | `gate_val_rows` |
| — | Batch probabilities: A, D, and the three baselines (formulas in §3). | `main` |
| G6 | Non-finite feature counts recorded per feature (A and D). Every probability array (A, LGBM, logit leg, D, three baselines) checked for shape, NaN, inf, negative, >1 and \|sum−1\|>1e-6. Any invalid array aborts. Nothing is replaced or hidden. | `finiteness`, `prob_check` |
| G7 | Baselines recomputed with the unmodified `Baselines().fit(t,h)` / `.predict(t,va)`. Point metrics must match the D3 JSON to 1e-12 (n, macro-F1, AUC, Brier, ECE, accuracy, log loss); per-class values, confusion matrix and reliability must match exactly; `describe()` must equal D3's `fit_on_train`. The baselines' val rows must equal `va`. | `compare_d3` |
| G8 | One-day streaming cross-check (§5). | `stream_check` |
| — | Save `d5_val_probabilities.npz` (validation rows only) **before** the bootstrap. | `main` |
| Metrics | For A and D per horizon: `full_report` (accuracy, macro-F1, per-class P/R/F1/support, AUC, Brier, log loss, ECE, confusion, reliability), prediction distribution, and paired bootstraps (§4). The three baselines get `full_report` and prediction distribution. | metrics loop |
| End | G2 re-check; FinBERT cache re-check; write metrics JSON, metadata JSON, and sha256 file (LF line endings). | `main` |

### G3 bundle checks, per horizon
- `cfg == configs/real.yaml`; horizons `[5, 15, 30]`; 42 columns.
- D model:
  - `cols == range(42)`; feature names equal the D4 metadata.
  - The model string in the bundle is byte-equal to `lgbm_D_h{h}.txt`.
  - `Booster.current_iteration()` equals D4's `best_iteration`; `num_feature == 42`.
  - logit `W` is 3×42, `b` is 3, and `w == fusion.blend_logit_weight` (0.25).
- A model: `W` is 3×8 and `b` is 3.
- Both A and D calibrators are the identity.

## 3. Probability reconstruction (consistent with D4)

| Model | Formula in the script | Existing implementation it mirrors |
|---|---|---|
| A | `softmax(X[:, cols_A] @ W_A.T + b_A)`. `W`, `b` already fold in the train-fitted scaler. Softmax is `exp(z−max)/sum`. | `BaselineLogistic.predict_proba` (identical up to floating point) and the `Predictor` A path |
| D | `(1−w)·Booster(model_str).predict(X) + w·softmax(nan_to_num(X) @ W.T + b)` with w = 0.25 | `FusionBlend.predict_proba`. The model string is truncated at `best_iteration`, the same as `FusionLGBM.predict_proba(num_iteration=best)`. |
| Baselines | `Baselines().fit(t,h).predict(t,va)` → majority, prior and momentum, **kept separate** | `evaluation/baselines.py` (D3) |

- **Label order:** DOWN=0, NEUTRAL=1, UP=2, the same as `CLASS_NAMES`, LightGBM, scikit-learn and D3.
- **No calibration:** the identity calibrator is not applied in batch, since it is a no-op, and no `Calibrator` is constructed.
- **NaN handling:** `nan_to_num` appears only in D's logistic leg, exactly as in training. NaN counts are reported (G6). NaNs in A features would make P_A NaN, so G6 aborts rather than hiding them.

## 4. Paired day-block bootstrap

- **Code:** the existing `block_bootstrap(P, y, day, B=1000, seed=0, P_ref=…)` gives CIs for macro-F1, AUC, Brier, ECE and accuracy, plus `diff_vs_ref_ci` (model − ref).
  - Within one call, the model and the reference use the **same** resampled indices, so the comparison is paired.
  - Every call re-seeds with 0, so every comparison uses the same 1000 day draws.
- **Comparisons:**
  - A vs majority, prior and momentum
  - D vs majority, prior, momentum and **model A**
  - 7 comparisons × 3 horizons = **21 paired comparisons**
- **Log loss:** `block_bootstrap` has no log loss. `logloss_bootstrap` adds it, reusing `block_bootstrap`'s exact sampling procedure (`default_rng(0)`, then `rng.choice(days, 27, replace=True)` once per replicate).
  - Alignment is **verified at runtime**: the Brier CI recomputed inside `logloss_bootstrap` must equal `block_bootstrap`'s Brier CI to 1e-12, or the run aborts.
  - Per-row loss terms are the same terms whose mean `full_report` and `brier_multiclass` compute.
- **Alignment:** every array (P_A, P_D, three baselines, y, day) is built from the same `va` index array. Lengths are asserted before bootstrapping, and the baselines' val rows are asserted equal to `va` (G5b).
- **Direction:** lower is better for Brier, log loss and ECE; higher is better for AUC, macro-F1 and accuracy. Differences are always model − reference.
- **Multiplicity:** 21 comparisons × 6 metrics = 126 difference CIs, with no correction applied. These are **exploratory** and labelled that way in the output (`multiple_comparisons`).
- **Baseline CIs:** standalone baseline CIs are not recomputed. The D3 JSON values apply, and identical point metrics are verified by G7.

### Bootstrap API, confirmed from source (final review)

`evaluation/metrics_ext.py` (untracked, added by the Session 2 patch; reverse-patch check exact):

```python
def block_bootstrap(P, y, day, B: int = 1000, seed: int = 0, P_ref=None, alpha: float = 0.05) -> dict
```

- **Arguments:** `seed` and `alpha` are supported keyword arguments. The script calls `block_bootstrap(P, y, day, B=B, seed=SEED, P_ref=Pref, alpha=ALPHA)`.
- **Random generator:** `rng = np.random.default_rng(seed)` is created **inside every call**, so it resets on each call.
  - Each replicate draws `rng.choice(days, len(days), replace=True)` exactly once, with `days = np.unique(day)`.
  - Rows are concatenated in the sampled day order.
- **Return value:** `{"n_days": int, "B": int, "ci": {macro_f1, auc_ovr, brier, ece, accuracy: [lo, hi]}}`. When `P_ref` is given, it also contains `"diff_vs_ref_ci": {the same 5 keys: [lo, hi]}` for (P − P_ref) on the same resample. Percentiles come from `np.nanpercentile`.
- **No log loss** is included, which is why the supplement exists.
- **`logloss_bootstrap`** reproduces the same steps: `default_rng(0)`, the same `np.unique(day)`, the same `by_day` index lists, one `rng.choice` per replicate, and the same concatenation order.
  - It now also uses `np.nanpercentile` (changed from `np.percentile` in the final review, for exact parity).
  - It recomputes Brier on each replicate and asserts its CI equals `block_bootstrap`'s Brier CI to 1e-12. Brier is computed as row-wise `((P−Y)²).sum(1)` means over the same indices.

## 5. One-day streaming cross-check (G8)

**G8 is BLOCKING.** If streaming and batch probabilities disagree, the run stops before any metrics are computed. The full G8 diagnostics are kept in `d5_gate_failure.json`:
- day, bars, events
- coverage, max/median error and the share of errors above 1e-6 for each mode and horizon
- non-finite live rows

The check is not bypassed, and model behaviour is never altered to force agreement.

- **Day:** the **first validation day** (fixed rule, day index 82), all 10 symbols, every bar of that day. That includes warm-up bars, which the Predictor needs. Only `split=="val"` rows are replayed.
- **Bar order:** `bars_sorted = bars.sort_values(["symbol","timestamp"])`, asserted to be row-aligned with `ds.table`. `mret = market_return(bars_sorted)`.
- **Mechanism:** `replay(Predictor(bundle, mode), …)` for A (no news) and D (with news). This is the same mechanism as `tests/test_predictor.py`.
- **News events:**
  - Only events up to the end of that day are passed, and `replay` pushes each event strictly before the bar it precedes.
  - Raw event vectors, including the history features, come from the batch event table, as in the existing test.
  - `event_memory.npz` is **not** used: the history features are already inside the event vectors, and the memory only serves the live slow path.
- **Pass thresholds** (copied from the existing tests):
  - D: coverage ≥ 0.9, median max-abs error < 1e-9, and fewer than 0.2% of bars with error > 1e-6
  - A: coverage ≥ 0.9 and max error < 1e-8
- **On failure:** abort before metrics and document. Nothing is forced to match.
- **Scope:** this check verifies implementation consistency, not performance. Its numbers are not used for evaluation.

## 6. Static review findings

### Searches for potentially dangerous calls

| Term | Occurrences | Verdict |
|---|---|---|
| `fit` | `BL.Baselines().fit(t, h)` (line 437). The rest are comments, strings, `describe()` keys and `fit_on_train`. | Baselines.fit is the D3 procedure: class counts and momentum tercile counts on **train labels only**, not a trainable FLASH model. **Not visible in this file:** `prepare()` internally refits RegimePCA/GMM/HMM on train, with val used for n_components, exactly as in D4. This refit is in-memory and is gated by G4 against the bundle. Nothing is saved. |
| `fit_transform` | none | — |
| `predict` | `Booster.predict(Xv)` (val rows), `bl.predict(t, va)` (val rows), and `replay(Predictor…)` (one val day), which calls `Predictor.on_bar`. The rest are comments and strings. | Validation only. |
| `predict_proba` | only in a comment | — |
| `score` | only in strings | `experiments.score` is **not** imported. |
| `evaluate` | none called directly. `full_report` and `block_bootstrap` call `src.evaluation.metrics.evaluate` internally. | These are the metric functions only, called on val arrays. |
| `train` | the strings "train"/"val" in the guards and in `split_days` counts. No training functions. | `train.py`, `fit_variant`, `FusionBlend`, `BaselineLogistic`, `LGBMClassifier` and `Calibrator` are **not imported**. |
| `test` | comments and strings only | No `rows("test")`, no `split=="test"` selection, and no test day counts (removed during review). |
| `FLASH_TEST_UNLOCK` | read only, at line 389: aborts if set | Never written. No `--unlock-test`; `check_split_allowed` is not used. |
| `Calibrator` | none | No calibration is fitted. |

### Writes

**Completion criterion.** A run is complete **only** if all of the following hold:
- the exit code is 0
- `d5_outputs.sha256` exists
- `"status": "complete"` appears in both `d5_run_metadata.json` and `d5_validation_metrics.json`

`status` is set to `complete` only after:
- every metric and bootstrap has finished
- the final D4-artifact re-check (G2) passes
- the FinBERT-cache re-check passes; a changed cache raises an error

Any other outcome produces `d5_gate_failure.json`. Nothing is ever deleted or cleaned up: there are no `unlink`, `remove` or `rmtree` calls. Partial outputs are kept and labelled as partial.

There are exactly five write sites, all under `reports/data_prep/stage_d5_validation_evaluation/`:
- `np.savez_compressed(OUT_PROBS)`
- `OUT_FAIL.write_text`
- `OUT_METRICS.write_text`
- `OUT_META.write_text`
- `OUT_SHA.write_text`

There are no `mkdir`, `joblib.dump` or `resolve()` calls (`resolve()` would create directories), and no writes to `models/`, `models_real/`, D3 or D4 directories. D4 artifacts are opened read-only (`joblib.load`, `read_text`, hashing).

### Other checks
- **Syntax:** `ast.parse` and `compile` pass. An AST undefined-name scan found only `__file__`, which is a module global.
- **Test-split guards:** `_NoTestDataset.rows` and a runtime wrapper on `evaluation.baselines.rows` both raise for any split other than train/val.
  - The wrapper replaces the module attribute in memory only; the source file is untouched.
  - `Baselines.fit` resolves `rows` from the module globals, so it is guarded too.

## 7. Test-isolation safeguards

1. Two runtime guards raise on any non-train/val row request (Dataset and baselines).
2. Selection is explicitly `split=="val"` & `valid_x` & label ≥ 0. G5 asserts that every selected index is val, inside the val day range, over 27 days, and that row counts and supports equal the expected values.
3. The env guard aborts if `FLASH_TEST_UNLOCK` is set. There is no unlock path.
4. No test-capable entry points are imported (`fit_variant`, `score`, `run_experiments`, `benchmark`, `demo`, `train.py`, `check_split_allowed`).
5. The streaming replay covers one validation day only, with events ≤ that day's end.
6. Outputs contain validation rows only (`d5_val_probabilities.npz`) plus aggregates.
7. **Accepted in-memory scope:** `prepare()` and `past_return_z` (inside `Baselines`) compute over the full chronological table in memory, including test-period rows. This is the same as D3 and D4 and was accepted in D4. No test-period row is predicted, calibrated, scored or written.

## 8. Unresolved technical concerns

| # | Concern | Handling |
|---|---|---|
| C1 | **G4 could fail** if the PCA/GMM/HMM refit is not bit-stable (threaded BLAS). | The run aborts before metrics. A fallback (posteriors from the bundle's exported PCA/HMM) would need separate approval. |
| C2 | **G8 could fail** if `XF[val]` has NaNs in D features: `Predictor` has no `nan_to_num`, unlike training. | **Blocking, as confirmed by the user.** The run aborts, and G6 records the NaN counts so the cause is identifiable. |
| C3 | **G7 could fail** if `prepare()`'s table differs from D3's `build_table(load_market(...))`. Both use the same loader and function, so equality is expected. | The run aborts. |
| C4 | **Runtime (estimate):** D3 took about 63 min for 6 paired and 3 unpaired bootstraps. D5 has 21 paired bootstraps, so roughly **3 h** in total, plus about 3 min for `prepare()`. | Run in the background with a log. Probabilities are saved before the bootstrap. |
| C5 | Shock/calm subgroup metrics (proposed in the preflight) are **not** implemented, because they were not requested in this stage. | They can be added later if wanted. |
| C6 | Standalone baseline bootstrap CIs are taken from D3, not recomputed. | Identical point metrics are verified by G7. |
| C7 | Only one streaming day (the first val day) is checked. | It is a consistency spot-check, not exhaustive. |

## 9. Interpretation restrictions (these will be written into the final D5 report)

1. D4 probabilities are **uncalibrated** (identity calibrator). Brier, log loss, ECE and reliability are labelled accordingly.
2. Validation was already used for PCA component selection and for LightGBM early stopping in model D.
3. The D3 baselines were also evaluated on validation.
4. D5 is therefore **not an independent generalisation assessment**.
5. Model D differs from model A in model family, feature count (42 vs 8), feature composition (market, regime, news, history and derived features) and blending.
6. A D-over-A improvement **cannot** establish that news features add predictive value.
7. Isolating the news contribution would need a separately authorised ablation (for example, variant C).
8. The test split remains locked and unexamined.

## 10. Proposed execution command (only after explicit approval)

```
cd C:\Users\prsan\Projects\FLASH
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe reports/data_prep/stage_d5_validation_evaluation/d5_validation_evaluation.py \
    > reports/data_prep/stage_d5_validation_evaluation/d5_validation_evaluation.log 2>&1
```

**Expected outputs if all gates pass:**
- `d5_val_probabilities.npz`
- `d5_validation_metrics.json`
- `d5_run_metadata.json`
- `d5_outputs.sha256`
- `d5_validation_evaluation.log`

**On a gate failure:** `d5_gate_failure.json` plus the log, exit code 2, and no metrics.

After the run, I will write `STAGE_D5_VALIDATION_EVALUATION_REPORT.md`.

## 11. Final read-only review: changes (script still NOT executed)

| # | Issue found | Correction (minimal) |
|---|---|---|
| F1 | A non-gate error would end the run without writing a failure record. Such errors include a guard `RuntimeError`, the alignment checks (previously `SystemExit`, which `except` does not catch), the final G2 re-check, or a bootstrap crash. After `d5_val_probabilities.npz` had been saved, a failed run could then look like partial evidence of completion. | Both phases are wrapped in `except Exception` → `_fail()`, which writes `d5_gate_failure.json` with status, type, message, traceback and diagnostics, and exits with code 2, 3 or 4. The alignment exits now raise `RuntimeError`. A `status` field (`incomplete` → `complete`) is set only at the very end. A FinBERT-cache change at the end now raises an error instead of only being recorded. Nothing is deleted. |
| F2 | G8's `ok` flags were `numpy.bool_`, which would be serialised to JSON as the string `"True"`. | Wrapped in `bool()`. Logic unchanged. |
| F3 | `logloss_bootstrap` used `np.percentile` while `block_bootstrap` uses `np.nanpercentile`. The results are identical when there are no NaNs. | Switched to `np.nanpercentile` for exact parity. |

**Unchanged and confirmed:**
- The `block_bootstrap` signature, keyword arguments, return keys, and per-call reset of the random generator (§4).
- Paired comparisons: A vs majority, prior and momentum; D vs majority, prior, momentum and A. That is 7 × 3 horizons = 21. `n_cmp` counts one per reference per model per horizon, giving 21.
- All arrays are indexed by the same `va`. Every comparison re-seeds with 0, so all comparisons share the identical 1000 day resamples.
- `metrics_per_comparison = 6`: the 5 `diff_vs_ref_ci` keys plus log loss.
