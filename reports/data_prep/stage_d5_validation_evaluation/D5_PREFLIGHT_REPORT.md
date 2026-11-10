# Stage D5 Preflight: Validation-Only Evaluation of the D4 Models

- **Date:** 2026-11-09
- **Repo:** `C:\Users\prsan\Projects\FLASH`
- **Branch:** `data-pipeline`
- **HEAD:** `ffa1cda72861d22ac9f2ec71e029858e582cc9d3`
- **Status:** planning only. No evaluation script was created or run. No predictions or metrics were computed, no models were loaded for prediction, and the test split was not touched.

Each finding below is marked **[verified]** (read in code or artifacts, with a reference), **[proposed]** (the plan) or **[open]** (uncertain, needs a decision or a runtime check).

---

## 1. Objective and scope

The goal is to evaluate the six D4 models on the **validation split only** and compare them with the D3 validation baselines:
- Model A: standalone L2 logistic regression on price features.
- Model D: `FusionBlend`, i.e. LightGBM blended with an L2 logistic regression on all features.
- Horizons: 5, 15 and 30 bars.

**Out of scope:**
- test-split access of any kind
- training or retraining
- fitting or changing calibration
- threshold or gate optimisation
- model selection decisions
- running `scripts/benchmark.py`, `scripts/demo.py` or `scripts/train.py`
- source, configuration or dependency changes
- D6

## 2. D4 artifact compatibility findings

| Item | Finding |
|---|---|
| Bundle location | `models_real/stage_d4/bundle.joblib` (sha `fd91c20e…`), `event_memory.npz` (sha `9f2d2e39…`). Hashes re-verified in §13. **[verified]** |
| Default loaders | `load_bundle(cfg)` / `load_memory(cfg)` read `resolve(cfg,"models")/…` = `models_real/bundle.joblib`, **not** `stage_d4/`. `resolve()` also creates directories ([predictor.py:207-219](../../../src/inference/predictor.py), [config.py](../../../src/config.py)). **Do not use them.** **[verified]** |
| Manual loading | `bundle = joblib.load("models_real/stage_d4/bundle.joblib")`. This is a trusted local pickle written by D4. **[verified in D4]** |
| Bundle contents | Keys `A, D, cfg, col_names (42), groups, pca, hmm, horizons [5,15,30], stage_d4`. `D[h] = {model (LightGBM text truncated at best_iteration), cal, cols = range(42), logit {W 3×42, b, w = 0.25}}`. `A[h] = {W 3×8, b, cal}`. Every `cal` = `{"method":"temperature","maps":[1.0]}`. **[verified in D4]** |
| `Predictor` | `Predictor(bundle,"A"/"D")` constructs. It is a **streaming, per-bar** engine (`on_bar`), however, so it is not suitable for batch evaluation of about 90k rows per horizon. **[verified]** |
| Batch reconstruction | The bundle holds everything needed for **batch** probabilities: LightGBM booster from `D[h]["model"]` via `load_booster` ([fusion_lgbm.py:38-39](../../../src/models/fusion_lgbm.py)), the logit leg `W,b,w`, and A's `W,b`. The planned formulas are in §4.3. **[verified]** |
| LightGBM best iteration | `model_string()` saves only `num_iteration=best_iteration` ([fusion_lgbm.py:31-32](../../../src/models/fusion_lgbm.py)), so `Booster.predict` on the stored string uses exactly the early-stopped trees. **[verified]** |
| Class ordering | Labels are `DOWN, NEUTRAL, UP = 0, 1, 2` ([market_features.py:31-32](../../../src/features/market_features.py)). All training sets contained all 3 classes (D4 metadata), so scikit-learn `classes_ = [0,1,2]` and LightGBM multiclass columns are 0..2. Probability column *i* is therefore class *i*, matching `CLASS_NAMES = ["DOWN","NEUTRAL","UP"]` in metrics ([metrics.py:7](../../../src/evaluation/metrics.py)). **[verified]** Runtime assertion still proposed (§9). |
| Event memory | **Not needed for batch evaluation.** The history features (`h_mean`, `h_std`, `h_sim`) that models consume are rebuilt inside `prepare()` → `build_events` → `history_features` from causally known events. `event_memory.npz` only serves the live slow path. **Exclude it from D5.** **[verified]** |

## 3. Validation data selection and label mapping

**Features**
- Batch features must be rebuilt with the **same, unmodified** `prepare()` used in D4 (`get_data` → `prepare(cfg, bars, news, encoder, finbert_cache)`), using the cache-only FinBERT guard.
- As in D4 (and as accepted there), `prepare()` builds the **full chronological table in memory**, including test-period rows. **[verified]**

**Row selection** (`Dataset.rows("val", h)`, [experiments.py:85-88](../../../src/evaluation/experiments.py)):
- rows where `split == "val"`, `valid_x` is true, and `y{h} >= 0`
- labels are `y{h}`, with −1 meaning no label (horizon past session end, or no sigma)

**Labels**
- Labels are triple-barrier labels per symbol-day ([market_features.py:101-120](../../../src/features/market_features.py)).
- UP if the upper barrier `close·exp(+k·σ·√h)` is hit first within h bars, DOWN if the lower one is, NEUTRAL otherwise, with k = 1.0. **[verified]**

**Split**
- `split_days` splits all **136** trading days chronologically ([align.py:100-107](../../../src/data/align.py)). 510,000 bars ÷ (10 symbols × 375) = 136.
  - train = round(0.6·136) = 82 days (2024-06-18..10-14)
  - val = round(0.2·136) = 27 days (10-15..11-25)
  - test = the remaining 27 days (11-26..2025-01-02)
  - This matches D1/D2. **[verified]**
- The last val day has day index 108 (82 + 27 − 1), matching D4's event-memory cutoff tmin 40874 = 108·375 + 374.
- Validation day-level blocks: 27 days, the same as D3.
- **Erratum:** the D4 report (§3.1) says the split is "derived from all 109 trading days". The correct total is 136; 109 is train + val. The D4 report is left unchanged as a D4 artifact; this note corrects it.

**Expected validation row counts** (equal to D3's `n` and to D4's early-stopping `n_val`):

| h | val rows |
|---|---|
| 5 | 91,800 |
| 15 | 89,100 |
| 30 | 85,050 |

D5 will assert these counts. **[verified from D3 JSON and D4 metadata]**

**Validation class supports** (D3): DOWN / NEUTRAL / UP

| h | DOWN | NEUTRAL | UP |
|---|---|---|---|
| 5 | 17,581 | 56,517 | 17,702 |
| 15 | 19,697 | 49,554 | 19,849 |
| 30 | 20,006 | 45,035 | 20,009 |

All three classes are present at every horizon, so OvR AUC is defined.

## 4. Proposed evaluation methodology [proposed]

### 4.1 Pipeline
1. **Guards.** Abort if `FLASH_TEST_UNLOCK` is set. Abort if any output path exists. Use a `_NoTestDataset` subclass whose `rows()` raises for anything other than `"train"` / `"val"`.
2. **Integrity before starting.** Hash `bundle.joblib` and abort on mismatch with D4 (`fd91c20e…`). Hash the FinBERT cache (`43447937…`) before and after `prepare()`.
3. **Load data.** `get_data(cfg, "data/raw/market", "data/raw/news.csv")`.
4. **Rebuild features.** `prepare(...)` with the cache-only encoder.
5. **Reproducibility gate.** The refitted `ds.regime.pca.export()` and `ds.regime.hmm.export()` must equal the bundle's `pca` and `hmm` (`np.allclose`, rtol = 1e-9). `ds.col_names` must equal `bundle["col_names"]`. PCA uses a full solver, the GMM/HMM are seeded (`gmm_seed = 7`), and PCA, GMM and HMM all refit on the identical train rows, so equality is expected.
   - **If the gate fails, abort without computing metrics and report.** A mismatch would mean the regime features differ from those the models were trained on.
6. **Predict per horizon.** Use `va = ds.rows("val", h)` and `X = ds.XF[va]`, then compute A and D probabilities (§4.3). No other split is ever indexed.
7. **Validate the probabilities.** All values finite, each in [0,1], each row summing to 1 within 1e-6, shape (n,3). Abort on failure.
8. **Compute metrics** with the existing `full_report` and `block_bootstrap` (§5, §6).
9. **Write outputs** (§10).

### 4.2 Calibration handling
- Raw probabilities are used, i.e. after the identity calibrator, which is mathematically a no-op.
- The driver does **not** construct `Calibrator`, and no temperature, threshold or gate is fitted or changed.
- The `Predictor` confidence gate (0.5) is **not** applied. Metrics use the argmax and the full probability vectors, exactly as D3 did.

### 4.3 Batch probability formulas (mirroring the training-time code)

- **A:** `p = softmax(X[:, cols_A] @ W_A.T + b_A)`.
  - `W_A`, `b_A` already fold in the train-fitted `StandardScaler` (`BaselineLogistic.export`, [baseline_logistic.py:24-28](../../../src/models/baseline_logistic.py)).
  - `cols_A = variant_cols(ds,"A")`, i.e. the 8 price features, and must equal the D4 metadata feature list.
  - Mathematically identical to `BaselineLogistic.predict_proba`.
- **D:** `P = (1−w)·Booster(D[h]["model"]).predict(X) + w·softmax(nan_to_num(X) @ W.T + b)`, with w = 0.25.
  - This mirrors `FusionBlend.predict_proba` ([fusion_lgbm.py:59-63](../../../src/models/fusion_lgbm.py)), including `nan_to_num` on the logit leg as in training.
  - **[open]** The streaming `Predictor` does *not* apply `nan_to_num` (an existing quirk). D5 will count NaNs in `XF[val]` and report them. If there are none, the two paths coincide.
- **Optional cross-check [proposed, needs approval].** Replay one or two **validation** days through the streaming `Predictor` (`replay()`) and compare its outputs with the batch probabilities (median absolute difference below 1e-9), the same check `tests/test_predictor.py` does on synthetic data. This confirms the batch reconstruction matches the deployed inference path. It processes validation bars only. The cost is small, but it does need `news` event plumbing (the `events` argument), so it may be skipped without loss of the main results.

### 4.4 Subgroup analysis [proposed]

Metrics are also split into two subgroups, using the existing `Dataset.shock()` definition ([experiments.py:91-93](../../../src/evaluation/experiments.py)):
- **news-shock:** latest usable news ≤ 30 min old
- **calm:** no recent news

This shows where D's news features could matter. Subgroup sizes are reported; AUC is NaN-guarded if a class is missing (`evaluate` already does this).

## 5. Metric definitions

All metrics use the existing, unmodified implementations in `src/evaluation/metrics.py` and `evaluation/metrics_ext.py` (the same code as D3). y ∈ {0,1,2}, P is n×3, and pred = argmax P.

| # | Metric | Definition (as implemented) | Source |
|---|---|---|---|
| 1 | Accuracy | mean(pred == y) | `evaluate` |
| 2 | Macro-F1 | sklearn `f1_score(average="macro", labels=[0,1,2])` | `evaluate` |
| 3–4 | Per-class precision / recall / F1 / support | `precision_recall_fscore_support(labels=[0,1,2], zero_division=0)`. A never-predicted class gets precision 0 (not undefined). | `full_report` |
| 5 | OvR ROC-AUC | `roc_auc_score(multi_class="ovr", average="macro", labels=[0,1,2])`; NaN if a class is absent | `evaluate` |
| 6 | Multiclass Brier | mean over rows of Σ_c (P_c − 1[y=c])²; range [0,2] | `brier_multiclass` |
| 7 | Log loss | −mean log(clip(P[y], 1e-12, 1)) | `full_report` |
| 8 | Prediction distribution | counts and shares of argmax ∈ {DOWN, NEUTRAL, UP}, plus the mean predicted probability per class | **new aggregation in the D5 driver** (simple bincount and mean, not a new metric definition) |
| 9 | Confusion matrix | `confusion_matrix(y, pred, labels=[0,1,2])`, rows = true, columns = predicted | `conf_matrix` |
| + | ECE (10 bins, top-class) and per-class reliability curves | reported for parity with D3 | `ece`, `full_report` |
| + | Day-block bootstrap 95% CIs | B = 1000, seed 0, 27 val days; for macro-F1, AUC, Brier, ECE and accuracy (no CI for log loss, as in D3) | `block_bootstrap` |

**Brier, log loss and ECE will be labelled "uncalibrated probabilities"** everywhere in the outputs.

## 6. D3 baseline comparison methodology [proposed]

### Comparability [verified]

| Aspect | D3 | D5 |
|---|---|---|
| Table | `build_table(bars, cfg)` | the same `build_table` (inside `prepare`) |
| Row filter | `split=="val" & valid_x & y≥0` | the same |
| `n` per horizon | identical (§3) | identical (§3) |
| Metric code | `full_report` / `block_bootstrap` (B = 1000, seed 0) | the same code |

Point metrics are therefore directly comparable.

### Paired differences [open: needs a decision]

D3 saved only aggregate metrics, not per-row baseline probabilities. A **paired** bootstrap CI of (model − baseline), resampling the same days for both, needs the baseline probabilities per row.

- **Option P1 (recommended).** Recompute the three baselines in the D5 driver with the unmodified `evaluation/baselines.Baselines().fit(table, h)` and `.predict(table, val_rows)`.
  - These are deterministic counts on train labels, the same code as D3, not a new predictive model.
  - Assert that their `full_report` point metrics equal the D3 JSON (tolerance 1e-12) before using them.
  - Then compute `block_bootstrap(P_model, y, day, P_ref=P_baseline)` against **prior** (the strongest calibrated constant) and **momentum**: two reference calls per model and horizon.
- **Option P2.** Compare against the D3 JSON point values and CIs only (unpaired). No baseline recomputation, but CI overlap is a weak, conservative test.

`block_bootstrap` accepts one `P_ref` per call, so P1 runs it twice per model and horizon.

**Bars to beat** (from D3, h = 5 / 15 / 30):

| Metric | h = 5 | h = 15 | h = 30 |
|---|---|---|---|
| Brier < | 0.5466 | 0.5925 | 0.6089 (momentum) |
| Log loss < | 0.9319 | 0.9949 | 1.0176 (momentum) |
| AUC > | 0.519 | 0.508 | 0.512 (momentum) |
| Macro-F1 > | 0.254 | 0.238 | 0.231 (every baseline predicts all-NEUTRAL) |
| Accuracy (all-NEUTRAL reference) | 0.616 | 0.556 | 0.530 |

### Interpretation rules (fixed now, before seeing results)

- **"Improves over baselines"** requires the paired bootstrap CI of (model − baseline) to exclude 0 in the favourable direction for Brier or log-loss-equivalent probabilistic metrics, *and* for AUC. Brier is the primary metric, AUC secondary.
- **Accuracy and macro-F1 are secondary.** Accuracy rewards predicting NEUTRAL.
- **NEUTRAL concentration** is flagged if the argmax NEUTRAL share is ≥ 95%, or if UP/DOWN recall is below 5%.
- **Detection of UP and DOWN** is assessed with per-class recall and precision, plus the one-vs-rest reliability curves.

### Model A vs Model D [proposed, with the stated limitation]

- Report A and D side by side, with a paired bootstrap of D − A (`P_ref = P_A`).
- **D > A does NOT show that news adds predictive value.** D differs from A in several ways at once:
  1. model class (gradient-boosted trees vs linear)
  2. features: 42 vs 8. D adds cross-sectional market features (`cov20`, `beta20`, `mkt_ret1`, `mkt_vol20`, `vol_z`, `tod`), HMM regime posteriors, news features, history features and derived interactions.
  3. the logistic blend leg
  4. early stopping on validation (A uses no validation)
- Any D − A gap mixes all four factors. Isolating news needs the existing variant **C** (LightGBM on market + derived + regime, no news) or a news ablation. Those require **training new models**, which is outside D5's authorisation. They are listed as a future option only.
- The shock/calm subgroup split (§4.4) is descriptive evidence, not causal evidence.

## 7. Validation-bias limitations (mandatory caveat)

Validation data has already been used for:
1. `RegimePCA` component selection (n_components = 4 on val explained variance; D2 caveat L8)
2. LightGBM early stopping for all three D models (best_iter 174 / 121 / 107 chosen on val multi_logloss)
3. D3 baseline evaluation and its review
4. planning and review discussions that have seen D3 val numbers

**D5 validation metrics are therefore not an independent estimate of generalisation performance.** D in particular is optimistically biased, because its tree count was selected on the same rows. A is less directly affected: it uses no validation data in fitting, and its 8 price features do not include the val-selected PCA/HMM regime columns. A's results are still not independent, because validation has been inspected in D3 and in this planning. Only D's regime columns carry the PCA selection bias.

Further caveats:
- The 27 validation days come from one six-week window (2024-10-15 .. 11-25), so bootstrap CIs reflect day-to-day variation only, not regime change.
- D5 results must not be described as final, unbiased or test performance anywhere in the outputs. The test split stays locked and will not be used to resolve this.

## 8. Probability calibration limitations

- Every D4 calibrator is the identity, `{"method":"temperature","maps":[1.0]}`, so the probabilities are **uncalibrated**.
- Brier, log loss, ECE and the reliability curves therefore measure raw model probabilities and will be labelled that way. They may look worse than a calibrated model would.
- No calibrator will be fitted, and no thresholds or gates optimised. Any calibration decision, and any validation re-use it implies, belongs to a later, separately approved stage.
- The D3 prior baseline is near-perfectly calibrated by construction (ECE ≈ 0.01), so Brier comparisons against it are a demanding bar for uncalibrated models.

## 9. Test-set isolation strategy [proposed]

1. **No test indexing.** The driver calls `ds.rows()` only with `"val"`, plus `"train"` if baseline recomputation (P1) is approved, because `Baselines.fit` uses train labels. `_NoTestDataset.rows()` raises on any other split. The baselines module's own `rows()` is only called with `"train"`/`"val"`.
2. **No test-related imports.** No import of `fit_variant`, `score`, `run_experiments`, `scripts/benchmark.py`, `scripts/demo.py`, `scripts/train.py` or `check_split_allowed`.
3. **Environment guard.** Abort if `FLASH_TEST_UNLOCK` is set. The variable is never set or modified.
4. **Assertions on indices.**
   - The predicted indices are a subset of `split == "val"` rows; this is asserted.
   - The count equals the expected n.
   - The max day index of the predicted rows is ≤ the last val day.
5. **Static review before execution.** I'll grep the driver for `test`, `predict`, `rows(` and `Calibrator` and show the result before asking to run.
6. **Accepted in-memory scope (as in D4).** `prepare()` computes features, labels and HMM posteriors for test-period rows in memory. None of these are predicted on, scored or written.
7. **Output check.** No output file will contain test-split rows. Only aggregates and validation probabilities are written.

## 10. Proposed script structure and output files [proposed, not created]

Driver: `reports/data_prep/stage_d5_validation_evaluation/d5_val_eval_driver.py`

```
guards (env, no-overwrite, bundle sha == D4, FinBERT cache sha)
load cfg (configs/real.yaml) + bundle (joblib.load, stage_d4 path)
get_data -> prepare(cache-only encoder) -> ds (+ _NoTestDataset)
reproducibility gate: pca/hmm exports, col_names, variant_cols(A/D) == bundle/D4 metadata
for h in (5, 15, 30):
    va = ds.rows("val", h); assert len(va) == expected[h]; y = ds.y(va, h); day = table.day_idx[va]
    P_A = softmax(XF[va][:, colsA] @ W_A.T + b_A)
    P_D = 0.75 * Booster(D[h].model).predict(XF[va]) + 0.25 * softmax(nan_to_num(XF[va]) @ W.T + b)
    validate probabilities; count NaNs in XF[va]
    [P1] baselines = Baselines().fit(table, h).predict(table, va); assert == D3 JSON point metrics
    for each model: full_report + prediction distribution + block_bootstrap
                    (+ paired diffs vs prior, vs momentum, and D vs A)
    shock/calm subgroup evaluate()
write JSON / log / probabilities / sha256 (LF line endings)
```

Planned outputs, all new and all in `reports/data_prep/stage_d5_validation_evaluation/`:

| File | Content |
|---|---|
| `d5_val_eval_driver.py` | the driver (written only after approval of this plan) |
| `d5_validation_metrics.json` | all metrics by model and horizon, CIs, paired diffs, distributions, subgroup results, gate results |
| `d5_validation_eval.log` | stdout/stderr, command, timestamps, exit code |
| `d5_val_probabilities.npz` | **optional:** per-row val probabilities for A and D plus y, day and row ids, for reproducibility; no test rows |
| `d5_outputs.sha256` | checksums, written with `newline="\n"` (avoids the D4 CRLF issue) |
| `STAGE_D5_VALIDATION_EVALUATION_REPORT.md` | final report, written after the run |

No files are written to `models_real/`, `models/`, `reports/real/` or `reports/baselines/`.

**Estimated runtime [estimate, not measured]:**
- about 2.5 min for `prepare()` (measured in D4) plus seconds for prediction
- bootstrap dominates: D3 took about 63 min for 9 model-horizon pairs with one reference each
- D5 with P1 has 6 model-horizon pairs × (1 + 2 baseline refs + D-vs-A) bootstrap passes
- total roughly **1.5–2.5 h**; B could be reduced if that is preferred

## 11. Risks and unresolved questions

| # | Item | Status |
|---|---|---|
| R1 | **The reproducibility gate could fail** if PCA, GMM or HMM refitting is not bit-stable (threaded BLAS). Mitigation: abort before metrics. The fallback would compute regime posteriors directly from the bundle's exported PCA and HMM (`ForwardFilter`), which needs approval. | open |
| R2 | NaNs in `XF[val]`: `nan_to_num` is used in the D logit leg in training but not in `Predictor`. D5 follows training semantics and reports the NaN count. | open (runtime check) |
| R3 | Paired baseline comparison needs baseline recomputation (P1) vs D3-JSON-only (P2). | **decision needed** |
| R4 | Optional streaming-`Predictor` cross-check on 1–2 validation days. | **decision needed** |
| R5 | Whether to save per-row validation probabilities (`.npz`, about 3–5 MB). | **decision needed** |
| R6 | Bootstrap B (1000, like D3) vs a smaller B for runtime. | **decision needed** (default 1000) |
| R7 | Split arithmetic is consistent (136 days → 82/27/27). The D4 report's "109 trading days" wording is an erratum (§3). D5 will also log the per-split day counts. | resolved (documented) |
| R8 | The A vs D comparison cannot isolate news value (§6). Isolating it needs variant C / ablation training, which is not authorised. | documented limitation |
| R9 | Validation re-use bias (§7); uncalibrated probabilities (§8). | documented limitation |
| R10 | `LGBMDeprecationWarning` (`eval_set`) is irrelevant for prediction; `Booster.predict` is unaffected. | none |
| R11 | Loading `bundle.joblib` executes pickle. It was produced locally by D4 and its sha is checked before load. | mitigated |

## 12. Exact execution command (only after approval of this plan and of the driver)

```
cd C:\Users\prsan\Projects\FLASH
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe reports/data_prep/stage_d5_validation_evaluation/d5_val_eval_driver.py \
    > reports/data_prep/stage_d5_validation_evaluation/d5_validation_eval.log 2>&1
```

`FLASH_TEST_UNLOCK` must remain unset. The run will be in the background, with the exit code recorded.

## 13. Preflight integrity state (at report time)

See the accompanying chat summary for the verification output. This report is the only new file, and `reports/data_prep/stage_d5_validation_evaluation/` is the only new directory.

---

**Stopped after preflight.** Nothing has been written beyond this report: no driver, no predictions, no metrics. Awaiting approval of the plan and decisions on R3–R6.
